"""Fair pairwise-baseline vs pairwise/multi-phase MotionFiLM training.

All modes use the same block split and preprocessing.

baseline:
    Independent pairwise samples. Nine micro-batches are accumulated before
    each optimizer update, so every update sees the same number of pairs as
    MotionFiLM.

pairwise_motionfilm:
    The same independent pairwise samples and nine-micro-batch accumulation as
    baseline, with image-conditioned MotionEncoder + FiLM enabled. No phase ID
    or multi-phase trajectory loss is used.

motionfilm:
    One sample contains one phase-0 fixed image and phase-1..9 moving images.
    Registration losses are averaged over phases and coupled by motion-code
    and low-resolution DVF trajectory losses.

Validation is always performed on the same flattened pairwise validation set.
The test set is evaluated once, after loading the best validation checkpoint.
"""
import csv
import os
import random
from argparse import ArgumentParser

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from ldm.data.xcat_multiphase import (
    MultiPhaseDataset,
    PairwisePhaseDataset,
    collate_multiphase,
    collate_pairwise,
)
from ldm.util import instantiate_from_config
from TransModels.LDMMorph import LDMMorph
from utils.utils import (
    MSE,
    SpatialTransform,
    bending_energy_loss,
    jacobian_determinant_vxm,
    jacobian_neg_loss,
    smoothloss,
)


SEED = 42
NUM_PHASES = 9


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = ArgumentParser()
parser.add_argument("--ldm_ckpt", "--resume", dest="ldm_ckpt", required=True)
parser.add_argument("--ldm_config", required=True)
parser.add_argument(
    "--data_root",
    default="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
)
parser.add_argument(
    "--cond_mode",
    choices=["baseline", "pairwise_motionfilm", "motionfilm"],
    default="motionfilm",
)
parser.add_argument(
    "--no_phase_id",
    action="store_true",
    help="Disable phase embedding; MotionFiLM only uses MotionEncoder(moving, fixed).",
)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--iteration", type=int, default=24001)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--smth_labda", type=float, default=0.4)
parser.add_argument("--bending_w", type=float, default=0.0)
parser.add_argument("--jacdet_w", type=float, default=0.0)
parser.add_argument("--lambda_z_acc", type=float, default=0.005)
parser.add_argument("--lambda_dvf_acc", type=float, default=0.001)
parser.add_argument("--alpha_motion_gap", type=float, default=2.0)
parser.add_argument("--loss_type", choices=["ncc", "mse"], default="ncc")
parser.add_argument("--fg_thr", type=float, default=0.05)
parser.add_argument("--t_enc", type=int, default=1)
parser.add_argument("--checkpoint", type=int, default=2500)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--vis_interval", type=int, default=1000)
parser.add_argument("--no_ldm", action="store_true")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument(
    "--save_dir",
    default="./logs/Fair_MultiPhase_Ablation",
)


def load_ldm(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[LDM] restored {ckpt_path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def body_mask(image, threshold=0.05):
    from scipy.ndimage import binary_fill_holes, label

    array = image.detach().cpu().numpy()
    output = np.zeros_like(array, dtype=np.float32)
    for batch_index in range(array.shape[0]):
        mask = array[batch_index, 0] > threshold
        mask = binary_fill_holes(mask)
        labels, count = label(mask)
        if count > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == int(sizes.argmax())
        output[batch_index, 0] = mask.astype(np.float32)
    return torch.from_numpy(output).to(image.device)


def masked_mse(prediction, target, mask):
    numerator = (((prediction - target) ** 2) * mask).sum()
    return numerator / mask.sum().clamp(min=1.0)


def ncc_loss(fixed, moving, win_size=15, mask=None):
    if fixed.shape != moving.shape:
        raise ValueError(f"NCC shape mismatch: {fixed.shape} vs {moving.shape}")
    pad = win_size // 2
    batch, channels, height, width = fixed.shape
    fixed_pad = F.pad(fixed, [pad] * 4, mode="reflect")
    moving_pad = F.pad(moving, [pad] * 4, mode="reflect")
    fixed_patch = (
        fixed_pad.unfold(2, win_size, 1)
        .unfold(3, win_size, 1)
        .contiguous()
        .view(batch, channels, height, width, -1)
    )
    moving_patch = (
        moving_pad.unfold(2, win_size, 1)
        .unfold(3, win_size, 1)
        .contiguous()
        .view(batch, channels, height, width, -1)
    )
    fixed_centered = fixed_patch - fixed_patch.mean(dim=-1, keepdim=True)
    moving_centered = moving_patch - moving_patch.mean(dim=-1, keepdim=True)
    covariance = (fixed_centered * moving_centered).mean(dim=-1)
    fixed_var = fixed_centered.square().mean(dim=-1)
    moving_var = moving_centered.square().mean(dim=-1)
    eps = 1e-8
    ncc = covariance / (
        torch.sqrt(fixed_var.clamp(min=eps))
        * torch.sqrt(moving_var.clamp(min=eps))
        + eps
    )
    if mask is None:
        return 1.0 - ncc.mean()
    mask = mask.to(ncc.dtype)
    return 1.0 - (ncc * mask).sum() / mask.sum().clamp(min=1.0)


def ncc_global(x, y, mask):
    mask = mask.to(x.dtype)
    count = mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mean_x = (x * mask).sum(dim=(1, 2, 3), keepdim=True) / count
    mean_y = (y * mask).sum(dim=(1, 2, 3), keepdim=True) / count
    centered_x = (x - mean_x) * mask
    centered_y = (y - mean_y) * mask
    var_x = centered_x.square().sum(dim=(1, 2, 3)) / count.flatten()
    var_y = centered_y.square().sum(dim=(1, 2, 3)) / count.flatten()
    covariance = (centered_x * centered_y).sum(dim=(1, 2, 3)) / count.flatten()
    return covariance / (torch.sqrt(var_x.clamp(min=1e-8) * var_y.clamp(min=1e-8)) + 1e-8)


@torch.no_grad()
def extract_pair_scores(ldm_model, moving, fixed, t_enc=1):
    """Extract paired LDM features using exactly the same noise for the pair."""
    moving_latent = ldm_model.get_first_stage_encoding(
        ldm_model.encode_first_stage(moving)
    ).detach()
    fixed_latent = ldm_model.get_first_stage_encoding(
        ldm_model.encode_first_stage(fixed)
    ).detach()
    timestep = torch.full(
        (moving.shape[0],),
        t_enc,
        device=moving.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(moving_latent)
    moving_noisy = ldm_model.q_sample(moving_latent, t=timestep, noise=noise)
    fixed_noisy = ldm_model.q_sample(fixed_latent, t=timestep, noise=noise)
    moving_output = ldm_model.apply_model(
        moving_noisy, t=timestep, cond=None, return_ids=True
    )[1][0]
    fixed_output = ldm_model.apply_model(
        fixed_noisy, t=timestep, cond=None, return_ids=True
    )[1][0]
    return (
        torch.cat(
            [moving_output[0], moving_output[2], fixed_output[0], fixed_output[2]],
            dim=1,
        ),
        torch.cat(
            [moving_output[3], moving_output[5], fixed_output[3], fixed_output[5]],
            dim=1,
        ),
        torch.cat(
            [moving_output[6], moving_output[8], fixed_output[6], fixed_output[8]],
            dim=1,
        ),
        torch.cat(
            [moving_output[9], moving_output[11], fixed_output[9], fixed_output[11]],
            dim=1,
        ),
    )


def model_forward(model, moving, fixed, scores, phase_id):
    output = model(moving, fixed, *scores, phase_id=phase_id)
    if len(output) == 3:
        displacement, _, motion_code = output
    else:
        displacement, _ = output
        motion_code = None
    return displacement, motion_code


def encode_image(ldm_model, image):
    return ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(image))


def registration_losses(
    opt,
    ldm_model,
    transform,
    latent_mse,
    moving,
    fixed,
    foreground,
    displacement,
    fixed_latent,
):
    _, warped = transform(moving, displacement.permute(0, 2, 3, 1))
    if opt.loss_type == "ncc":
        image_loss = ncc_loss(warped, fixed, mask=foreground)
    else:
        image_loss = masked_mse(warped, fixed, foreground)
    warped_latent = checkpoint(
        lambda value: encode_image(ldm_model, value),
        warped,
        use_reentrant=False,
    )
    latent_loss = latent_mse(warped_latent, fixed_latent)
    smooth_loss = smoothloss(displacement)
    bend_loss = bending_energy_loss(displacement)
    jac_loss = jacobian_neg_loss(displacement)
    total = (
        opt.beta * image_loss
        + (1.0 - opt.beta) * latent_loss
        + opt.smth_labda * smooth_loss
        + opt.bending_w * bend_loss
        + opt.jacdet_w * jac_loss
    )
    metrics = {
        "image": image_loss.detach(),
        "latent": latent_loss.detach(),
        "smooth": smooth_loss.detach(),
        "bend": bend_loss.detach(),
        "jac": jac_loss.detach(),
    }
    return total, metrics


def jacobian_values(displacement):
    values = []
    for index in range(displacement.shape[0]):
        dvf = displacement[index].detach().cpu().numpy().copy()
        height, width = dvf.shape[-2:]
        dvf[0] *= height / 2.0
        dvf[1] *= width / 2.0
        determinant = jacobian_determinant_vxm(dvf)
        values.append(
            (
                float((determinant < 0).mean()),
                int((determinant < 0).sum()),
                float(determinant.min()),
            )
        )
    return values


def unpack_pairwise(batch):
    if len(batch) < 4:
        raise ValueError("collate_pairwise must return fixed, moving, phase_id, names")
    return batch[0], batch[1], batch[2], batch[3]


def make_loaders(opt):
    common = dict(data_root=opt.data_root, normalize=True)
    if opt.cond_mode in ("baseline", "pairwise_motionfilm"):
        train_set = PairwisePhaseDataset(split="train", flip_p=0.5, **common)
        train_collate = collate_pairwise
    else:
        train_set = MultiPhaseDataset(split="train", flip_p=0.5, **common)
        train_collate = collate_multiphase

    train_loader = Data.DataLoader(
        train_set,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=train_collate,
    )
    val_set = PairwisePhaseDataset(split="val", flip_p=0.0, **common)
    test_set = PairwisePhaseDataset(split="test", flip_p=0.0, **common)
    val_loader = Data.DataLoader(
        val_set,
        batch_size=opt.bs,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_pairwise,
    )
    test_loader = Data.DataLoader(
        test_set,
        batch_size=opt.bs,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_pairwise,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(opt, model, ldm_model, transform, loader, split_name):
    model.eval()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    torch.manual_seed(opt.seed + 1000)
    torch.cuda.manual_seed(opt.seed + 1000)

    before_values = []
    after_values = []
    negative_ratios = []
    folding_counts = []
    minimum_jacobians = []

    for batch in loader:
        fixed, moving, phase_id, _ = unpack_pairwise(batch)
        fixed = fixed.cuda().float()
        moving = moving.cuda().float()
        phase_id = phase_id.cuda().long()
        foreground = body_mask(fixed, opt.fg_thr)
        scores = extract_pair_scores(ldm_model, moving, fixed, opt.t_enc)
        model_phase = (
            phase_id
            if opt.cond_mode == "motionfilm" and not opt.no_phase_id
            else None
        )
        displacement, _ = model_forward(model, moving, fixed, scores, model_phase)
        _, warped = transform(moving, displacement.permute(0, 2, 3, 1))

        for index in range(fixed.shape[0]):
            item_mask = foreground[index:index + 1]
            item_fixed = fixed[index:index + 1]
            before_values.append(
                1.0
                - ncc_loss(
                    item_fixed,
                    moving[index:index + 1],
                    mask=item_mask,
                ).item()
            )
            after_values.append(
                1.0
                - ncc_loss(
                    item_fixed,
                    warped[index:index + 1],
                    mask=item_mask,
                ).item()
            )
        for negative, count, minimum in jacobian_values(displacement):
            negative_ratios.append(negative)
            folding_counts.append(count)
            minimum_jacobians.append(minimum)

    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    model.train()
    result = {
        "split": split_name,
        "ncc_before": float(np.mean(before_values)),
        "ncc_after": float(np.mean(after_values)),
        "ncc_delta": float(np.mean(after_values) - np.mean(before_values)),
        "neg_ratio": float(np.mean(negative_ratios)),
        "folds_per_image": float(np.mean(folding_counts)),
        "min_jac": float(np.min(minimum_jacobians)),
        "samples": len(after_values),
    }
    print(
        f"[{split_name}] samples={result['samples']} "
        f"NCC={result['ncc_before']:.5f}->{result['ncc_after']:.5f} "
        f"delta={result['ncc_delta']:+.5f} "
        f"negR={100.0 * result['neg_ratio']:.4f}% "
        f"folds/image={result['folds_per_image']:.2f} "
        f"minJ={result['min_jac']:.4f}"
    )
    return result


def append_eval_csv(path, step, result):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                [
                    "step", "split", "samples", "ncc_before", "ncc_after",
                    "ncc_delta", "neg_ratio", "folds_per_image", "min_jac",
                ]
            )
        writer.writerow(
            [
                step,
                result["split"],
                result["samples"],
                result["ncc_before"],
                result["ncc_after"],
                result["ncc_delta"],
                result["neg_ratio"],
                result["folds_per_image"],
                result["min_jac"],
            ]
        )


def visualize_training_sample(
    step,
    opt,
    model,
    ldm_model,
    transform,
    vis_samples,
    vis_dir,
):
    """Render an 8-row x 3-col figure for three fixed vis samples.

    vis_samples is the dict returned by `build_visualization_samples`, with keys
    'fixed' [1, 1, H, W] and 'moving' [1, 3, 1, H, W] where the 3 channels are
    phases 1, 5, 9 of the same slice.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    model.eval()
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)

    fixed = vis_samples["fixed"].cuda().float()
    moving = vis_samples["moving"].cuda().float()  # [1, 3, 1, H, W]
    phase_ids = vis_samples["phase_ids"].cuda().long()
    slice_label = vis_samples["label"]
    phases = moving.shape[1]
    height, width = moving.shape[-2:]

    score_cache = [
        extract_pair_scores(ldm_model, moving[:, p], fixed, opt.t_enc)
        for p in range(phases)
    ]

    warped_list = []
    dvf_list = []
    jac_list = []
    ncc_after = []
    ncc_before = []
    neg_jac = []

    with torch.no_grad():
        for p in range(phases):
            model_phase = (
            phase_ids[:, p]
            if opt.cond_mode == "motionfilm" and not opt.no_phase_id
            else None
        )
            D, _ = model_forward(model, moving[:, p], fixed, score_cache[p], model_phase)
            _, warped = transform(moving[:, p], D.permute(0, 2, 3, 1))

            dvf = D[0].detach().cpu().numpy().copy()
            dvf[0] *= height / 2.0
            dvf[1] *= width / 2.0
            jac = jacobian_determinant_vxm(dvf)

            warped_list.append(warped[0, 0].detach().cpu().numpy())
            dvf_list.append((dvf[0], dvf[1]))
            jac_list.append(jac)
            ncc_after.append(
                1.0
                - ncc_loss(
                    fixed, warped, mask=body_mask(fixed, opt.fg_thr)
                ).item()
            )
            ncc_before.append(
                1.0
                - ncc_loss(
                    fixed, moving[:, p], mask=body_mask(fixed, opt.fg_thr)
                ).item()
            )
            neg_jac.append(float((jac < 0).mean()))

    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    if was_training:
        model.train()

    fig, axes = plt.subplots(
        nrows=8, ncols=phases, figsize=(3.0 * phases, 18.0),
        gridspec_kw={"hspace": 0.35, "wspace": 0.05},
    )
    row_titles = [
        "Moving",
        "Fixed",
        "Warped",
        "Diff |Fixed-Moving|",
        "Diff |Fixed-Warped|",
        "DVF-X (pixel)",
        "DVF-Y (pixel)",
        "Jacobian det (>=0 valid)",
    ]
    vmin_img, vmax_img = 0.0, 1.0
    for p in range(phases):
        axes[0, p].imshow(moving[0, p, 0].cpu().numpy(), cmap="gray", vmin=vmin_img, vmax=vmax_img)
        axes[0, p].set_title(f"phase {int(phase_ids[0, p].item()) + 1}", fontsize=9)
        axes[1, p].imshow(fixed[0, 0].cpu().numpy(), cmap="gray", vmin=vmin_img, vmax=vmax_img)
        axes[2, p].imshow(warped_list[p], cmap="gray", vmin=vmin_img, vmax=vmax_img)
        axes[3, p].imshow(
            np.abs(fixed[0, 0].cpu().numpy() - moving[0, p, 0].cpu().numpy()),
            cmap="magma",
        )
        axes[4, p].imshow(
            np.abs(fixed[0, 0].cpu().numpy() - warped_list[p]),
            cmap="magma",
        )
        dvf_max = max(
            np.abs(dvf_list[p][0]).max(), np.abs(dvf_list[p][1]).max(), 1e-6
        )
        axes[5, p].imshow(dvf_list[p][0], cmap="seismic", vmin=-dvf_max, vmax=dvf_max)
        axes[6, p].imshow(dvf_list[p][1], cmap="seismic", vmin=-dvf_max, vmax=dvf_max)
        jac_abs = max(np.abs(jac_list[p].min()), np.abs(jac_list[p].max()), 1e-3)
        axes[7, p].imshow(
            jac_list[p],
            cmap="RdBu_r",
            vmin=-jac_abs,
            vmax=jac_abs,
        )
        for row in range(8):
            axes[row, p].set_xticks([])
            axes[row, p].set_yticks([])

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=9)
    fig.suptitle(
        f"{opt.cond_mode} | step {step} | {slice_label} | "
        f"NCC before/after: "
        + " / ".join(
            f"p{int(phase_ids[0, p].item()) + 1}:{ncc_before[p]:.3f}->{ncc_after[p]:.3f}"
            for p in range(phases)
        )
        + f" | negR: "
        + " / ".join(f"{neg_jac[p] * 100:.2f}%" for p in range(phases)),
        fontsize=10,
    )

    out_path = os.path.join(vis_dir, f"step_{step:06d}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_visualization_samples(opt):
    """Load one fixed (block 2 / slice 0 / phase 0) and its phase 1, 5, 9 movings.

    Both modes use the same triplet so visual comparisons are apples-to-apples.
    """
    vis_split = PairwisePhaseDataset(
        data_root=opt.data_root, split="val", flip_p=0.0, normalize=True
    )
    base_samples = vis_split.base_samples
    target_block = 2
    target_slice = 0
    base_idx = None
    for index, (block_id, slice_id) in enumerate(base_samples):
        if block_id == target_block and slice_id == target_slice:
            base_idx = index
            break
    if base_idx is None:
        raise RuntimeError(
            f"Could not find block {target_block} slice {target_slice} in val split"
        )

    base_samples[base_idx]
    block_id, slice_id = base_samples[base_idx]
    fixed_np = vis_split._load_npy(
        vis_split._raw_index(block_id, 0, slice_id), vis_split.fixed_dir
    )
    fixed_np_min = fixed_np.min()
    fixed_np_max = fixed_np.max()
    if fixed_np_max - fixed_np_min > 1e-6:
        fixed_np = (fixed_np - fixed_np_min) / (fixed_np_max - fixed_np_min)
    fixed_t = torch.from_numpy(fixed_np).float().unsqueeze(0).unsqueeze(0)

    phase_indices = [0, 4, 8]  # phase 1, 5, 9
    moving_seq = []
    for phase_index in phase_indices:
        moving_np = vis_split._load_npy(
            vis_split._raw_index(block_id, phase_index, slice_id), vis_split.moving_dir
        )
        if fixed_np_max - fixed_np_min > 1e-6:
            moving_np = (moving_np - fixed_np_min) / (fixed_np_max - fixed_np_min)
        moving_seq.append(moving_np)
    moving_t = torch.from_numpy(np.stack(moving_seq, axis=0)).float()[:, None]
    moving_t = moving_t.unsqueeze(0)  # [1, 3, 1, H, W]

    return {
        "fixed": fixed_t,
        "moving": moving_t,
        "phase_ids": torch.tensor(phase_indices, dtype=torch.long).unsqueeze(0),
        "label": f"block{block_id}_slice{slice_id:02d}",
    }


def next_cycling(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_pairwise_step(
    opt, model, ldm_model, transform, latent_mse, optimizer, iterator, loader
):
    optimizer.zero_grad(set_to_none=True)
    metric_lists = {key: [] for key in ["image", "latent", "smooth", "bend", "jac"]}
    total_values = []

    for _ in range(NUM_PHASES):
        batch, iterator = next_cycling(iterator, loader)
        fixed, moving, _, _ = unpack_pairwise(batch)
        fixed = fixed.cuda().float()
        moving = moving.cuda().float()
        foreground = body_mask(fixed, opt.fg_thr)
        scores = extract_pair_scores(ldm_model, moving, fixed, opt.t_enc)
        displacement, _ = model_forward(model, moving, fixed, scores, phase_id=None)
        with torch.no_grad():
            fixed_latent = encode_image(ldm_model, fixed)
        loss, metrics = registration_losses(
            opt, ldm_model, transform, latent_mse,
            moving, fixed, foreground, displacement, fixed_latent,
        )
        (loss / NUM_PHASES).backward()
        total_values.append(loss.detach())
        for key in metric_lists:
            metric_lists[key].append(metrics[key])

    optimizer.step()
    summary = {key: torch.stack(v).mean().item() for key, v in metric_lists.items()}
    summary["total"] = torch.stack(total_values).mean().item()
    summary["z_acc"] = 0.0
    summary["dvf_acc"] = 0.0
    return summary, iterator


def train_motionfilm_step(
    opt, model, ldm_model, transform, latent_mse, optimizer, batch
):
    fixed, moving_sequence, phase_ids = batch[0], batch[1], batch[2]
    fixed = fixed.cuda().float()
    moving_sequence = moving_sequence.cuda().float()
    phase_ids = phase_ids.cuda().long()
    batch_size, phases = moving_sequence.shape[:2]
    if phases != NUM_PHASES:
        raise ValueError(f"Expected 9 phases, got {phases}")
    if phase_ids.min().item() < 0 or phase_ids.max().item() > 8:
        raise ValueError("phase_ids must be in [0, 8]")

    foreground = body_mask(fixed, opt.fg_thr)
    score_cache = [
        extract_pair_scores(ldm_model, moving_sequence[:, p], fixed, opt.t_enc)
        for p in range(phases)
    ]
    optimizer.zero_grad(set_to_none=True)

    motion_codes = []
    dvf_low_list = []
    for p in range(phases):
        model_phase = None if opt.no_phase_id else phase_ids[:, p]

        D, mc = model_forward(
            model,
            moving_sequence[:, p],
            fixed,
            score_cache[p],
            model_phase,
        )
        if mc is None:
            raise RuntimeError("MotionFiLM mode did not return a motion code")
        motion_codes.append(mc)
        dvf_low_list.append(F.interpolate(D, size=(64, 64), mode="bilinear", align_corners=True))

    mc_seq  = torch.stack(motion_codes, dim=1)
    with torch.no_grad():
        mc_jump = (
            mc_seq[:, 1:] - mc_seq[:, :-1]
        ).norm(dim=-1).mean()

        mc_std = mc_seq.std(dim=1).mean()
        mc_norm = mc_seq.norm(dim=-1).mean()
        mc_ratio = mc_std / (mc_norm + 1e-8)
    print(f"MotionFiLM motion code statistics: {mc_jump.item():.8f} {mc_std.item():.8f} {mc_norm.item():.8f} {mc_ratio.item():.8e}")
    
    dvf_seq = torch.stack(dvf_low_list, dim=1)
    mc_acc  = mc_seq[:, 2:] - 2.0 * mc_seq[:, 1:-1] + mc_seq[:, :-2]
    dvf_acc = dvf_seq[:, 2:] - 2.0 * dvf_seq[:, 1:-1] + dvf_seq[:, :-2]

    with torch.no_grad():
        gaps = torch.stack(
            [1.0 - ncc_global(moving_sequence[:, p], moving_sequence[:, p + 1], foreground)
             for p in range(phases - 1)], dim=1)
        gap_acc = 0.5 * (gaps[:, :-1] + gaps[:, 1:])
        w = torch.exp(-opt.alpha_motion_gap * gap_acc)

    L_z_acc   = (w * mc_acc.square().sum(dim=-1)).mean()
    L_dvf_acc = dvf_acc.abs().mean()
    L_traj = opt.lambda_z_acc * L_z_acc + opt.lambda_dvf_acc * L_dvf_acc
    if L_traj.requires_grad:
        L_traj.backward()

    del mc_seq, dvf_seq, mc_acc, dvf_acc

    with torch.no_grad():
        fixed_latent = encode_image(ldm_model, fixed)

    metric_lists = {k: [] for k in ["image", "latent", "smooth", "bend", "jac"]}
    reg_losses = []
    for p in range(phases):
        model_phase = None if opt.no_phase_id else phase_ids[:, p]

        D, _ = model_forward(
            model,
            moving_sequence[:, p],
            fixed,
            score_cache[p],
            model_phase,
        )
        loss, metrics = registration_losses(
            opt, ldm_model, transform, latent_mse,
            moving_sequence[:, p], fixed, foreground, D, fixed_latent,
        )
        (loss / phases).backward()
        reg_losses.append(loss.detach())
        for k in metric_lists:
            metric_lists[k].append(metrics[k])

    optimizer.step()
    summary = {k: torch.stack(v).mean().item() for k, v in metric_lists.items()}
    summary["total"]  = torch.stack(reg_losses).mean().item() + L_traj.detach().item()
    summary["z_acc"]  = L_z_acc.detach().item()
    summary["dvf_acc"] = L_dvf_acc.detach().item()
    return summary


def train():
    opt = parser.parse_args()
    seed_everything(opt.seed)
    opt.save_dir = os.path.join(opt.save_dir, opt.cond_mode)
    os.makedirs(opt.save_dir, exist_ok=True)
    print(f"[Mode] {opt.cond_mode} | save_dir={opt.save_dir}")
    print(
        f"[Conditioning] use_motion_film={opt.cond_mode in ('pairwise_motionfilm', 'motionfilm')} | "
        f"use_phase_embedding={opt.cond_mode == 'motionfilm' and not opt.no_phase_id}"
    )

    train_loader, val_loader, test_loader = make_loaders(opt)
    if opt.cond_mode in ("baseline", "pairwise_motionfilm"):
        train_pairs = len(train_loader.dataset)
        print(f"[Data] train={train_pairs} independent pairs")
    else:
        bases = len(train_loader.dataset)
        print(f"[Data] train={bases} base samples ({bases * NUM_PHASES} pairs)")
    print(
        f"[Data] val={len(val_loader.dataset)} independent pairs | "
        f"test={len(test_loader.dataset)} independent pairs | "
        f"pairs/update={opt.bs * NUM_PHASES}"
    )

    ldm_model = load_ldm(opt.ldm_config, opt.ldm_ckpt)
    model = LDMMorph(
        128 * 2,
        192 * 2,
        320 * 2,
        448 * 2,
        use_ldm=not opt.no_ldm,
        use_motion_film=opt.cond_mode in ("pairwise_motionfilm", "motionfilm"),
    ).cuda()
    transform = SpatialTransform().cuda()
    for parameter in transform.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    latent_mse = MSE().loss

    train_csv = os.path.join(opt.save_dir, "train_log.csv")
    eval_csv = os.path.join(opt.save_dir, "eval_log.csv")
    with open(train_csv, "w", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "step", "loss_total", "L_image", "L_latent", "L_smooth",
                "L_bend", "L_jac", "L_z_acc", "L_dvf_acc",
            ]
        )

    best_path = os.path.join(opt.save_dir, "best_val.pth")
    best_val_ncc = -float("inf")
    train_iterator = iter(train_loader)

    vis_samples = build_visualization_samples(opt)
    vis_dir = os.path.join(opt.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(
        f"[Vis] deterministic samples = {vis_samples['label']} | "
        f"phase_ids = {vis_samples['phase_ids'][0].tolist()} | "
        f"-> {vis_dir}"
    )

    for step in range(1, opt.iteration + 1):
        model.train()
        if opt.cond_mode in ("baseline", "pairwise_motionfilm"):
            metrics, train_iterator = train_pairwise_step(
                opt,
                model,
                ldm_model,
                transform,
                latent_mse,
                optimizer,
                train_iterator,
                train_loader,
            )
        else:
            batch, train_iterator = next_cycling(train_iterator, train_loader)
            metrics = train_motionfilm_step(
                opt,
                model,
                ldm_model,
                transform,
                latent_mse,
                optimizer,
                batch,
            )

        if step == 1 or step % opt.log_interval == 0:
            print(
                f"step={step} L={metrics['total']:.6f} "
                f"img={metrics['image']:.6f} latent={metrics['latent']:.6f} "
                f"smooth={metrics['smooth']:.6f} bend={metrics['bend']:.6f} "
                f"jac={metrics['jac']:.6f} z_acc={metrics['z_acc']:.6f} "
                f"dvf_acc={metrics['dvf_acc']:.6f}"
            )
            with open(train_csv, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        step,
                        metrics["total"],
                        metrics["image"],
                        metrics["latent"],
                        metrics["smooth"],
                        metrics["bend"],
                        metrics["jac"],
                        metrics["z_acc"],
                        metrics["dvf_acc"],
                    ]
                )

        if step % opt.checkpoint == 0 or step == opt.iteration:
            checkpoint_path = os.path.join(opt.save_dir, f"step_{step:06d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            validation = evaluate(
                opt,
                model,
                ldm_model,
                transform,
                val_loader,
                "val",
            )
            append_eval_csv(eval_csv, step, validation)
            if validation["ncc_after"] > best_val_ncc:
                best_val_ncc = validation["ncc_after"]
                torch.save(model.state_dict(), best_path)
                print(f"[Best] val NCC={best_val_ncc:.6f} -> {best_path}")

        if step == 1 or step % opt.vis_interval == 0:
            try:
                vis_path = visualize_training_sample(
                    step, opt, model, ldm_model, transform,
                    vis_samples, vis_dir,
                )
                print(f"[Vis] step {step} -> {vis_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[Vis] step {step} failed: {exc}")

    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location="cuda"), strict=True)
    test_result = evaluate(
        opt,
        model,
        ldm_model,
        transform,
        test_loader,
        "test",
    )
    append_eval_csv(eval_csv, "FINAL_BEST", test_result)
    print(f"[Done] best validation model tested once: {best_path}")


if __name__ == "__main__":
    train()
