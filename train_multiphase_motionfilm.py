"""Fair pairwise-baseline vs multi-phase MotionFiLM training.

Both modes use the same block split and preprocessing.

baseline:
    Independent pairwise samples. Nine micro-batches are accumulated before
    each optimizer update, so every update sees the same number of pairs as
    MotionFiLM.

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
parser.add_argument("--cond_mode", choices=["baseline", "motionfilm"], default="motionfilm")
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
parser.add_argument("--log_interval", type=int, default=50)
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
    if opt.cond_mode == "baseline":
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
        model_phase = phase_id if opt.cond_mode == "motionfilm" else None
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


def next_cycling(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_baseline_step(
    opt, model, ldm_model, transform, latent_mse, optimizer, iterator, loader
):
    optimizer.zero_grad(set_to_none=True)
    metric_lists = {key: [] for key in ["image", "latent", "smooth", "bend", "jac"]}
    total_values = []

    # Nine independent micro-batches = the same pair count as one 9-phase update.
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
            opt,
            ldm_model,
            transform,
            latent_mse,
            moving,
            fixed,
            foreground,
            displacement,
            fixed_latent,
        )
        (loss / NUM_PHASES).backward()
        total_values.append(loss.detach())
        for key in metric_lists:
            metric_lists[key].append(metrics[key])

    optimizer.step()
    summary = {key: torch.stack(values).mean().item() for key, values in metric_lists.items()}
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
        extract_pair_scores(
            ldm_model,
            moving_sequence[:, phase],
            fixed,
            opt.t_enc,
        )
        for phase in range(phases)
    ]
    optimizer.zero_grad(set_to_none=True)

    motion_codes = []
    low_resolution_dvfs = []
    for phase in range(phases):
        displacement, motion_code = model_forward(
            model,
            moving_sequence[:, phase],
            fixed,
            score_cache[phase],
            phase_ids[:, phase],
        )
        if motion_code is None:
            raise RuntimeError("MotionFiLM mode did not return a motion code")
        motion_codes.append(motion_code)
        low_resolution_dvfs.append(
            F.interpolate(
                displacement,
                size=(64, 64),
                mode="bilinear",
                align_corners=True,
            )
        )

    motion_sequence = torch.stack(motion_codes, dim=1)
    dvf_sequence = torch.stack(low_resolution_dvfs, dim=1)
    motion_acceleration = (
        motion_sequence[:, 2:]
        - 2.0 * motion_sequence[:, 1:-1]
        + motion_sequence[:, :-2]
    )
    with torch.no_grad():
        gaps = torch.stack(
            [
                1.0
                - ncc_global(
                    moving_sequence[:, phase],
                    moving_sequence[:, phase + 1],
                    foreground,
                )
                for phase in range(phases - 1)
            ],
            dim=1,
        )
        gap_acceleration = 0.5 * (gaps[:, :-1] + gaps[:, 1:])
        weights = torch.exp(-opt.alpha_motion_gap * gap_acceleration)
    z_acc_loss = (
        weights * motion_acceleration.square().sum(dim=-1)
    ).mean()
    dvf_acceleration = (
        dvf_sequence[:, 2:]
        - 2.0 * dvf_sequence[:, 1:-1]
        + dvf_sequence[:, :-2]
    )
    dvf_acc_loss = dvf_acceleration.abs().mean()
    trajectory_loss = (
        opt.lambda_z_acc * z_acc_loss
        + opt.lambda_dvf_acc * dvf_acc_loss
    )
    if trajectory_loss.requires_grad:
        trajectory_loss.backward()

    del motion_sequence, dvf_sequence, motion_acceleration, dvf_acceleration
    with torch.no_grad():
        fixed_latent = encode_image(ldm_model, fixed)

    metric_lists = {key: [] for key in ["image", "latent", "smooth", "bend", "jac"]}
    registration_totals = []
    for phase in range(phases):
        moving = moving_sequence[:, phase]
        displacement, _ = model_forward(
            model,
            moving,
            fixed,
            score_cache[phase],
            phase_ids[:, phase],
        )
        loss, metrics = registration_losses(
            opt,
            ldm_model,
            transform,
            latent_mse,
            moving,
            fixed,
            foreground,
            displacement,
            fixed_latent,
        )
        (loss / phases).backward()
        registration_totals.append(loss.detach())
        for key in metric_lists:
            metric_lists[key].append(metrics[key])

    optimizer.step()
    summary = {key: torch.stack(values).mean().item() for key, values in metric_lists.items()}
    registration_mean = torch.stack(registration_totals).mean().item()
    summary["z_acc"] = z_acc_loss.detach().item()
    summary["dvf_acc"] = dvf_acc_loss.detach().item()
    summary["total"] = registration_mean + trajectory_loss.detach().item()
    return summary


def train():
    opt = parser.parse_args()
    seed_everything(opt.seed)
    opt.save_dir = os.path.join(opt.save_dir, opt.cond_mode)
    os.makedirs(opt.save_dir, exist_ok=True)
    print(f"[Mode] {opt.cond_mode} | save_dir={opt.save_dir}")

    train_loader, val_loader, test_loader = make_loaders(opt)
    if opt.cond_mode == "baseline":
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
        use_motion_film=opt.cond_mode == "motionfilm",
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

    for step in range(1, opt.iteration + 1):
        model.train()
        if opt.cond_mode == "baseline":
            metrics, train_iterator = train_baseline_step(
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

