"""Stage-two training for cross-phase attention and its continuation control.

Both modes start from the same Pairwise MotionFiLM checkpoint and consume the
same same-slice nine-phase batches.

attention_residual:
    Freeze every Pairwise MotionFiLM parameter and train only a lightweight
    cross-phase attention module plus a zero-initialized residual DVF head.

residual_only:
    Freeze the same Pairwise MotionFiLM model and train the same phasewise FFN,
    local encoder, and residual head, while bypassing cross-phase attention.

continued_control:
    Add no new module and continue updating Pairwise MotionFiLM for the same
    data exposure and optimizer-update budget.
"""

import csv
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data

from ldm.data.xcat_multiphase import (
    MultiPhaseDataset,
    collate_multiphase,
)
from TransModels.LDMMorph import LDMMorph
from TransModels.PhaseAttentionResidual import CrossPhaseAttentionResidual
from train_multiphase_motionfilm import (
    NUM_PHASES,
    body_mask,
    encode_image,
    extract_pair_scores,
    jacobian_values,
    load_ldm,
    model_forward,
    ncc_loss,
    registration_losses,
    seed_everything,
)
from utils.utils import MSE, SpatialTransform


parser = ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["attention_residual", "residual_only", "continued_control"],
    required=True,
)
parser.add_argument("--pairwise_ckpt", required=True)
parser.add_argument("--ldm_ckpt", "--resume", dest="ldm_ckpt", required=True)
parser.add_argument("--ldm_config", required=True)
parser.add_argument(
    "--data_root",
    default="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
)
parser.add_argument("--save_dir", default="./logs/PhaseAttention_Ablation")
parser.add_argument("--iteration", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--sequences_per_update", type=int, default=2)
parser.add_argument("--checkpoint", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--smth_labda", type=float, default=0.4)
parser.add_argument("--bending_w", type=float, default=0.0)
parser.add_argument("--jacdet_w", type=float, default=0.0)
parser.add_argument("--loss_type", choices=["ncc", "mse"], default="ncc")
parser.add_argument("--fg_thr", type=float, default=0.05)
parser.add_argument("--t_enc", type=int, default=1)
parser.add_argument("--delta_w", type=float, default=1e-3)
parser.add_argument("--attention_heads", type=int, default=4)
parser.add_argument("--residual_channels", type=int, default=32)
parser.add_argument("--residual_size", type=int, default=128)
parser.add_argument("--no_ldm", action="store_true")


REFINEMENT_MODES = {"attention_residual", "residual_only"}


def load_pairwise_model(path, use_ldm=True):
    model = LDMMorph(
        128 * 2,
        192 * 2,
        320 * 2,
        448 * 2,
        use_ldm=use_ldm,
        use_motion_film=True,
    ).cuda()
    payload = torch.load(path, map_location="cuda")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "pairwise_model_state_dict" in payload:
        state = payload["pairwise_model_state_dict"]
    else:
        state = payload
    model.load_state_dict(state, strict=True)
    print(f"[Pairwise] loaded {path} | strict=True")
    return model


def freeze_model(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def make_loaders(opt):
    common = dict(data_root=opt.data_root, normalize=True)
    train_set = MultiPhaseDataset(split="train", flip_p=0.5, **common)
    val_set = MultiPhaseDataset(split="val", flip_p=0.0, **common)
    test_set = MultiPhaseDataset(split="test", flip_p=0.0, **common)
    train_loader = Data.DataLoader(
        train_set,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=collate_multiphase,
    )
    val_loader = Data.DataLoader(
        val_set,
        batch_size=opt.bs,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_multiphase,
    )
    test_loader = Data.DataLoader(
        test_set,
        batch_size=opt.bs,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_multiphase,
    )
    return train_loader, val_loader, test_loader


def next_cycling(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def frozen_pairwise_sequence(opt, model, ldm_model, transform, fixed, moving):
    displacements = []
    warped_images = []
    motion_codes = []
    for phase in range(moving.shape[1]):
        moving_phase = moving[:, phase]
        scores = extract_pair_scores(ldm_model, moving_phase, fixed, opt.t_enc)
        displacement, motion_code = model_forward(
            model,
            moving_phase,
            fixed,
            scores,
            phase_id=None,
        )
        if motion_code is None:
            raise RuntimeError("Pairwise MotionFiLM did not return motion_code")
        _, warped = transform(
            moving_phase,
            displacement.permute(0, 2, 3, 1),
        )
        displacements.append(displacement.detach())
        warped_images.append(warped.detach())
        motion_codes.append(motion_code.detach())
    return displacements, warped_images, torch.stack(motion_codes, dim=1)


def train_attention_sequence(
    opt,
    base_model,
    refiner,
    ldm_model,
    transform,
    latent_mse,
    batch,
    loss_scale,
):
    fixed, moving = batch[0].cuda().float(), batch[1].cuda().float()
    if moving.shape[1] != NUM_PHASES:
        raise ValueError(f"Expected {NUM_PHASES} phases, got {moving.shape[1]}")
    foreground = body_mask(fixed, opt.fg_thr)
    base_dvfs, base_warped, motion_codes = frozen_pairwise_sequence(
        opt, base_model, ldm_model, transform, fixed, moving
    )
    with torch.no_grad():
        fixed_latent = encode_image(ldm_model, fixed)

    totals = []
    residual_values = []
    metric_values = {key: [] for key in ["image", "latent", "smooth", "bend", "jac"]}
    for phase in range(NUM_PHASES):
        refine_function = (
            refiner.refine_phase
            if opt.mode == "attention_residual"
            else refiner.refine_phase_residual_only
        )
        refined_dvf, residual, _ = refine_function(
            moving=moving[:, phase],
            fixed=fixed,
            pairwise_warped=base_warped[phase],
            pairwise_dvf=base_dvfs[phase],
            motion_codes=motion_codes,
            phase_index=phase,
        )
        registration, metrics = registration_losses(
            opt,
            ldm_model,
            transform,
            latent_mse,
            moving[:, phase],
            fixed,
            foreground,
            refined_dvf,
            fixed_latent,
        )
        residual_penalty = residual.abs().mean()
        loss = registration + opt.delta_w * residual_penalty
        (loss * loss_scale / NUM_PHASES).backward()
        totals.append(loss.detach())
        residual_values.append(residual_penalty.detach())
        for key in metric_values:
            metric_values[key].append(metrics[key])

    result = {key: torch.stack(value).mean().item() for key, value in metric_values.items()}
    result["total"] = torch.stack(totals).mean().item()
    result["residual"] = torch.stack(residual_values).mean().item()
    return result


def train_control_sequence(
    opt,
    model,
    ldm_model,
    transform,
    latent_mse,
    batch,
    loss_scale,
):
    fixed, moving = batch[0].cuda().float(), batch[1].cuda().float()
    if moving.shape[1] != NUM_PHASES:
        raise ValueError(f"Expected {NUM_PHASES} phases, got {moving.shape[1]}")
    foreground = body_mask(fixed, opt.fg_thr)
    with torch.no_grad():
        fixed_latent = encode_image(ldm_model, fixed)

    totals = []
    metric_values = {key: [] for key in ["image", "latent", "smooth", "bend", "jac"]}
    for phase in range(NUM_PHASES):
        moving_phase = moving[:, phase]
        scores = extract_pair_scores(ldm_model, moving_phase, fixed, opt.t_enc)
        displacement, _ = model_forward(
            model,
            moving_phase,
            fixed,
            scores,
            phase_id=None,
        )
        loss, metrics = registration_losses(
            opt,
            ldm_model,
            transform,
            latent_mse,
            moving_phase,
            fixed,
            foreground,
            displacement,
            fixed_latent,
        )
        (loss * loss_scale / NUM_PHASES).backward()
        totals.append(loss.detach())
        for key in metric_values:
            metric_values[key].append(metrics[key])

    result = {key: torch.stack(value).mean().item() for key, value in metric_values.items()}
    result["total"] = torch.stack(totals).mean().item()
    result["residual"] = 0.0
    return result


@torch.no_grad()
def evaluate_sequence(
    opt,
    base_model,
    refiner,
    ldm_model,
    transform,
    loader,
    split_name,
):
    base_model.eval()
    if refiner is not None:
        refiner.eval()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    torch.manual_seed(opt.seed + 1000)
    torch.cuda.manual_seed(opt.seed + 1000)

    before_values = []
    coarse_values = []
    after_values = []
    negative_ratios = []
    folding_counts = []
    minimum_jacobians = []
    residual_values = []

    for batch in loader:
        fixed, moving = batch[0].cuda().float(), batch[1].cuda().float()
        foreground = body_mask(fixed, opt.fg_thr)
        if refiner is not None:
            base_dvfs, base_warped, motion_codes = frozen_pairwise_sequence(
                opt, base_model, ldm_model, transform, fixed, moving
            )

        for phase in range(NUM_PHASES):
            moving_phase = moving[:, phase]
            if refiner is None:
                scores = extract_pair_scores(
                    ldm_model, moving_phase, fixed, opt.t_enc
                )
                displacement, _ = model_forward(
                    base_model,
                    moving_phase,
                    fixed,
                    scores,
                    phase_id=None,
                )
                _, warped = transform(
                    moving_phase,
                    displacement.permute(0, 2, 3, 1),
                )
                coarse_warped = warped
                residual_values.append(0.0)
            else:
                refine_function = (
                    refiner.refine_phase
                    if opt.mode == "attention_residual"
                    else refiner.refine_phase_residual_only
                )
                displacement, residual, _ = refine_function(
                    moving=moving_phase,
                    fixed=fixed,
                    pairwise_warped=base_warped[phase],
                    pairwise_dvf=base_dvfs[phase],
                    motion_codes=motion_codes,
                    phase_index=phase,
                )
                _, warped = transform(
                    moving_phase,
                    displacement.permute(0, 2, 3, 1),
                )
                coarse_warped = base_warped[phase]
                residual_values.append(residual.abs().mean().item())

            for index in range(fixed.shape[0]):
                item_mask = foreground[index:index + 1]
                item_fixed = fixed[index:index + 1]
                before_values.append(
                    1.0
                    - ncc_loss(
                        item_fixed,
                        moving_phase[index:index + 1],
                        mask=item_mask,
                    ).item()
                )
                coarse_values.append(
                    1.0
                    - ncc_loss(
                        item_fixed,
                        coarse_warped[index:index + 1],
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
    result = {
        "split": split_name,
        "samples": len(after_values),
        "ncc_before": float(np.mean(before_values)),
        "ncc_coarse": float(np.mean(coarse_values)),
        "ncc_after": float(np.mean(after_values)),
        "neg_ratio": float(np.mean(negative_ratios)),
        "folds_per_image": float(np.mean(folding_counts)),
        "min_jac": float(np.min(minimum_jacobians)),
        "residual_abs": float(np.mean(residual_values)),
    }
    print(
        f"[{split_name}] samples={result['samples']} "
        f"NCC={result['ncc_before']:.5f}->{result['ncc_after']:.5f} "
        f"coarse={result['ncc_coarse']:.5f} "
        f"negR={100.0 * result['neg_ratio']:.4f}% "
        f"folds/image={result['folds_per_image']:.2f} "
        f"minJ={result['min_jac']:.4f} "
        f"|delta|={result['residual_abs']:.7f}"
    )
    return result


def append_eval(path, step, result):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                [
                    "step", "split", "samples", "ncc_before", "ncc_coarse",
                    "ncc_after", "neg_ratio", "folds_per_image", "min_jac",
                    "residual_abs",
                ]
            )
        writer.writerow(
            [
                step,
                result["split"],
                result["samples"],
                result["ncc_before"],
                result["ncc_coarse"],
                result["ncc_after"],
                result["neg_ratio"],
                result["folds_per_image"],
                result["min_jac"],
                result["residual_abs"],
            ]
        )


def checkpoint_payload(opt, model, refiner, step, best_val):
    payload = {
        "mode": opt.mode,
        "step": step,
        "best_val_ncc": best_val,
        "config": vars(opt),
    }
    if opt.mode in REFINEMENT_MODES:
        payload["pairwise_model_state_dict"] = model.state_dict()
        state_key = (
            "attention_residual_state_dict"
            if opt.mode == "attention_residual"
            else "residual_only_state_dict"
        )
        payload[state_key] = refiner.state_dict()
    else:
        payload["model_state_dict"] = model.state_dict()
    return payload


def restore_checkpoint(path, opt, model, refiner):
    payload = torch.load(path, map_location="cuda")
    if opt.mode in REFINEMENT_MODES:
        model.load_state_dict(payload["pairwise_model_state_dict"], strict=True)
        state_key = (
            "attention_residual_state_dict"
            if opt.mode == "attention_residual"
            else "residual_only_state_dict"
        )
        refiner.load_state_dict(payload[state_key], strict=True)
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)


def train():
    opt = parser.parse_args()
    if opt.bs != 1:
        raise ValueError("Use --bs 1; sequences_per_update controls diversity")
    if opt.sequences_per_update < 1:
        raise ValueError("sequences_per_update must be >= 1")
    seed_everything(opt.seed)
    opt.save_dir = os.path.join(opt.save_dir, opt.mode)
    os.makedirs(opt.save_dir, exist_ok=True)

    print(f"[Mode] {opt.mode} | save_dir={opt.save_dir}")
    print(
        f"[Budget] sequences/update={opt.sequences_per_update} | "
        f"pairs/update={opt.sequences_per_update * NUM_PHASES} | "
        f"updates={opt.iteration}"
    )
    train_loader, val_loader, test_loader = make_loaders(opt)
    print(
        f"[Data] train={len(train_loader.dataset)} sequences | "
        f"val={len(val_loader.dataset)} | test={len(test_loader.dataset)}"
    )

    ldm_model = load_ldm(opt.ldm_config, opt.ldm_ckpt)
    base_model = load_pairwise_model(
        opt.pairwise_ckpt,
        use_ldm=not opt.no_ldm,
    )
    transform = SpatialTransform().cuda()
    for parameter in transform.parameters():
        parameter.requires_grad_(False)
    latent_mse = MSE().loss

    refiner = None
    if opt.mode in REFINEMENT_MODES:
        freeze_model(base_model)
        refiner = CrossPhaseAttentionResidual(
            code_dim=16,
            num_heads=opt.attention_heads,
            hidden_channels=opt.residual_channels,
            residual_size=opt.residual_size,
        ).cuda()
        if opt.mode == "residual_only":
            for parameter in refiner.code_norm.parameters():
                parameter.requires_grad_(False)
            for parameter in refiner.phase_attention.parameters():
                parameter.requires_grad_(False)
        trainable = [
            parameter for parameter in refiner.parameters()
            if parameter.requires_grad
        ]
    else:
        base_model.train()
        for parameter in base_model.parameters():
            parameter.requires_grad_(True)
        trainable = list(base_model.parameters())

    parameter_count = sum(parameter.numel() for parameter in trainable)
    print(f"[Trainable] {parameter_count / 1e6:.3f}M parameters")
    optimizer = torch.optim.Adam(trainable, lr=opt.lr)

    train_csv = os.path.join(opt.save_dir, "train_log.csv")
    eval_csv = os.path.join(opt.save_dir, "eval_log.csv")
    with open(train_csv, "w", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "step", "loss_total", "L_image", "L_latent", "L_smooth",
                "L_bend", "L_jac", "L_residual",
            ]
        )

    iterator = iter(train_loader)
    best_path = os.path.join(opt.save_dir, "best_val.pth")
    last_path = os.path.join(opt.save_dir, "last.pth")
    best_val = -float("inf")

    for step in range(1, opt.iteration + 1):
        if opt.mode in REFINEMENT_MODES:
            base_model.eval()
            refiner.train()
        else:
            base_model.train()
        optimizer.zero_grad(set_to_none=True)
        update_metrics = []
        for _ in range(opt.sequences_per_update):
            batch, iterator = next_cycling(iterator, train_loader)
            loss_scale = 1.0 / opt.sequences_per_update
            if opt.mode in REFINEMENT_MODES:
                metrics = train_attention_sequence(
                    opt,
                    base_model,
                    refiner,
                    ldm_model,
                    transform,
                    latent_mse,
                    batch,
                    loss_scale,
                )
            else:
                metrics = train_control_sequence(
                    opt,
                    base_model,
                    ldm_model,
                    transform,
                    latent_mse,
                    batch,
                    loss_scale,
                )
            update_metrics.append(metrics)
        optimizer.step()

        averaged = {
            key: float(np.mean([item[key] for item in update_metrics]))
            for key in update_metrics[0]
        }
        if step == 1 or step % opt.log_interval == 0:
            print(
                f"step={step} L={averaged['total']:.6f} "
                f"img={averaged['image']:.6f} "
                f"latent={averaged['latent']:.6f} "
                f"smooth={averaged['smooth']:.6f} "
                f"bend={averaged['bend']:.6f} "
                f"jac={averaged['jac']:.6f} "
                f"residual={averaged['residual']:.7f}"
            )
            with open(train_csv, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        step,
                        averaged["total"],
                        averaged["image"],
                        averaged["latent"],
                        averaged["smooth"],
                        averaged["bend"],
                        averaged["jac"],
                        averaged["residual"],
                    ]
                )

        if step % opt.checkpoint == 0 or step == opt.iteration:
            validation = evaluate_sequence(
                opt,
                base_model,
                refiner,
                ldm_model,
                transform,
                val_loader,
                "val",
            )
            append_eval(eval_csv, step, validation)
            if validation["ncc_after"] > best_val:
                best_val = validation["ncc_after"]
                torch.save(
                    checkpoint_payload(opt, base_model, refiner, step, best_val),
                    best_path,
                )
                print(f"[Best] val NCC={best_val:.6f} -> {best_path}")

    torch.save(
        checkpoint_payload(opt, base_model, refiner, opt.iteration, best_val),
        last_path,
    )
    restore_checkpoint(best_path, opt, base_model, refiner)
    test_result = evaluate_sequence(
        opt,
        base_model,
        refiner,
        ldm_model,
        transform,
        test_loader,
        "test",
    )
    append_eval(eval_csv, "FINAL_BEST", test_result)
    print(f"[Done] best validation model tested once: {best_path}")


if __name__ == "__main__":
    train()
