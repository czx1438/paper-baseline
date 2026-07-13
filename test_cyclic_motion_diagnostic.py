"""Diagnose whether cyclic multi-phase registration is worth training.

This script performs no optimization. It evaluates the existing Baseline and
Pairwise MotionFiLM models on the same XCAT multi-phase split and answers three
questions:

1. Are the nine independently predicted phase-to-phase0 DVFs temporally
   coherent?
2. Can a model trained with phase0 as fixed register adjacent phase pairs at
   all? This is an out-of-distribution stress test, not a final benchmark.
3. Do compositions of adjacent DVFs agree with the direct phase-to-phase0 DVF,
   and does the complete phase0->...->phase9->phase0 cycle close?

The Baseline and Pairwise MotionFiLM models share exactly the same cached LDM
scores for every image pair. Flow composition uses the repository's original
SpatialTransform, preserving its normalized-coordinate convention.
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from ldm.data.xcat_multiphase import MultiPhaseDataset, collate_multiphase
from TransModels.LDMMorph import LDMMorph
from train_multiphase_motionfilm import (
    NUM_PHASES,
    body_mask,
    extract_pair_scores,
    load_ldm,
    model_forward,
    ncc_loss,
    seed_everything,
)
from utils.utils import SpatialTransform, jacobian_determinant_vxm


MODEL_NAMES = ("baseline", "pairwise")
TOTAL_PHASES = NUM_PHASES + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--pairwise_ckpt", required=True)
    parser.add_argument("--ldm_config", required=True)
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument(
        "--data_root",
        default="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--t_enc", type=int, default=1)
    parser.add_argument("--fg_thr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual_slices", default="0,9,18")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--no_ldm", action="store_true")
    parser.add_argument(
        "--save_dir",
        default="./logs/cyclic_motion_diagnostic",
    )
    return parser.parse_args()


def extract_model_state(payload, path):
    if not isinstance(payload, dict):
        return payload
    for key in (
        "model_state_dict",
        "state_dict",
        "pairwise_model_state_dict",
    ):
        if key in payload:
            return payload[key]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError(f"Cannot find a registration state_dict in {path}")


def build_model(use_motion_film, use_ldm):
    return LDMMorph(
        128 * 2,
        192 * 2,
        320 * 2,
        448 * 2,
        use_ldm=use_ldm,
        use_motion_film=use_motion_film,
    ).cuda()


def load_registration_model(path, use_motion_film, use_ldm):
    payload = torch.load(path, map_location="cpu")
    model = build_model(
        use_motion_film=use_motion_film,
        use_ldm=use_ldm,
    )
    model.load_state_dict(extract_model_state(payload, path), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"[Model] loaded {path} | "
        f"use_motion_film={use_motion_film} | strict=True"
    )
    return model


def parse_name(name):
    match = re.search(r"block(\d+)_slice(\d+)", str(name))
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def masked_mse(prediction, target, mask):
    numerator = (((prediction - target) ** 2) * mask).sum()
    return float((numerator / mask.sum().clamp(min=1.0)).item())


def to_pixel_displacement(displacement):
    result = displacement.clone()
    height, width = result.shape[-2:]
    result[:, 0] *= height / 2.0
    result[:, 1] *= width / 2.0
    return result


def vector_norm_map_px(displacement):
    displacement_px = to_pixel_displacement(displacement)
    return torch.sqrt(displacement_px.square().sum(dim=1) + 1e-12)


def masked_map_mean(value_map, mask):
    mask_2d = mask[:, 0]
    return float(
        ((value_map * mask_2d).sum() / mask_2d.sum().clamp(min=1.0)).item()
    )


def flow_difference_metrics(first, second, mask):
    norm_map = vector_norm_map_px(first - second)
    return {
        "dvf_difference_px": masked_map_mean(norm_map, mask),
        "dvf_difference_max_px": float(norm_map.max().item()),
    }


def jacobian_result(displacement):
    dvf = displacement[0].detach().cpu().numpy().copy()
    height, width = dvf.shape[-2:]
    dvf[0] *= height / 2.0
    dvf[1] *= width / 2.0
    determinant = jacobian_determinant_vxm(dvf)
    magnitude = np.sqrt(dvf[0] ** 2 + dvf[1] ** 2)
    return {
        "neg_ratio": float((determinant < 0).mean()),
        "fold_count": int((determinant < 0).sum()),
        "min_jac": float(determinant.min()),
        "mean_jac": float(determinant.mean()),
        "mean_dvf_px": float(magnitude.mean()),
        "max_dvf_px": float(magnitude.max()),
        "jacobian_map": determinant,
    }


def warp_tensor(transform, tensor, displacement):
    _, warped = transform(
        tensor,
        displacement.permute(0, 2, 3, 1),
    )
    return warped


def compose_displacements(transform, first, second):
    """Compose backward sampling flows using the repository convention.

    If ``first`` maps target A coordinates into source B and ``second`` maps
    target B coordinates into source C, the result maps target A into source C:

        composed(x) = first(x) + second(x + first(x)).
    """
    second_on_first = warp_tensor(transform, second, first)
    return first + second_on_first


def registration_row(
    model_name,
    pair_type,
    block_id,
    slice_id,
    source_phase,
    target_phase,
    source,
    target,
    target_mask,
    displacement,
    warped,
):
    jacobian = jacobian_result(displacement)
    ncc_before = 1.0 - ncc_loss(target, source, mask=target_mask).item()
    ncc_after = 1.0 - ncc_loss(target, warped, mask=target_mask).item()
    return {
        "model": model_name,
        "pair_type": pair_type,
        "block_id": block_id,
        "slice_id": slice_id,
        "source_phase": source_phase,
        "target_phase": target_phase,
        "ncc_before": ncc_before,
        "ncc_after": ncc_after,
        "ncc_delta": ncc_after - ncc_before,
        "mse_before": masked_mse(source, target, target_mask),
        "mse_after": masked_mse(warped, target, target_mask),
        **{
            key: value
            for key, value in jacobian.items()
            if key != "jacobian_map"
        },
    }


def temporal_metrics(displacement_sequence, foreground):
    """Temporal diagnostics for direct phase1..9-to-phase0 spoke flows."""
    zero = torch.zeros_like(displacement_sequence[:, :1])
    closed = torch.cat([zero, displacement_sequence, zero], dim=1)
    velocity = closed[:, 1:] - closed[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]

    velocity_px = velocity.clone()
    acceleration_px = acceleration.clone()
    height, width = displacement_sequence.shape[-2:]
    velocity_px[:, :, 0] *= height / 2.0
    velocity_px[:, :, 1] *= width / 2.0
    acceleration_px[:, :, 0] *= height / 2.0
    acceleration_px[:, :, 1] *= width / 2.0

    velocity_norm = torch.sqrt(velocity_px.square().sum(dim=2) + 1e-12)
    acceleration_norm = torch.sqrt(
        acceleration_px.square().sum(dim=2) + 1e-12
    )
    mask = foreground[:, 0, None]
    velocity_mean = (
        (velocity_norm * mask).sum()
        / (mask.sum() * velocity_norm.shape[1]).clamp(min=1.0)
    )
    acceleration_mean = (
        (acceleration_norm * mask).sum()
        / (mask.sum() * acceleration_norm.shape[1]).clamp(min=1.0)
    )

    internal_velocity = velocity_norm[:, 1:-1]
    internal_velocity_mean = (
        (internal_velocity * mask).sum()
        / (mask.sum() * internal_velocity.shape[1]).clamp(min=1.0)
    )
    phase9_gap = vector_norm_map_px(displacement_sequence[:, -1])
    return {
        "cycle_velocity_px": float(velocity_mean.item()),
        "internal_velocity_px": float(internal_velocity_mean.item()),
        "cycle_acceleration_px": float(acceleration_mean.item()),
        "phase9_to_phase0_spoke_gap_px": masked_map_mean(
            phase9_gap, foreground
        ),
    }


def motion_code_metrics(motion_codes):
    """Measure whether the frozen pairwise motion tokens differ by phase."""
    if motion_codes.shape[0] != 1:
        raise ValueError("Motion-code diagnostics require batch size 1")
    codes = motion_codes[0]
    normalized_codes = F.normalize(codes, dim=-1)
    cosine = normalized_codes @ normalized_codes.transpose(0, 1)
    off_diagonal = ~torch.eye(
        codes.shape[0], dtype=torch.bool, device=codes.device
    )

    layer_normalized = F.layer_norm(codes, (codes.shape[-1],))
    layer_normalized = F.normalize(layer_normalized, dim=-1)
    normalized_cosine = layer_normalized @ layer_normalized.transpose(0, 1)
    consecutive = (codes[1:] - codes[:-1]).norm(dim=-1)
    return {
        "motion_code_std": float(codes.std(dim=0).mean().item()),
        "motion_code_norm": float(codes.norm(dim=-1).mean().item()),
        "motion_code_std_norm_ratio": float(
            (codes.std(dim=0).mean() / codes.norm(dim=-1).mean().clamp(min=1e-8)).item()
        ),
        "motion_code_consecutive_l2": float(consecutive.mean().item()),
        "motion_code_offdiag_cosine": float(cosine[off_diagonal].mean().item()),
        "layernorm_offdiag_cosine": float(
            normalized_cosine[off_diagonal].mean().item()
        ),
        "layernorm_offdiag_cosine_min": float(
            normalized_cosine[off_diagonal].min().item()
        ),
        "layernorm_offdiag_cosine_max": float(
            normalized_cosine[off_diagonal].max().item()
        ),
    }


def aggregate(rows, group_fields, metric_fields):
    if not rows:
        return []
    groups = sorted(
        {
            tuple(row[field] for field in group_fields)
            for row in rows
        }
    )
    output = []
    for group in groups:
        subset = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == group
        ]
        record = {
            field: value for field, value in zip(group_fields, group)
        }
        record["samples"] = len(subset)
        for metric in metric_fields:
            values = np.asarray(
                [row[metric] for row in subset], dtype=np.float64
            )
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0
            )
            record[f"{metric}_median"] = float(np.median(values))
        output.append(record)
    return output


def slice_level_paired_statistics(composition_rows):
    output = []
    for model_name in MODEL_NAMES:
        model_rows = [
            row for row in composition_rows if row["model"] == model_name
        ]
        slice_ids = sorted({row["slice_id"] for row in model_rows})
        ncc_differences = []
        mse_differences = []
        for slice_id in slice_ids:
            subset = [
                row for row in model_rows if row["slice_id"] == slice_id
            ]
            ncc_differences.append(
                float(np.mean([row["ncc_composed"] - row["ncc_direct"] for row in subset]))
            )
            mse_differences.append(
                float(np.mean([row["mse_composed"] - row["mse_direct"] for row in subset]))
            )
        result = {
            "comparison": f"{model_name}_composed_minus_direct",
            "n_slices": len(slice_ids),
            "ncc_difference_mean": float(np.mean(ncc_differences)),
            "ncc_difference_std": float(np.std(ncc_differences, ddof=1)),
            "mse_difference_mean": float(np.mean(mse_differences)),
        }
        try:
            from scipy.stats import wilcoxon

            result["wilcoxon_ncc_p"] = float(
                wilcoxon(ncc_differences).pvalue
            )
            result["wilcoxon_mse_p"] = float(
                wilcoxon(mse_differences).pvalue
            )
        except Exception as error:
            result["wilcoxon_error"] = str(error)
        output.append(result)
    return output


@torch.no_grad()
def predict_shared_pair(
    args,
    source,
    target,
    baseline,
    pairwise,
    ldm_model,
):
    """Run both registration models with one shared LDM score extraction."""
    scores = extract_pair_scores(ldm_model, source, target, args.t_enc)
    baseline_dvf, baseline_code = model_forward(
        baseline, source, target, scores, phase_id=None
    )
    pairwise_dvf, pairwise_code = model_forward(
        pairwise, source, target, scores, phase_id=None
    )
    if baseline_code is not None:
        raise RuntimeError("Baseline unexpectedly returned a motion code")
    if pairwise_code is None:
        raise RuntimeError("Pairwise MotionFiLM returned no motion code")
    return {
        "baseline": (baseline_dvf, None),
        "pairwise": (pairwise_dvf, pairwise_code),
    }


@torch.no_grad()
def evaluate(
    args,
    dataset,
    baseline,
    pairwise,
    ldm_model,
    transform,
):
    loader = Data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_multiphase,
    )
    wanted_slices = {
        int(value.strip())
        for value in args.visual_slices.split(",")
        if value.strip()
    }

    direct_rows = []
    adjacent_rows = []
    composition_rows = []
    temporal_rows = []
    cycle_rows = []
    motion_code_rows = []
    figure_cache = {}

    for batch_index, batch in enumerate(loader):
        fixed = batch[0].cuda().float()
        moving_sequence = batch[1].cuda().float()
        names = batch[3]
        if fixed.shape[0] != 1:
            raise ValueError("Use --bs 1")
        if moving_sequence.shape[1] != NUM_PHASES:
            raise ValueError(f"Expected {NUM_PHASES} moving phases")

        images = [fixed] + [
            moving_sequence[:, phase] for phase in range(NUM_PHASES)
        ]
        phase0_mask = body_mask(fixed, args.fg_thr)
        block_id, slice_id = parse_name(names[0])
        if slice_id < 0 and hasattr(dataset, "base_samples"):
            block_id, slice_id = dataset.base_samples[batch_index]

        direct_dvfs = {name: [] for name in MODEL_NAMES}
        direct_warped = {name: [] for name in MODEL_NAMES}
        direct_codes = []

        # Direct spoke flows: phase p -> phase 0.
        for source_phase in range(1, TOTAL_PHASES):
            predictions = predict_shared_pair(
                args,
                images[source_phase],
                images[0],
                baseline,
                pairwise,
                ldm_model,
            )
            for model_name in MODEL_NAMES:
                displacement, motion_code = predictions[model_name]
                warped = warp_tensor(
                    transform, images[source_phase], displacement
                )
                direct_dvfs[model_name].append(displacement)
                direct_warped[model_name].append(warped)
                direct_rows.append(
                    registration_row(
                        model_name=model_name,
                        pair_type="direct_to_phase0",
                        block_id=block_id,
                        slice_id=slice_id,
                        source_phase=source_phase,
                        target_phase=0,
                        source=images[source_phase],
                        target=images[0],
                        target_mask=phase0_mask,
                        displacement=displacement,
                        warped=warped,
                    )
                )
                if model_name == "pairwise":
                    direct_codes.append(motion_code)

        for model_name in MODEL_NAMES:
            sequence = torch.stack(direct_dvfs[model_name], dim=1)
            temporal_rows.append(
                {
                    "model": model_name,
                    "block_id": block_id,
                    "slice_id": slice_id,
                    **temporal_metrics(sequence, phase0_mask),
                }
            )

        motion_codes = torch.stack(direct_codes, dim=1)
        motion_code_rows.append(
            {
                "model": "pairwise",
                "block_id": block_id,
                "slice_id": slice_id,
                **motion_code_metrics(motion_codes),
            }
        )

        # Adjacent edges p -> p-1, followed by the closing edge 0 -> 9.
        adjacent_dvfs = {name: [] for name in MODEL_NAMES}
        for source_phase in range(1, TOTAL_PHASES):
            target_phase = source_phase - 1
            target_mask = body_mask(images[target_phase], args.fg_thr)
            predictions = predict_shared_pair(
                args,
                images[source_phase],
                images[target_phase],
                baseline,
                pairwise,
                ldm_model,
            )
            for model_name in MODEL_NAMES:
                displacement, _ = predictions[model_name]
                warped = warp_tensor(
                    transform, images[source_phase], displacement
                )
                adjacent_dvfs[model_name].append(displacement)
                adjacent_rows.append(
                    registration_row(
                        model_name=model_name,
                        pair_type="adjacent",
                        block_id=block_id,
                        slice_id=slice_id,
                        source_phase=source_phase,
                        target_phase=target_phase,
                        source=images[source_phase],
                        target=images[target_phase],
                        target_mask=target_mask,
                        displacement=displacement,
                        warped=warped,
                    )
                )

        closing_predictions = predict_shared_pair(
            args,
            images[0],
            images[NUM_PHASES],
            baseline,
            pairwise,
            ldm_model,
        )
        phase9_mask = body_mask(images[NUM_PHASES], args.fg_thr)
        closing_dvfs = {}
        for model_name in MODEL_NAMES:
            displacement, _ = closing_predictions[model_name]
            closing_dvfs[model_name] = displacement
            warped = warp_tensor(transform, images[0], displacement)
            adjacent_rows.append(
                registration_row(
                    model_name=model_name,
                    pair_type="cycle_closing",
                    block_id=block_id,
                    slice_id=slice_id,
                    source_phase=0,
                    target_phase=NUM_PHASES,
                    source=images[0],
                    target=images[NUM_PHASES],
                    target_mask=phase9_mask,
                    displacement=displacement,
                    warped=warped,
                )
            )

        cached_models = {}
        for model_name in MODEL_NAMES:
            composed = torch.zeros_like(direct_dvfs[model_name][0])
            composed_warped = []
            composed_dvfs = []
            composed_ncc = []

            for source_phase in range(1, TOTAL_PHASES):
                composed = compose_displacements(
                    transform,
                    composed,
                    adjacent_dvfs[model_name][source_phase - 1],
                )
                source = images[source_phase]
                warped = warp_tensor(transform, source, composed)
                direct_dvf = direct_dvfs[model_name][source_phase - 1]
                direct_image = direct_warped[model_name][source_phase - 1]
                ncc_direct = 1.0 - ncc_loss(
                    fixed, direct_image, mask=phase0_mask
                ).item()
                ncc_composed = 1.0 - ncc_loss(
                    fixed, warped, mask=phase0_mask
                ).item()
                composition_jacobian = jacobian_result(composed)
                composition_rows.append(
                    {
                        "model": model_name,
                        "block_id": block_id,
                        "slice_id": slice_id,
                        "source_phase": source_phase,
                        "ncc_direct": ncc_direct,
                        "ncc_composed": ncc_composed,
                        "ncc_composed_minus_direct": ncc_composed - ncc_direct,
                        "mse_direct": masked_mse(
                            direct_image, fixed, phase0_mask
                        ),
                        "mse_composed": masked_mse(
                            warped, fixed, phase0_mask
                        ),
                        **flow_difference_metrics(
                            composed, direct_dvf, phase0_mask
                        ),
                        "composed_neg_ratio": composition_jacobian["neg_ratio"],
                        "composed_fold_count": composition_jacobian["fold_count"],
                        "composed_min_jac": composition_jacobian["min_jac"],
                    }
                )
                composed_warped.append(warped)
                composed_dvfs.append(composed.clone())
                composed_ncc.append(ncc_composed)

            full_cycle = compose_displacements(
                transform, composed, closing_dvfs[model_name]
            )
            closure_image = warp_tensor(transform, fixed, full_cycle)
            closure_norm = vector_norm_map_px(full_cycle)
            closure_jacobian = jacobian_result(full_cycle)
            cycle_rows.append(
                {
                    "model": model_name,
                    "block_id": block_id,
                    "slice_id": slice_id,
                    "cycle_closure_px": masked_map_mean(
                        closure_norm, phase0_mask
                    ),
                    "cycle_closure_max_px": float(closure_norm.max().item()),
                    "cycle_identity_ncc": 1.0
                    - ncc_loss(
                        fixed, closure_image, mask=phase0_mask
                    ).item(),
                    "cycle_identity_mse": masked_mse(
                        closure_image, fixed, phase0_mask
                    ),
                    "cycle_neg_ratio": closure_jacobian["neg_ratio"],
                    "cycle_fold_count": closure_jacobian["fold_count"],
                    "cycle_min_jac": closure_jacobian["min_jac"],
                }
            )

            if slice_id in wanted_slices:
                cached_models[model_name] = {
                    "direct_warped": [x.detach().cpu() for x in direct_warped[model_name]],
                    "composed_warped": [x.detach().cpu() for x in composed_warped],
                    "direct_ncc": [
                        1.0
                        - ncc_loss(
                            fixed,
                            image,
                            mask=phase0_mask,
                        ).item()
                        for image in direct_warped[model_name]
                    ],
                    "composed_ncc": composed_ncc,
                }

        if slice_id in wanted_slices:
            figure_cache[slice_id] = {
                "fixed": fixed.detach().cpu(),
                "moving": moving_sequence.detach().cpu(),
                "models": cached_models,
            }

        print(
            f"[Eval] sequence {batch_index + 1}/{len(loader)} "
            f"block={block_id} slice={slice_id}"
        )

    return {
        "direct_rows": direct_rows,
        "adjacent_rows": adjacent_rows,
        "composition_rows": composition_rows,
        "temporal_rows": temporal_rows,
        "cycle_rows": cycle_rows,
        "motion_code_rows": motion_code_rows,
        "figure_cache": figure_cache,
    }


def save_figures(args, figure_cache):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for slice_id, cache in sorted(figure_cache.items()):
        fixed = cache["fixed"][0, 0].numpy()
        moving = cache["moving"]
        for model_name in MODEL_NAMES:
            model_cache = cache["models"][model_name]
            fig, axes = plt.subplots(6, NUM_PHASES, figsize=(27, 18))
            for phase in range(NUM_PHASES):
                moving_image = moving[0, phase, 0].numpy()
                direct = model_cache["direct_warped"][phase][0, 0].numpy()
                composed = model_cache["composed_warped"][phase][0, 0].numpy()
                images = (
                    moving_image,
                    fixed,
                    direct,
                    composed,
                    np.abs(fixed - direct),
                    np.abs(fixed - composed),
                )
                for row_index, image in enumerate(images):
                    axes[row_index, phase].imshow(
                        image,
                        cmap="gray" if row_index < 4 else "magma",
                    )
                    axes[row_index, phase].set_xticks([])
                    axes[row_index, phase].set_yticks([])
                axes[2, phase].set_title(
                    f"P{phase + 1} NCC={model_cache['direct_ncc'][phase]:.4f}",
                    fontsize=8,
                )
                axes[3, phase].set_title(
                    f"P{phase + 1} NCC={model_cache['composed_ncc'][phase]:.4f}",
                    fontsize=8,
                )
            row_labels = (
                "Moving",
                "Fixed P0",
                "Direct warped",
                "Adjacent-composed warped",
                "Direct |F-W|",
                "Composed |F-W|",
            )
            for row_index, label in enumerate(row_labels):
                axes[row_index, 0].set_ylabel(label, fontsize=9)
            fig.suptitle(
                f"{args.split} slice {slice_id}: {model_name} direct vs adjacent composition",
                fontsize=14,
            )
            path = os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_{model_name}_cyclic_diagnostic.png",
            )
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Figure] {path}")


def print_summary(
    direct_summary,
    adjacent_summary,
    composition_summary,
    temporal_summary,
    cycle_summary,
    motion_code_summary,
    paired_statistics,
):
    print("\n" + "=" * 112)
    print("DIRECT PHASE-TO-PHASE0 REGISTRATION")
    print("=" * 112)
    for row in direct_summary:
        print(
            f"{row['model']:<10} "
            f"NCC={row['ncc_before_mean']:.6f}->{row['ncc_after_mean']:.6f} "
            f"negR={100.0 * row['neg_ratio_mean']:.4f}% "
            f"folds/image={row['fold_count_mean']:.2f}"
        )

    print("\n" + "=" * 112)
    print("ADJACENT-PAIR OOD STRESS TEST")
    print("=" * 112)
    for row in adjacent_summary:
        print(
            f"{row['model']:<10} {row['pair_type']:<14} "
            f"NCC={row['ncc_before_mean']:.6f}->{row['ncc_after_mean']:.6f} "
            f"delta={row['ncc_delta_mean']:+.6f} "
            f"negR={100.0 * row['neg_ratio_mean']:.4f}%"
        )

    print("\n" + "=" * 112)
    print("ADJACENT COMPOSITION VS DIRECT FLOW")
    print("=" * 112)
    for row in composition_summary:
        print(
            f"{row['model']:<10} "
            f"direct={row['ncc_direct_mean']:.6f} "
            f"composed={row['ncc_composed_mean']:.6f} "
            f"delta={row['ncc_composed_minus_direct_mean']:+.6f} "
            f"DVF disagreement={row['dvf_difference_px_mean']:.4f}px"
        )

    print("\n" + "=" * 112)
    print("DIRECT-SPOKE TEMPORAL DIAGNOSTICS")
    print("=" * 112)
    for row in temporal_summary:
        print(
            f"{row['model']:<10} "
            f"velocity={row['cycle_velocity_px_mean']:.4f}px "
            f"acceleration={row['cycle_acceleration_px_mean']:.4f}px "
            f"P9->P0 spoke gap={row['phase9_to_phase0_spoke_gap_px_mean']:.4f}px"
        )

    print("\n" + "=" * 112)
    print("FULL ADJACENT CYCLE CLOSURE")
    print("=" * 112)
    for row in cycle_summary:
        print(
            f"{row['model']:<10} "
            f"closure={row['cycle_closure_px_mean']:.4f}px "
            f"identity NCC={row['cycle_identity_ncc_mean']:.6f} "
            f"negR={100.0 * row['cycle_neg_ratio_mean']:.4f}%"
        )

    if motion_code_summary:
        row = motion_code_summary[0]
        print("\n" + "=" * 112)
        print("PAIRWISE MOTION-CODE SEPARABILITY")
        print("=" * 112)
        print(
            f"std/norm={row['motion_code_std_norm_ratio_mean']:.6e} "
            f"raw cosine={row['motion_code_offdiag_cosine_mean']:.6f} "
            f"after-LayerNorm cosine={row['layernorm_offdiag_cosine_mean']:.6f}"
        )

    print("\nSLICE-LEVEL COMPOSITION TESTS")
    for result in paired_statistics:
        suffix = ""
        if "wilcoxon_ncc_p" in result:
            suffix = f" | p={result['wilcoxon_ncc_p']:.6g}"
        print(
            f"{result['comparison']}: "
            f"NCC={result['ncc_difference_mean']:+.6f}{suffix}"
        )

    print("\nInterpretation note:")
    print(
        "Adjacent pairs are out of the training distribution because the current "
        "models were trained with phase0 as fixed. Poor adjacent results alone do "
        "not disprove cyclic modeling; they mean an adjacent-pair model must be "
        "trained before using path consistency."
    )


def main():
    args = parse_args()
    if args.bs != 1:
        raise ValueError("Use --bs 1; flow composition is evaluated per slice")
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed + 1000)

    ldm_model = load_ldm(args.ldm_config, args.ldm_ckpt)
    baseline = load_registration_model(
        args.baseline_ckpt,
        use_motion_film=False,
        use_ldm=not args.no_ldm,
    )
    pairwise = load_registration_model(
        args.pairwise_ckpt,
        use_motion_film=True,
        use_ldm=not args.no_ldm,
    )
    transform = SpatialTransform().cuda().eval()
    for parameter in transform.parameters():
        parameter.requires_grad_(False)

    dataset = MultiPhaseDataset(
        data_root=args.data_root,
        split=args.split,
        flip_p=0.0,
        normalize=True,
    )
    print(
        f"[Data] split={args.split} sequences={len(dataset)} "
        f"direct_pairs={len(dataset) * NUM_PHASES} "
        f"adjacent_pairs={len(dataset) * TOTAL_PHASES}"
    )

    results = evaluate(
        args,
        dataset,
        baseline,
        pairwise,
        ldm_model,
        transform,
    )

    direct_summary = aggregate(
        results["direct_rows"],
        ("model",),
        (
            "ncc_before",
            "ncc_after",
            "ncc_delta",
            "mse_after",
            "neg_ratio",
            "fold_count",
            "min_jac",
            "mean_dvf_px",
        ),
    )
    adjacent_summary = aggregate(
        results["adjacent_rows"],
        ("model", "pair_type"),
        (
            "ncc_before",
            "ncc_after",
            "ncc_delta",
            "mse_after",
            "neg_ratio",
            "fold_count",
            "min_jac",
        ),
    )
    adjacent_per_edge = aggregate(
        results["adjacent_rows"],
        ("model", "pair_type", "source_phase", "target_phase"),
        ("ncc_before", "ncc_after", "ncc_delta", "neg_ratio"),
    )
    composition_summary = aggregate(
        results["composition_rows"],
        ("model",),
        (
            "ncc_direct",
            "ncc_composed",
            "ncc_composed_minus_direct",
            "mse_direct",
            "mse_composed",
            "dvf_difference_px",
            "dvf_difference_max_px",
            "composed_neg_ratio",
            "composed_fold_count",
            "composed_min_jac",
        ),
    )
    composition_per_phase = aggregate(
        results["composition_rows"],
        ("model", "source_phase"),
        (
            "ncc_direct",
            "ncc_composed",
            "ncc_composed_minus_direct",
            "dvf_difference_px",
            "composed_neg_ratio",
        ),
    )
    temporal_summary = aggregate(
        results["temporal_rows"],
        ("model",),
        (
            "cycle_velocity_px",
            "internal_velocity_px",
            "cycle_acceleration_px",
            "phase9_to_phase0_spoke_gap_px",
        ),
    )
    cycle_summary = aggregate(
        results["cycle_rows"],
        ("model",),
        (
            "cycle_closure_px",
            "cycle_closure_max_px",
            "cycle_identity_ncc",
            "cycle_identity_mse",
            "cycle_neg_ratio",
            "cycle_fold_count",
            "cycle_min_jac",
        ),
    )
    motion_code_summary = aggregate(
        results["motion_code_rows"],
        ("model",),
        (
            "motion_code_std",
            "motion_code_norm",
            "motion_code_std_norm_ratio",
            "motion_code_consecutive_l2",
            "motion_code_offdiag_cosine",
            "layernorm_offdiag_cosine",
            "layernorm_offdiag_cosine_min",
            "layernorm_offdiag_cosine_max",
        ),
    )
    paired_statistics = slice_level_paired_statistics(
        results["composition_rows"]
    )

    output_tables = {
        "direct_pair_metrics.csv": results["direct_rows"],
        "direct_summary.csv": direct_summary,
        "adjacent_pair_metrics.csv": results["adjacent_rows"],
        "adjacent_summary.csv": adjacent_summary,
        "adjacent_per_edge_summary.csv": adjacent_per_edge,
        "composition_metrics.csv": results["composition_rows"],
        "composition_summary.csv": composition_summary,
        "composition_per_phase_summary.csv": composition_per_phase,
        "temporal_per_slice.csv": results["temporal_rows"],
        "temporal_summary.csv": temporal_summary,
        "cycle_per_slice.csv": results["cycle_rows"],
        "cycle_summary.csv": cycle_summary,
        "motion_code_per_slice.csv": results["motion_code_rows"],
        "motion_code_summary.csv": motion_code_summary,
    }
    for filename, rows in output_tables.items():
        write_csv(os.path.join(args.save_dir, filename), rows)

    report = {
        "warning": (
            "Adjacent-pair evaluation is an OOD stress test because the current "
            "models were trained with phase0 as fixed."
        ),
        "direct_summary": direct_summary,
        "adjacent_summary": adjacent_summary,
        "composition_summary": composition_summary,
        "temporal_summary": temporal_summary,
        "cycle_summary": cycle_summary,
        "motion_code_summary": motion_code_summary,
        "paired_statistics": paired_statistics,
    }
    with open(
        os.path.join(args.save_dir, "diagnostic_report.json"),
        "w",
    ) as handle:
        json.dump(report, handle, indent=2)

    if not args.no_figures:
        save_figures(args, results["figure_cache"])
    print_summary(
        direct_summary,
        adjacent_summary,
        composition_summary,
        temporal_summary,
        cycle_summary,
        motion_code_summary,
        paired_statistics,
    )
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
