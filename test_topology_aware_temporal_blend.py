"""Topology-aware local temporal blending for multi-phase registration.

This validation-only script compares three deterministic post-processors:

  1. direct: the original phase-to-phase0 DVF;
  2. global: a fixed global blend with the adjacent-phase DVF average;
  3. risk: the same temporal correction applied selectively near low-Jacobian
     regions of the direct DVF.

For phase p, the adjacent temporal reference and refined field are

    T_p = 0.5 * (D_{p-1} + D_{p+1}),
    D'_p(x) = D_p(x) + alpha * w_p(x) * (T_p(x) - D_p(x)),

where D_0 = D_10 = 0. The soft risk weight w is derived from the direct
Jacobian determinant, spatially dilated and blurred, and constrained to a
dilated foreground support. This avoids improving the global folding metric by
editing only empty background.

No model is trained and no checkpoint is written. Use validation data to choose
the deterministic configuration. Run test data only after freezing the rule.
"""

import argparse
import csv
import json
import os

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


SOURCE_NAMES = ("baseline", "pairwise")


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
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Use val for method selection; reserve test for final evaluation.",
    )
    parser.add_argument("--global_alpha", type=float, default=0.10)
    parser.add_argument("--risk_alphas", default="0.1,0.2,0.4,0.6,1.0")
    parser.add_argument("--risk_thresholds", default="0,0.1,0.2")
    parser.add_argument("--risk_temperature", type=float, default=0.05)
    parser.add_argument("--risk_dilate", type=int, default=9)
    parser.add_argument("--risk_blur", type=int, default=15)
    parser.add_argument("--support_dilate", type=int, default=31)
    parser.add_argument(
        "--ncc_budgets",
        default="0.0001,0.0005,0.001",
        help="Allowed mean NCC drop from direct for selecting a configuration.",
    )
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
        default="./logs/Topology_Aware_Temporal_Blend_val",
    )
    return parser.parse_args()


def parse_float_list(text, name):
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"Invalid {name}: {text}") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def validate_kernel(value, name):
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")


def float_token(value):
    text = f"{value:.3f}"
    return text.replace("-", "m").replace(".", "p")


def make_configurations(global_alpha, risk_alphas, risk_thresholds):
    configurations = [
        {
            "method": "direct",
            "config": "direct",
            "alpha": 0.0,
            "risk_threshold": float("nan"),
        },
        {
            "method": "global",
            "config": f"global_a{float_token(global_alpha)}",
            "alpha": global_alpha,
            "risk_threshold": float("nan"),
        },
    ]
    for threshold in risk_thresholds:
        for alpha in risk_alphas:
            configurations.append(
                {
                    "method": "risk",
                    "config": (
                        f"risk_t{float_token(threshold)}_a{float_token(alpha)}"
                    ),
                    "alpha": alpha,
                    "risk_threshold": threshold,
                }
            )
    return configurations


def extract_model_state(payload, path):
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "state_dict", "pairwise_model_state_dict"):
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
    model = build_model(use_motion_film=use_motion_film, use_ldm=use_ldm)
    model.load_state_dict(extract_model_state(payload, path), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"[Model] loaded {path} | use_motion_film={use_motion_film} | strict=True"
    )
    return model


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def masked_mse(prediction, target, mask):
    numerator = (((prediction - target) ** 2) * mask).sum()
    return float((numerator / mask.sum().clamp(min=1.0)).item())


def masked_ssim(prediction, target, mask, window=11):
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denominator = (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp(min=1e-8)
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / denominator
    return float(((ssim_map * mask).sum() / mask.sum().clamp(min=1.0)).item())


def displacement_to_pixel_numpy(displacement):
    dvf = displacement[0].detach().cpu().numpy().copy()
    height, width = dvf.shape[-2:]
    dvf[0] *= height / 2.0
    dvf[1] *= width / 2.0
    return dvf


def jacobian_map(displacement):
    return jacobian_determinant_vxm(displacement_to_pixel_numpy(displacement))


def resize_mask_numpy(mask, shape):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    if tensor.shape[-2:] != tuple(shape):
        tensor = F.interpolate(tensor, size=shape, mode="nearest")
    return tensor[0, 0].numpy() > 0.5


def jacobian_result(displacement, foreground):
    determinant = jacobian_map(displacement)
    dvf = displacement_to_pixel_numpy(displacement)
    magnitude = np.sqrt(dvf[0] ** 2 + dvf[1] ** 2)
    foreground_np = foreground[0, 0].detach().cpu().numpy() > 0.5
    foreground_np = resize_mask_numpy(foreground_np, determinant.shape)
    foreground_count = max(int(foreground_np.sum()), 1)
    foreground_negative = np.logical_and(determinant < 0, foreground_np)
    return {
        "neg_ratio": float((determinant < 0).mean()),
        "fold_count": int((determinant < 0).sum()),
        "fg_neg_ratio": float(foreground_negative.sum() / foreground_count),
        "fg_fold_count": int(foreground_negative.sum()),
        "min_jac": float(determinant.min()),
        "fg_min_jac": float(determinant[foreground_np].min())
        if foreground_np.any()
        else float("nan"),
        "mean_jac": float(determinant.mean()),
        "mean_dvf_px": float(magnitude.mean()),
        "max_dvf_px": float(magnitude.max()),
        "jacobian_map": determinant,
    }


def smooth_support(foreground, support_dilate, blur):
    support = F.max_pool2d(
        foreground,
        kernel_size=support_dilate,
        stride=1,
        padding=support_dilate // 2,
    )
    support = F.avg_pool2d(
        support,
        kernel_size=blur,
        stride=1,
        padding=blur // 2,
    )
    return support.clamp(0.0, 1.0)


def build_risk_mask(
    displacement,
    foreground,
    threshold,
    temperature,
    risk_dilate,
    risk_blur,
    support_dilate,
):
    determinant = jacobian_map(displacement)
    determinant_t = torch.from_numpy(determinant.astype(np.float32))[None, None]
    determinant_t = determinant_t.to(
        device=displacement.device,
        dtype=displacement.dtype,
    )
    if determinant_t.shape[-2:] != displacement.shape[-2:]:
        determinant_t = F.interpolate(
            determinant_t,
            size=displacement.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    risk = torch.sigmoid((float(threshold) - determinant_t) / temperature)
    risk = F.max_pool2d(
        risk,
        kernel_size=risk_dilate,
        stride=1,
        padding=risk_dilate // 2,
    )
    risk = F.avg_pool2d(
        risk,
        kernel_size=risk_blur,
        stride=1,
        padding=risk_blur // 2,
    )
    support = smooth_support(foreground, support_dilate, risk_blur)
    return (risk * support).clamp(0.0, 1.0)


def risk_mask_metrics(risk_mask, foreground):
    denominator = foreground.sum().clamp(min=1.0)
    return {
        "risk_weight_mean": float(risk_mask.mean().item()),
        "risk_weight_fg_mean": float(
            ((risk_mask * foreground).sum() / denominator).item()
        ),
        "risk_weight_max": float(risk_mask.max().item()),
        "risk_fraction_gt_0p5": float((risk_mask > 0.5).float().mean().item()),
        "risk_fg_fraction_gt_0p5": float(
            ((((risk_mask > 0.5).float()) * foreground).sum() / denominator).item()
        ),
    }


def temporal_linear_reference(direct_sequence):
    zero = torch.zeros_like(direct_sequence[:, :1])
    closed = torch.cat([zero, direct_sequence, zero], dim=1)
    return 0.5 * (closed[:, :-2] + closed[:, 2:])


def build_risk_sequence(
    direct_sequence,
    foreground,
    threshold,
    temperature,
    risk_dilate,
    risk_blur,
    support_dilate,
):
    masks = []
    for phase in range(NUM_PHASES):
        masks.append(
            build_risk_mask(
                direct_sequence[:, phase],
                foreground,
                threshold,
                temperature,
                risk_dilate,
                risk_blur,
                support_dilate,
            )
        )
    return torch.stack(masks, dim=1)


def refine_sequence(direct_sequence, temporal_reference, configuration, risk_masks):
    if configuration["method"] == "direct":
        return direct_sequence
    correction = temporal_reference - direct_sequence
    if configuration["method"] == "global":
        return direct_sequence + configuration["alpha"] * correction
    threshold = configuration["risk_threshold"]
    return direct_sequence + configuration["alpha"] * risk_masks[threshold] * correction


def to_pixel_sequence(displacement_sequence):
    result = displacement_sequence.clone()
    height, width = result.shape[-2:]
    result[:, :, 0] *= height / 2.0
    result[:, :, 1] *= width / 2.0
    return result


def sequence_difference_px(first, second):
    difference = to_pixel_sequence(first - second)
    magnitude = torch.sqrt(difference.square().sum(dim=2) + 1e-12)
    return float(magnitude.mean().item()), float(magnitude.max().item())


def masked_temporal_mean(value_map, foreground):
    mask = foreground[:, 0, None]
    denominator = (mask.sum() * value_map.shape[1]).clamp(min=1.0)
    return float(((value_map * mask).sum() / denominator).item())


def trajectory_metrics(displacement_sequence, direct_sequence, foreground):
    sequence_px = to_pixel_sequence(displacement_sequence)
    zero = torch.zeros_like(sequence_px[:, :1])
    closed = torch.cat([zero, sequence_px, zero], dim=1)
    velocity = closed[:, 1:] - closed[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    velocity_norm = torch.sqrt(velocity.square().sum(dim=2) + 1e-12)
    acceleration_norm = torch.sqrt(acceleration.square().sum(dim=2) + 1e-12)
    endpoint_norm = torch.sqrt(sequence_px[:, -1].square().sum(dim=1) + 1e-12)
    endpoint = float(
        ((endpoint_norm * foreground[:, 0]).sum() / foreground[:, 0].sum().clamp(min=1.0)).item()
    )
    change_mean, change_max = sequence_difference_px(
        displacement_sequence,
        direct_sequence,
    )
    return {
        "phase9_to_phase0_gap_px": endpoint,
        "temporal_velocity_px": masked_temporal_mean(velocity_norm, foreground),
        "temporal_acceleration_px": masked_temporal_mean(
            acceleration_norm,
            foreground,
        ),
        "change_from_direct_px": change_mean,
        "change_from_direct_max_px": change_max,
    }


def warp_tensor(transform, image, displacement):
    _, warped = transform(image, displacement.permute(0, 2, 3, 1))
    return warped


@torch.no_grad()
def raw_sequences(args, fixed, moving_sequence, baseline, pairwise, ldm_model):
    score_cache = [
        extract_pair_scores(ldm_model, moving_sequence[:, phase], fixed, args.t_enc)
        for phase in range(NUM_PHASES)
    ]
    output = {source: [] for source in SOURCE_NAMES}
    for phase in range(NUM_PHASES):
        moving = moving_sequence[:, phase]
        baseline_dvf, _ = model_forward(
            baseline,
            moving,
            fixed,
            score_cache[phase],
            phase_id=None,
        )
        pairwise_dvf, motion_code = model_forward(
            pairwise,
            moving,
            fixed,
            score_cache[phase],
            phase_id=None,
        )
        if motion_code is None:
            raise RuntimeError("Pairwise MotionFiLM returned no motion code")
        output["baseline"].append(baseline_dvf)
        output["pairwise"].append(pairwise_dvf)
    return {
        source: torch.stack(displacements, dim=1)
        for source, displacements in output.items()
    }


def registration_metrics(fixed, moving, warped, foreground, displacement):
    ncc_before = 1.0 - ncc_loss(fixed, moving, mask=foreground).item()
    ncc_after = 1.0 - ncc_loss(fixed, warped, mask=foreground).item()
    jacobian = jacobian_result(displacement, foreground)
    return {
        "ncc_before": ncc_before,
        "ncc_after": ncc_after,
        "ncc_delta": ncc_after - ncc_before,
        "mse_before": masked_mse(moving, fixed, foreground),
        "mse_after": masked_mse(warped, fixed, foreground),
        "ssim_before": masked_ssim(moving, fixed, foreground),
        "ssim_after": masked_ssim(warped, fixed, foreground),
        **{key: value for key, value in jacobian.items() if key != "jacobian_map"},
    }


@torch.no_grad()
def evaluate(
    args,
    configurations,
    risk_thresholds,
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
        int(value.strip()) for value in args.visual_slices.split(",") if value.strip()
    }
    pair_rows = []
    trajectory_rows = []
    risk_rows = []
    figure_cache = {}

    seed_everything(args.seed + 1000)
    for batch_index, batch in enumerate(loader):
        fixed = batch[0].cuda().float()
        moving_sequence = batch[1].cuda().float()
        names = batch[3]
        block_id = int(batch[4][0].item())
        slice_id = int(batch[5][0].item())
        if fixed.shape[0] != 1:
            raise ValueError("Use --bs 1")
        foreground = body_mask(fixed, args.fg_thr)
        direct_sequences = raw_sequences(
            args,
            fixed,
            moving_sequence,
            baseline,
            pairwise,
            ldm_model,
        )

        for source in SOURCE_NAMES:
            direct_sequence = direct_sequences[source]
            temporal_reference = temporal_linear_reference(direct_sequence)
            risk_masks = {
                threshold: build_risk_sequence(
                    direct_sequence,
                    foreground,
                    threshold,
                    args.risk_temperature,
                    args.risk_dilate,
                    args.risk_blur,
                    args.support_dilate,
                )
                for threshold in risk_thresholds
            }
            for threshold, masks in risk_masks.items():
                for phase in range(NUM_PHASES):
                    risk_rows.append(
                        {
                            "source": source,
                            "risk_threshold": threshold,
                            "block_id": block_id,
                            "slice_id": slice_id,
                            "pairname": names[0],
                            "phase": phase + 1,
                            **risk_mask_metrics(masks[:, phase], foreground),
                        }
                    )

            for configuration in configurations:
                displacement_sequence = refine_sequence(
                    direct_sequence,
                    temporal_reference,
                    configuration,
                    risk_masks,
                )
                base_fields = {
                    "source": source,
                    "method": configuration["method"],
                    "config": configuration["config"],
                    "alpha": configuration["alpha"],
                    "risk_threshold": configuration["risk_threshold"],
                    "block_id": block_id,
                    "slice_id": slice_id,
                    "pairname": names[0],
                }
                trajectory_rows.append(
                    {
                        **base_fields,
                        **trajectory_metrics(
                            displacement_sequence,
                            direct_sequence,
                            foreground,
                        ),
                    }
                )
                for phase in range(NUM_PHASES):
                    moving = moving_sequence[:, phase]
                    displacement = displacement_sequence[:, phase]
                    warped = warp_tensor(transform, moving, displacement)
                    pair_rows.append(
                        {
                            **base_fields,
                            "phase": phase + 1,
                            **registration_metrics(
                                fixed,
                                moving,
                                warped,
                                foreground,
                                displacement,
                            ),
                        }
                    )

            if slice_id in wanted_slices:
                figure_cache[(slice_id, source)] = {
                    "fixed": fixed.detach().cpu(),
                    "moving": moving_sequence.detach().cpu(),
                    "foreground": foreground.detach().cpu(),
                    "direct": direct_sequence.detach().cpu(),
                }
        print(f"[Topology-aware] sequence {batch_index + 1}/{len(loader)}")

    return pair_rows, trajectory_rows, risk_rows, figure_cache


PAIR_METRICS = (
    "ncc_before",
    "ncc_after",
    "ncc_delta",
    "mse_after",
    "ssim_after",
    "neg_ratio",
    "fold_count",
    "fg_neg_ratio",
    "fg_fold_count",
    "min_jac",
    "fg_min_jac",
    "mean_dvf_px",
    "max_dvf_px",
)


TRAJECTORY_METRICS = (
    "phase9_to_phase0_gap_px",
    "temporal_velocity_px",
    "temporal_acceleration_px",
    "change_from_direct_px",
    "change_from_direct_max_px",
)


RISK_METRICS = (
    "risk_weight_mean",
    "risk_weight_fg_mean",
    "risk_weight_max",
    "risk_fraction_gt_0p5",
    "risk_fg_fraction_gt_0p5",
)


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std, float(np.median(values))


def aggregate(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, subset in sorted(grouped.items()):
        record = {name: value for name, value in zip(group_keys, key)}
        record["samples"] = len(subset)
        first = subset[0]
        for name in ("method", "alpha", "risk_threshold"):
            if name in first and name not in record:
                record[name] = first[name]
        for metric in metric_keys:
            mean, std, median = finite_stats([row[metric] for row in subset])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_median"] = median
        output.append(record)
    return output


def add_direct_differences(summary_rows, extra_keys=()):
    direct = {}
    for row in summary_rows:
        if row["method"] == "direct":
            key = (row["source"],) + tuple(row[name] for name in extra_keys)
            direct[key] = row
    for row in summary_rows:
        key = (row["source"],) + tuple(row[name] for name in extra_keys)
        reference = direct[key]
        for metric in (
            "ncc_after",
            "ssim_after",
            "mse_after",
            "neg_ratio",
            "fold_count",
            "fg_neg_ratio",
            "fg_fold_count",
        ):
            row[f"{metric}_change_from_direct"] = (
                row[f"{metric}_mean"] - reference[f"{metric}_mean"]
            )
    return summary_rows


def pareto_front(summary_rows, topology_metric):
    output = []
    metric_key = f"{topology_metric}_mean"
    for source in SOURCE_NAMES:
        subset = [row for row in summary_rows if row["source"] == source]
        for candidate in subset:
            dominated = False
            for other in subset:
                accuracy_no_worse = other["ncc_after_mean"] >= candidate["ncc_after_mean"] - 1e-12
                topology_no_worse = other[metric_key] <= candidate[metric_key] + 1e-12
                strictly_better = (
                    other["ncc_after_mean"] > candidate["ncc_after_mean"] + 1e-12
                    or other[metric_key] < candidate[metric_key] - 1e-12
                )
                if accuracy_no_worse and topology_no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                record = dict(candidate)
                record["pareto_topology_metric"] = topology_metric
                output.append(record)
    return sorted(output, key=lambda row: (row["source"], -row["ncc_after_mean"]))


def select_under_budgets(summary_rows, budgets, topology_metric):
    output = []
    topology_key = f"{topology_metric}_mean"
    for source in SOURCE_NAMES:
        subset = [row for row in summary_rows if row["source"] == source]
        direct = next(row for row in subset if row["method"] == "direct")
        global_row = next(row for row in subset if row["method"] == "global")
        for budget in budgets:
            eligible = [
                row
                for row in subset
                if row["ncc_after_mean"] >= direct["ncc_after_mean"] - budget - 1e-12
            ]
            selected = min(
                eligible,
                key=lambda row: (
                    row[topology_key],
                    row["fg_fold_count_mean"],
                    row["neg_ratio_mean"],
                    -row["ncc_after_mean"],
                ),
            )
            output.append(
                {
                    "source": source,
                    "topology_metric": topology_metric,
                    "ncc_drop_budget": budget,
                    "selected_config": selected["config"],
                    "selected_method": selected["method"],
                    "selected_alpha": selected["alpha"],
                    "selected_risk_threshold": selected["risk_threshold"],
                    "direct_ncc": direct["ncc_after_mean"],
                    "selected_ncc": selected["ncc_after_mean"],
                    "ncc_change": selected["ncc_after_change_from_direct"],
                    "direct_topology": direct[topology_key],
                    "selected_topology": selected[topology_key],
                    "topology_change": (
                        selected[topology_key] - direct[topology_key]
                    ),
                    "direct_fg_folds": direct["fg_fold_count_mean"],
                    "selected_fg_folds": selected["fg_fold_count_mean"],
                    "fg_folds_change": selected["fg_fold_count_change_from_direct"],
                    "direct_global_neg_ratio": direct["neg_ratio_mean"],
                    "selected_global_neg_ratio": selected["neg_ratio_mean"],
                    "global_neg_ratio_change": selected[
                        "neg_ratio_change_from_direct"
                    ],
                    "global_config": global_row["config"],
                    "global_ncc_change": global_row["ncc_after_change_from_direct"],
                    "global_topology_change": (
                        global_row[topology_key] - direct[topology_key]
                    ),
                }
            )
    return output


def paired_statistics(pair_rows, configurations):
    indexed = {
        (row["source"], row["config"], row["slice_id"], row["phase"]): row
        for row in pair_rows
    }
    slice_ids = sorted({row["slice_id"] for row in pair_rows})
    metrics = (
        "ncc_after",
        "ssim_after",
        "mse_after",
        "neg_ratio",
        "fold_count",
        "fg_neg_ratio",
        "fg_fold_count",
    )
    slice_rows = []
    results = []
    for source in SOURCE_NAMES:
        for configuration in configurations:
            if configuration["method"] == "direct":
                continue
            config = configuration["config"]
            differences_by_metric = {metric: [] for metric in metrics}
            for slice_id in slice_ids:
                record = {
                    "comparison": f"{source}:{config}_minus_direct",
                    "source": source,
                    "config": config,
                    "method": configuration["method"],
                    "alpha": configuration["alpha"],
                    "risk_threshold": configuration["risk_threshold"],
                    "slice_id": slice_id,
                }
                for metric in metrics:
                    differences = []
                    for phase in range(1, NUM_PHASES + 1):
                        current = indexed[(source, config, slice_id, phase)]
                        direct = indexed[(source, "direct", slice_id, phase)]
                        differences.append(current[metric] - direct[metric])
                    value = float(np.mean(differences))
                    record[f"{metric}_difference"] = value
                    differences_by_metric[metric].append(value)
                slice_rows.append(record)
            result = {
                "comparison": f"{source}:{config}_minus_direct",
                "source": source,
                "config": config,
                "method": configuration["method"],
                "alpha": configuration["alpha"],
                "risk_threshold": configuration["risk_threshold"],
                "n_slices": len(slice_ids),
            }
            for metric, values in differences_by_metric.items():
                result[f"{metric}_difference_mean"] = float(np.mean(values))
                result[f"{metric}_difference_std"] = float(np.std(values, ddof=1))
                try:
                    from scipy.stats import wilcoxon

                    result[f"wilcoxon_{metric}_p"] = float(wilcoxon(values).pvalue)
                except Exception as error:
                    result[f"wilcoxon_{metric}_error"] = str(error)
            results.append(result)
    return slice_rows, results


def configuration_lookup(configurations):
    return {configuration["config"]: configuration for configuration in configurations}


def selected_configurations(budget_rows, source, preferred_budget=0.001):
    source_rows = [row for row in budget_rows if row["source"] == source]
    selected = min(
        source_rows,
        key=lambda row: abs(row["ncc_drop_budget"] - preferred_budget),
    )
    return selected["selected_config"]


def save_tradeoff_figures(args, summary_rows, pareto_rows, budget_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for source in SOURCE_NAMES:
        rows = [row for row in summary_rows if row["source"] == source]
        pareto_configs = {
            row["config"] for row in pareto_rows if row["source"] == source
        }
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = {"direct": "black", "global": "tab:blue", "risk": "tab:green"}
        markers = {"direct": "*", "global": "s", "risk": "o"}
        for row in rows:
            color = colors[row["method"]]
            alpha_value = 1.0 if row["config"] in pareto_configs else 0.35
            axes[0].scatter(
                100.0 * row["fg_neg_ratio_mean"],
                row["ncc_after_mean"],
                color=color,
                marker=markers[row["method"]],
                alpha=alpha_value,
                s=70,
            )
            axes[1].scatter(
                100.0 * row["neg_ratio_mean"],
                row["ncc_after_mean"],
                color=color,
                marker=markers[row["method"]],
                alpha=alpha_value,
                s=70,
            )
        axes[0].set_xlabel("Foreground negative Jacobian ratio (%)")
        axes[1].set_xlabel("Global negative Jacobian ratio (%)")
        for axis in axes:
            axis.set_ylabel("Mean NCC")
            axis.grid(alpha=0.3)
        selected = selected_configurations(budget_rows, source)
        fig.suptitle(
            f"{source}: topology-aware temporal blending | budget 0.001: {selected}"
        )
        path = os.path.join(figure_dir, f"{source}_topology_aware_tradeoff.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Figure] {path}")


@torch.no_grad()
def save_registration_figures(
    args,
    configurations,
    budget_rows,
    figure_cache,
    transform,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = configuration_lookup(configurations)
    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for (slice_id, source), cache in sorted(figure_cache.items()):
        selected_name = selected_configurations(budget_rows, source)
        names = ["direct"]
        global_name = next(
            configuration["config"]
            for configuration in configurations
            if configuration["method"] == "global"
        )
        names.append(global_name)
        if selected_name not in names:
            names.append(selected_name)

        fixed = cache["fixed"].cuda().float()
        moving = cache["moving"].cuda().float()
        foreground = cache["foreground"].cuda().float()
        direct = cache["direct"].cuda().float()
        temporal = temporal_linear_reference(direct)
        rendered = {}
        for name in names:
            configuration = lookup[name]
            risk_masks = {}
            if configuration["method"] == "risk":
                risk_masks[configuration["risk_threshold"]] = build_risk_sequence(
                    direct,
                    foreground,
                    configuration["risk_threshold"],
                    args.risk_temperature,
                    args.risk_dilate,
                    args.risk_blur,
                    args.support_dilate,
                )
            sequence = refine_sequence(direct, temporal, configuration, risk_masks)
            warped_list = []
            ncc_list = []
            for phase in range(NUM_PHASES):
                warped = warp_tensor(transform, moving[:, phase], sequence[:, phase])
                warped_list.append(warped.detach().cpu())
                ncc_list.append(
                    1.0 - ncc_loss(fixed, warped, mask=foreground).item()
                )
            rendered[name] = {
                "sequence": sequence.detach().cpu(),
                "warped": warped_list,
                "ncc": ncc_list,
            }

        fixed_np = fixed[0, 0].detach().cpu().numpy()
        moving_cpu = moving.detach().cpu()
        for name in names[1:]:
            fig, axes = plt.subplots(8, NUM_PHASES, figsize=(27, 23))
            current = rendered[name]
            direct_render = rendered["direct"]
            row_labels = (
                "Moving",
                "Fixed",
                "Direct warped",
                f"{name} warped",
                "Direct |F-W|",
                f"{name} |F-W|",
                "Direct Jacobian",
                f"{name} Jacobian",
            )
            for phase in range(NUM_PHASES):
                moving_np = moving_cpu[0, phase, 0].numpy()
                direct_warped = direct_render["warped"][phase][0, 0].numpy()
                current_warped = current["warped"][phase][0, 0].numpy()
                direct_jac = jacobian_map(direct_render["sequence"][:, phase])
                current_jac = jacobian_map(current["sequence"][:, phase])
                images = (
                    moving_np,
                    fixed_np,
                    direct_warped,
                    current_warped,
                    np.abs(fixed_np - direct_warped),
                    np.abs(fixed_np - current_warped),
                )
                for row_index, image in enumerate(images):
                    axes[row_index, phase].imshow(
                        image,
                        cmap="gray" if row_index < 4 else "magma",
                    )
                limit = max(
                    1.0,
                    float(np.abs(direct_jac).max()),
                    float(np.abs(current_jac).max()),
                )
                axes[6, phase].imshow(
                    direct_jac,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                axes[7, phase].imshow(
                    current_jac,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                axes[2, phase].set_title(
                    f"P{phase + 1} direct={direct_render['ncc'][phase]:.4f}",
                    fontsize=8,
                )
                axes[3, phase].set_title(
                    f"{name}={current['ncc'][phase]:.4f}",
                    fontsize=7,
                )
                for row_index in range(8):
                    axes[row_index, phase].set_xticks([])
                    axes[row_index, phase].set_yticks([])
            for row_index, label in enumerate(row_labels):
                axes[row_index, 0].set_ylabel(label, fontsize=8)
            fig.suptitle(f"{args.split} slice {slice_id}: {source} {name}")
            path = os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_{source}_{name}.png",
            )
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Figure] {path}")


def print_summary(summary_rows, budget_rows, pareto_rows):
    print("\n" + "=" * 122)
    print("TOPOLOGY-AWARE TEMPORAL BLEND SUMMARY")
    print("=" * 122)
    pareto_configs = {
        (row["source"], row["config"]) for row in pareto_rows
    }
    for source in SOURCE_NAMES:
        print(f"\n[{source}]")
        rows = sorted(
            [row for row in summary_rows if row["source"] == source],
            key=lambda row: (-row["ncc_after_mean"], row["fg_neg_ratio_mean"]),
        )
        for row in rows:
            marker = "*" if (source, row["config"]) in pareto_configs else " "
            print(
                f"{marker} {row['config']:<26} "
                f"NCC={row['ncc_after_mean']:.6f} "
                f"dNCC={row['ncc_after_change_from_direct']:+.6f} "
                f"fgNeg={100.0 * row['fg_neg_ratio_mean']:.4f}% "
                f"dFgNeg={100.0 * row['fg_neg_ratio_change_from_direct']:+.4f}pp "
                f"globalNeg={100.0 * row['neg_ratio_mean']:.4f}%"
            )
    print("\nBUDGET-CONSTRAINED SELECTION (primary topology = foreground negR)")
    for row in budget_rows:
        print(
            f"{row['source']:<9} budget={row['ncc_drop_budget']:.4f} "
            f"selected={row['selected_config']:<26} "
            f"dNCC={row['ncc_change']:+.6f} "
            f"dFgNeg={100.0 * row['topology_change']:+.4f}pp "
            f"dFgFolds={row['fg_folds_change']:+.1f}"
        )


def main():
    args = parse_args()
    risk_alphas = sorted(set(parse_float_list(args.risk_alphas, "--risk_alphas")))
    risk_thresholds = sorted(
        set(parse_float_list(args.risk_thresholds, "--risk_thresholds"))
    )
    budgets = sorted(set(parse_float_list(args.ncc_budgets, "--ncc_budgets")))
    if not 0.0 <= args.global_alpha <= 1.0:
        raise ValueError("--global_alpha must be in [0, 1]")
    if min(risk_alphas) < 0.0 or max(risk_alphas) > 1.0:
        raise ValueError("Every risk alpha must be in [0, 1]")
    if min(budgets) < 0.0:
        raise ValueError("NCC budgets must be non-negative")
    if args.risk_temperature <= 0.0:
        raise ValueError("--risk_temperature must be positive")
    validate_kernel(args.risk_dilate, "--risk_dilate")
    validate_kernel(args.risk_blur, "--risk_blur")
    validate_kernel(args.support_dilate, "--support_dilate")
    if args.bs != 1:
        raise ValueError("Use --bs 1")

    configurations = make_configurations(
        args.global_alpha,
        risk_alphas,
        risk_thresholds,
    )
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed + 1000)
    print(
        f"[Topology-aware] split={args.split} | global_alpha={args.global_alpha} | "
        f"risk_alphas={risk_alphas} | thresholds={risk_thresholds}"
    )
    print(
        f"[Risk mask] temperature={args.risk_temperature} "
        f"dilate={args.risk_dilate} blur={args.risk_blur} "
        f"support_dilate={args.support_dilate}"
    )

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
        f"configurations={len(configurations)}"
    )
    pair_rows, trajectory_rows, risk_rows, figure_cache = evaluate(
        args,
        configurations,
        risk_thresholds,
        dataset,
        baseline,
        pairwise,
        ldm_model,
        transform,
    )

    summary_rows = aggregate(pair_rows, ("source", "config"), PAIR_METRICS)
    summary_rows = add_direct_differences(summary_rows)
    phase_rows = aggregate(
        pair_rows,
        ("source", "config", "phase"),
        PAIR_METRICS,
    )
    phase_rows = add_direct_differences(phase_rows, extra_keys=("phase",))
    slice_rows = aggregate(
        pair_rows,
        ("source", "config", "slice_id"),
        PAIR_METRICS,
    )
    slice_rows = add_direct_differences(slice_rows, extra_keys=("slice_id",))
    trajectory_summary = aggregate(
        trajectory_rows,
        ("source", "config"),
        TRAJECTORY_METRICS,
    )
    risk_summary = aggregate(
        risk_rows,
        ("source", "risk_threshold"),
        RISK_METRICS,
    )
    pareto_rows = pareto_front(summary_rows, topology_metric="fg_neg_ratio")
    budget_rows = select_under_budgets(
        summary_rows,
        budgets,
        topology_metric="fg_neg_ratio",
    )
    slice_difference_rows, paired_results = paired_statistics(
        pair_rows,
        configurations,
    )

    write_csv(os.path.join(args.save_dir, "per_pair_metrics.csv"), pair_rows)
    write_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.save_dir, "per_phase_summary.csv"), phase_rows)
    write_csv(os.path.join(args.save_dir, "per_slice_summary.csv"), slice_rows)
    write_csv(
        os.path.join(args.save_dir, "trajectory_per_sequence.csv"),
        trajectory_rows,
    )
    write_csv(
        os.path.join(args.save_dir, "trajectory_summary.csv"),
        trajectory_summary,
    )
    write_csv(os.path.join(args.save_dir, "risk_mask_summary.csv"), risk_summary)
    write_csv(os.path.join(args.save_dir, "pareto_front.csv"), pareto_rows)
    write_csv(os.path.join(args.save_dir, "budget_selection.csv"), budget_rows)
    write_csv(
        os.path.join(args.save_dir, "per_slice_differences.csv"),
        slice_difference_rows,
    )
    with open(os.path.join(args.save_dir, "paired_statistics.json"), "w") as handle:
        json.dump(json_safe(paired_results), handle, indent=2)
    report = {
        "warning": (
            "Validation-only deterministic post-processing. Foreground is an "
            "automatic intensity mask, not an anatomical ground-truth mask."
        ),
        "split": args.split,
        "configurations": configurations,
        "risk_temperature": args.risk_temperature,
        "risk_dilate": args.risk_dilate,
        "risk_blur": args.risk_blur,
        "support_dilate": args.support_dilate,
        "ncc_budgets": budgets,
        "summary": summary_rows,
        "risk_mask_summary": risk_summary,
        "pareto_front": pareto_rows,
        "budget_selection": budget_rows,
        "trajectory_summary": trajectory_summary,
        "paired_statistics": paired_results,
    }
    with open(os.path.join(args.save_dir, "diagnostic_report.json"), "w") as handle:
        json.dump(json_safe(report), handle, indent=2)

    if not args.no_figures:
        save_tradeoff_figures(args, summary_rows, pareto_rows, budget_rows)
        save_registration_figures(
            args,
            configurations,
            budget_rows,
            figure_cache,
            transform,
        )
    print_summary(summary_rows, budget_rows, pareto_rows)
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
