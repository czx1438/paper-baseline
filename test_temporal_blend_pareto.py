"""Validation-only Pareto sweep for local temporal DVF regularization.

This script performs no network training. Existing Baseline and Pairwise
MotionFiLM checkpoints predict direct phase-to-phase0 displacement fields for
the nine moving phases. A local temporal reference is constructed from the two
adjacent phase fields:

    T_p = 0.5 * (D_{p-1} + D_{p+1}),

with D_0 = D_10 = 0 because phase 0 is the fixed image and phase 10 is the next
cycle's phase 0. The evaluated field is

    D_p(alpha) = (1 - alpha) * D_p + alpha * T_p.

Alpha 0 is exactly the direct registration field. Alpha 1 is exactly periodic
linear leave-one-phase-out interpolation. Intermediate alphas test whether a
small local temporal correction can reduce folding without materially reducing
registration accuracy.

Use the validation split to choose alpha. Run the test split only once after
the method and alpha selection rule are fixed.
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


TOTAL_CYCLE_PHASES = NUM_PHASES + 1
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
        help="Use val for alpha selection and reserve test for final evaluation.",
    )
    parser.add_argument(
        "--alphas",
        default="0:1:0.05",
        help="Either start:end:step or comma-separated values.",
    )
    parser.add_argument(
        "--ncc_budgets",
        default="0.0005,0.001,0.002,0.005",
        help="Maximum allowed mean NCC drop from alpha=0.",
    )
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--t_enc", type=int, default=1)
    parser.add_argument("--fg_thr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual_slices", default="0,9,18")
    parser.add_argument("--visual_alphas", default="0,0.1,0.25,0.5,1")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--no_ldm", action="store_true")
    parser.add_argument(
        "--save_dir",
        default="./logs/Temporal_Blend_Pareto_val",
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


def parse_alphas(text):
    if ":" in text:
        parts = [float(value.strip()) for value in text.split(":")]
        if len(parts) != 3:
            raise ValueError("--alphas range must be start:end:step")
        start, end, step = parts
        if step <= 0 or end < start:
            raise ValueError("--alphas requires end >= start and step > 0")
        count = int(np.floor((end - start) / step + 1e-9)) + 1
        values = [start + index * step for index in range(count)]
        if values[-1] < end - 1e-9:
            values.append(end)
    else:
        values = parse_float_list(text, "--alphas")
    values = sorted({round(float(value), 10) for value in values})
    if values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("Every alpha must be in [0, 1]")
    if not any(abs(value) < 1e-9 for value in values):
        raise ValueError("--alphas must include 0 for the direct reference")
    if not any(abs(value - 1.0) < 1e-9 for value in values):
        raise ValueError("--alphas must include 1 for the periodic-linear endpoint")
    return values


def resolve_visual_alphas(requested, available):
    resolved = []
    for value in requested:
        nearest = min(available, key=lambda candidate: abs(candidate - value))
        if abs(nearest - value) > 1e-7:
            raise ValueError(
                f"visual alpha {value} is not present in --alphas; nearest={nearest}"
            )
        if nearest not in resolved:
            resolved.append(nearest)
    return resolved


def alpha_label(alpha):
    return f"alpha_{alpha:.4f}".replace(".", "p")


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
    return float(
        ((value_map * mask).sum() / (mask.sum() * value_map.shape[1]).clamp(min=1.0)).item()
    )


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


def temporal_linear_reference(direct_sequence):
    zero = torch.zeros_like(direct_sequence[:, :1])
    closed = torch.cat([zero, direct_sequence, zero], dim=1)
    return 0.5 * (closed[:, :-2] + closed[:, 2:])


def blend_sequence(direct_sequence, temporal_reference, alpha):
    return direct_sequence + float(alpha) * (temporal_reference - direct_sequence)


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
    jacobian = jacobian_result(displacement)
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
    alphas,
    visual_alphas,
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
        if moving_sequence.shape[1] != NUM_PHASES:
            raise ValueError(f"Expected {NUM_PHASES} moving phases")
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
            source_cache = None
            if slice_id in wanted_slices:
                source_cache = {
                    "fixed": fixed.detach().cpu(),
                    "moving": moving_sequence.detach().cpu(),
                    "alphas": {},
                }

            for alpha in alphas:
                displacement_sequence = blend_sequence(
                    direct_sequence,
                    temporal_reference,
                    alpha,
                )
                trajectory_rows.append(
                    {
                        "source": source,
                        "alpha": alpha,
                        "block_id": block_id,
                        "slice_id": slice_id,
                        "pairname": names[0],
                        **trajectory_metrics(
                            displacement_sequence,
                            direct_sequence,
                            foreground,
                        ),
                    }
                )
                alpha_cache = None
                if source_cache is not None and alpha in visual_alphas:
                    alpha_cache = {"dvfs": [], "warped": [], "ncc": []}

                for phase_index in range(NUM_PHASES):
                    moving = moving_sequence[:, phase_index]
                    displacement = displacement_sequence[:, phase_index]
                    warped = warp_tensor(transform, moving, displacement)
                    metrics = registration_metrics(
                        fixed,
                        moving,
                        warped,
                        foreground,
                        displacement,
                    )
                    pair_rows.append(
                        {
                            "source": source,
                            "alpha": alpha,
                            "block_id": block_id,
                            "slice_id": slice_id,
                            "pairname": names[0],
                            "phase": phase_index + 1,
                            **metrics,
                        }
                    )
                    if alpha_cache is not None:
                        alpha_cache["dvfs"].append(displacement.detach().cpu())
                        alpha_cache["warped"].append(warped.detach().cpu())
                        alpha_cache["ncc"].append(metrics["ncc_after"])
                if alpha_cache is not None:
                    source_cache["alphas"][alpha] = alpha_cache

            if source_cache is not None:
                figure_cache[(slice_id, source)] = source_cache

        print(f"[Pareto] sequence {batch_index + 1}/{len(loader)}")

    return pair_rows, trajectory_rows, figure_cache


PAIR_METRICS = (
    "ncc_before",
    "ncc_after",
    "ncc_delta",
    "mse_after",
    "ssim_after",
    "neg_ratio",
    "fold_count",
    "min_jac",
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
        for metric in metric_keys:
            mean, std, median = finite_stats([row[metric] for row in subset])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_median"] = median
        output.append(record)
    return output


def add_direct_differences(summary_rows, group_keys):
    direct = {}
    for row in summary_rows:
        if abs(float(row["alpha"])) < 1e-9:
            key = tuple(row[name] for name in group_keys if name != "alpha")
            direct[key] = row
    for row in summary_rows:
        key = tuple(row[name] for name in group_keys if name != "alpha")
        reference = direct[key]
        row["ncc_change_from_direct"] = (
            row["ncc_after_mean"] - reference["ncc_after_mean"]
        )
        row["ssim_change_from_direct"] = (
            row["ssim_after_mean"] - reference["ssim_after_mean"]
        )
        row["mse_change_from_direct"] = (
            row["mse_after_mean"] - reference["mse_after_mean"]
        )
        row["neg_ratio_change_from_direct"] = (
            row["neg_ratio_mean"] - reference["neg_ratio_mean"]
        )
        row["fold_count_change_from_direct"] = (
            row["fold_count_mean"] - reference["fold_count_mean"]
        )
        denominator = reference["fold_count_mean"]
        row["fold_count_relative_change"] = (
            row["fold_count_change_from_direct"] / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
    return summary_rows


def pareto_front(summary_rows):
    output = []
    for source in SOURCE_NAMES:
        subset = [row for row in summary_rows if row["source"] == source]
        for candidate in subset:
            dominated = False
            for other in subset:
                accuracy_no_worse = other["ncc_after_mean"] >= candidate["ncc_after_mean"] - 1e-12
                topology_no_worse = other["neg_ratio_mean"] <= candidate["neg_ratio_mean"] + 1e-12
                strictly_better = (
                    other["ncc_after_mean"] > candidate["ncc_after_mean"] + 1e-12
                    or other["neg_ratio_mean"] < candidate["neg_ratio_mean"] - 1e-12
                )
                if accuracy_no_worse and topology_no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                output.append(dict(candidate))
    return sorted(output, key=lambda row: (row["source"], row["alpha"]))


def select_under_budgets(summary_rows, budgets):
    output = []
    for source in SOURCE_NAMES:
        subset = [row for row in summary_rows if row["source"] == source]
        direct = min(subset, key=lambda row: abs(row["alpha"]))
        for budget in budgets:
            eligible = [
                row
                for row in subset
                if row["ncc_after_mean"] >= direct["ncc_after_mean"] - budget - 1e-12
            ]
            selected = min(
                eligible,
                key=lambda row: (
                    row["neg_ratio_mean"],
                    row["fold_count_mean"],
                    -row["ncc_after_mean"],
                ),
            )
            output.append(
                {
                    "source": source,
                    "ncc_drop_budget": budget,
                    "selected_alpha": selected["alpha"],
                    "direct_ncc": direct["ncc_after_mean"],
                    "selected_ncc": selected["ncc_after_mean"],
                    "actual_ncc_change": selected["ncc_change_from_direct"],
                    "direct_neg_ratio": direct["neg_ratio_mean"],
                    "selected_neg_ratio": selected["neg_ratio_mean"],
                    "neg_ratio_change": selected["neg_ratio_change_from_direct"],
                    "direct_fold_count": direct["fold_count_mean"],
                    "selected_fold_count": selected["fold_count_mean"],
                    "fold_count_change": selected["fold_count_change_from_direct"],
                    "fold_count_relative_change": selected["fold_count_relative_change"],
                    "selected_ssim": selected["ssim_after_mean"],
                    "selected_mse": selected["mse_after_mean"],
                }
            )
    return output


def paired_statistics(pair_rows, alphas):
    indexed = {
        (row["source"], row["alpha"], row["slice_id"], row["phase"]): row
        for row in pair_rows
    }
    slice_ids = sorted({row["slice_id"] for row in pair_rows})
    metrics = ("ncc_after", "ssim_after", "mse_after", "neg_ratio", "fold_count")
    slice_rows = []
    results = []
    for source in SOURCE_NAMES:
        for alpha in alphas:
            if abs(alpha) < 1e-9:
                continue
            differences_by_metric = {metric: [] for metric in metrics}
            for slice_id in slice_ids:
                record = {
                    "comparison": f"{source}:alpha_{alpha:.4f}_minus_alpha_0",
                    "source": source,
                    "alpha": alpha,
                    "slice_id": slice_id,
                }
                for metric in metrics:
                    phase_differences = []
                    for phase in range(1, NUM_PHASES + 1):
                        current = indexed[(source, alpha, slice_id, phase)]
                        direct = indexed[(source, 0.0, slice_id, phase)]
                        phase_differences.append(current[metric] - direct[metric])
                    difference = float(np.mean(phase_differences))
                    record[f"{metric}_difference"] = difference
                    differences_by_metric[metric].append(difference)
                slice_rows.append(record)

            result = {
                "comparison": f"{source}:alpha_{alpha:.4f}_minus_alpha_0",
                "source": source,
                "alpha": alpha,
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


def save_tradeoff_curves(args, summary_rows, pareto_rows, budgets_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for source in SOURCE_NAMES:
        rows = sorted(
            [row for row in summary_rows if row["source"] == source],
            key=lambda row: row["alpha"],
        )
        pareto_alphas = {
            row["alpha"] for row in pareto_rows if row["source"] == source
        }
        source_budgets = [
            row for row in budgets_rows if row["source"] == source
        ]
        alphas = [row["alpha"] for row in rows]
        ncc = [row["ncc_after_mean"] for row in rows]
        neg_ratio = [100.0 * row["neg_ratio_mean"] for row in rows]
        folds = [row["fold_count_mean"] for row in rows]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes[0, 0].plot(alphas, ncc, marker="o")
        axes[0, 0].set_xlabel("alpha")
        axes[0, 0].set_ylabel("Mean NCC")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 1].plot(alphas, neg_ratio, marker="o", color="tab:red")
        axes[0, 1].set_xlabel("alpha")
        axes[0, 1].set_ylabel("Negative Jacobian ratio (%)")
        axes[0, 1].grid(alpha=0.3)
        axes[1, 0].plot(alphas, folds, marker="o", color="tab:orange")
        axes[1, 0].set_xlabel("alpha")
        axes[1, 0].set_ylabel("Folds per image")
        axes[1, 0].grid(alpha=0.3)

        for row in rows:
            color = "tab:green" if row["alpha"] in pareto_alphas else "tab:blue"
            axes[1, 1].scatter(
                100.0 * row["neg_ratio_mean"],
                row["ncc_after_mean"],
                color=color,
                s=45,
            )
            axes[1, 1].annotate(
                f"{row['alpha']:.2f}",
                (100.0 * row["neg_ratio_mean"], row["ncc_after_mean"]),
                fontsize=7,
            )
        axes[1, 1].set_xlabel("Negative Jacobian ratio (%)")
        axes[1, 1].set_ylabel("Mean NCC")
        axes[1, 1].grid(alpha=0.3)
        budget_text = "\n".join(
            f"budget {row['ncc_drop_budget']:.4f}: alpha={row['selected_alpha']:.2f}"
            for row in source_budgets
        )
        axes[1, 1].text(
            0.02,
            0.02,
            budget_text,
            transform=axes[1, 1].transAxes,
            fontsize=8,
            va="bottom",
        )
        fig.suptitle(f"{source}: local temporal blend validation tradeoff")
        path = os.path.join(figure_dir, f"{source}_pareto_tradeoff.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Figure] {path}")


def save_registration_figures(args, visual_alphas, figure_cache):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for (slice_id, source), cache in sorted(figure_cache.items()):
        fixed = cache["fixed"][0, 0].numpy()
        moving_sequence = cache["moving"]
        direct = cache["alphas"][0.0]
        for alpha in visual_alphas:
            current = cache["alphas"][alpha]
            fig, axes = plt.subplots(8, NUM_PHASES, figsize=(27, 23))
            row_labels = (
                "Moving",
                "Fixed",
                "Direct warped",
                f"alpha={alpha:.2f} warped",
                "Direct |F-W|",
                f"alpha={alpha:.2f} |F-W|",
                "Direct Jacobian",
                f"alpha={alpha:.2f} Jacobian",
            )
            for phase_index in range(NUM_PHASES):
                moving = moving_sequence[0, phase_index, 0].numpy()
                direct_warped = direct["warped"][phase_index][0, 0].numpy()
                current_warped = current["warped"][phase_index][0, 0].numpy()
                direct_jacobian = jacobian_result(
                    direct["dvfs"][phase_index]
                )["jacobian_map"]
                current_jacobian = jacobian_result(
                    current["dvfs"][phase_index]
                )["jacobian_map"]
                images = (
                    moving,
                    fixed,
                    direct_warped,
                    current_warped,
                    np.abs(fixed - direct_warped),
                    np.abs(fixed - current_warped),
                )
                for row_index, image in enumerate(images):
                    cmap = "gray" if row_index < 4 else "magma"
                    axes[row_index, phase_index].imshow(image, cmap=cmap)
                jacobian_limit = max(
                    1.0,
                    float(np.abs(direct_jacobian).max()),
                    float(np.abs(current_jacobian).max()),
                )
                axes[6, phase_index].imshow(
                    direct_jacobian,
                    cmap="RdBu_r",
                    vmin=-jacobian_limit,
                    vmax=jacobian_limit,
                )
                axes[7, phase_index].imshow(
                    current_jacobian,
                    cmap="RdBu_r",
                    vmin=-jacobian_limit,
                    vmax=jacobian_limit,
                )
                axes[2, phase_index].set_title(
                    f"P{phase_index + 1} direct={direct['ncc'][phase_index]:.4f}",
                    fontsize=8,
                )
                axes[3, phase_index].set_title(
                    f"alpha={alpha:.2f}: {current['ncc'][phase_index]:.4f}",
                    fontsize=8,
                )
                for row_index in range(8):
                    axes[row_index, phase_index].set_xticks([])
                    axes[row_index, phase_index].set_yticks([])
            for row_index, label in enumerate(row_labels):
                axes[row_index, 0].set_ylabel(label, fontsize=8)
            fig.suptitle(
                f"{args.split} slice {slice_id}: {source} temporal blend alpha={alpha:.2f}",
                fontsize=14,
            )
            path = os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_{source}_{alpha_label(alpha)}.png",
            )
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Figure] {path}")


def validate_endpoints(summary_rows):
    for source in SOURCE_NAMES:
        direct = next(
            row
            for row in summary_rows
            if row["source"] == source and abs(row["alpha"]) < 1e-9
        )
        linear = next(
            row
            for row in summary_rows
            if row["source"] == source and abs(row["alpha"] - 1.0) < 1e-9
        )
        print(
            f"[Endpoint] {source} alpha=0 direct NCC={direct['ncc_after_mean']:.6f} | "
            f"alpha=1 periodic-linear NCC={linear['ncc_after_mean']:.6f}"
        )


def print_summary(summary_rows, budget_rows, pareto_rows):
    print("\n" + "=" * 118)
    print("TEMPORAL BLEND PARETO SUMMARY")
    print("=" * 118)
    for source in SOURCE_NAMES:
        print(f"\n[{source}]")
        rows = sorted(
            [row for row in summary_rows if row["source"] == source],
            key=lambda row: row["alpha"],
        )
        pareto_alphas = {
            row["alpha"] for row in pareto_rows if row["source"] == source
        }
        for row in rows:
            marker = "*" if row["alpha"] in pareto_alphas else " "
            print(
                f"{marker} alpha={row['alpha']:.2f} "
                f"NCC={row['ncc_after_mean']:.6f} "
                f"dNCC={row['ncc_change_from_direct']:+.6f} "
                f"negR={100.0 * row['neg_ratio_mean']:.4f}% "
                f"dNegR={100.0 * row['neg_ratio_change_from_direct']:+.4f}pp "
                f"folds={row['fold_count_mean']:.1f} "
                f"dFolds={row['fold_count_change_from_direct']:+.1f}"
            )
    print("\nBUDGET-CONSTRAINED SELECTION")
    for row in budget_rows:
        print(
            f"{row['source']:<9} budget={row['ncc_drop_budget']:.4f} "
            f"alpha={row['selected_alpha']:.2f} "
            f"dNCC={row['actual_ncc_change']:+.6f} "
            f"dNegR={100.0 * row['neg_ratio_change']:+.4f}pp "
            f"dFolds={row['fold_count_change']:+.1f}"
        )
    print("\n* denotes a non-dominated NCC/negative-Jacobian Pareto point")


def main():
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    budgets = sorted(set(parse_float_list(args.ncc_budgets, "--ncc_budgets")))
    if min(budgets) < 0:
        raise ValueError("NCC budgets must be non-negative")
    requested_visual_alphas = parse_float_list(
        args.visual_alphas,
        "--visual_alphas",
    )
    visual_alphas = resolve_visual_alphas(requested_visual_alphas, alphas)
    if args.bs != 1:
        raise ValueError("Use --bs 1")
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed + 1000)

    print(f"[Pareto] split={args.split} | alphas={alphas}")
    print(f"[Pareto] NCC budgets={budgets} | visual alphas={visual_alphas}")
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
        f"pairs/alpha/model={len(dataset) * NUM_PHASES}"
    )
    pair_rows, trajectory_rows, figure_cache = evaluate(
        args,
        alphas,
        visual_alphas,
        dataset,
        baseline,
        pairwise,
        ldm_model,
        transform,
    )

    summary_rows = aggregate(pair_rows, ("source", "alpha"), PAIR_METRICS)
    summary_rows = add_direct_differences(summary_rows, ("source", "alpha"))
    phase_rows = aggregate(
        pair_rows,
        ("source", "alpha", "phase"),
        PAIR_METRICS,
    )
    phase_rows = add_direct_differences(
        phase_rows,
        ("source", "alpha", "phase"),
    )
    slice_rows = aggregate(
        pair_rows,
        ("source", "alpha", "slice_id"),
        PAIR_METRICS,
    )
    slice_rows = add_direct_differences(
        slice_rows,
        ("source", "alpha", "slice_id"),
    )
    trajectory_summary = aggregate(
        trajectory_rows,
        ("source", "alpha"),
        TRAJECTORY_METRICS,
    )
    pareto_rows = pareto_front(summary_rows)
    budget_rows = select_under_budgets(summary_rows, budgets)
    slice_difference_rows, paired_results = paired_statistics(pair_rows, alphas)

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
            "This is a validation-only post-processing sweep. All alpha values "
            "use the current phase's direct DVF; alpha is not a held-out predictor."
        ),
        "split": args.split,
        "alphas": alphas,
        "ncc_budgets": budgets,
        "sequences": len(dataset),
        "summary": summary_rows,
        "pareto_front": pareto_rows,
        "budget_selection": budget_rows,
        "trajectory_summary": trajectory_summary,
        "paired_statistics": paired_results,
    }
    with open(os.path.join(args.save_dir, "diagnostic_report.json"), "w") as handle:
        json.dump(json_safe(report), handle, indent=2)

    validate_endpoints(summary_rows)
    if not args.no_figures:
        save_tradeoff_curves(args, summary_rows, pareto_rows, budget_rows)
        save_registration_figures(args, visual_alphas, figure_cache)
    print_summary(summary_rows, budget_rows, pareto_rows)
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
