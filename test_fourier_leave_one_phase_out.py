"""Leave-one-phase-out diagnosis of periodic multi-phase DVF structure.

This script performs no network training. For every same-slice sequence, the
existing Baseline and Pairwise MotionFiLM models first predict nine direct
phase-to-phase0 DVFs. One phase DVF is then hidden, and its field is predicted
using only the other phases with one of the following temporal models:

  * periodic linear interpolation
  * periodic cubic spline interpolation
  * anchored Fourier models of configurable order

The hidden phase image is used only after prediction to measure registration
quality. Its direct DVF is never used by the temporal predictor. Direct model
output is reported as a reference, not as deformation ground truth.

Phase 0 is the fixed image. Phases 1..9 lie at t=p/10, and phase 10 is the next
cycle's phase 0. The anchored Fourier representation is

    u(t) = sum_k A_k sin(2*pi*k*t) + B_k (cos(2*pi*k*t) - 1),

which enforces u(0)=u(1)=0.
"""

import argparse
import csv
import json
import math
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
        help="Use val while selecting the temporal model; reserve test for once.",
    )
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument(
        "--interpolators",
        default="linear,cubic",
        help="Comma-separated subset of: linear,cubic",
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
        default="./logs/fourier_leave_one_phase_out_val",
    )
    return parser.parse_args()


def parse_orders(text):
    orders = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not orders:
        raise ValueError("--orders must contain at least one positive integer")
    if min(orders) < 1:
        raise ValueError("Every Fourier order must be >= 1")
    max_identifiable_order = (NUM_PHASES - 1) // 2
    if max(orders) > max_identifiable_order:
        raise ValueError(
            f"Order {max(orders)} is not identifiable after hiding one of "
            f"{NUM_PHASES} moving phases; maximum is {max_identifiable_order}"
        )
    return orders


def parse_interpolators(text):
    values = [value.strip().lower() for value in text.split(",") if value.strip()]
    allowed = {"linear", "cubic"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown interpolators: {unknown}; allowed={sorted(allowed)}")
    return list(dict.fromkeys(values))


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
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def method_names(orders, interpolators):
    names = ["direct_reference"]
    if "linear" in interpolators:
        names.append("periodic_linear")
    if "cubic" in interpolators:
        names.append("periodic_cubic")
    names.extend(f"fourier_k{order}" for order in orders)
    return names


def fourier_design(phases, order, device, dtype):
    phases = torch.as_tensor(phases, device=device, dtype=dtype)
    theta = 2.0 * math.pi * phases / float(TOTAL_CYCLE_PHASES)
    columns = []
    for harmonic in range(1, order + 1):
        columns.append(torch.sin(harmonic * theta))
        columns.append(torch.cos(harmonic * theta) - 1.0)
    return torch.stack(columns, dim=1)


def observed_phase_fields(displacement_sequence, held_phase):
    """Return observed phases and fields, including the phase0 identity field."""
    zero = torch.zeros_like(displacement_sequence[:, 0])
    phases = [0] + [
        phase for phase in range(1, NUM_PHASES + 1) if phase != held_phase
    ]
    fields = [zero]
    fields.extend(displacement_sequence[:, phase - 1] for phase in phases[1:])
    return phases, torch.stack(fields, dim=1)


def to_pixel_sequence(displacement_sequence):
    result = displacement_sequence.clone()
    height, width = result.shape[-2:]
    result[:, :, 0] *= height / 2.0
    result[:, :, 1] *= width / 2.0
    return result


def to_pixel_displacement(displacement):
    result = displacement.clone()
    height, width = result.shape[-2:]
    result[:, 0] *= height / 2.0
    result[:, 1] *= width / 2.0
    return result


def vector_error_px(first, second):
    difference = to_pixel_displacement(first - second)
    magnitude = torch.sqrt(difference.square().sum(dim=1) + 1e-12)
    return float(magnitude.mean().item()), float(magnitude.max().item())


def sequence_vector_error_px(first, second):
    difference = to_pixel_sequence(first - second)
    magnitude = torch.sqrt(difference.square().sum(dim=2) + 1e-12)
    return float(magnitude.mean().item())


def predict_fourier(displacement_sequence, held_phase, order):
    """Fit observed DVFs and predict one hidden phase without using its DVF."""
    phases, observed = observed_phase_fields(displacement_sequence, held_phase)
    design = fourier_design(
        phases,
        order,
        device=displacement_sequence.device,
        dtype=displacement_sequence.dtype,
    )
    coefficients = torch.einsum(
        "kn,bnchw->bkchw",
        torch.linalg.pinv(design),
        observed,
    )
    hidden_design = fourier_design(
        [held_phase],
        order,
        device=displacement_sequence.device,
        dtype=displacement_sequence.dtype,
    )[0]
    predicted = torch.einsum("k,bkchw->bchw", hidden_design, coefficients)
    fitted_observed = torch.einsum("nk,bkchw->bnchw", design, coefficients)
    fit_error_px = sequence_vector_error_px(fitted_observed, observed)
    nonzero_design = design[1:]
    condition = float(torch.linalg.cond(nonzero_design).item())
    return predicted, fit_error_px, condition


def phase_field(displacement_sequence, phase):
    if phase in (0, TOTAL_CYCLE_PHASES):
        return torch.zeros_like(displacement_sequence[:, 0])
    return displacement_sequence[:, phase - 1]


def predict_periodic_linear(displacement_sequence, held_phase):
    left = phase_field(displacement_sequence, held_phase - 1)
    right = phase_field(displacement_sequence, held_phase + 1)
    return 0.5 * (left + right)


def periodic_cubic_weights(held_phase):
    """Return periodic cubic-spline weights over all observed phase fields."""
    source_phases = [0] + [
        phase for phase in range(1, NUM_PHASES + 1) if phase != held_phase
    ]
    x = np.asarray(source_phases, dtype=np.float64) / TOTAL_CYCLE_PHASES
    count = len(source_phases)
    intervals = np.empty(count, dtype=np.float64)
    intervals[:-1] = x[1:] - x[:-1]
    intervals[-1] = x[0] + 1.0 - x[-1]

    # Solve the periodic cubic-spline system once for every identity-basis
    # input. The evaluated row is then a set of interpolation weights that can
    # be applied to an arbitrary dense displacement field.
    system = np.zeros((count, count), dtype=np.float64)
    basis = np.eye(count, dtype=np.float64)
    right_hand_side = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        previous = (index - 1) % count
        following = (index + 1) % count
        h_previous = intervals[previous]
        h_following = intervals[index]
        system[index, previous] = h_previous
        system[index, index] = 2.0 * (h_previous + h_following)
        system[index, following] = h_following
        right_hand_side[index] = 6.0 * (
            (basis[following] - basis[index]) / h_following
            - (basis[index] - basis[previous]) / h_previous
        )
    second_derivative_weights = np.linalg.solve(system, right_hand_side)

    target = float(held_phase) / TOTAL_CYCLE_PHASES
    left_index = count - 1
    for index in range(count - 1):
        if x[index] <= target <= x[index + 1]:
            left_index = index
            break
    right_index = (left_index + 1) % count
    left_x = x[left_index]
    right_x = x[right_index] if right_index != 0 else x[0] + 1.0
    target_x = target if target >= left_x else target + 1.0
    interval = right_x - left_x
    left_fraction = (right_x - target_x) / interval
    right_fraction = (target_x - left_x) / interval
    weights = (
        left_fraction * basis[left_index]
        + right_fraction * basis[right_index]
        + (left_fraction ** 3 - left_fraction)
        * second_derivative_weights[left_index]
        * interval ** 2
        / 6.0
        + (right_fraction ** 3 - right_fraction)
        * second_derivative_weights[right_index]
        * interval ** 2
        / 6.0
    )
    return source_phases, weights


def predict_periodic_cubic(displacement_sequence, held_phase):
    phases, observed = observed_phase_fields(displacement_sequence, held_phase)
    weight_phases, weights = periodic_cubic_weights(held_phase)
    if phases != weight_phases:
        raise RuntimeError("Internal cubic interpolation phase ordering mismatch")
    weights = torch.as_tensor(
        weights,
        device=displacement_sequence.device,
        dtype=displacement_sequence.dtype,
    )
    return torch.einsum("n,bnchw->bchw", weights, observed)


def predict_methods(displacement_sequence, held_phase, orders, interpolators):
    output = {}
    if "linear" in interpolators:
        output["periodic_linear"] = (
            predict_periodic_linear(displacement_sequence, held_phase),
            float("nan"),
            float("nan"),
        )
    if "cubic" in interpolators:
        output["periodic_cubic"] = (
            predict_periodic_cubic(displacement_sequence, held_phase),
            float("nan"),
            float("nan"),
        )
    for order in orders:
        predicted, fit_error_px, condition = predict_fourier(
            displacement_sequence,
            held_phase,
            order,
        )
        output[f"fourier_k{order}"] = (
            predicted,
            fit_error_px,
            condition,
        )
    return output


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
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp(min=1e-8)
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
def evaluate(args, orders, interpolators, dataset, baseline, pairwise, ldm_model, transform):
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
    rows = []
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
            source_cache = None
            if slice_id in wanted_slices:
                source_cache = {
                    "fixed": fixed.detach().cpu(),
                    "moving": moving_sequence.detach().cpu(),
                    "methods": {
                        name: {"dvfs": [], "warped": [], "ncc": []}
                        for name in method_names(orders, interpolators)
                    },
                }

            for held_phase in range(1, NUM_PHASES + 1):
                moving = moving_sequence[:, held_phase - 1]
                direct = direct_sequence[:, held_phase - 1]
                direct_warped = warp_tensor(transform, moving, direct)
                direct_metrics = registration_metrics(
                    fixed,
                    moving,
                    direct_warped,
                    foreground,
                    direct,
                )
                direct_gain = direct_metrics["ncc_after"] - direct_metrics["ncc_before"]
                rows.append(
                    {
                        "source": source,
                        "method": "direct_reference",
                        "block_id": block_id,
                        "slice_id": slice_id,
                        "pairname": names[0],
                        "held_phase": held_phase,
                        "uses_held_phase_dvf": 1,
                        "observed_fit_error_px": 0.0,
                        "design_condition": float("nan"),
                        "dvf_difference_to_direct_px": 0.0,
                        "dvf_difference_to_direct_max_px": 0.0,
                        "recovered_direct_ncc_gain": 1.0,
                        "ncc_gap_to_direct": 0.0,
                        **direct_metrics,
                    }
                )

                if source_cache is not None:
                    source_cache["methods"]["direct_reference"]["dvfs"].append(
                        direct.detach().cpu()
                    )
                    source_cache["methods"]["direct_reference"]["warped"].append(
                        direct_warped.detach().cpu()
                    )
                    source_cache["methods"]["direct_reference"]["ncc"].append(
                        direct_metrics["ncc_after"]
                    )

                predictions = predict_methods(
                    direct_sequence,
                    held_phase,
                    orders,
                    interpolators,
                )
                for method, (predicted, fit_error_px, condition) in predictions.items():
                    warped = warp_tensor(transform, moving, predicted)
                    metrics = registration_metrics(
                        fixed,
                        moving,
                        warped,
                        foreground,
                        predicted,
                    )
                    difference_mean, difference_max = vector_error_px(
                        predicted,
                        direct,
                    )
                    recovered_gain = float("nan")
                    if abs(direct_gain) > 1e-8:
                        recovered_gain = (
                            metrics["ncc_after"] - metrics["ncc_before"]
                        ) / direct_gain
                    rows.append(
                        {
                            "source": source,
                            "method": method,
                            "block_id": block_id,
                            "slice_id": slice_id,
                            "pairname": names[0],
                            "held_phase": held_phase,
                            "uses_held_phase_dvf": 0,
                            "observed_fit_error_px": fit_error_px,
                            "design_condition": condition,
                            "dvf_difference_to_direct_px": difference_mean,
                            "dvf_difference_to_direct_max_px": difference_max,
                            "recovered_direct_ncc_gain": recovered_gain,
                            "ncc_gap_to_direct": (
                                metrics["ncc_after"] - direct_metrics["ncc_after"]
                            ),
                            **metrics,
                        }
                    )
                    if source_cache is not None:
                        source_cache["methods"][method]["dvfs"].append(
                            predicted.detach().cpu()
                        )
                        source_cache["methods"][method]["warped"].append(
                            warped.detach().cpu()
                        )
                        source_cache["methods"][method]["ncc"].append(
                            metrics["ncc_after"]
                        )

            if source_cache is not None:
                figure_cache[(slice_id, source)] = source_cache

        print(f"[LOPO] sequence {batch_index + 1}/{len(loader)}")

    return rows, figure_cache


SUMMARY_METRICS = (
    "ncc_before",
    "ncc_after",
    "ncc_delta",
    "ncc_gap_to_direct",
    "recovered_direct_ncc_gain",
    "mse_after",
    "ssim_after",
    "neg_ratio",
    "fold_count",
    "min_jac",
    "mean_dvf_px",
    "max_dvf_px",
    "dvf_difference_to_direct_px",
    "observed_fit_error_px",
    "design_condition",
)


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std, float(np.median(values))


def aggregate(rows, group_keys, metric_keys=SUMMARY_METRICS):
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


def paired_statistics(rows, methods):
    indexed = {
        (row["source"], row["method"], row["slice_id"], row["held_phase"]): row
        for row in rows
    }
    comparisons = []
    for source in SOURCE_NAMES:
        for method in methods:
            if method != "direct_reference":
                comparisons.append((source, method, "direct_reference"))
        reference = "periodic_cubic" if "periodic_cubic" in methods else "periodic_linear"
        if reference in methods:
            for method in methods:
                if method.startswith("fourier_"):
                    comparisons.append((source, method, reference))

    slice_rows = []
    results = []
    slice_ids = sorted({row["slice_id"] for row in rows})
    metrics = ("ncc_after", "ssim_after", "mse_after", "neg_ratio", "fold_count")
    for source, left, right in comparisons:
        per_metric = {metric: [] for metric in metrics}
        for slice_id in slice_ids:
            record = {
                "comparison": f"{source}:{left}_minus_{right}",
                "source": source,
                "left": left,
                "right": right,
                "slice_id": slice_id,
            }
            for metric in metrics:
                differences = []
                for phase in range(1, NUM_PHASES + 1):
                    left_row = indexed[(source, left, slice_id, phase)]
                    right_row = indexed[(source, right, slice_id, phase)]
                    differences.append(left_row[metric] - right_row[metric])
                value = float(np.mean(differences))
                record[f"{metric}_difference"] = value
                per_metric[metric].append(value)
            slice_rows.append(record)

        result = {
            "comparison": f"{source}:{left}_minus_{right}",
            "source": source,
            "left": left,
            "right": right,
            "n_slices": len(slice_ids),
        }
        for metric, values in per_metric.items():
            result[f"{metric}_difference_mean"] = float(np.mean(values))
            result[f"{metric}_difference_std"] = float(np.std(values, ddof=1))
            try:
                from scipy.stats import wilcoxon

                result[f"wilcoxon_{metric}_p"] = float(wilcoxon(values).pvalue)
            except Exception as error:
                result[f"wilcoxon_{metric}_error"] = str(error)
        results.append(result)
    return slice_rows, results


def save_figures(args, methods, figure_cache):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for (slice_id, source), cache in sorted(figure_cache.items()):
        fixed = cache["fixed"][0, 0].numpy()
        moving_sequence = cache["moving"]
        direct = cache["methods"]["direct_reference"]
        for method in methods:
            if method == "direct_reference":
                continue
            predicted = cache["methods"][method]
            fig, axes = plt.subplots(8, NUM_PHASES, figsize=(27, 23))
            row_labels = (
                "Moving",
                "Fixed",
                "Direct warped",
                f"LOPO {method} warped",
                "Direct |F-W|",
                f"LOPO {method} |F-W|",
                "Direct Jacobian",
                f"LOPO {method} Jacobian",
            )
            for phase_index in range(NUM_PHASES):
                moving = moving_sequence[0, phase_index, 0].numpy()
                direct_warped = direct["warped"][phase_index][0, 0].numpy()
                predicted_warped = predicted["warped"][phase_index][0, 0].numpy()
                direct_jacobian = jacobian_result(
                    direct["dvfs"][phase_index]
                )["jacobian_map"]
                predicted_jacobian = jacobian_result(
                    predicted["dvfs"][phase_index]
                )["jacobian_map"]
                images = (
                    moving,
                    fixed,
                    direct_warped,
                    predicted_warped,
                    np.abs(fixed - direct_warped),
                    np.abs(fixed - predicted_warped),
                )
                for row_index, image in enumerate(images):
                    cmap = "gray" if row_index < 4 else "magma"
                    axes[row_index, phase_index].imshow(image, cmap=cmap)
                jacobian_limit = max(
                    1.0,
                    float(np.abs(direct_jacobian).max()),
                    float(np.abs(predicted_jacobian).max()),
                )
                axes[6, phase_index].imshow(
                    direct_jacobian,
                    cmap="RdBu_r",
                    vmin=-jacobian_limit,
                    vmax=jacobian_limit,
                )
                axes[7, phase_index].imshow(
                    predicted_jacobian,
                    cmap="RdBu_r",
                    vmin=-jacobian_limit,
                    vmax=jacobian_limit,
                )
                axes[2, phase_index].set_title(
                    f"P{phase_index + 1} direct={direct['ncc'][phase_index]:.4f}",
                    fontsize=8,
                )
                axes[3, phase_index].set_title(
                    f"held-out={predicted['ncc'][phase_index]:.4f}",
                    fontsize=8,
                )
                for row_index in range(8):
                    axes[row_index, phase_index].set_xticks([])
                    axes[row_index, phase_index].set_yticks([])
            for row_index, label in enumerate(row_labels):
                axes[row_index, 0].set_ylabel(label, fontsize=8)
            fig.suptitle(
                f"{args.split} slice {slice_id}: {source} {method} leave-one-phase-out",
                fontsize=14,
            )
            path = os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_{source}_{method}.png",
            )
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Figure] {path}")


def print_summary(summary_rows, paired_results):
    print("\n" + "=" * 120)
    print("LEAVE-ONE-PHASE-OUT SUMMARY")
    print("=" * 120)
    for row in summary_rows:
        print(
            f"{row['source']:<9} {row['method']:<20} "
            f"NCC={row['ncc_after_mean']:.6f} "
            f"gap={row['ncc_gap_to_direct_mean']:+.6f} "
            f"recovery={100.0 * row['recovered_direct_ncc_gain_mean']:.2f}% "
            f"SSIM={row['ssim_after_mean']:.6f} "
            f"negR={100.0 * row['neg_ratio_mean']:.4f}% "
            f"DVFdiff={row['dvf_difference_to_direct_px_mean']:.4f}px"
        )
    print("\nPAIRED SLICE-LEVEL COMPARISONS")
    for result in paired_results:
        p_value = result.get("wilcoxon_ncc_after_p", float("nan"))
        print(
            f"{result['comparison']}: "
            f"NCC={result['ncc_after_difference_mean']:+.6f} "
            f"negR={100.0 * result['neg_ratio_difference_mean']:+.5f}pp "
            f"p_NCC={p_value:.6g}"
        )


def main():
    args = parse_args()
    orders = parse_orders(args.orders)
    interpolators = parse_interpolators(args.interpolators)
    methods = method_names(orders, interpolators)
    if args.bs != 1:
        raise ValueError("Use --bs 1")
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed + 1000)

    print(
        f"[LOPO] split={args.split} | orders={orders} | "
        f"interpolators={interpolators}"
    )
    print("[LOPO] direct_reference uses the hidden phase; all other methods do not")
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
        f"held-out predictions/model={len(dataset) * NUM_PHASES}"
    )
    rows, figure_cache = evaluate(
        args,
        orders,
        interpolators,
        dataset,
        baseline,
        pairwise,
        ldm_model,
        transform,
    )

    summary_rows = aggregate(rows, ("source", "method"))
    phase_rows = aggregate(rows, ("source", "method", "held_phase"))
    per_slice_rows = aggregate(rows, ("source", "method", "slice_id"))
    slice_difference_rows, paired_results = paired_statistics(rows, methods)

    write_csv(os.path.join(args.save_dir, "per_prediction_metrics.csv"), rows)
    write_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.save_dir, "per_phase_summary.csv"), phase_rows)
    write_csv(os.path.join(args.save_dir, "per_slice_summary.csv"), per_slice_rows)
    write_csv(
        os.path.join(args.save_dir, "per_slice_differences.csv"),
        slice_difference_rows,
    )
    with open(os.path.join(args.save_dir, "paired_statistics.json"), "w") as handle:
        json.dump(json_safe(paired_results), handle, indent=2)
    report = {
        "warning": (
            "Direct model DVFs are references, not deformation ground truth. "
            "This experiment tests temporal predictability and image alignment."
        ),
        "split": args.split,
        "orders": orders,
        "interpolators": interpolators,
        "sequences": len(dataset),
        "summary": summary_rows,
        "paired_statistics": paired_results,
    }
    with open(os.path.join(args.save_dir, "diagnostic_report.json"), "w") as handle:
        json.dump(json_safe(report), handle, indent=2)

    if not args.no_figures:
        save_figures(args, methods, figure_cache)
    print_summary(summary_rows, paired_results)
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
