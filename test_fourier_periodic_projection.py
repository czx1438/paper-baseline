"""Offline periodic Fourier projection for multi-phase registration DVFs.

This script does not train a network. It evaluates raw Baseline and Pairwise
MotionFiLM DVFs, then projects each same-slice nine-phase DVF sequence onto a
low-order periodic basis. Every raw model uses the same cached LDM pair scores.

For phase angle theta, the projected displacement is

    u(theta) = sum_k A_k sin(k theta) + B_k (cos(k theta) - 1)

so u(0) = u(2*pi) = 0 by construction. Phase 0 is the fixed image and phases
1..9 are located at theta = 2*pi*p/10.
"""

import argparse
import csv
import json
import math
import os
import re

import numpy as np
import torch
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
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--orders", default="1,2")
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
        default="./logs/fourier_periodic_projection",
    )
    return parser.parse_args()


def parse_orders(text):
    orders = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not orders:
        raise ValueError("--orders must contain at least one positive integer")
    if min(orders) < 1:
        raise ValueError("Every Fourier order must be >= 1")
    if max(orders) * 2 > NUM_PHASES:
        raise ValueError(
            f"Order {max(orders)} has too many coefficients for {NUM_PHASES} phases"
        )
    return orders


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
    raise ValueError(f"Cannot find a model state_dict in {path}")


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


def fourier_design(order, device, dtype):
    phases = torch.arange(1, TOTAL_CYCLE_PHASES, device=device, dtype=dtype)
    theta = 2.0 * math.pi * phases / float(TOTAL_CYCLE_PHASES)
    columns = []
    for harmonic in range(1, order + 1):
        columns.append(torch.sin(harmonic * theta))
        columns.append(torch.cos(harmonic * theta) - 1.0)
    return torch.stack(columns, dim=1)


def fourier_projection_matrix(order, device, dtype):
    design = fourier_design(order, device=device, dtype=dtype)
    return design @ torch.linalg.pinv(design)


def project_periodic(displacement_sequence, order):
    """Project [B, 9, 2, H, W] DVFs onto a periodic order-K basis."""
    projection = fourier_projection_matrix(
        order,
        device=displacement_sequence.device,
        dtype=displacement_sequence.dtype,
    )
    return torch.einsum("ts,bschw->btchw", projection, displacement_sequence)


def to_pixel_displacement(displacement_sequence):
    result = displacement_sequence.clone()
    height, width = result.shape[-2:]
    result[:, :, 0] *= height / 2.0
    result[:, :, 1] *= width / 2.0
    return result


def displacement_difference_px(first, second):
    difference = to_pixel_displacement(first - second)
    return torch.sqrt((difference ** 2).sum(dim=2) + 1e-12)


def trajectory_metrics(displacement_sequence, raw_sequence):
    """Metrics over [B, 9, 2, H, W], including phase-0 closure endpoints."""
    sequence_px = to_pixel_displacement(displacement_sequence)
    zero = torch.zeros_like(sequence_px[:, :1])
    closed = torch.cat([zero, sequence_px, zero], dim=1)
    velocity = closed[:, 1:] - closed[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    velocity_norm = torch.sqrt((velocity ** 2).sum(dim=2) + 1e-12)
    acceleration_norm = torch.sqrt((acceleration ** 2).sum(dim=2) + 1e-12)
    endpoint_norm = torch.sqrt((sequence_px[:, -1] ** 2).sum(dim=1) + 1e-12)
    projection_change = displacement_difference_px(
        displacement_sequence, raw_sequence
    )
    return {
        "phase9_to_phase0_gap_px": float(endpoint_norm.mean().item()),
        "temporal_velocity_px": float(velocity_norm.mean().item()),
        "temporal_acceleration_px": float(acceleration_norm.mean().item()),
        "projection_change_px": float(projection_change.mean().item()),
        "projection_change_max_px": float(projection_change.max().item()),
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
        "magnitude_map": magnitude,
    }


def masked_mse(prediction, target, mask):
    numerator = (((prediction - target) ** 2) * mask).sum()
    return float((numerator / mask.sum().clamp(min=1.0)).item())


def parse_name(name):
    match = re.search(r"block(\d+)_slice(\d+)", str(name))
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def method_name(source, order):
    return f"{source}_raw" if order == 0 else f"{source}_fourier_k{order}"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
            baseline, moving, fixed, score_cache[phase], phase_id=None
        )
        pairwise_dvf, motion_code = model_forward(
            pairwise, moving, fixed, score_cache[phase], phase_id=None
        )
        if motion_code is None:
            raise RuntimeError("Pairwise MotionFiLM returned no motion code")
        output["baseline"].append(baseline_dvf)
        output["pairwise"].append(pairwise_dvf)
    return {
        source: torch.stack(displacements, dim=1)
        for source, displacements in output.items()
    }


@torch.no_grad()
def evaluate(args, orders, dataset, baseline, pairwise, ldm_model, transform):
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
    pair_rows = []
    trajectory_rows = []
    figure_cache = {}

    seed_everything(args.seed + 1000)
    for batch_index, batch in enumerate(loader):
        fixed = batch[0].cuda().float()
        moving_sequence = batch[1].cuda().float()
        phase_ids = batch[2].cuda().long()
        names = batch[3]
        if fixed.shape[0] != 1:
            raise ValueError("Use --bs 1")
        if moving_sequence.shape[1] != NUM_PHASES:
            raise ValueError(f"Expected {NUM_PHASES} moving phases")

        foreground = body_mask(fixed, args.fg_thr)
        block_id, slice_id = parse_name(names[0])
        if slice_id < 0 and hasattr(dataset, "base_samples"):
            block_id, slice_id = dataset.base_samples[batch_index]

        raw = raw_sequences(
            args,
            fixed,
            moving_sequence,
            baseline,
            pairwise,
            ldm_model,
        )
        all_sequences = {}
        for source in SOURCE_NAMES:
            all_sequences[method_name(source, 0)] = raw[source]
            for order in orders:
                all_sequences[method_name(source, order)] = project_periodic(
                    raw[source], order
                )

        cached_methods = {}
        for name, displacement_sequence in all_sequences.items():
            source = name.split("_")[0]
            order = 0 if name.endswith("_raw") else int(name.rsplit("k", 1)[1])
            trajectory = trajectory_metrics(
                displacement_sequence,
                raw[source],
            )
            trajectory_rows.append(
                {
                    "method": name,
                    "source": source,
                    "fourier_order": order,
                    "block_id": block_id,
                    "slice_id": slice_id,
                    **trajectory,
                }
            )

            method_dvfs = []
            method_warped = []
            method_ncc = []
            for phase in range(NUM_PHASES):
                moving = moving_sequence[:, phase]
                displacement = displacement_sequence[:, phase]
                _, warped = transform(
                    moving,
                    displacement.permute(0, 2, 3, 1),
                )
                jacobian = jacobian_result(displacement)
                ncc_before = 1.0 - ncc_loss(
                    fixed, moving, mask=foreground
                ).item()
                ncc_after = 1.0 - ncc_loss(
                    fixed, warped, mask=foreground
                ).item()
                phase_id = int(phase_ids[0, phase].item()) + 1
                phase_change = displacement_difference_px(
                    displacement[:, None], raw[source][:, phase:phase + 1]
                )
                pair_rows.append(
                    {
                        "method": name,
                        "source": source,
                        "fourier_order": order,
                        "block_id": block_id,
                        "slice_id": slice_id,
                        "phase": phase_id,
                        "ncc_before": ncc_before,
                        "ncc_after": ncc_after,
                        "ncc_delta": ncc_after - ncc_before,
                        "mse_before": masked_mse(moving, fixed, foreground),
                        "mse_after": masked_mse(warped, fixed, foreground),
                        "projection_change_px": float(phase_change.mean().item()),
                        **{
                            key: value
                            for key, value in jacobian.items()
                            if not key.endswith("_map")
                        },
                    }
                )
                if slice_id in wanted_slices:
                    method_dvfs.append(displacement.detach().cpu())
                    method_warped.append(warped.detach().cpu())
                    method_ncc.append(ncc_after)
            if slice_id in wanted_slices:
                cached_methods[name] = {
                    "dvfs": method_dvfs,
                    "warped": method_warped,
                    "ncc": method_ncc,
                }

        if slice_id in wanted_slices:
            figure_cache[slice_id] = {
                "fixed": fixed.detach().cpu(),
                "moving": moving_sequence.detach().cpu(),
                "methods": cached_methods,
            }
        print(f"[Eval] sequence {batch_index + 1}/{len(loader)}")

    return pair_rows, trajectory_rows, figure_cache


def aggregate(rows, group_key, metric_keys):
    output = []
    groups = sorted({row[group_key] for row in rows})
    for group in groups:
        subset = [row for row in rows if row[group_key] == group]
        record = {group_key: group, "samples": len(subset)}
        for key in metric_keys:
            values = np.asarray([row[key] for row in subset], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1))
            record[f"{key}_median"] = float(np.median(values))
        output.append(record)
    return output


def phase_summaries(rows):
    output = []
    for method in sorted({row["method"] for row in rows}):
        for phase in range(1, NUM_PHASES + 1):
            subset = [
                row
                for row in rows
                if row["method"] == method and row["phase"] == phase
            ]
            output.append(
                {
                    "method": method,
                    "phase": phase,
                    "samples": len(subset),
                    "ncc_after_mean": float(np.mean([x["ncc_after"] for x in subset])),
                    "ncc_after_std": float(np.std([x["ncc_after"] for x in subset], ddof=1)),
                    "neg_ratio_mean": float(np.mean([x["neg_ratio"] for x in subset])),
                    "fold_count_mean": float(np.mean([x["fold_count"] for x in subset])),
                    "projection_change_px_mean": float(
                        np.mean([x["projection_change_px"] for x in subset])
                    ),
                }
            )
    return output


def paired_statistics(rows, orders):
    indexed = {
        (row["method"], row["slice_id"], row["phase"]): row for row in rows
    }
    comparisons = []
    for source in SOURCE_NAMES:
        raw_name = method_name(source, 0)
        for order in orders:
            comparisons.append((method_name(source, order), raw_name))
    for order in orders:
        comparisons.append((method_name("pairwise", order), "baseline_raw"))

    slice_rows = []
    results = []
    slice_ids = sorted({row["slice_id"] for row in rows})
    for left, right in comparisons:
        ncc_differences = []
        neg_differences = []
        mse_differences = []
        for slice_id in slice_ids:
            ncc_values = []
            neg_values = []
            mse_values = []
            for phase in range(1, NUM_PHASES + 1):
                left_row = indexed[(left, slice_id, phase)]
                right_row = indexed[(right, slice_id, phase)]
                ncc_values.append(left_row["ncc_after"] - right_row["ncc_after"])
                neg_values.append(left_row["neg_ratio"] - right_row["neg_ratio"])
                mse_values.append(left_row["mse_after"] - right_row["mse_after"])
            ncc_difference = float(np.mean(ncc_values))
            neg_difference = float(np.mean(neg_values))
            mse_difference = float(np.mean(mse_values))
            ncc_differences.append(ncc_difference)
            neg_differences.append(neg_difference)
            mse_differences.append(mse_difference)
            slice_rows.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "slice_id": slice_id,
                    "ncc_difference": ncc_difference,
                    "mse_difference": mse_difference,
                    "neg_ratio_difference": neg_difference,
                }
            )
        result = {
            "comparison": f"{left}_minus_{right}",
            "n_slices": len(slice_ids),
            "ncc_difference_mean": float(np.mean(ncc_differences)),
            "ncc_difference_std": float(np.std(ncc_differences, ddof=1)),
            "mse_difference_mean": float(np.mean(mse_differences)),
            "neg_ratio_difference_mean": float(np.mean(neg_differences)),
        }
        try:
            from scipy.stats import wilcoxon

            result["wilcoxon_ncc_p"] = float(wilcoxon(ncc_differences).pvalue)
            result["wilcoxon_mse_p"] = float(wilcoxon(mse_differences).pvalue)
            result["wilcoxon_neg_ratio_p"] = float(
                wilcoxon(neg_differences).pvalue
            )
        except Exception as error:
            result["wilcoxon_error"] = str(error)
        results.append(result)
    return slice_rows, results


def save_figures(args, orders, figure_cache):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for slice_id, cache in sorted(figure_cache.items()):
        fixed = cache["fixed"][0, 0].numpy()
        moving_sequence = cache["moving"]
        for source in SOURCE_NAMES:
            methods = [method_name(source, 0)] + [
                method_name(source, order) for order in orders
            ]
            labels = ["Raw"] + [f"Fourier K={order}" for order in orders]
            rows = ["Moving", "Fixed"]
            rows += [f"{label} warped" for label in labels]
            rows += [f"{label} |F-W|" for label in labels]
            rows += [f"{label} Jacobian" for label in labels]
            error_start = 2 + len(methods)
            jacobian_start = 2 + 2 * len(methods)
            fig_height = max(20, 3 * len(rows))
            fig, axes = plt.subplots(
                len(rows), NUM_PHASES, figsize=(27, fig_height)
            )
            for phase in range(NUM_PHASES):
                moving = moving_sequence[0, phase, 0].numpy()
                warped = {
                    name: cache["methods"][name]["warped"][phase][0, 0].numpy()
                    for name in methods
                }
                jacobians = {
                    name: jacobian_result(
                        cache["methods"][name]["dvfs"][phase]
                    )["jacobian_map"]
                    for name in methods
                }
                images = [moving, fixed]
                images += [warped[name] for name in methods]
                images += [np.abs(fixed - warped[name]) for name in methods]
                for row_index, image in enumerate(images):
                    cmap = "gray" if row_index < error_start else "magma"
                    axes[row_index, phase].imshow(image, cmap=cmap)
                jac_limit = max(
                    1.0,
                    max(float(np.abs(value).max()) for value in jacobians.values()),
                )
                for offset, name in enumerate(methods):
                    axes[jacobian_start + offset, phase].imshow(
                        jacobians[name],
                        cmap="RdBu_r",
                        vmin=-jac_limit,
                        vmax=jac_limit,
                    )
                for offset, name in enumerate(methods):
                    ncc_value = cache["methods"][name]["ncc"][phase]
                    axes[2 + offset, phase].set_title(
                        f"P{phase + 1} NCC={ncc_value:.4f}", fontsize=7
                    )
                for row_index in range(len(rows)):
                    axes[row_index, phase].set_xticks([])
                    axes[row_index, phase].set_yticks([])
            for row_index, label in enumerate(rows):
                axes[row_index, 0].set_ylabel(label, fontsize=8)
            fig.suptitle(
                f"{args.split} slice {slice_id}: {source} periodic projection",
                fontsize=14,
            )
            path = os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_{source}_fourier.png",
            )
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Figure] {path}")


def print_summary(summary_rows, trajectory_summary, paired_results):
    print("\n" + "=" * 110)
    print("REGISTRATION SUMMARY")
    print("=" * 110)
    for row in summary_rows:
        print(
            f"{row['method']:<24} "
            f"NCC={row['ncc_before_mean']:.6f}->{row['ncc_after_mean']:.6f} "
            f"MSE={row['mse_after_mean']:.7f} "
            f"negR={100.0 * row['neg_ratio_mean']:.4f}% "
            f"folds/image={row['fold_count_mean']:.2f}"
        )
    print("\n" + "=" * 110)
    print("TRAJECTORY SUMMARY")
    print("=" * 110)
    for row in trajectory_summary:
        print(
            f"{row['method']:<24} "
            f"gap={row['phase9_to_phase0_gap_px_mean']:.4f}px "
            f"velocity={row['temporal_velocity_px_mean']:.4f}px "
            f"acceleration={row['temporal_acceleration_px_mean']:.4f}px "
            f"change={row['projection_change_px_mean']:.4f}px"
        )
    print("\nPAIRED SLICE-LEVEL COMPARISONS")
    for result in paired_results:
        suffix = ""
        if "wilcoxon_ncc_p" in result:
            suffix = f" | p={result['wilcoxon_ncc_p']:.6g}"
        print(
            f"{result['comparison']}: "
            f"NCC={result['ncc_difference_mean']:+.6f} "
            f"negR={100.0 * result['neg_ratio_difference_mean']:+.5f}pp"
            f"{suffix}"
        )


def main():
    args = parse_args()
    orders = parse_orders(args.orders)
    if args.bs != 1:
        raise ValueError("Use --bs 1")
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed + 1000)

    print(f"[Fourier] orders={orders} | total cycle phases={TOTAL_CYCLE_PHASES}")
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
        f"pairs={len(dataset) * NUM_PHASES}"
    )
    pair_rows, trajectory_rows, figure_cache = evaluate(
        args,
        orders,
        dataset,
        baseline,
        pairwise,
        ldm_model,
        transform,
    )

    summary_rows = aggregate(
        pair_rows,
        "method",
        (
            "ncc_before",
            "ncc_after",
            "ncc_delta",
            "mse_after",
            "neg_ratio",
            "fold_count",
            "min_jac",
            "mean_dvf_px",
            "max_dvf_px",
            "projection_change_px",
        ),
    )
    trajectory_summary = aggregate(
        trajectory_rows,
        "method",
        (
            "phase9_to_phase0_gap_px",
            "temporal_velocity_px",
            "temporal_acceleration_px",
            "projection_change_px",
            "projection_change_max_px",
        ),
    )
    phase_rows = phase_summaries(pair_rows)
    slice_rows, paired_results = paired_statistics(pair_rows, orders)

    write_csv(os.path.join(args.save_dir, "per_pair_metrics.csv"), pair_rows)
    write_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.save_dir, "per_phase_summary.csv"), phase_rows)
    write_csv(
        os.path.join(args.save_dir, "trajectory_per_slice.csv"), trajectory_rows
    )
    write_csv(
        os.path.join(args.save_dir, "trajectory_summary.csv"), trajectory_summary
    )
    write_csv(
        os.path.join(args.save_dir, "per_slice_differences.csv"), slice_rows
    )
    with open(
        os.path.join(args.save_dir, "paired_statistics.json"), "w"
    ) as handle:
        json.dump(paired_results, handle, indent=2)

    if not args.no_figures:
        save_figures(args, orders, figure_cache)
    print_summary(summary_rows, trajectory_summary, paired_results)
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
