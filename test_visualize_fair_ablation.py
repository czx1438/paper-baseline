"""Fair test and visualization for baseline vs multi-phase MotionFiLM.

This script is intentionally aligned with train_multiphase_motionfilm.py:
  * same hard block split (train 0/1/3/5, val 2, test 4)
  * same fixed-based min-max normalization
  * same foreground-masked 15x15 local NCC
  * same paired moving/fixed LDM feature extraction with shared diffusion noise
  * same SpatialTransform and Jacobian convention
  * baseline: use_motion_film=False, phase_id=None
  * MotionFiLM-no-phase: use_motion_film=True, phase_id=None

It evaluates both checkpoints in one loop. Therefore each pair is processed
with exactly the same LDM features and diffusion noise for both models.
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

from ldm.data.xcat_multiphase import (
    MultiPhaseDataset,
    PairwisePhaseDataset,
    collate_multiphase,
    collate_pairwise,
)
from TransModels.LDMMorph import LDMMorph
from utils.utils import SpatialTransform, jacobian_determinant_vxm

# Reuse the exact feature extraction and NCC implementation used in training.
from train_multiphase_motionfilm import (
    body_mask,
    extract_pair_scores,
    load_ldm,
    model_forward,
    ncc_loss,
)


BLOCK_SPLIT = {
    "train": {0, 1, 3, 5},
    "val": {2},
    "test": {4},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--motionfilm_ckpt", required=True)
    parser.add_argument("--ldm_config", required=True)
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument(
        "--data_root",
        default="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--t_enc", type=int, default=1)
    parser.add_argument("--fg_thr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--no_ldm", action="store_true")
    parser.add_argument(
        "--motion_use_phase_id",
        action="store_true",
        help="Pass phase IDs to MotionFiLM. Leave OFF for the no-phase model.",
    )
    parser.add_argument(
        "--visual_slices",
        default="0,9,18",
        help="Comma-separated slice IDs from the selected block split.",
    )
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument(
        "--save_dir",
        default="./logs/fair_ablation_test",
    )
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_registration_model(path, use_motion_film, use_ldm=True):
    model = LDMMorph(
        128 * 2,
        192 * 2,
        320 * 2,
        448 * 2,
        use_ldm=use_ldm,
        use_motion_film=use_motion_film,
    ).cuda()
    payload = torch.load(path, map_location="cuda")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    print(
        f"[Model] loaded {path} | "
        f"use_motion_film={use_motion_film} | strict=True"
    )
    return model


def masked_mse_value(prediction, target, mask):
    value = (((prediction - target) ** 2) * mask).sum()
    return float((value / mask.sum().clamp(min=1.0)).item())


def jacobian_metrics(displacement):
    """Return one metric dict per image, matching training's convention."""
    results = []
    for index in range(displacement.shape[0]):
        dvf = displacement[index].detach().cpu().numpy().copy()
        height, width = dvf.shape[-2:]
        dvf[0] *= height / 2.0
        dvf[1] *= width / 2.0
        determinant = jacobian_determinant_vxm(dvf)
        magnitude = np.sqrt(dvf[0] ** 2 + dvf[1] ** 2)
        results.append(
            {
                "neg_ratio": float((determinant < 0).mean()),
                "fold_count": int((determinant < 0).sum()),
                "min_jac": float(determinant.min()),
                "mean_jac": float(determinant.mean()),
                "max_jac": float(determinant.max()),
                "mean_dvf_px": float(magnitude.mean()),
                "max_dvf_px": float(magnitude.max()),
            }
        )
    return results


def parse_pairname(name, fallback_phase):
    match = re.search(r"block(\d+)_slice(\d+)_phase(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return -1, -1, int(fallback_phase)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, model_name):
    subset = [row for row in rows if row["model"] == model_name]

    def stats(key):
        values = np.asarray([row[key] for row in subset], dtype=np.float64)
        return float(values.mean()), float(values.std(ddof=1)), float(np.median(values))

    result = {"model": model_name, "samples": len(subset)}
    for key in [
        "ncc_before",
        "ncc_after",
        "ncc_delta",
        "mse_before",
        "mse_after",
        "neg_ratio",
        "fold_count",
        "min_jac",
        "mean_dvf_px",
        "max_dvf_px",
    ]:
        mean, std, median = stats(key)
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = std
        result[f"{key}_median"] = median
    return result


def phase_summaries(rows):
    output = []
    for model_name in ["baseline", "motionfilm"]:
        for phase in range(1, 10):
            subset = [
                row
                for row in rows
                if row["model"] == model_name and row["phase"] == phase
            ]
            for metric in ["ncc_after", "ncc_delta", "neg_ratio", "fold_count"]:
                values = np.asarray([row[metric] for row in subset], dtype=np.float64)
                if metric == "ncc_after":
                    record = {
                        "model": model_name,
                        "phase": phase,
                        "samples": len(subset),
                    }
                    output.append(record)
                output[-1][f"{metric}_mean"] = float(values.mean())
                output[-1][f"{metric}_std"] = float(values.std(ddof=1))
    return output


def paired_rows(all_rows):
    by_key = {}
    for row in all_rows:
        key = row["pairname"]
        by_key.setdefault(key, {})[row["model"]] = row

    output = []
    for pairname in sorted(by_key):
        pair = by_key[pairname]
        if "baseline" not in pair or "motionfilm" not in pair:
            continue
        baseline = pair["baseline"]
        motion = pair["motionfilm"]
        output.append(
            {
                "pairname": pairname,
                "block_id": baseline["block_id"],
                "slice_id": baseline["slice_id"],
                "phase": baseline["phase"],
                "ncc_before": baseline["ncc_before"],
                "baseline_ncc_after": baseline["ncc_after"],
                "motionfilm_ncc_after": motion["ncc_after"],
                "motion_minus_baseline_ncc": (
                    motion["ncc_after"] - baseline["ncc_after"]
                ),
                "baseline_neg_ratio": baseline["neg_ratio"],
                "motionfilm_neg_ratio": motion["neg_ratio"],
                "motion_minus_baseline_neg_ratio": (
                    motion["neg_ratio"] - baseline["neg_ratio"]
                ),
                "baseline_fold_count": baseline["fold_count"],
                "motionfilm_fold_count": motion["fold_count"],
            }
        )
    return output


def paired_statistics(comparison_rows):
    """Compute paired tests on 19 slice means, not 171 correlated pairs."""
    slice_groups = {}
    for row in comparison_rows:
        slice_groups.setdefault(row["slice_id"], []).append(row)

    slice_rows = []
    for slice_id, rows in sorted(slice_groups.items()):
        slice_rows.append(
            {
                "slice_id": slice_id,
                "ncc_difference": float(
                    np.mean([row["motion_minus_baseline_ncc"] for row in rows])
                ),
                "neg_ratio_difference": float(
                    np.mean(
                        [row["motion_minus_baseline_neg_ratio"] for row in rows]
                    )
                ),
            }
        )

    result = {
        "unit": "slice_mean_over_9_phases",
        "n_slices": len(slice_rows),
        "motion_minus_baseline_ncc_mean": float(
            np.mean([row["ncc_difference"] for row in slice_rows])
        ),
        "motion_minus_baseline_ncc_std": float(
            np.std([row["ncc_difference"] for row in slice_rows], ddof=1)
        ),
        "motion_minus_baseline_neg_ratio_mean": float(
            np.mean([row["neg_ratio_difference"] for row in slice_rows])
        ),
    }
    try:
        from scipy.stats import wilcoxon

        ncc_values = [row["ncc_difference"] for row in slice_rows]
        jac_values = [row["neg_ratio_difference"] for row in slice_rows]
        result["wilcoxon_ncc_p"] = float(
            wilcoxon(ncc_values, alternative="two-sided").pvalue
        )
        result["wilcoxon_neg_ratio_p"] = float(
            wilcoxon(jac_values, alternative="two-sided").pvalue
        )
    except Exception as error:
        result["wilcoxon_error"] = str(error)
    return slice_rows, result


@torch.no_grad()
def evaluate_both(args, baseline, motionfilm, ldm_model, transform):
    dataset = PairwisePhaseDataset(
        data_root=args.data_root,
        split=args.split,
        flip_p=0.0,
        normalize=True,
        block_split=BLOCK_SPLIT,
    )
    loader = Data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pairwise,
    )
    print(
        f"[Data] split={args.split} blocks={sorted(dataset.blocks)} "
        f"pairs={len(dataset)}"
    )

    rows = []
    seed_everything(args.seed)
    for batch_index, batch in enumerate(loader):
        fixed, moving, phase_ids, names = batch
        fixed = fixed.cuda().float()
        moving = moving.cuda().float()
        phase_ids = phase_ids.cuda().long()
        foreground = body_mask(fixed, args.fg_thr)

        # Shared once by both models: exactly identical LDM features/noise.
        scores = extract_pair_scores(ldm_model, moving, fixed, args.t_enc)
        baseline_dvf, _ = model_forward(
            baseline, moving, fixed, scores, phase_id=None
        )
        motion_phase = phase_ids if args.motion_use_phase_id else None
        motion_dvf, _ = model_forward(
            motionfilm, moving, fixed, scores, phase_id=motion_phase
        )
        _, baseline_warped = transform(
            moving, baseline_dvf.permute(0, 2, 3, 1)
        )
        _, motion_warped = transform(
            moving, motion_dvf.permute(0, 2, 3, 1)
        )
        baseline_jac = jacobian_metrics(baseline_dvf)
        motion_jac = jacobian_metrics(motion_dvf)

        for item in range(fixed.shape[0]):
            item_fixed = fixed[item:item + 1]
            item_moving = moving[item:item + 1]
            item_mask = foreground[item:item + 1]
            phase = int(phase_ids[item].item()) + 1
            block_id, slice_id, name_phase = parse_pairname(names[item], phase)
            phase = name_phase
            ncc_before = 1.0 - ncc_loss(
                item_fixed, item_moving, mask=item_mask
            ).item()
            mse_before = masked_mse_value(item_moving, item_fixed, item_mask)

            for model_name, warped, jac in [
                ("baseline", baseline_warped[item:item + 1], baseline_jac[item]),
                ("motionfilm", motion_warped[item:item + 1], motion_jac[item]),
            ]:
                row = {
                    "model": model_name,
                    "pairname": names[item],
                    "block_id": block_id,
                    "slice_id": slice_id,
                    "phase": phase,
                    "ncc_before": ncc_before,
                    "ncc_after": 1.0
                    - ncc_loss(item_fixed, warped, mask=item_mask).item(),
                    "mse_before": mse_before,
                    "mse_after": masked_mse_value(warped, item_fixed, item_mask),
                }
                row["ncc_delta"] = row["ncc_after"] - row["ncc_before"]
                row.update(jac)
                rows.append(row)

        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            print(
                f"[Eval] batch {batch_index + 1}/{len(loader)} | "
                f"processed pairs={min((batch_index + 1) * args.bs, len(dataset))}"
            )
    return rows


def jacobian_map(displacement):
    dvf = displacement[0].detach().cpu().numpy().copy()
    height, width = dvf.shape[-2:]
    dvf[0] *= height / 2.0
    dvf[1] *= width / 2.0
    return jacobian_determinant_vxm(dvf), np.sqrt(dvf[0] ** 2 + dvf[1] ** 2)


@torch.no_grad()
def visualize_slice(
    args,
    slice_id,
    dataset,
    baseline,
    motionfilm,
    ldm_model,
    transform,
    figure_dir,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_index = None
    for index, (_, candidate_slice) in enumerate(dataset.base_samples):
        if candidate_slice == slice_id:
            dataset_index = index
            break
    if dataset_index is None:
        print(f"[Figure] slice {slice_id} not found; skipped")
        return

    batch = collate_multiphase([dataset[dataset_index]])
    fixed, moving_sequence, phase_ids, names = batch[:4]
    fixed = fixed.cuda().float()
    moving_sequence = moving_sequence.cuda().float()
    phase_ids = phase_ids.cuda().long()
    foreground = body_mask(fixed, args.fg_thr)

    baseline_warped = []
    motion_warped = []
    baseline_jac = []
    motion_jac = []
    baseline_mag = []
    motion_mag = []
    baseline_ncc = []
    motion_ncc = []

    # Deterministic and shared feature realization for both models.
    seed_everything(args.seed + slice_id)
    for phase in range(9):
        moving = moving_sequence[:, phase]
        scores = extract_pair_scores(ldm_model, moving, fixed, args.t_enc)
        baseline_dvf, _ = model_forward(
            baseline, moving, fixed, scores, phase_id=None
        )
        motion_phase = phase_ids[:, phase] if args.motion_use_phase_id else None
        motion_dvf, _ = model_forward(
            motionfilm, moving, fixed, scores, phase_id=motion_phase
        )
        _, baseline_image = transform(
            moving, baseline_dvf.permute(0, 2, 3, 1)
        )
        _, motion_image = transform(
            moving, motion_dvf.permute(0, 2, 3, 1)
        )
        base_jac, base_mag = jacobian_map(baseline_dvf)
        film_jac, film_mag = jacobian_map(motion_dvf)
        baseline_warped.append(baseline_image[0, 0].cpu().numpy())
        motion_warped.append(motion_image[0, 0].cpu().numpy())
        baseline_jac.append(base_jac)
        motion_jac.append(film_jac)
        baseline_mag.append(base_mag)
        motion_mag.append(film_mag)
        baseline_ncc.append(
            1.0 - ncc_loss(fixed, baseline_image, mask=foreground).item()
        )
        motion_ncc.append(
            1.0 - ncc_loss(fixed, motion_image, mask=foreground).item()
        )

    fixed_np = fixed[0, 0].cpu().numpy()
    fig, axes = plt.subplots(10, 9, figsize=(27, 27))
    row_names = [
        "Moving",
        "Fixed",
        "Baseline warped",
        "MotionFiLM warped",
        "Baseline |F-W|",
        "MotionFiLM |F-W|",
        "Baseline DVF magnitude",
        "MotionFiLM DVF magnitude",
        "Baseline Jacobian",
        "MotionFiLM Jacobian",
    ]
    for phase in range(9):
        moving_np = moving_sequence[0, phase, 0].cpu().numpy()
        axes[0, phase].imshow(moving_np, cmap="gray", vmin=0, vmax=1)
        axes[1, phase].imshow(fixed_np, cmap="gray", vmin=0, vmax=1)
        axes[2, phase].imshow(baseline_warped[phase], cmap="gray", vmin=0, vmax=1)
        axes[3, phase].imshow(motion_warped[phase], cmap="gray", vmin=0, vmax=1)
        axes[4, phase].imshow(
            np.abs(fixed_np - baseline_warped[phase]), cmap="magma", vmin=0
        )
        axes[5, phase].imshow(
            np.abs(fixed_np - motion_warped[phase]), cmap="magma", vmin=0
        )
        magnitude_max = max(
            float(baseline_mag[phase].max()),
            float(motion_mag[phase].max()),
            1e-6,
        )
        axes[6, phase].imshow(
            baseline_mag[phase], cmap="viridis", vmin=0, vmax=magnitude_max
        )
        axes[7, phase].imshow(
            motion_mag[phase], cmap="viridis", vmin=0, vmax=magnitude_max
        )
        jac_limit = max(
            float(np.abs(baseline_jac[phase]).max()),
            float(np.abs(motion_jac[phase]).max()),
            1.0,
        )
        axes[8, phase].imshow(
            baseline_jac[phase], cmap="RdBu_r", vmin=-jac_limit, vmax=jac_limit
        )
        axes[9, phase].imshow(
            motion_jac[phase], cmap="RdBu_r", vmin=-jac_limit, vmax=jac_limit
        )
        axes[0, phase].set_title(f"Phase {phase + 1}")
        axes[2, phase].set_title(f"B NCC={baseline_ncc[phase]:.3f}", fontsize=8)
        axes[3, phase].set_title(f"M NCC={motion_ncc[phase]:.3f}", fontsize=8)
        for row in range(10):
            axes[row, phase].set_xticks([])
            axes[row, phase].set_yticks([])

    for row, name in enumerate(row_names):
        axes[row, 0].set_ylabel(name, fontsize=9)
    mode_label = "with_phase" if args.motion_use_phase_id else "no_phase"
    fig.suptitle(
        f"{args.split} {names[0]} | Baseline vs MotionFiLM-{mode_label}",
        fontsize=14,
    )
    path = os.path.join(
        figure_dir,
        f"{args.split}_{names[0]}_baseline_vs_motionfilm.png",
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] saved {path}")


def print_summary(summary_rows, paired_result):
    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    for row in summary_rows:
        print(
            f"{row['model']:<11} "
            f"NCC={row['ncc_before_mean']:.6f}->{row['ncc_after_mean']:.6f} "
            f"delta={row['ncc_delta_mean']:+.6f} "
            f"negR={100.0 * row['neg_ratio_mean']:.4f}% "
            f"folds/image={row['fold_count_mean']:.2f} "
            f"minJ(median)={row['min_jac_median']:.4f}"
        )
    print(
        "Paired slice-level MotionFiLM - Baseline NCC = "
        f"{paired_result['motion_minus_baseline_ncc_mean']:+.6f}"
    )
    if "wilcoxon_ncc_p" in paired_result:
        print(f"Wilcoxon p (19 slice means, NCC) = {paired_result['wilcoxon_ncc_p']:.6g}")
        print(
            "Wilcoxon p (19 slice means, negR) = "
            f"{paired_result['wilcoxon_neg_ratio_p']:.6g}"
        )


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    print(f"[Mode] split={args.split} | MotionFiLM phase ID={args.motion_use_phase_id}")
    print(f"[Output] {args.save_dir}")
    ldm_model = load_ldm(args.ldm_config, args.ldm_ckpt)
    baseline = load_registration_model(
        args.baseline_ckpt,
        use_motion_film=False,
        use_ldm=not args.no_ldm,
    )
    motionfilm = load_registration_model(
        args.motionfilm_ckpt,
        use_motion_film=True,
        use_ldm=not args.no_ldm,
    )
    transform = SpatialTransform().cuda().eval()
    for parameter in transform.parameters():
        parameter.requires_grad_(False)

    rows = evaluate_both(
        args, baseline, motionfilm, ldm_model, transform
    )
    summary_rows = [
        summarize(rows, "baseline"),
        summarize(rows, "motionfilm"),
    ]
    phase_rows = phase_summaries(rows)
    comparison_rows = paired_rows(rows)
    slice_rows, paired_result = paired_statistics(comparison_rows)

    write_csv(os.path.join(args.save_dir, "per_pair_metrics.csv"), rows)
    write_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.save_dir, "per_phase_summary.csv"), phase_rows)
    write_csv(os.path.join(args.save_dir, "paired_comparison.csv"), comparison_rows)
    write_csv(os.path.join(args.save_dir, "per_slice_differences.csv"), slice_rows)
    with open(os.path.join(args.save_dir, "paired_statistics.json"), "w") as handle:
        json.dump(paired_result, handle, indent=2)

    if not args.no_figures:
        multi_dataset = MultiPhaseDataset(
            data_root=args.data_root,
            split=args.split,
            flip_p=0.0,
            normalize=True,
            block_split=BLOCK_SPLIT,
        )
        slice_ids = [
            int(value.strip())
            for value in args.visual_slices.split(",")
            if value.strip()
        ]
        for slice_id in slice_ids:
            visualize_slice(
                args,
                slice_id,
                multi_dataset,
                baseline,
                motionfilm,
                ldm_model,
                transform,
                figure_dir,
            )

    print_summary(summary_rows, paired_result)
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()

