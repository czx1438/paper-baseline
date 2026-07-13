"""Fair test and visualization for phase-attention stage-two ablation.

Compares the original baseline, frozen Pairwise MotionFiLM coarse output,
attention/residual refinement, residual-only refinement, and budget-matched
continued-training control. Every model uses the same test sequence and the
same cached LDM pair features.
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
import torch.utils.data as Data

from ldm.data.xcat_multiphase import MultiPhaseDataset, collate_multiphase
from TransModels.LDMMorph import LDMMorph
from TransModels.PhaseAttentionResidual import CrossPhaseAttentionResidual
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


MODEL_NAMES = (
    "baseline",
    "coarse_pairwise",
    "attention_residual",
    "residual_only",
    "continued_control",
)
MODEL_LABELS = {
    "baseline": "Baseline",
    "coarse_pairwise": "Pairwise coarse",
    "attention_residual": "Attention",
    "residual_only": "Residual-only",
    "continued_control": "Control",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--attention_ckpt", required=True)
    parser.add_argument("--residual_ckpt", required=True)
    parser.add_argument("--control_ckpt", required=True)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Use the stage-two training seed; evaluation uses seed + 1000.",
    )
    parser.add_argument("--visual_slices", default="0,9,18")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--no_ldm", action="store_true")
    parser.add_argument(
        "--save_dir",
        default="./logs/fair_test_phase_attention_stage2",
    )
    return parser.parse_args()


def build_registration_model(use_ldm, use_motion_film):
    return LDMMorph(
        128 * 2,
        192 * 2,
        320 * 2,
        448 * 2,
        use_ldm=use_ldm,
        use_motion_film=use_motion_film,
    ).cuda()


def extract_model_state(payload, path):
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "state_dict"):
        if key in payload:
            return payload[key]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError(f"Cannot find a model state_dict in {path}")


def load_models(args):
    baseline_payload = torch.load(args.baseline_ckpt, map_location="cpu")
    attention_payload = torch.load(args.attention_ckpt, map_location="cpu")
    residual_payload = torch.load(args.residual_ckpt, map_location="cpu")
    control_payload = torch.load(args.control_ckpt, map_location="cpu")
    if attention_payload.get("mode") != "attention_residual":
        raise ValueError("--attention_ckpt is not an attention_residual checkpoint")
    if residual_payload.get("mode") != "residual_only":
        raise ValueError("--residual_ckpt is not a residual_only checkpoint")
    if control_payload.get("mode") != "continued_control":
        raise ValueError("--control_ckpt is not a continued_control checkpoint")

    baseline = build_registration_model(
        use_ldm=not args.no_ldm,
        use_motion_film=False,
    )
    baseline.load_state_dict(
        extract_model_state(baseline_payload, args.baseline_ckpt), strict=True
    )

    coarse = build_registration_model(
        use_ldm=not args.no_ldm,
        use_motion_film=True,
    )
    coarse.load_state_dict(
        attention_payload["pairwise_model_state_dict"], strict=True
    )

    config = attention_payload.get("config", {})
    refiner = CrossPhaseAttentionResidual(
        code_dim=16,
        num_heads=int(config.get("attention_heads", 4)),
        hidden_channels=int(config.get("residual_channels", 32)),
        residual_size=int(config.get("residual_size", 128)),
    ).cuda()
    refiner.load_state_dict(
        attention_payload["attention_residual_state_dict"], strict=True
    )

    residual_config = residual_payload.get("config", {})
    residual_refiner = CrossPhaseAttentionResidual(
        code_dim=16,
        num_heads=int(residual_config.get("attention_heads", 4)),
        hidden_channels=int(residual_config.get("residual_channels", 32)),
        residual_size=int(residual_config.get("residual_size", 128)),
    ).cuda()
    residual_refiner.load_state_dict(
        residual_payload["residual_only_state_dict"], strict=True
    )

    control = build_registration_model(
        use_ldm=not args.no_ldm,
        use_motion_film=True,
    )
    control.load_state_dict(control_payload["model_state_dict"], strict=True)

    residual_coarse_state = residual_payload["pairwise_model_state_dict"]
    attention_coarse_state = attention_payload["pairwise_model_state_dict"]
    for key in attention_coarse_state:
        if not torch.equal(attention_coarse_state[key], residual_coarse_state[key]):
            raise ValueError(
                "Attention and residual-only checkpoints do not contain the "
                "same frozen Pairwise MotionFiLM state"
            )

    for model in (baseline, coarse, refiner, residual_refiner, control):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    print(f"[Baseline] loaded {args.baseline_ckpt} | strict=True")
    print(
        f"[Attention] step={attention_payload.get('step')} "
        f"best_val={attention_payload.get('best_val_ncc')} | strict=True"
    )
    print(
        f"[Control] step={control_payload.get('step')} "
        f"best_val={control_payload.get('best_val_ncc')} | strict=True"
    )
    print(
        f"[Residual-only] step={residual_payload.get('step')} "
        f"best_val={residual_payload.get('best_val_ncc')} | strict=True"
    )
    return baseline, coarse, refiner, residual_refiner, control


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


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sequence_outputs(
    args,
    fixed,
    moving_sequence,
    baseline,
    coarse,
    refiner,
    residual_refiner,
    control,
    ldm_model,
    transform,
):
    score_cache = [
        extract_pair_scores(ldm_model, moving_sequence[:, phase], fixed, args.t_enc)
        for phase in range(NUM_PHASES)
    ]

    baseline_dvfs = []
    baseline_warped = []
    coarse_dvfs = []
    coarse_warped = []
    motion_codes = []
    control_dvfs = []
    control_warped = []
    for phase in range(NUM_PHASES):
        moving = moving_sequence[:, phase]
        baseline_dvf, _ = model_forward(
            baseline, moving, fixed, score_cache[phase], phase_id=None
        )
        coarse_dvf, motion_code = model_forward(
            coarse, moving, fixed, score_cache[phase], phase_id=None
        )
        control_dvf, _ = model_forward(
            control, moving, fixed, score_cache[phase], phase_id=None
        )
        if motion_code is None:
            raise RuntimeError("The coarse Pairwise MotionFiLM returned no motion code")
        _, baseline_image = transform(
            moving, baseline_dvf.permute(0, 2, 3, 1)
        )
        _, coarse_image = transform(
            moving, coarse_dvf.permute(0, 2, 3, 1)
        )
        _, control_image = transform(
            moving, control_dvf.permute(0, 2, 3, 1)
        )
        baseline_dvfs.append(baseline_dvf)
        baseline_warped.append(baseline_image)
        coarse_dvfs.append(coarse_dvf)
        coarse_warped.append(coarse_image)
        motion_codes.append(motion_code)
        control_dvfs.append(control_dvf)
        control_warped.append(control_image)

    motion_codes = torch.stack(motion_codes, dim=1)
    attention_dvfs = []
    attention_warped = []
    residuals = []
    attention_weights = None
    for phase in range(NUM_PHASES):
        refined_dvf, residual, weights = refiner.refine_phase(
            moving=moving_sequence[:, phase],
            fixed=fixed,
            pairwise_warped=coarse_warped[phase],
            pairwise_dvf=coarse_dvfs[phase],
            motion_codes=motion_codes,
            phase_index=phase,
        )
        _, refined_image = transform(
            moving_sequence[:, phase], refined_dvf.permute(0, 2, 3, 1)
        )
        attention_dvfs.append(refined_dvf)
        attention_warped.append(refined_image)
        residuals.append(residual)
        if attention_weights is None:
            attention_weights = weights

    residual_only_dvfs = []
    residual_only_warped = []
    residual_only_residuals = []
    for phase in range(NUM_PHASES):
        refined_dvf, residual, _ = residual_refiner.refine_phase_residual_only(
            moving=moving_sequence[:, phase],
            fixed=fixed,
            pairwise_warped=coarse_warped[phase],
            pairwise_dvf=coarse_dvfs[phase],
            motion_codes=motion_codes,
            phase_index=phase,
        )
        _, refined_image = transform(
            moving_sequence[:, phase], refined_dvf.permute(0, 2, 3, 1)
        )
        residual_only_dvfs.append(refined_dvf)
        residual_only_warped.append(refined_image)
        residual_only_residuals.append(residual)

    return {
        "baseline": (baseline_dvfs, baseline_warped),
        "coarse_pairwise": (coarse_dvfs, coarse_warped),
        "attention_residual": (attention_dvfs, attention_warped),
        "residual_only": (residual_only_dvfs, residual_only_warped),
        "continued_control": (control_dvfs, control_warped),
        "residuals": {
            "attention_residual": residuals,
            "residual_only": residual_only_residuals,
        },
        "attention_weights": attention_weights,
    }


@torch.no_grad()
def evaluate(
    args,
    dataset,
    baseline,
    coarse,
    refiner,
    residual_refiner,
    control,
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
    rows = []
    figure_cache = {}
    wanted_slices = {
        int(value.strip())
        for value in args.visual_slices.split(",")
        if value.strip()
    }

    seed_everything(args.seed + 1000)
    for batch_index, batch in enumerate(loader):
        fixed = batch[0].cuda().float()
        moving_sequence = batch[1].cuda().float()
        phase_ids = batch[2].cuda().long()
        names = batch[3]
        if fixed.shape[0] != 1:
            raise ValueError("Use --bs 1 so each visualization is one slice sequence")
        if moving_sequence.shape[1] != NUM_PHASES:
            raise ValueError(f"Expected {NUM_PHASES} phases")

        foreground = body_mask(fixed, args.fg_thr)
        outputs = sequence_outputs(
            args,
            fixed,
            moving_sequence,
            baseline,
            coarse,
            refiner,
            residual_refiner,
            control,
            ldm_model,
            transform,
        )
        block_id, slice_id = parse_name(names[0])
        if slice_id < 0 and hasattr(dataset, "base_samples"):
            block_id, slice_id = dataset.base_samples[batch_index]

        for phase in range(NUM_PHASES):
            moving = moving_sequence[:, phase]
            ncc_before = 1.0 - ncc_loss(
                fixed, moving, mask=foreground
            ).item()
            mse_before = masked_mse(moving, fixed, foreground)
            phase_id = int(phase_ids[0, phase].item()) + 1
            for model_name in MODEL_NAMES:
                displacement = outputs[model_name][0][phase]
                warped = outputs[model_name][1][phase]
                jac = jacobian_result(displacement)
                row = {
                    "model": model_name,
                    "block_id": block_id,
                    "slice_id": slice_id,
                    "phase": phase_id,
                    "ncc_before": ncc_before,
                    "ncc_after": 1.0
                    - ncc_loss(fixed, warped, mask=foreground).item(),
                    "mse_before": mse_before,
                    "mse_after": masked_mse(warped, fixed, foreground),
                    "neg_ratio": jac["neg_ratio"],
                    "fold_count": jac["fold_count"],
                    "min_jac": jac["min_jac"],
                    "mean_dvf_px": jac["mean_dvf_px"],
                    "max_dvf_px": jac["max_dvf_px"],
                    "residual_abs": (
                        float(
                            outputs["residuals"][model_name][phase]
                            .abs()
                            .mean()
                            .item()
                        )
                        if model_name in outputs["residuals"]
                        else 0.0
                    ),
                }
                row["ncc_delta"] = row["ncc_after"] - row["ncc_before"]
                rows.append(row)

        if slice_id in wanted_slices:
            figure_cache[slice_id] = {
                "fixed": fixed.detach().cpu(),
                "moving": moving_sequence.detach().cpu(),
                "outputs": {
                    name: (
                        [item.detach().cpu() for item in outputs[name][0]],
                        [item.detach().cpu() for item in outputs[name][1]],
                    )
                    for name in MODEL_NAMES
                },
                "attention_weights": outputs["attention_weights"].detach().cpu(),
            }

        print(f"[Eval] sequence {batch_index + 1}/{len(loader)}")
    return rows, figure_cache


def summaries(rows):
    output = []
    for model_name in MODEL_NAMES:
        subset = [row for row in rows if row["model"] == model_name]
        record = {"model": model_name, "samples": len(subset)}
        for key in (
            "ncc_before",
            "ncc_after",
            "ncc_delta",
            "mse_after",
            "neg_ratio",
            "fold_count",
            "min_jac",
            "mean_dvf_px",
            "residual_abs",
        ):
            values = np.asarray([row[key] for row in subset], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1))
            record[f"{key}_median"] = float(np.median(values))
        output.append(record)
    return output


def phase_summaries(rows):
    output = []
    for model_name in MODEL_NAMES:
        for phase in range(1, NUM_PHASES + 1):
            subset = [
                row
                for row in rows
                if row["model"] == model_name and row["phase"] == phase
            ]
            output.append(
                {
                    "model": model_name,
                    "phase": phase,
                    "samples": len(subset),
                    "ncc_after_mean": float(np.mean([x["ncc_after"] for x in subset])),
                    "ncc_after_std": float(np.std([x["ncc_after"] for x in subset], ddof=1)),
                    "neg_ratio_mean": float(np.mean([x["neg_ratio"] for x in subset])),
                    "fold_count_mean": float(np.mean([x["fold_count"] for x in subset])),
                }
            )
    return output


def paired_statistics(rows):
    indexed = {
        (row["model"], row["slice_id"], row["phase"]): row for row in rows
    }
    comparisons = [
        ("residual_only", "baseline"),
        ("attention_residual", "baseline"),
        ("coarse_pairwise", "baseline"),
        ("attention_residual", "residual_only"),
        ("attention_residual", "coarse_pairwise"),
        ("residual_only", "coarse_pairwise"),
        ("attention_residual", "continued_control"),
        ("continued_control", "coarse_pairwise"),
    ]
    slice_rows = []
    results = []
    for left, right in comparisons:
        differences = []
        neg_differences = []
        for slice_id in sorted({row["slice_id"] for row in rows}):
            ncc_values = []
            neg_values = []
            for phase in range(1, NUM_PHASES + 1):
                left_row = indexed[(left, slice_id, phase)]
                right_row = indexed[(right, slice_id, phase)]
                ncc_values.append(left_row["ncc_after"] - right_row["ncc_after"])
                neg_values.append(left_row["neg_ratio"] - right_row["neg_ratio"])
            ncc_difference = float(np.mean(ncc_values))
            neg_difference = float(np.mean(neg_values))
            differences.append(ncc_difference)
            neg_differences.append(neg_difference)
            slice_rows.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "slice_id": slice_id,
                    "ncc_difference": ncc_difference,
                    "neg_ratio_difference": neg_difference,
                }
            )
        result = {
            "comparison": f"{left}_minus_{right}",
            "n_slices": len(differences),
            "ncc_difference_mean": float(np.mean(differences)),
            "ncc_difference_std": float(np.std(differences, ddof=1)),
            "neg_ratio_difference_mean": float(np.mean(neg_differences)),
        }
        try:
            from scipy.stats import wilcoxon

            result["wilcoxon_ncc_p"] = float(wilcoxon(differences).pvalue)
            result["wilcoxon_neg_ratio_p"] = float(
                wilcoxon(neg_differences).pvalue
            )
        except Exception as error:
            result["wilcoxon_error"] = str(error)
        results.append(result)
    return slice_rows, results


def save_figures(args, figure_cache):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    for slice_id, cache in sorted(figure_cache.items()):
        fixed = cache["fixed"][0, 0].numpy()
        moving_sequence = cache["moving"]
        rows = ["Moving", "Fixed"]
        rows += [f"{MODEL_LABELS[name]} warped" for name in MODEL_NAMES]
        rows += [f"{MODEL_LABELS[name]} |F-W|" for name in MODEL_NAMES]
        rows += [f"{MODEL_LABELS[name]} Jacobian" for name in MODEL_NAMES]
        error_start = 2 + len(MODEL_NAMES)
        jacobian_start = 2 + 2 * len(MODEL_NAMES)
        fig, axes = plt.subplots(len(rows), NUM_PHASES, figsize=(27, 43))
        for phase in range(NUM_PHASES):
            moving = moving_sequence[0, phase, 0].numpy()
            warped = {
                name: cache["outputs"][name][1][phase][0, 0].numpy()
                for name in MODEL_NAMES
            }
            jacobians = {
                name: jacobian_result(cache["outputs"][name][0][phase])["jacobian_map"]
                for name in MODEL_NAMES
            }
            images = [moving, fixed]
            images += [warped[name] for name in MODEL_NAMES]
            images += [np.abs(fixed - warped[name]) for name in MODEL_NAMES]
            for row_index, image in enumerate(images):
                cmap = "gray" if row_index < error_start else "magma"
                axes[row_index, phase].imshow(image, cmap=cmap)
            jac_limit = max(
                1.0,
                max(float(np.abs(value).max()) for value in jacobians.values()),
            )
            for offset, name in enumerate(MODEL_NAMES):
                axes[jacobian_start + offset, phase].imshow(
                    jacobians[name],
                    cmap="RdBu_r",
                    vmin=-jac_limit,
                    vmax=jac_limit,
                )
            axes[0, phase].set_title(f"Phase {phase + 1}")
            for row_index in range(len(rows)):
                axes[row_index, phase].set_xticks([])
                axes[row_index, phase].set_yticks([])
        for row_index, label in enumerate(rows):
            axes[row_index, 0].set_ylabel(label, fontsize=9)
        fig.suptitle(
            f"{args.split} slice {slice_id}: Stage-2 fair comparison", fontsize=14
        )
        figure_path = os.path.join(
            figure_dir, f"{args.split}_slice{slice_id:02d}_stage2.png"
        )
        fig.savefig(figure_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        all_head_weights = cache["attention_weights"][0].numpy()
        weights = all_head_weights.mean(axis=0)
        np.save(
            os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_attention_weights.npy",
            ),
            all_head_weights,
        )
        np.savetxt(
            os.path.join(
                figure_dir,
                f"{args.split}_slice{slice_id:02d}_attention_mean.csv",
            ),
            weights,
            delimiter=",",
        )
        fig, axis = plt.subplots(figsize=(7, 6))
        value_min = float(weights.min())
        value_max = float(weights.max())
        if value_max - value_min < 1e-8:
            value_min -= 1e-8
            value_max += 1e-8
        image = axis.imshow(
            weights, cmap="viridis", vmin=value_min, vmax=value_max
        )
        axis.set_xticks(range(NUM_PHASES), range(1, NUM_PHASES + 1))
        axis.set_yticks(range(NUM_PHASES), range(1, NUM_PHASES + 1))
        axis.set_xlabel("Key/value phase")
        axis.set_ylabel("Query phase")
        axis.set_title(f"Slice {slice_id}: mean cross-phase attention")
        fig.colorbar(image, ax=axis)
        attention_path = os.path.join(
            figure_dir, f"{args.split}_slice{slice_id:02d}_attention.png"
        )
        fig.savefig(attention_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        head_count = all_head_weights.shape[0]
        fig, axes = plt.subplots(1, head_count, figsize=(5 * head_count, 4.5))
        axes = np.atleast_1d(axes)
        for head, axis in enumerate(axes):
            head_weights = all_head_weights[head]
            image = axis.imshow(
                head_weights,
                cmap="viridis",
                vmin=float(head_weights.min()),
                vmax=float(head_weights.max()),
            )
            axis.set_title(f"Head {head + 1}")
            axis.set_xticks(range(NUM_PHASES), range(1, NUM_PHASES + 1))
            axis.set_yticks(range(NUM_PHASES), range(1, NUM_PHASES + 1))
            fig.colorbar(image, ax=axis, fraction=0.046)
        fig.suptitle(f"Slice {slice_id}: per-head cross-phase attention")
        head_path = os.path.join(
            figure_dir, f"{args.split}_slice{slice_id:02d}_attention_heads.png"
        )
        fig.savefig(head_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Figure] {figure_path}")
        print(f"[Figure] {attention_path}")
        print(f"[Figure] {head_path}")


def print_summary(summary_rows, paired_results):
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    for row in summary_rows:
        print(
            f"{row['model']:<22} "
            f"NCC={row['ncc_before_mean']:.6f}->{row['ncc_after_mean']:.6f} "
            f"delta={row['ncc_delta_mean']:+.6f} "
            f"negR={100.0 * row['neg_ratio_mean']:.4f}% "
            f"folds/image={row['fold_count_mean']:.2f} "
            f"minJ(median)={row['min_jac_median']:.4f}"
        )
    for result in paired_results:
        suffix = ""
        if "wilcoxon_ncc_p" in result:
            suffix = f" | Wilcoxon p={result['wilcoxon_ncc_p']:.6g}"
        print(
            f"{result['comparison']}: "
            f"NCC={result['ncc_difference_mean']:+.6f}{suffix}"
        )


def main():
    args = parse_args()
    if args.bs != 1:
        raise ValueError("Use --bs 1")
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed + 1000)

    ldm_model = load_ldm(args.ldm_config, args.ldm_ckpt)
    baseline, coarse, refiner, residual_refiner, control = load_models(args)
    transform = SpatialTransform().cuda().eval()
    for parameter in transform.parameters():
        parameter.requires_grad_(False)

    dataset = MultiPhaseDataset(
        data_root=args.data_root,
        split=args.split,
        flip_p=0.0,
        normalize=True,
    )
    print(f"[Data] split={args.split} sequences={len(dataset)} pairs={len(dataset) * 9}")
    rows, figure_cache = evaluate(
        args,
        dataset,
        baseline,
        coarse,
        refiner,
        residual_refiner,
        control,
        ldm_model,
        transform,
    )
    summary_rows = summaries(rows)
    phase_rows = phase_summaries(rows)
    slice_rows, paired_results = paired_statistics(rows)

    write_csv(os.path.join(args.save_dir, "per_pair_metrics.csv"), rows)
    write_csv(os.path.join(args.save_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.save_dir, "per_phase_summary.csv"), phase_rows)
    write_csv(os.path.join(args.save_dir, "per_slice_differences.csv"), slice_rows)
    with open(
        os.path.join(args.save_dir, "paired_statistics.json"), "w"
    ) as handle:
        json.dump(paired_results, handle, indent=2)

    if not args.no_figures:
        save_figures(args, figure_cache)
    print_summary(summary_rows, paired_results)
    print(f"\n[Done] results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
