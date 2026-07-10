import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_phase(pairname):
    m = re.search(r"phase(\d+)", str(pairname))
    if m is None:
        return -1
    return int(m.group(1))


def load_stats(path, method):
    df = pd.read_csv(path)
    df["Method"] = method
    df["Phase"] = df["Pairname"].apply(parse_phase)

    numeric_cols = [
        "NCC_After", "NCC_Delta",
        "SSIM_After", "SSIM_Delta",
        "NCC_HeartROI_After", "NCC_HeartROI_Delta",
        "NCC_LiverROI_After", "NCC_LiverROI_Delta",
        "SSIM_HeartROI_After", "SSIM_HeartROI_Delta",
        "SSIM_LiverROI_After", "SSIM_LiverROI_Delta",
        "Dice_Heart_After", "Dice_Heart_Delta",
        "Dice_Liver_After", "Dice_Liver_Delta",
        "Min_Jac", "N_Foldings", "Jac_Neg_Ratio",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def mean_summary(df):
    metrics = [
        "NCC_After",
        "SSIM_After",
        "NCC_HeartROI_After",
        "NCC_LiverROI_After",
        "SSIM_HeartROI_After",
        "SSIM_LiverROI_After",
        "Dice_Heart_After",
        "Dice_Liver_After",
        "Jac_Neg_Ratio",
        "N_Foldings",
        "Min_Jac",
    ]
    return df.groupby("Method")[metrics].mean().reset_index()


def phase_summary(df):
    metrics = [
        "NCC_After",
        "NCC_Delta",
        "SSIM_After",
        "SSIM_Delta",
        "NCC_HeartROI_After",
        "NCC_LiverROI_After",
        "SSIM_HeartROI_After",
        "SSIM_LiverROI_After",
        "Dice_Heart_After",
        "Dice_Liver_After",
        "Jac_Neg_Ratio",
        "N_Foldings",
        "Min_Jac",
    ]
    return df.groupby(["Method", "Phase"])[metrics].mean().reset_index()


def hard_phase_summary(df):
    hard = df[df["Phase"].isin([3, 4, 5, 6])].copy()
    return mean_summary(hard)


def plot_phase_curve(phase_df, metric, out_path, ylabel=None):
    if metric not in phase_df.columns:
        return
    plt.figure(figsize=(8, 5))

    for method in phase_df["Method"].unique():
        sub = phase_df[phase_df["Method"] == method].sort_values("Phase")
        if metric in sub.columns:
            plt.plot(sub["Phase"], sub[metric], marker="o", label=method)

    plt.xlabel("Phase")
    plt.ylabel(ylabel or metric)
    plt.title(f"{metric} by phase")
    plt.xticks(range(1, 10))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def print_rank(summary, metric, higher_better=True):
    s = summary[["Method", metric]].copy()
    s = s.sort_values(metric, ascending=not higher_better)
    print(f"\n[{metric}]")
    print(s.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_012", required=True)
    parser.add_argument("--path_0", required=True)
    parser.add_argument("--path_013", required=True)
    parser.add_argument("--path_023", required=True)
    parser.add_argument("--path_123", required=True)
    parser.add_argument("--path_baseline", required=True)
    parser.add_argument("--out_dir", default="./analysis_compare_6methods")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_012 = load_stats(args.path_012, "012")
    df_0 = load_stats(args.path_0, "0")
    df_013 = load_stats(args.path_013, "013")
    df_023 = load_stats(args.path_023, "023")
    df_123 = load_stats(args.path_123, "123")
    df_baseline = load_stats(args.path_baseline, "baseline")

    df_all = pd.concat([df_012, df_0, df_013, df_023, df_123, df_baseline], ignore_index=True)

    overall = mean_summary(df_all)
    phase = phase_summary(df_all)
    hard = hard_phase_summary(df_all)

    df_all.to_csv(os.path.join(args.out_dir, "all_raw_merged.csv"), index=False)
    overall.to_csv(os.path.join(args.out_dir, "summary_overall.csv"), index=False)
    phase.to_csv(os.path.join(args.out_dir, "summary_by_phase.csv"), index=False)
    hard.to_csv(os.path.join(args.out_dir, "summary_hard_phase_03_06.csv"), index=False)

    print("\n================ Overall Summary ================")
    print(overall.to_string(index=False))

    print("\n================ Hard Phase 03-06 Summary ================")
    print(hard.to_string(index=False))

    print_rank(overall, "NCC_After", higher_better=True)
    print_rank(overall, "SSIM_After", higher_better=True)
    print_rank(overall, "NCC_HeartROI_After", higher_better=True)
    print_rank(overall, "NCC_LiverROI_After", higher_better=True)
    print_rank(overall, "SSIM_HeartROI_After", higher_better=True)
    print_rank(overall, "SSIM_LiverROI_After", higher_better=True)
    print_rank(overall, "Dice_Heart_After", higher_better=True)
    print_rank(overall, "Dice_Liver_After", higher_better=True)
    print_rank(overall, "Jac_Neg_Ratio", higher_better=False)
    print_rank(overall, "N_Foldings", higher_better=False)
    print_rank(overall, "Min_Jac", higher_better=True)

    plot_phase_curve(
        phase,
        "NCC_After",
        os.path.join(args.out_dir, "phase_curve_ncc_after.png"),
        ylabel="NCC after"
    )
    plot_phase_curve(
        phase,
        "SSIM_After",
        os.path.join(args.out_dir, "phase_curve_ssim_after.png"),
        ylabel="SSIM after"
    )
    plot_phase_curve(
        phase,
        "NCC_HeartROI_After",
        os.path.join(args.out_dir, "phase_curve_heart_ncc_after.png"),
        ylabel="Heart ROI NCC after"
    )
    plot_phase_curve(
        phase,
        "NCC_LiverROI_After",
        os.path.join(args.out_dir, "phase_curve_liver_ncc_after.png"),
        ylabel="Liver ROI NCC after"
    )
    plot_phase_curve(
        phase,
        "SSIM_HeartROI_After",
        os.path.join(args.out_dir, "phase_curve_heart_ssim_after.png"),
        ylabel="Heart ROI SSIM after"
    )
    plot_phase_curve(
        phase,
        "SSIM_LiverROI_After",
        os.path.join(args.out_dir, "phase_curve_liver_ssim_after.png"),
        ylabel="Liver ROI SSIM after"
    )
    plot_phase_curve(
        phase,
        "Dice_Heart_After",
        os.path.join(args.out_dir, "phase_curve_heart_dice_after.png"),
        ylabel="Heart Dice after"
    )
    plot_phase_curve(
        phase,
        "Dice_Liver_After",
        os.path.join(args.out_dir, "phase_curve_liver_dice_after.png"),
        ylabel="Liver Dice after"
    )
    plot_phase_curve(
        phase,
        "Jac_Neg_Ratio",
        os.path.join(args.out_dir, "phase_curve_jac_neg_ratio.png"),
        ylabel="Jacobian negative ratio (%)"
    )
    plot_phase_curve(
        phase,
        "N_Foldings",
        os.path.join(args.out_dir, "phase_curve_n_foldings.png"),
        ylabel="Number of Foldings"
    )
    plot_phase_curve(
        phase,
        "Min_Jac",
        os.path.join(args.out_dir, "phase_curve_min_jac.png"),
        ylabel="Min Jacobian"
    )

    # 重点表：hard phase 每个 phase 展开
    hard_phase = phase[phase["Phase"].isin([3, 4, 5, 6])].copy()
    hard_phase.to_csv(os.path.join(args.out_dir, "hard_phase_03_06_by_phase.csv"), index=False)

    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()