"""
LDM 特征可视化脚本
用于检验 LDM 提取的特征是否对 moving/fixed 图像具有区分性。

用法:
    python visualize_ldm_features.py --ldm_ckpt <LDM_checkpoint_path> --morph_ckpt <Morph_checkpoint_path>

示例:
    python visualize_ldm_features.py \
        --ldm_ckpt logs/2026-05-21T23-33-39_xcat_motion-ldm/checkpoints/last.ckpt \
        --morph_ckpt logs/XCAT_TransMorph_Smooth_1.0_beta_0.8_xcat_motion_603/NCCVal_0.9208_Epoch_024000.pth \
        --xcat_path xcat_data \
        --output_dir logs/ldm_feature_visualization
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.optim import Adam
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldm.util import instantiate_from_config, default
import TransModels.LDMMorph as LDMMorph
from utils.utils import Dataset_XCAT_Registration, SpatialTransform


def load_ldm(config_path, ckpt_path):
    """加载 LDM 模型"""
    print(f"[LDM] Loading config from {config_path}")
    print(f"[LDM] Loading checkpoint from {ckpt_path}")
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model


def extract_ldm_features(model, mov_img, fix_img, t_enc=1):
    """
    提取 moving 和 fixed 图像的 LDM 中间层特征。

    流程:
        Image -> encode -> latent_z -> 加噪声 -> U-Net去噪器 -> skip connections特征

    返回:
        mov_score0~3, fix_score0~3: 各尺度的 moving/fixed 特征
        mov_z, fix_z: latent 空间特征
    """
    with torch.no_grad():
        mov_z = model.get_first_stage_encoding(model.encode_first_stage(mov_img)).detach()
        fix_z = model.get_first_stage_encoding(model.encode_first_stage(fix_img)).detach()

        noise = torch.randn_like(mov_z)
        mov_noisy = model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
        fix_noisy = model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)

        mov_out = model.apply_model(mov_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
        fix_out = model.apply_model(fix_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

        mov_score0 = torch.cat((mov_out[1][0][0],  mov_out[1][0][2]),  dim=1)
        mov_score1 = torch.cat((mov_out[1][0][3],  mov_out[1][0][5]),  dim=1)
        mov_score2 = torch.cat((mov_out[1][0][6],  mov_out[1][0][8]),  dim=1)
        mov_score3 = torch.cat((mov_out[1][0][9],  mov_out[1][0][11]), dim=1)

        fix_score0 = torch.cat((fix_out[1][0][0],  fix_out[1][0][2]),  dim=1)
        fix_score1 = torch.cat((fix_out[1][0][3],  fix_out[1][0][5]),  dim=1)
        fix_score2 = torch.cat((fix_out[1][0][6],  fix_out[1][0][8]),  dim=1)
        fix_score3 = torch.cat((fix_out[1][0][9],  fix_out[1][0][11]), dim=1)

    return (mov_score0, mov_score1, mov_score2, mov_score3,
            fix_score0, fix_score1, fix_score2, fix_score3,
            mov_z, fix_z)


def visualize_feature_maps(mov_score0, mov_score1, mov_score2, mov_score3,
                           fix_score0, fix_score1, fix_score2, fix_score3,
                           mov_img, fix_img,
                           mov_z, fix_z,
                           pairname="", save_path=None):
    """
    可视化 LDM 特征图。

    每个尺度展示:
    - Moving 特征（取通道平均 + std）
    - Fixed 特征（取通道平均 + std）
    - |Moving - Fixed| 差异图（越大说明区分性越好）
    """
    mov_scores = [mov_score0, mov_score1, mov_score2, mov_score3]
    fix_scores = [fix_score0, fix_score1, fix_score2, fix_score3]
    scale_names = ["Scale 0 (H, W)", "Scale 1 (H/2, W/2)", "Scale 2 (H/4, W/4)", "Scale 3 (H/8, W/8)"]

    n_rows = 4  # 4 个尺度
    n_cols = 5  # mov_mean, mov_std, fix_mean, fix_std, |diff|_mean
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

    stats = []

    for row in range(n_rows):
        ms = mov_scores[row][0].cpu()   # [C, H, W]
        fs = fix_scores[row][0].cpu()
        C, H, W = ms.shape

        mov_mean = ms.mean(dim=0).numpy()    # [H, W]
        mov_std  = ms.std(dim=0).numpy()
        fix_mean = fs.mean(dim=0).numpy()
        fix_std  = fs.std(dim=0).numpy()
        diff     = np.abs(mov_mean - fix_mean)

        stats.append({
            "scale": scale_names[row],
            "mov_mean": mov_mean.mean(),
            "fix_mean": fix_mean.mean(),
            "diff_mean": diff.mean(),
            "diff_max": diff.max(),
            "n_channels": C,
        })

        imgs = [mov_mean, mov_std, fix_mean, fix_std, diff]
        titles = [
            f"Mov mean\n(shape {H}x{W})",
            f"Mov std\n({C}ch)",
            f"Fix mean",
            f"Fix std\n({C}ch)",
            f"|Mov - Fix| mean\n(diff={diff.mean():.4f})"
        ]

        for col in range(n_cols):
            ax = axes[row, col]
            img = imgs[col]
            vmax = max(abs(img.min()), abs(img.max()))
            ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(titles[col], fontsize=9)
            ax.axis('off')

    plt.suptitle(f"LDM Feature Maps — {pairname}", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()

    return stats


def visualize_latent_space(mov_z, fix_z, pairname="", save_path=None):
    """可视化 latent 空间的重建质量"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    mz = mov_z[0, 0].cpu().numpy()  # 取第一个通道
    fz = fix_z[0, 0].cpu().numpy()
    diff = np.abs(mz - fz)

    vmax = max(mz.max(), fz.max())
    axes[0].imshow(mz, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title(f"Moving latent (z)\nmean={mz.mean():.4f} std={mz.std():.4f}")
    axes[0].axis('off')

    axes[1].imshow(fz, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title(f"Fixed latent (z)\nmean={fz.mean():.4f} std={fz.std():.4f}")
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f"|Moving - Fixed| latent\nmean={diff.mean():.4f} max={diff.max():.4f}")
    axes[2].axis('off')

    plt.suptitle(f"Latent Space — {pairname}", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


def visualize_discriminativeness_over_samples(scores_list, pairnames, save_path=None):
    """
    汇总图: 多个样本的 moving/fixed 特征差异均值
    用于快速对比不同样本的区分性
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (scores, name) in enumerate(scores_list):
        mov_scores = scores[:4]
        fix_scores = scores[4:8]

        diffs = []
        for ms, fs in zip(mov_scores, fix_scores):
            diff = torch.abs(ms[0].mean(dim=0) - fs[0].mean(dim=0)).mean().item()
            diffs.append(diff)

        x = np.arange(4) + i * 0.8
        ax.bar(x, diffs, width=0.6, label=name[:30])

    ax.set_xticks(np.arange(4) + 1.2)
    ax.set_xticklabels(["Scale 0", "Scale 1", "Scale 2", "Scale 3"])
    ax.set_ylabel("|Moving - Fixed| mean")
    ax.set_title("Discriminativeness across samples (higher = more discriminative)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize LDM features for registration")
    parser.add_argument("--ldm_ckpt", type=str,
                        default="logs/2026-05-21T23-33-39_xcat_motion-ldm/checkpoints/last.ckpt",
                        help="LDM checkpoint path")
    parser.add_argument("--morph_ckpt", type=str, default=None,
                        help="LDMMorph checkpoint path (optional, not needed for feature extraction)")
    parser.add_argument("--ldm_config", type=str,
                        default="configs/latent-diffusion/xcat_motion-ldm.yaml",
                        help="LDM config yaml path")
    parser.add_argument("--xcat_path", type=str,
                        default="xcat_data",
                        help="XCAT data root")
    parser.add_argument("--output_dir", type=str,
                        default="logs/ldm_feature_visualization",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=6,
                        help="Number of samples to visualize")
    parser.add_argument("--t_enc", type=int, default=1,
                        help="Timestep for LDM feature extraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1. 加载 LDM
    ldm_model = load_ldm(args.ldm_config, args.ldm_ckpt)

    # 2. 加载数据集（取验证集，因为验证集没有运动增强）
    print(f"[Data] Loading XCAT dataset from {args.xcat_path}")
    val_dataset = Dataset_XCAT_Registration(
        data_root=args.xcat_path,
        split='val',
        motion_types=['identity'],
        flip_p=0.0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    print(f"[Data] {len(val_dataset)} validation samples")

    # 3. 逐样本提取并可视化
    all_stats = []
    all_scores = []

    for idx, (mov, fix, _, _, name) in enumerate(val_loader):
        if idx >= args.num_samples:
            break

        mov = mov.cuda().float()
        fix = fix.cuda().float()

        pairname = f"{idx:03d}_{name[0]}"
        print(f"\n[Processing] {pairname}")

        # 提取 LDM 特征
        (mov_s0, mov_s1, mov_s2, mov_s3,
         fix_s0, fix_s1, fix_s2, fix_s3,
         mov_z, fix_z) = extract_ldm_features(ldm_model, mov, fix, t_enc=args.t_enc)

        # 可视化每个尺度的特征图
        vis_path = os.path.join(args.output_dir, f"feature_maps_{pairname}.png")
        stats = visualize_feature_maps(
            mov_s0, mov_s1, mov_s2, mov_s3,
            fix_s0, fix_s1, fix_s2, fix_s3,
            mov, fix, mov_z, fix_z,
            pairname=pairname, save_path=vis_path
        )
        all_stats.append(stats)

        # 可视化 latent 空间
        lat_path = os.path.join(args.output_dir, f"latent_{pairname}.png")
        visualize_latent_space(mov_z, fix_z, pairname=pairname, save_path=lat_path)

        all_scores.append((mov_s0, mov_s1, mov_s2, mov_s3,
                           fix_s0, fix_s1, fix_s2, fix_s3))

    # 4. 汇总对比图
    print("\n[Summary] Generating discriminativeness comparison...")
    summary_path = os.path.join(args.output_dir, "discriminativeness_comparison.png")
    pairnames = [f"sample_{i}" for i in range(len(all_scores))]
    visualize_discriminativeness_over_samples(all_scores, pairnames, save_path=summary_path)

    # 5. 打印统计摘要
    print("\n" + "=" * 60)
    print("Discriminativeness Summary (|Moving - Fixed| mean per scale)")
    print("=" * 60)
    print(f"{'Sample':<15} {'Scale0':>10} {'Scale1':>10} {'Scale2':>10} {'Scale3':>10}")
    print("-" * 60)
    for i, stats in enumerate(all_stats):
        vals = [s["diff_mean"] for s in stats]
        print(f"sample_{i:<7} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")

    print(f"\n[Output] All visualizations saved to: {args.output_dir}/")
    print("\n=== How to interpret ===")
    print("1. Scale 0-3 show features at different resolutions (coarse to fine)")
    print("2. The '|Mov - Fix|' heatmap (last column) shows feature difference")
    print("3. HIGHER difference = more discriminative = better for registration")
    print("4. If difference is near zero, LDM features are NOT useful for that sample")
    print("5. Latent |Mov - Fix| shows reconstruction-level differences")


if __name__ == "__main__":
    main()
