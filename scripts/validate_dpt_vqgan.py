#!/usr/bin/env python3
"""
DPT-VQGAN 解耦验证脚本
==============================================================
验证三项指标：
    (a) 运动互换：G(a_fixed, m_j) 是否重建出第 j 相位（逐图对比 + 拼图）
    (b) 解剖稳定性：同被试不同相位 a 的差异（给数值）
    (c) 相位可分性：从 m_k 预测相位的准确率

用法：
    python scripts/validate_dpt_vqgan.py \
        --ckpt logs/xcat-dpt-vqgan/checkpoints/last.ckpt \
        --output viz_dpt_validation
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from ldm.models.dpt_vqgan import DPTVQGAN
from ldm.data.xcat_seq_grouped import (
    XCATSeqGroupedValidation, xcat_seq_grouped_collate_fn
)
from torch.utils.data import DataLoader


def load_model(ckpt_path):
    """加载训练好的 DPT-VQGAN checkpoint"""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    hparams = ckpt.get("hyper_parameters", {})

    # 用 checkpoint 中的 hparams 重建模型（确保 num_phases 等参数一致）
    model = DPTVQGAN(**hparams)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def calc_psnr(img1, img2, eps=1e-6):
    """计算 PSNR"""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse < eps:
        return 100.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def validate_decoupling(model, dataloader, output_dir, num_subjects=3):
    """
    执行三项验证。

    ⚠️ 假设：dataloader 每 batch 是一个被试的多张图（形状同训练时）
    ⚠️ 假设：phase_mlp 输出维度 = model.num_phases
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    num_phases = model.num_phases

    results = {
        "anat_stability": [],
        "phase_accuracy": [],
        "swap_psnr": [],
    }

    batch_iter = iter(dataloader)
    subject_count = 0

    while subject_count < num_subjects:
        try:
            batch = next(batch_iter)
        except StopIteration:
            break

        images = batch["images"].to(device)   # (B, num_frames, 1, H, W)
        phases = batch["phases"]              # (B, num_frames)
        B, num_frames = images.shape[:2]

        for b in range(B):
            subject_count += 1
            if subject_count > num_subjects:
                break

            subj_images = images[b]   # (num_frames, 1, H, W)
            subj_phases = phases[b].cpu().numpy()

            # ---- (a) 运动互换可视化 ----
            with torch.no_grad():
                # 整 batch 一次编码
                x_flat = subj_images.view(-1, 1, subj_images.shape[1], subj_images.shape[2])
                a_all_flat, _, _ = model.encode_anat(x_flat)   # (NF, C, H, W)
                m_all_flat = model.encode_motion(x_flat)       # (NF, m_dim)

                C, H, W = a_all_flat.shape[1:]
                a_all = a_all_flat.view(num_frames, C, H, W)
                m_all = m_all_flat.view(num_frames, model.m_dim)

                # 取帧0的解剖 a_fixed
                a_fixed = a_all[0:1]   # (1, C, H, W)

                # 对每个 j>0，做 G(a_fixed, m_j) vs I_j
                n_swap = num_frames - 1
                fig, axes = plt.subplots(n_swap, 3, figsize=(9, 3 * n_swap))
                psnr_list = []

                for j in range(1, num_frames):
                    m_j = m_all[j:j+1]   # (1, m_dim)
                    x_swap = model.decode(a_fixed, m_j)   # G(a_fixed, m_j)
                    x_true = subj_images[j:j+1]

                    # 原始帧0图像（参考）
                    x_ref = subj_images[0:1]

                    vmax = x_true.max().item()
                    psnr = calc_psnr(x_swap, x_true)

                    ax0 = axes[j-1, 0] if n_swap > 1 else axes[0]
                    ax1 = axes[j-1, 1] if n_swap > 1 else axes[1]
                    ax2 = axes[j-1, 2] if n_swap > 1 else axes[2]

                    ax0.imshow(x_ref[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
                    ax0.set_title(f"Ref: I_0 (phase {subj_phases[0]})")
                    ax0.axis('off')

                    ax1.imshow(x_swap[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
                    ax1.set_title(f"G(a_0, m_{j}) PSNR={psnr:.1f}dB")
                    ax1.axis('off')

                    ax2.imshow(x_true[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
                    ax2.set_title(f"Ground Truth: I_{j} (phase {subj_phases[j]})")
                    ax2.axis('off')

                    psnr_list.append(psnr)

                avg_psnr = np.mean(psnr_list)
                results["swap_psnr"].append(avg_psnr)
                plt.suptitle(f"Subject {subject_count} | Motion Swap | Avg PSNR={avg_psnr:.2f}dB", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/swap_s{subject_count:02d}.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  [Swap] Subject {subject_count}: avg PSNR = {avg_psnr:.2f} dB | per-frame: " +
                      " ".join([f"m{j}={p:.1f}" for j, p in enumerate(psnr_list, 1)]))

            # ---- (b) 解剖稳定性 ----
            with torch.no_grad():
                diffs = []
                for f in range(1, num_frames):
                    d = torch.abs(a_all[0] - a_all[f]).mean().item()
                    diffs.append(d)
                avg_diff = np.mean(diffs)
                results["anat_stability"].append(avg_diff)

            # 画 a 的差异热力图
            fig, axes = plt.subplots(1, num_frames - 1, figsize=(3 * (num_frames - 1), 3))
            if num_frames - 1 == 1:
                axes = [axes]
            for i, f in enumerate(range(1, num_frames)):
                diff_map = torch.abs(a_all[0] - a_all[f])[0].cpu().numpy()
                im = axes[i].imshow(diff_map, cmap='hot', vmin=0, vmax=diff_map.max() if diff_map.max() > 0 else 1)
                axes[i].set_title(f"a_0 vs a_{f}\nL1={diffs[i]:.4f}")
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            plt.suptitle(f"Subject {subject_count} | Anatomical Stability (a)", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/anat_s{subject_count:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [Anat] Subject {subject_count}: avg a L1 diff = {avg_diff:.4f}")

            # ---- (c) 相位可分性 ----
            with torch.no_grad():
                # encode_motion 返回 m_k（m_dim 维），不是 gamma/beta concat
                m_all_for_mlp = m_all_flat  # (NF, m_dim)
                phases_target = torch.from_numpy(subj_phases).long().to(device)
                logits = model.phase_mlp(m_all_for_mlp)   # (NF, num_phases)
                pred = logits.argmax(dim=1).cpu().numpy()
                acc = (pred == subj_phases).mean()
                results["phase_accuracy"].append(acc)

            # 画相位预测混淆矩阵
            fig, ax = plt.subplots(figsize=(6, 5))
            confusion = np.zeros((num_phases, num_phases), dtype=int)
            for true_p, pred_p in zip(subj_phases, pred):
                if true_p < num_phases and pred_p < num_phases:
                    confusion[true_p, pred_p] += 1
            im = ax.imshow(confusion, cmap='Blues')
            ax.set_xlabel("Predicted Phase")
            ax.set_ylabel("True Phase")
            ax.set_title(f"Phase Prediction Accuracy: {acc:.1%}")
            ticks = list(range(num_phases))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/phase_pred_s{subject_count:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [Phase] Subject {subject_count}: acc = {acc:.1%}")

            # 打印相位预测详情
            for f in range(num_frames):
                correct = "✓" if pred[f] == subj_phases[f] else "✗"
                print(f"       frame {f} (phase {subj_phases[f]}): pred={pred[f]} {correct}")

            print()

    # ---- 汇总统计 ----
    print("\n" + "=" * 60)
    print("=== 解耦验证汇总 ===")
    print("=" * 60)
    print(f"(a) 运动互换 - 平均 PSNR: {np.mean(results['swap_psnr']):.2f} dB")
    print(f"    (越高说明 a_fixed+m_j 重建第 j 相位越准确，解耦越成功）")
    print(f"(b) 解剖稳定性 - 平均 a L1 差异: {np.mean(results['anat_stability']):.4f}")
    print(f"    (越低说明同一被试不同相位共享相同解剖）")
    print(f"(c) 相位可分性 - 平均相位准确率: {np.mean(results['phase_accuracy']):.1%}")
    print(f"    (高准确率说明 m_k 成功编码了相位/运动信息）")

    np.save(f"{output_dir}/metrics.npy", {
        "swap_psnr": results["swap_psnr"],
        "anat_stability": results["anat_stability"],
        "phase_accuracy": results["phase_accuracy"],
    })
    print(f"\n数值结果已保存至 {output_dir}/metrics.npy")
    print(f"对比图已保存至 {output_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description="DPT-VQGAN 解耦验证")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--output", type=str, default="viz_dpt_validation",
                        help="输出目录")
    parser.add_argument("--num_subjects", type=int, default=3,
                        help="验证的被试数量")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt)
    model.to(device)

    dataset = XCATSeqGroupedValidation()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=xcat_seq_grouped_collate_fn,
    )

    print(f"\n验证集：{len(dataset)} 被试")
    print(f"每被试：{dataset.num_frames} 张图")
    print(f"相位数（num_phases）：{model.num_phases}")
    print(f"将验证 {args.num_subjects} 个被试\n")

    results = validate_decoupling(model, loader, args.output, args.num_subjects)


if __name__ == "__main__":
    main()
