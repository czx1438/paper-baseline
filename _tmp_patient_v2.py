"""完整验证 师兄解读: 1026 = 6 patient × 19 z-stack × 9 time_phase"""
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT_FX = "/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/fixed/fixed"
ROOT_MV = "/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/moving/moving"
fx_files = sorted([f for f in os.listdir(ROOT_FX) if f.endswith(".npy")])
mv_files = sorted([f for f in os.listdir(ROOT_MV) if f.endswith(".npy")])
fx = np.stack([np.load(os.path.join(ROOT_FX, f)) for f in fx_files])
mv = np.stack([np.load(os.path.join(ROOT_MV, f)) for f in mv_files])

OUT = "/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main"
print(f"moving shape: {mv.shape}, fixed shape: {fx.shape}")

# ============================================================
# 图 1: file[0..18] 完整 z-stack (19 张)
# ============================================================
fig, axes = plt.subplots(3, 7, figsize=(18, 8))
for i in range(19):
    ax = axes[i // 7, i % 7]
    ax.imshow(mv[i], cmap="gray", vmin=0, vmax=mv.max())
    ax.set_title(f"mv[{i}] z={i}", fontsize=10)
    ax.axis("off")
# 隐藏多余 axes
for j in range(19, 21):
    axes[j // 7, j % 7].axis("off")
plt.suptitle("图1: moving[0..18] 完整 19-slice z-stack\n这是数据中 file[0..18] 的真实样子", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_01_zstack_0_18.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_01_zstack_0_18.png")

# ============================================================
# 图 2: file[19..37] vs file[0..18] 比较 (验证同 z-stack 还是跨 z-stack)
# ============================================================
fig, axes = plt.subplots(4, 10, figsize=(22, 9))
# 上 2 行: mv[0..18]
for i in range(19):
    ax = axes[i // 10, i % 10]
    ax.imshow(mv[i], cmap="gray", vmin=0, vmax=mv.max())
    ax.set_title(f"mv[{i}]", fontsize=9)
    ax.axis("off")
# 下 2 行: mv[19..37]
for i in range(19):
    ax = axes[2 + i // 10, i % 10]
    ax.imshow(mv[i+19], cmap="gray", vmin=0, vmax=mv.max())
    ax.set_title(f"mv[{i+19}]", fontsize=9)
    ax.axis("off")
# 隐藏余下
for j in range(19, 20):
    axes[j // 10, j % 10].axis("off")
for j in range(19, 20):
    axes[2 + j // 10, j % 10].axis("off")
plt.suptitle("图2: 上=file[0..18] (z-stack 0), 下=file[19..37] (z-stack 1)\n对比两个相邻 19-slice 3D 体", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_02_two_zstacks.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_02_two_zstacks.png")

# ============================================================
# 图 3: 师兄解读: file[0..8] 应该是同 z 不同 t (如果对, 应该 9 张很相似)
# 实际: file[0..8] 是 z=0..8 跨越整个 3D 体
# ============================================================
fig, axes = plt.subplots(2, 9, figsize=(20, 5))
for i in range(9):
    axes[0, i].imshow(mv[i], cmap="gray", vmin=0, vmax=mv[:19].max())
    axes[0, i].set_title(f"mv[{i}]", fontsize=11)
    axes[0, i].axis("off")
    axes[1, i].imshow(mv[i+9], cmap="gray", vmin=0, vmax=mv[:19].max())
    axes[1, i].set_title(f"mv[{i+9}]", fontsize=11)
    axes[1, i].axis("off")
plt.suptitle("图3: 师兄解读预测 file[0..8] 是同 z=0 不同 t (应相似), file[9..17] 是 z=1 不同 t\n实际看图说话: file[0..8] 跨越 z=0..8 (变化大)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_03_first18_alternative.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_03_first18_alternative.png")

# ============================================================
# 图 4: 6 个 patient × 19 z-stack × 9 time_phase 排布 (按 1026 = 6 × 171)
# file[i] = p × 171 + stack × 19 + z
# ============================================================
fig, axes = plt.subplots(6, 9, figsize=(24, 14))
vmax = mv[:1026].max()
for p in range(6):
    for s in range(9):
        # patient p, z-stack s 的 z=0
        idx = p * 171 + s * 19
        if idx < 1026:
            axes[p, s].imshow(mv[idx], cmap="gray", vmin=0, vmax=vmax)
            axes[p, s].set_title(f"p{p},s{s}\nf[{idx}]", fontsize=8)
            axes[p, s].axis("off")
plt.suptitle("图4: 按 1026 = 6 patient × 9 z-stack 排布\n每个 cell = (patient p, z-stack s) 的 z=0 切片\n排布: file[i] = p × 171 + s × 19 + z", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_04_6patient_9stack.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_04_6patient_9stack.png")

# ============================================================
# 图 5: 同一个 patient (p=0) 的 9 个 z-stack 各取 z=0..18 (9 行 × 19 列)
# ============================================================
fig, axes = plt.subplots(9, 19, figsize=(38, 18))
vmax_p0 = mv[:171].max()
for s in range(9):
    for z in range(19):
        idx = s * 19 + z
        axes[s, z].imshow(mv[idx], cmap="gray", vmin=0, vmax=vmax_p0)
        axes[s, z].set_title(f"s{s},z{z}\nf[{idx}]", fontsize=6)
        axes[s, z].axis("off")
plt.suptitle("图5: Patient 0 (file[0..170]) 的 9 z-stack × 19 z-slice\n1 行 = 1 个 z-stack 的完整 19-slice, 共 9 行 = patient 0 的 9 个 z-stack\n如果师兄解读对, 9 行应该有微差异 (同 phantom 不同 time)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_05_patient0_full.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_05_patient0_full.png")

# ============================================================
# 图 6: 6 patient 各自的 z=0 (取 s=0, z=0), 看跨 patient 是否真不同 phantom
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for p in range(6):
    ax = axes[p // 3, p % 3]
    idx = p * 171 + 0 * 19 + 0  # s=0, z=0
    ax.imshow(mv[idx], cmap="gray", vmin=0, vmax=mv.max())
    ax.set_title(f"patient {p} (file[{idx}])", fontsize=12)
    ax.axis("off")
plt.suptitle("图6: 6 个 patient 的 z=0 (file[p×171])\n如果真不同 phantom, 应该 6 张视觉差异明显", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_06_six_patients.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_06_six_patients.png")

# ============================================================
# 图 7: 同一 patient 内 9 个 z-stack 的 z=0..8 (前 9 个 z), 看 stack 间差异
# ============================================================
fig, axes = plt.subplots(9, 9, figsize=(18, 18))
for s in range(9):
    for z in range(9):
        idx = s * 19 + z
        axes[s, z].imshow(mv[idx], cmap="gray", vmin=0, vmax=mv[:171].max())
        axes[s, z].set_title(f"s{s},z{z}\nf[{idx}]", fontsize=7)
        axes[s, z].axis("off")
plt.suptitle("图7: Patient 0 内 9 个 z-stack 的前 9 个 z 切片\n1 行 = 1 z-stack, 1 列 = 1 z 切片\n视觉上看: 行与行之间是否相似 (同 patient)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_07_patient0_9x9.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_07_patient0_9x9.png")

# ============================================================
# 图 8: 数值验证 heatmap - 6 patient 的 z=0 矩阵 (9 stack × 6 patient)
# ============================================================
# 取 6 patient × 9 z-stack, 每个的 z=0
fig, ax = plt.subplots(figsize=(14, 7))
vmax = mv[:1026].max()
for p in range(6):
    for s in range(9):
        idx = p * 171 + s * 19
        if idx < 1026:
            img = mv[idx]
            # 把每个图缩小并拼成网格
            pass

# 用 imshow 直接画一个大的 6×9 grid
fig, axes = plt.subplots(6, 9, figsize=(28, 18))
vmax = mv[:1026].max()
for p in range(6):
    for s in range(9):
        idx = p * 171 + s * 19
        axes[p, s].imshow(mv[idx], cmap="gray", vmin=0, vmax=vmax)
        axes[p, s].set_title(f"p{p},s{s},z=0", fontsize=10)
        axes[p, s].axis("off")
plt.suptitle("图8: 6 patient × 9 z-stack 的 z=0 完整网格 (54 张)\n肉眼对比: 同一 patient 的 9 行是否相似 (同 phantom), 跨 patient 是否不同 (不同 phantom)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/viz_08_54_z0_grid.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved viz_08_54_z0_grid.png")

# ============================================================
# 数值差异汇总
# ============================================================
print("\n" + "="*70)
print("数值差异汇总")
print("="*70)
print(f"{'对比':<35} | {'mean|diff|':>10}")
print("-"*50)

# patient 内 (同 phantom 不同 stack 同一 z)
print("\n[Patient 内] 9 个 z-stack 的 z=0 差异:")
s0_z0 = [mv[s * 19 + 0] for s in range(9)]
for i in range(9):
    for j in range(i+1, 9):
        d = np.mean(np.abs(s0_z0[i] - s0_z0[j]))
        print(f"  stack {i} vs stack {j} (patient 0): {d:.5f}")

# patient 间 (跨 phantom 同一 z)
print("\n[Patient 间] s=0, z=0 跨 patient:")
s0_z0_cross = [mv[p * 171 + 0] for p in range(6)]
for i in range(6):
    for j in range(i+1, 6):
        d = np.mean(np.abs(s0_z0_cross[i] - s0_z0_cross[j]))
        print(f"  patient {i} vs {j} (s=0,z=0): {d:.5f}")

print("\n=== 关键预测 vs 实际 ===")
print("师兄解读预测:")
print("  - patient 0 内的 9 个 z-stack 极相似 (同 phantom, 同 patient, 不同 t)")
print("  - patient 0 vs patient 1..5 有显著差异 (跨 phantom)")
print()
print("实测:")
in_p0 = [np.mean(np.abs(s0_z0[i] - s0_z0[j])) for i in range(9) for j in range(i+1, 9)]
in_p0_mean = np.mean(in_p0)
cross_p = [np.mean(np.abs(s0_z0_cross[i] - s0_z0_cross[j])) for i in range(6) for j in range(i+1, 6)]
cross_p_mean = np.mean(cross_p)
print(f"  patient 0 内 9 stack 差异均值: {in_p0_mean:.5f}")
print(f"  6 patient 跨 patient 差异均值: {cross_p_mean:.5f}")
print(f"  比例 (跨/内): {cross_p_mean/in_p0_mean:.2f}x")
print()
if cross_p_mean > in_p0_mean * 5:
    print("  → 师兄解读对! 跨 patient 显著大于 patient 内")
else:
    print(f"  → 师兄解读不成立: 跨/内比例仅 {cross_p_mean/in_p0_mean:.2f}x")
    print(f"    (一般\"不同 phantom\"应该有 10x+ 比例, 数据更像\"同 phantom 漂移\")")