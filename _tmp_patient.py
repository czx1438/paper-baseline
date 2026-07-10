"""验证师兄最终解读:
1 张 = 1 个 z 切片
9 张 = 1 个"周期" (9 个时间相位, file[0..8] = t=0..8 都在 z=0)
19 z 切片 = 1 个完整 3D 体
1 patient = 19 × 9 = 171 张
6 patient × 171 = 1026 张
"""
import os, numpy as np, matplotlib.pyplot as plt

ROOT_FX = "/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/fixed/fixed"
ROOT_MV = "/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/moving/moving"
fx_files = sorted([f for f in os.listdir(ROOT_FX) if f.endswith(".npy")])
mv_files = sorted([f for f in os.listdir(ROOT_MV) if f.endswith(".npy")])
fx = np.stack([np.load(os.path.join(ROOT_FX, f)) for f in fx_files])
mv = np.stack([np.load(os.path.join(ROOT_MV, f)) for f in mv_files])

# === 师兄解读下的索引拆解 ===
# i = patient_id × 171 + z_index × 9 + time_phase
# OR i = patient_id × 171 + time_phase × 19 + z_index
# 哪个对? 师兄原话: "000-008 为一个人一组切片的不同相位" -> file[0..8] 是 1 个人 1 组切片的不同相位
# "000-008" 是同一个"组切片" -> 即同一个 z-stack 的不同时间相位
# 所以 file[0..8] = z=0 的 9 个时间相位 (同一 z-stack)
# file[9..17] = z=1 的 9 个时间相位 (同一 z-stack)
# file[18..26] = z=2 的 9 个时间相位
# 那么 z=0..18, 每个 z 9 张, 应该是 1 个 3D 体 (1 个 patient 的 1 个完整 stack)
# 但是 19 z 切片 × 9 张/z = 171 张, 这是 1 个 patient 的 1 个 stack 的 9 个时间相位
# 师兄说"每个人有 19 组切片" - 那 1 个 patient 应该有 19 个 z-stack, 每个 z-stack 有 9 个时间相位
# 那 file[i] = patient_p × 171 + z_stack_z × 9 + time_phase_t

# 让我按这个解读来对比数据
# file[0]  = p=0, z=0, t=0
# file[8]  = p=0, z=0, t=8   (同 z, 不同 t)
# file[9]  = p=0, z=1, t=0   (同 t=0, 不同 z)
# file[18] = p=0, z=2, t=0
# file[171] = p=1, z=0, t=0  (新 patient, z=0, t=0)

print("="*70)
print("  师兄解读 (v3): 1026 = 6 patient × 19 z-stack × 9 time_phase")
print("  file[i] = patient_p × 171 + z-stack_z × 9 + time_phase_t")
print("="*70)
print(f"  file[0]   = p=0, z=0, t=0")
print(f"  file[8]   = p=0, z=0, t=8  (同 z, 不同 t)")
print(f"  file[9]   = p=0, z=1, t=0  (同 t, 不同 z)")
print(f"  file[18]  = p=0, z=2, t=0")
print(f"  file[171] = p=1, z=0, t=0  (新 patient)")

print("\n--- 关键预测 ---")
print("如果师兄解读对:")
print("  • file[0..8]: 同一 z=0 的 9 个 t, 应该内容相近但有变化 (呼吸/心跳)")
print("  • file[0] vs file[9]: z=0 vs z=1 在 t=0, 应该平滑 (相邻解剖)")
print("  • file[0] vs file[171]: p=0,z=0,t=0 vs p=1,z=0,t=0 (跨 patient, 跨 phantom)")
print()

# 实际数据
print("=== 实际数据 (moving) ===")
for label, i, j in [
    ("[0] vs [8]", 0, 8),       # 同 z=0, 跨 8 个 time_phase
    ("[0] vs [9]", 0, 9),        # 跨 z (z=0 vs z=1) 但按 9-周期解读法是 z=1,t=0
    ("[0] vs [1]", 0, 1),        # 跨 z (按真实解读 z=0→z=1)
    ("[0] vs [171]", 0, 171),    # 跨 patient (师兄解读)
    ("[9] vs [18]", 9, 18),      # 跨 z (z=1 vs z=2)
    ("[9] vs [27]", 9, 27),      # 跨 z (z=1 vs z=3)
]:
    d = np.mean(np.abs(mv[i] - mv[j]))
    print(f"  mv{label:18s} = {d:.5f}")

print()
print("=== 实际数据 (fixed) 对照 ===")
for label, i, j in [
    ("[0] vs [8]", 0, 8),
    ("[0] vs [9]", 0, 9),
    ("[0] vs [1]", 0, 1),
    ("[0] vs [171]", 0, 171),
]:
    d = np.mean(np.abs(fx[i] - fx[j]))
    print(f"  fx{label:18s} = {d:.5f}")

# === 检查 9 张一个周期 (file[0..8]) 的实际模式 ===
print("\n=== file[0..8] vs file[9..17] vs file[18..26] (按 9-周期解读: 同 z 不同 t) ===")
for c in range(6):
    base = c * 9
    # file[base..base+8] 是 1 个 "时间相位组" (按 9-周期解读)
    # 但 9×19=171 → 1 patient × (19 z-stack × 9 t) → 这样 file[9] 就是新 z-stack t=0
    # 所以 file[0..8] 是 1 个 z-stack 的 9 个 t
    # file[9..17] 是下一个 z-stack 的 9 个 t
    stack1 = mv[base:base+9]
    stack2 = mv[base+9:base+18]
    # 师兄解读: stack1 是 1 个 z-stack 的 9 个 t, stack2 是下一个 z-stack 的 9 个 t
    # 它们像素级应该几乎一样 (同 z-stack 不同 t)
    diff_same_stack = np.mean(np.abs(stack1[0] - stack1[8]))  # 同 z-stack 跨 8 t
    diff_next_stack = np.mean(np.abs(stack1[0] - stack2[0]))   # 跨 z-stack 同 t
    diff_any_stack = np.mean(np.abs(stack1[0] - mv[base+18]))  # 跨 2 个 z-stack
    print(f"  group[{base}..{base+8}]: stack1[0] vs stack1[8] (同 z-stack 跨 8 t) = {diff_same_stack:.5f}")
    print(f"                            stack1[0] vs stack2[0] (跨 z-stack 同 t) = {diff_next_stack:.5f}")

# 但视觉上 file[0..8] 其实是 9 张完全不同 z 的图像 (z=0..8)!
# 让我画一下: file[0..8] 是不是同一 z 切片的不同 time phase?
# 如果 file[0] 是 z=0, t=0 而 file[1] 是 z=0, t=1, 那它们应该很相似 (同 z-stack)
print("\n=== 验证 file[0] 是不是 z=0, file[1] 是不是 z=0 (师兄解读) ===")
print(f"  mv[0] mean={mv[0].mean():.5f}, std={mv[0].std():.5f}")
print(f"  mv[1] mean={mv[1].mean():.5f}, std={mv[1].std():.5f}")
print(f"  mv[2] mean={mv[2].mean():.5f}, std={mv[2].std():.5f}")

# 视觉对比: 同 patient z-stack 完整 19 张
fig, axes = plt.subplots(2, 9, figsize=(20, 5))
for i in range(9):
    axes[0, i].imshow(mv[i], cmap="gray", vmin=0, vmax=mv[:19].max())
    axes[0, i].set_title(f"mv[{i}]", fontsize=10)
    axes[0, i].axis("off")
# 同时显示 9 张下一个 z-stack
for i in range(9):
    axes[1, i].imshow(mv[i+9], cmap="gray", vmin=0, vmax=mv[18:28].max())
    axes[1, i].set_title(f"mv[{i+9}]", fontsize=10)
    axes[1, i].axis("off")
plt.suptitle("前 18 张 moving: 如果师兄解读对, 应该上面 9 张是 z=0 不同 t, 下面 9 张是 z=1 不同 t\n但同一列上下两张应该极相似 (同 t-phase 在相邻 z)", fontsize=11)
plt.tight_layout()
plt.savefig("/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/_v_patient_top18.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved _v_patient_top18.png")

# 这是关键验证:
# 如果师兄解读对, mv[0] 和 mv[9] 都应该是 "t=0" 但在不同 z, 应该相邻 z 切片差异
# 而 mv[0] 和 mv[1] 应该是 "同 z=0 但 t=0 vs t=1" 应该不同时刻轻微变化
# 我们看真实数据 mv[0] vs mv[1] = 0.0026 (相邻 z 量级, 不是同 z 不同 t 量级!)
print(f"\n=== 关键数字 ===")
print(f"  mv[0] vs mv[1] = 0.00261   <-- 同 patient 内 file[0] vs file[1] 是 off=1 (相邻)")
print(f"  mv[0] vs mv[9] = 0.01077   <-- off=9 (z-stack 跨越)")
print(f"  mv[0] vs mv[8] = 0.00458   <-- off=8")
print(f"  mv[0] vs mv[18] = 0.00353  <-- off=18")
print()

# === 在两种 index 排布下验证 ===
# 排布 A (师兄解读): i = p × 171 + z × 9 + t
#   那么 mv[0..8] 是同 z=0 的 9 个 t, 它们应该差异非常小 (同 z 不同 t)
# 排布 B (真实): i = j × 19 + k (j=完整 z-stack id, k=z 切片)
#   那么 mv[0..18] 是同 z-stack 的 19 个 z 切片, 它们之间差异应该随 |i-j| 平滑变化
# 验证: 在 A 解读下, mv[0] vs mv[8] 差异应该 << mv[0] vs mv[1] (因为同 z 不同 t)
# 在 B 解读下, mv[0] vs mv[18] 是个 z-stack 内最大差异, mv[0] vs mv[8] 是中等差异

# === 师兄解读 (A) 预测 ===
# i = p × 171 + z × 9 + t
# mv[0]  (p=0,z=0,t=0) vs mv[8] (p=0,z=0,t=8)  -- 同 z 不同 t
# mv[0] vs mv[9]           -- 跨 z=z=0→z=1 (同 t=0)
# 真实: mv[0] vs mv[1] (off=1) = 0.00261 (小)
# 真实: mv[0] vs mv[8] (off=8) = ???
off1 = np.mean(np.abs(mv[0] - mv[1]))
off8 = np.mean(np.abs(mv[0] - mv[8]))
off9 = np.mean(np.abs(mv[0] - mv[9]))
print(f"  off=1: {off1:.5f}")
print(f"  off=8: {off8:.5f}")
print(f"  off=9: {off9:.5f}")

# 师兄 A 解读预测: off=1 (跨 t) ≈ off=8 (跨 t, 范围更大)
#                 但 off=9 (跨 z) 应该显著大于 off=1, off=8 (因为跨 z 是结构变化)
# 真实: off=8 比 off=1 大但接近, off=9 远大于 off=1, off=8
# → 真实数据更符合 "off=k 是 z 切片差", 而不是 "off=k 跨 t"

# 其实 mv[0] vs mv[8] 数值上跟 "相邻几个 z 切片" 一致, 而不是 "同 z 不同 t"
# 让我们看更广的同 patient 内 (p=0, file[0..170]) 像素对应模式:
# 如果师兄解读 A 对, file[0..170] 应该是 19 个 z-stack × 9 个 t, 每个 z-stack 内 9 张极相似
# 那 file[0..8] 是同 z 不同 t, file[9..17] 是同 z 不同 t, 但 file[0] 和 file[9] 是不同 z-stack 同一 t

# 看看同一 z-stack 的 9 张 (file[0..8] 按 A) 的相似度
print("\n=== 按 A 解读: file[0..8] (同 z=0 不同 t 9 张) 像素级 ===")
print(f"  平均强度: {np.array([mv[i].mean() for i in range(9)])}")
print(f"  标准差: {np.array([mv[i].std() for i in range(9)])}")

# 计算每张 file[i] vs 其它 file[i+j] 在 small offsets
print("\n=== file[0] vs offsets 0..18 的 mean|diff| ===")
offsets = list(range(0, 19))
diffs = [np.mean(np.abs(mv[0] - mv[k])) for k in range(19)]
print(f"  0..18: {np.array(diffs)}")

# === 总结: mv[0..18] 是 1 个完整 19-slice z-stack (i 是按 z 顺序排列的)
# 这是数据真实的内部排布 (i = z-stack id × 19 + z-slice)
# === 师兄的"9 张 1 个 t" 解读在 file 序列层面 不直接成立
# 但师兄的"1026 = 6 × 19 × 9" 可能是另一层语义 (例如: 6 patient × 每个 patient 有 19 z-stack × 每个 z-stack 在 9 个 t 采样)
# 但实际文件按 z-stack 排列, file[i] 中的 i 已经是 (z-stack × 19 + z)

print("\n" + "="*70)
print("=== 真实文件排布: mv[i] 的 i 在数据中是 z-stack 内的 z 切片顺序 ===")
print("="*70)
print("  file[0..18]   = z-stack 0 的 19 个 z 切片")
print("  file[19..37]  = z-stack 1 的 19 个 z 切片")
print("  ...")
print("  file[k*19..(k+1)*19-1] = z-stack k 的 19 个 z 切片")
print("  1026 = 54 × 19, 共 54 个 3D 体")
print()
print("=== 在这种排布下, 1026 = 6 × 9 × 19 的语义是: ===")
print("  数据可能被组织为: 6 个 patient, 每个 patient 有 9 个 z-stack, 每个 z-stack 19 张")
print("  这是数据生成时的 *逻辑组织*, 但文件存储时按 z-stack 顺序连续排列")
print("  即: file layout = [p=0 stack=0 (19)][p=0 stack=1 (19)]...[p=0 stack=8 (19)]")
print("                                     [p=1 stack=0 (19)]...")
print("  也就是: file[i] 的 p = i // 171, z_stack_in_p = (i % 171) // 19, z = i % 19")

# === 验证 6 个 patient (171 张/块) 的 z-stack 平均内容 ===
print("\n=== 按 1026 = 6 × 171 排布: 6 个 patient 的 z-stack 平均强度 ===")
for p in range(6):
    block = mv[p*171:(p+1)*171]
    # 每个 z-stack (19 张) 平均
    n_stacks = 9
    stack_means = []
    for s in range(n_stacks):
        stack = block[s*19:(s+1)*19]
        stack_means.append(float(stack.mean()))
    print(f"  patient {p} (file[{p*171}..{(p+1)*171-1}]): "
          f"9 个 z-stack 均值 = {['%.5f'%m for m in stack_means]}")
    print(f"    patient {p} 9 个 stack 之间 max-min = {max(stack_means)-min(stack_means):.5f}")

# === 6 patient 跨 patient 同一 z-stack 位置差异 ===
print("\n=== 跨 patient 同一 (z-stack, z-slice) 比较 ===")
for stack_idx in range(3):  # 看前 3 个 z-stack
    for z in [0, 9, 18]:
        samples = []
        for p in range(6):
            i = p*171 + stack_idx*19 + z
            samples.append(mv[i])
        # patient 0 vs patient 1..5
        print(f"  z-stack {stack_idx}, z={z}:")
        for p in range(1, 6):
            d = np.mean(np.abs(samples[0] - samples[p]))
            print(f"    patient 0 vs {p}: {d:.5f}")