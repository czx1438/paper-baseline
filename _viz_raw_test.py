"""
可视化测试集全部 153 个样本的原始数据 (raw fixed + raw moving)。

测试集 = base_idx 97..113 = block 5, slice 0..18, 每个 slice 9 phases (1..9)
  fixed_idx = 855 + slice
  moving_idx = 855 + slice + (phase-1)*19

每个样本输出 1 张图: 2 列 (raw fixed | raw moving)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

SAVE_DIR = './logs/raw_test_visualization'
os.makedirs(SAVE_DIR, exist_ok=True)

FIXED_DIR = 'xcat_data/fixed/fixed'
MOVING_DIR = 'xcat_data/moving/moving'

# 测试集 base_idx 97..113:
#   base_idx 97  -> block=5, slice=2, fixed_idx=857
#   base_idx 98  -> block=5, slice=3, fixed_idx=858
#   ...
#   base_idx 113 -> block=5, slice=18, fixed_idx=873
# 共 17 slices × 9 phases = 153 样本
# (slice 0,1 不在测试集)

samples = []
for base_idx in range(97, 114):      # 97..113
    b = 5
    s = base_idx - 97 + 2            # slice 2..18
    fixed_idx = 5 * 171 + s          # = 855 + s = 857..873
    for p in range(1, 10):           # phase 1..9
        phase_id  = p - 1            # 0..8
        moving_idx = 5 * 171 + phase_id * 19 + s
        pairname = f"block{b}_slice{s:02d}_phase{p:02d}"
        samples.append({
            'pairname':   pairname,
            'base_idx':   base_idx,
            'block':      b,
            'slice':      s,
            'phase':      p,
            'phase_id':   phase_id,
            'fixed_idx':  fixed_idx,
            'moving_idx': moving_idx,
        })

print(f"Total test samples: {len(samples)}")

# 所有图像共一个 vmin/vmax (全局)
all_fixed_vals  = []
all_moving_vals = []
for s in samples:
    fi = np.load(os.path.join(FIXED_DIR,  f"{s['fixed_idx']:03d}.npy"))
    mi = np.load(os.path.join(MOVING_DIR, f"{s['moving_idx']:03d}.npy"))
    all_fixed_vals.append(fi)
    all_moving_vals.append(mi)

# 用全局 min/max 归一化显示
vmin_f = min(a.min() for a in all_fixed_vals)
vmax_f = max(a.max() for a in all_fixed_vals)
vmin_m = min(a.min() for a in all_moving_vals)
vmax_m = max(a.max() for a in all_moving_vals)
print(f"Fixed  global range: [{vmin_f:.4f}, {vmax_f:.4f}]")
print(f"Moving global range: [{vmin_m:.4f}, {vmax_m:.4f}]")

# 用于 0..1 归一化版本显示
all_vals = all_fixed_vals + all_moving_vals
global_min = min(a.min() for a in all_vals)
global_max = max(a.max() for a in all_vals)
print(f"Combined global range: [{global_min:.4f}, {global_max:.4f}]")


def norm01(arr):
    """Min-max 归一化到 [0,1]"""
    if global_max - global_min < 1e-6:
        return np.zeros_like(arr)
    return (arr - global_min) / (global_max - global_min)


# 按 slice 分组画总览
print("\n=== Drawing per-sample figures ===")
for k, s in enumerate(samples):
    fi = all_fixed_vals[k]
    mi = all_moving_vals[k]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Raw fixed
    ax = axes[0]
    im = ax.imshow(norm01(fi), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"RAW Fixed  |  {s['pairname']}\n"
                 f"fixed_idx={s['fixed_idx']:4d}  slice={s['slice']:2d}  "
                 f"block={s['block']}  phase=00(=fixed)\n"
                 f"mean={fi.mean():.4f}  min={fi.min():.4f}  max={fi.max():.4f}",
                 fontsize=10)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Raw moving
    ax = axes[1]
    im = ax.imshow(norm01(mi), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"RAW Moving  |  {s['pairname']}\n"
                 f"moving_idx={s['moving_idx']:4d}  slice={s['slice']:2d}  "
                 f"block={s['block']}  phase={s['phase']:02d}\n"
                 f"mean={mi.mean():.4f}  min={mi.min():.4f}  max={mi.max():.4f}",
                 fontsize=10)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"[{k+1}/153] Raw Data: {s['pairname']}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(SAVE_DIR, f"sample_{k:03d}_{s['pairname']}.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    if (k + 1) % 17 == 0 or k == len(samples) - 1:
        print(f"  [{k+1}/{len(samples)}] saved {out_path}")

print(f"\n✅ All 153 samples saved to {SAVE_DIR}/")

# ---- 额外: 每个 slice 的 9 phase 总览 (slices 2..18, 共 17 张) ----
print("\n=== Drawing per-slice overview (9 phases per slice) ===")
SLICE_SAVE = os.path.join(SAVE_DIR, 'per_slice_overview')
os.makedirs(SLICE_SAVE, exist_ok=True)

for s in range(2, 19):       # slice 2..18 (17 slices)
    slice_samples = [sp for sp in samples if sp['slice'] == s]
    n_phases = len(slice_samples)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"block5_slice{s:02d} — All 9 Phases Overview\n"
                 f"(fixed_idx={855+s}, moving_idx spans {855+s}..{855+s+152})",
                 fontsize=13, fontweight='bold', y=0.98)

    for idx, sp in enumerate(slice_samples):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        # 找 raw data
        fi = all_fixed_vals[samples.index(sp)]
        mi = all_moving_vals[samples.index(sp)]

        # 上半是 fixed，下半是 moving
        combined = np.vstack([norm01(fi), norm01(mi)])
        im = ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"phase {sp['phase']:02d}\nfi={sp['fixed_idx']} mi={sp['moving_idx']}\n"
                     f"fiμ={fi.mean():.3f} miμ={mi.mean():.3f}", fontsize=8)
        ax.axis('off')

    # 隐藏多余的 axes (如果有)
    for idx in range(n_phases, 9):
        row, col = divmod(idx, 3)
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(SLICE_SAVE, f"slice_{s:02d}_overview.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Slice {s:2d}: {out_path}")

print(f"\n✅ Slice overviews saved to {SLICE_SAVE}/")
