"""
训练集 block 1: slice 0, 1, 2 各 9 个相位的可视化
block 1 = indices 171..341
slice s -> fixed/{171+s:03d}, moving phase p -> {171+s+(p-1)*19:03d}
"""
import os, gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_ROOT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main'
FIXED_DIR  = os.path.join(DATA_ROOT, 'xcat_data/fixed/fixed')
MOVING_DIR = os.path.join(DATA_ROOT, 'xcat_data/moving/moving')
OUT_DIR    = os.path.join(DATA_ROOT, 'logs/raw_test_visualization/train_block1_s0_s1_s2')
os.makedirs(OUT_DIR, exist_ok=True)

SLICES = [0, 1, 2]
BLOCK_BASE = 171  # block 1 fixed starts at index 171

def ncc_local(a, b, win=15, stride=8):
    """局部窗口 NCC (下采样版)"""
    a_ds = a[::stride, ::stride].astype(np.float64)
    b_ds = b[::stride, ::stride].astype(np.float64)
    a_ds = (a_ds - a_ds.mean()) / (a_ds.std() + 1e-8)
    b_ds = (b_ds - b_ds.mean()) / (b_ds.std() + 1e-8)
    return float((a_ds * b_ds).mean())

def ncc_diff_stats(a, b):
    """像素级差分统计"""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return d.max(), d.mean(), (a - b).std()

all_nccs = {}
all_stats = {}

# ---- Per-slice large grid ----
for s in SLICES:
    gc.collect()
    fi = BLOCK_BASE + s
    f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))
    ms, mis, ds, nccs, stats = [], [], [], [], []
    for p in range(1, 10):
        mi = BLOCK_BASE + s + (p - 1) * 19
        m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
        ms.append(m)
        mis.append(mi)
        ds.append(np.abs(f.astype(np.float32) - m.astype(np.float32)))
        nccs.append(ncc_local(f, m))
        stats.append(ncc_diff_stats(f, m))
    all_nccs[s] = nccs
    all_stats[s] = stats

    gmin = min(f.min(), min(m.min() for m in ms))
    gmax = max(f.max(), max(m.max() for m in ms))

    fig, axes = plt.subplots(9, 3, figsize=(11, 35))
    fig.suptitle(
        f'Training Set | Block 1 | Slice {s} | Fixed: fixed/{fi:03d}.npy\n'
        f'Moving: phase1=/{mis[0]:03d} .. phase9=/{mis[-1]:03d}',
        fontsize=10, fontweight='bold', y=0.995
    )

    for row in range(9):
        axes[row, 0].imshow(f, cmap='gray', vmin=gmin, vmax=gmax)
        tag = ' (=Moving, same phase)' if row == 0 else ''
        axes[row, 0].set_title(f'Fixed\n/{fi:03d}{tag}', fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(ms[row], cmap='gray', vmin=gmin, vmax=gmax)
        axes[row, 1].set_title(f'Moving Phase {row+1:02d}\n/{mis[row]:03d}', fontsize=8)
        axes[row, 1].axis('off')

        im = axes[row, 2].imshow(ds[row], cmap='hot', vmin=0, vmax=ds[row].max())
        dmax, dmean, dstd = stats[row]
        axes[row, 2].set_title(
            f'|F-M|  NCC={nccs[row]:.3f}\nmax={dmax:.3f}  std={dstd:.3f}',
            fontsize=8
        )
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = f'{OUT_DIR}/train_block1_slice{s:02d}_phases.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    del f, ms, ds
    gc.collect()
    print(f'Slice {s}: NCCs={[round(n,3) for n in nccs]}')

# ---- Side-by-side comparison: 3 slices x 9 phases fixed | moving | diff ----
print("\nDrawing 3-slice combined grid...")
fig, axes = plt.subplots(9, 3 * len(SLICES), figsize=(6 * len(SLICES), 42))

# We need to reload all data for the combined view
combined_data = {}
for s in SLICES:
    gc.collect()
    fi = BLOCK_BASE + s
    f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))
    ms, mis, ds = [], [], []
    for p in range(1, 10):
        mi = BLOCK_BASE + s + (p - 1) * 19
        m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
        ms.append(m)
        mis.append(mi)
        ds.append(np.abs(f.astype(np.float32) - m.astype(np.float32)))
    combined_data[s] = (fi, f, ms, mis, ds)
    del m; gc.collect()

for col_i, s in enumerate(SLICES):
    fi, f, ms, mis, ds = combined_data[s]
    for row in range(9):
        ax = axes[row, col_i * 3 + 0]
        ax.imshow(f, cmap='gray', vmin=0, vmax=0.3)
        ax.set_title(f'Fixed /{fi:03d}\nblock=1 slice={s}', fontsize=7)
        ax.axis('off')

        ax = axes[row, col_i * 3 + 1]
        ax.imshow(ms[row], cmap='gray', vmin=0, vmax=0.3)
        ax.set_title(f'Moving /{mis[row]:03d}\nphase={row+1}', fontsize=7)
        ax.axis('off')

        ax = axes[row, col_i * 3 + 2]
        im = ax.imshow(ds[row], cmap='hot', vmin=0, vmax=ds[row].max())
        ax.set_title(f'|F-M|  NCC={all_nccs[s][row]:.3f}', fontsize=7)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0, col_i * 3 + 0].annotate(
        f'Block 1 | Slice {s}\nFixed /{fi:03d}',
        xy=(0.5, 1.05), xycoords='axes fraction',
        ha='center', va='bottom', fontsize=9, fontweight='bold'
    )

fig.suptitle(
    f'Training Set: Block 1 | Slices {SLICES}\n'
    f'9 Phases | Each row = Phase 1..9 | Cols: Fixed | Moving | Diff',
    fontsize=11, fontweight='bold', y=0.998
)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{OUT_DIR}/train_block1_3slices_9phases.png', dpi=100, bbox_inches='tight')
plt.close(fig); gc.collect()
for s in SLICES:
    del combined_data[s]
gc.collect()
print(f"  Saved: {OUT_DIR}/train_block1_3slices_9phases.png")

# ---- NCC curves ----
print("\nDrawing NCC curves...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col_i, s in enumerate(SLICES):
    fi = BLOCK_BASE + s
    nccs = all_nccs[s]
    ax = axes[col_i]
    bars = ax.bar(range(1, 10), nccs, color='#4A90D9', alpha=0.8, edgecolor='#1a4a7a')
    ax.plot(range(1, 10), nccs, 'ro-', linewidth=2, markersize=7)
    ax.axhline(nccs[0], color='red', linestyle='--', alpha=0.4, label=f'P1={nccs[0]:.3f}')
    ax.set_title(f'Block 1 | Slice {s}\nFixed /{fi:03d}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Phase', fontsize=9); ax.set_ylabel('NCC', fontsize=9)
    ax.set_xticks(range(1, 10))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=7)
    for bar, val in zip(bars, nccs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
fig.suptitle('Training Set: NCC(Fixed, Moving) vs Phase\nBlock 1 | Slices 0, 1, 2', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f'{OUT_DIR}/train_block1_ncc_curves.png', dpi=120, bbox_inches='tight')
plt.close(fig); gc.collect()
print(f"  Saved: {OUT_DIR}/train_block1_ncc_curves.png")

# ---- Heatmap ----
print("Drawing heatmap...")
ncc_matrix = np.array([[all_nccs[s][p-1] for s in SLICES] for p in range(1, 10)])
fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(ncc_matrix, cmap='RdYlGn', vmin=0, vmax=1.0, aspect='auto')
ax.set_xticks(range(3))
ax.set_xticklabels([f'Slice {s}\n(fixed/{BLOCK_BASE+s:03d})' for s in SLICES], fontsize=9)
ax.set_yticks(range(9))
ax.set_yticklabels([f'Phase {p}' for p in range(1, 10)], fontsize=9)
ax.set_title('NCC Heatmap\nBlock 1 | Slices 0, 1, 2 | Training Set', fontsize=10, fontweight='bold')
for r in range(9):
    for c in range(3):
        ax.text(c, r, f'{ncc_matrix[r, c]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if ncc_matrix[r, c] < 0.55 else 'black')
plt.colorbar(im, ax=ax, label='NCC'); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/train_block1_ncc_heatmap.png', dpi=120, bbox_inches='tight')
plt.close(fig); gc.collect()
print(f"  Saved: {OUT_DIR}/train_block1_ncc_heatmap.png")

# ---- Summary ----
print('\n' + '=' * 80)
print('TRAINING SET BLOCK 1 — NCC & DIFF SUMMARY')
print('=' * 80)
for s in SLICES:
    fi = BLOCK_BASE + s
    print(f'\nSlice {s} (fixed/{fi:03d}):')
    for p in range(1, 10):
        nc = all_nccs[s][p-1]
        dmax, dmean, dstd = all_stats[s][p-1]
        mi = BLOCK_BASE + s + (p - 1) * 19
        print(f'  Phase {p}  moving/{mi:03d}  NCC={nc:.4f}  |F-M|max={dmax:.4f}  std={dstd:.4f}')

print(f'\nAll outputs: {OUT_DIR}/')