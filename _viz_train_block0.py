"""
训练集 block 0: slice 3 和 slice 4 各 9 个相位的可视化
block 0 = indices 0..170
slice s -> fixed_idx = s, moving phase p -> s + (p-1)*19
"""
import os, gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_ROOT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main'
FIXED_DIR  = os.path.join(DATA_ROOT, 'xcat_data/fixed/fixed')
MOVING_DIR = os.path.join(DATA_ROOT, 'xcat_data/moving/moving')
OUT_DIR    = os.path.join(DATA_ROOT, 'logs/raw_test_visualization/train_block0_s3_s4')
os.makedirs(OUT_DIR, exist_ok=True)

SLICES = [3, 4]
# train block 0: indices 0..170 (int(171*0.7)=119? No, training uses range(0, 119) which is 119 files)
# Actually training range = range(0, int(n_fixed * 0.7)) = range(0, 119)
# So training covers block 0..4 partial: block 0 (0..170) is fully in train
# block 0 slice 3 -> fixed/003, slice 4 -> fixed/004

def ncc_light(a, b):
    """轻量 NCC: 降采样到 128x128"""
    a_ds = a[::4, ::4].astype(np.float64)
    b_ds = b[::4, ::4].astype(np.float64)
    a_ds = (a_ds - a_ds.mean()) / (a_ds.std() + 1e-8)
    b_ds = (b_ds - b_ds.mean()) / (b_ds.std() + 1e-8)
    return float(np.corrcoef(a_ds.ravel(), b_ds.ravel())[0, 1])

# ---- Per-slice large grid ----
all_nccs = {}
for s in SLICES:
    gc.collect()
    fi = s  # fixed/{s:03d}.npy
    f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))
    ms, mis, ds, nccs = [], [], [], []
    for p in range(1, 10):
        mi = s + (p - 1) * 19
        m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
        ms.append(m)
        mis.append(mi)
        ds.append(np.abs(f.astype(np.float32) - m.astype(np.float32)))
        nccs.append(ncc_light(f, m))
    all_nccs[s] = nccs

    gmin = min(f.min(), min(m.min() for m in ms))
    gmax = max(f.max(), max(m.max() for m in ms))

    fig, axes = plt.subplots(9, 3, figsize=(11, 35))
    fig.suptitle(
        f'Training Set | Block 0 | Slice {s} | Fixed: fixed/{fi:03d}.npy\n'
        f'Moving: phase1=/{mis[0]:03d} .. phase9=/{mis[-1]:03d}',
        fontsize=10, fontweight='bold', y=0.995
    )

    for row in range(9):
        axes[row, 0].imshow(f, cmap='gray', vmin=gmin, vmax=gmax)
        tag = ' (=Moving)' if row == 0 else ''
        axes[row, 0].set_title(f'Fixed\n/{fi:03d}{tag}', fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(ms[row], cmap='gray', vmin=gmin, vmax=gmax)
        axes[row, 1].set_title(f'Moving Phase {row+1:02d}\n/{mis[row]:03d}', fontsize=8)
        axes[row, 1].axis('off')

        im = axes[row, 2].imshow(ds[row], cmap='hot', vmin=0, vmax=ds[row].max())
        axes[row, 2].set_title(f'|F-M|  NCC={nccs[row]:.3f}', fontsize=8)
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = f'{OUT_DIR}/train_block0_slice{s:02d}_phases.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    del f, ms, ds
    gc.collect()
    print(f'Slice {s}: NCCs={[round(n,3) for n in nccs]}')

# ---- NCC curves ----
print("\nDrawing NCC curves...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for col_i, s in enumerate(SLICES):
    fi = s
    nccs = all_nccs[s]
    ax = axes[col_i]
    bars = ax.bar(range(1, 10), nccs, color='#4A90D9', alpha=0.8, edgecolor='#1a4a7a')
    ax.plot(range(1, 10), nccs, 'ro-', linewidth=2, markersize=7)
    ax.axhline(nccs[0], color='red', linestyle='--', alpha=0.4, label=f'P1={nccs[0]:.3f}')
    ax.set_title(f'Block 0 | Slice {s}\nFixed /{fi:03d}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Phase', fontsize=9); ax.set_ylabel('NCC', fontsize=9)
    ax.set_xticks(range(1, 10)); ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=7)
    for bar, val in zip(bars, nccs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
fig.suptitle('Training Set: NCC(Fixed, Moving) vs Phase\nBlock 0 | Slices 3, 4', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f'{OUT_DIR}/train_block0_ncc_curves.png', dpi=120, bbox_inches='tight')
plt.close(fig); gc.collect()
print(f"  Saved: {OUT_DIR}/train_block0_ncc_curves.png")

# ---- Heatmap ----
print("Drawing heatmap...")
ncc_matrix = np.array([[all_nccs[s][p-1] for s in SLICES] for p in range(1, 10)])
fig, ax = plt.subplots(figsize=(5, 7))
im = ax.imshow(ncc_matrix, cmap='RdYlGn', vmin=0, vmax=0.6, aspect='auto')
ax.set_xticks(range(2))
ax.set_xticklabels([f'Slice {s}\n(fixed/{s:03d})' for s in SLICES], fontsize=9)
ax.set_yticks(range(9))
ax.set_yticklabels([f'Phase {p}' for p in range(1, 10)], fontsize=9)
ax.set_title('NCC Heatmap\nBlock 0 | Slices 3, 4 | Training Set', fontsize=10, fontweight='bold')
for r in range(9):
    for c in range(2):
        ax.text(c, r, f'{ncc_matrix[r, c]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if ncc_matrix[r, c] < 0.35 else 'black')
plt.colorbar(im, ax=ax, label='NCC'); plt.tight_layout()
plt.savefig(f'{OUT_DIR}/train_block0_ncc_heatmap.png', dpi=120, bbox_inches='tight')
plt.close(fig); gc.collect()
print(f"  Saved: {OUT_DIR}/train_block0_ncc_heatmap.png")

# ---- Summary ----
print('\n' + '=' * 80)
print('TRAINING SET BLOCK 0 — NCC SUMMARY')
print('=' * 80)
for s in SLICES:
    fi = s
    print(f'Slice {s} (fixed/{fi:03d}): '
          + '  '.join([f'P{p}={all_nccs[s][p-1]:.3f}' for p in range(1,10)]))
print(f'\nAll outputs: {OUT_DIR}/')