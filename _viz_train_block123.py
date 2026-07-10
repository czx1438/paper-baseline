"""
训练集 block 1, 2, 3 所有 slices 的 9 相位 + fixed
按 phase 切分: 每个 block 输出 9 张 1x27 的图 + 1 张 NCC heatmap
"""
import os, gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_ROOT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main'
FIXED_DIR  = os.path.join(DATA_ROOT, 'xcat_data/fixed/fixed')
MOVING_DIR = os.path.join(DATA_ROOT, 'xcat_data/moving/moving')
OUT_DIR    = os.path.join(DATA_ROOT, 'logs/raw_test_visualization/train_block123_all_slices')
os.makedirs(OUT_DIR, exist_ok=True)

BLOCKS = {1: 171, 2: 342, 3: 513}
N_SLICES = 9
N_PHASES = 9

def ncc_local(a, b, stride=8):
    a_ds = a[::stride, ::stride].astype(np.float64)
    b_ds = b[::stride, ::stride].astype(np.float64)
    a_ds = (a_ds - a_ds.mean()) / (a_ds.std() + 1e-8)
    b_ds = (b_ds - b_ds.mean()) / (b_ds.std() + 1e-8)
    return float((a_ds * b_ds).mean())

# Pre-compute all NCCs (small memory)
print("Computing NCCs for all blocks...", flush=True)
ncc_tensor = np.zeros((3, N_SLICES, N_PHASES))
slice_idx = np.zeros((3, N_SLICES), dtype=int)

for b_i, (b_idx, b_start) in enumerate(BLOCKS.items()):
    gc.collect()
    for s in range(N_SLICES):
        fi = b_start + s
        slice_idx[b_i, s] = fi
        f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))
        for p in range(1, N_PHASES + 1):
            mi = b_start + s + (p - 1) * 19
            m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
            ncc_tensor[b_i, s, p-1] = ncc_local(f, m)
            del m
        del f
        gc.collect()
    print(f"  Block {b_idx} done", flush=True)

# =====================================================
# Per-block: 9 phases x 1 row x (3 cols * 9 slices) = 9 figures
# Each figure: 1 row x 27 cols (1 phase across all slices)
# Actually: 1 phase shows fixed + moving + diff for 9 slices = 1 row x 27
# =====================================================
for b_idx, b_start in BLOCKS.items():
    print(f"\n=== Drawing block {b_idx} ===", flush=True)
    b_i = list(BLOCKS.keys()).index(b_idx)

    for p in range(1, N_PHASES + 1):
        gc.collect()
        # Load fixed for this block (just slice 0 fixed for column header)
        # We need fixed + moving for all 9 slices at this phase
        fig, axes = plt.subplots(1, 3 * N_SLICES, figsize=(N_SLICES * 4.5, 2.5))

        for s in range(N_SLICES):
            fi = b_start + s
            mi = b_start + s + (p - 1) * 19
            f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))
            m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
            d = np.abs(f.astype(np.float32) - m.astype(np.float32))
            nc = ncc_tensor[b_i, s, p-1]

            ax = axes[s * 3 + 0]
            ax.imshow(f, cmap='gray', vmin=0, vmax=0.3)
            ax.set_title(f'Fixed\n/{fi:03d}  S{s}', fontsize=7)
            ax.axis('off')

            ax = axes[s * 3 + 1]
            ax.imshow(m, cmap='gray', vmin=0, vmax=0.3)
            ax.set_title(f'Moving\n/{mi:03d}  NCC={nc:.3f}', fontsize=7)
            ax.axis('off')

            ax = axes[s * 3 + 2]
            im = ax.imshow(d, cmap='hot', vmin=0, vmax=d.max())
            ax.set_title(f'|F-M|\nmax={d.max():.3f}', fontsize=7)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            del f, m, d; gc.collect()

        fig.suptitle(
            f'Training Set | Block {b_idx} | Phase {p} | All 9 Slices\n'
            f'Cols: Fixed | Moving | Diff  |  9 slices: S0..S8 (fixed/{b_start:03d}..{b_start+8:03d})',
            fontsize=10, fontweight='bold', y=1.02
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = f'{OUT_DIR}/train_block{b_idx}_phase{p:02d}_allslices.png'
        plt.savefig(out, dpi=80, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
    print(f"  Block {b_idx} all phases done", flush=True)

# =====================================================
# NCC heatmap: 27 rows x 9 cols
# =====================================================
print("\nDrawing NCC heatmap...", flush=True)
fig, ax = plt.subplots(figsize=(10, 14))
flat = ncc_tensor.reshape(3 * N_SLICES, N_PHASES)
im = ax.imshow(flat, cmap='RdYlGn', vmin=0, vmax=1.0, aspect='auto')

y_labels = []
for b_i, b_idx in enumerate(BLOCKS.keys()):
    for s in range(N_SLICES):
        y_labels.append(f'B{b_idx}S{s}\n/{slice_idx[b_i,s]:03d}')

ax.set_yticks(range(27))
ax.set_yticklabels(y_labels, fontsize=6)
ax.set_xticks(range(N_PHASES))
ax.set_xticklabels([f'P{p+1}' for p in range(N_PHASES)], fontsize=9)
ax.set_xlabel('Phase', fontsize=10)
ax.set_title('NCC(Fixed, Moving) Heatmap\nTraining Set | Blocks 1, 2, 3 | All Slices × 9 Phases',
             fontsize=11, fontweight='bold')

# Block separators
for b_i in range(1, 3):
    ax.axhline(y=b_i * N_SLICES - 0.5, color='black', linewidth=1.5)
ax.text(-0.6, 4, 'Block 1', fontsize=10, fontweight='bold', ha='right', va='center', color='red')
ax.text(-0.6, 13, 'Block 2', fontsize=10, fontweight='bold', ha='right', va='center', color='red')
ax.text(-0.6, 22, 'Block 3', fontsize=10, fontweight='bold', ha='right', va='center', color='red')

for r in range(27):
    for c in range(N_PHASES):
        v = flat[r, c]
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=5,
                color='white' if v < 0.55 else 'black')

plt.colorbar(im, ax=ax, label='NCC', fraction=0.03, pad=0.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/train_block123_ncc_heatmap.png', dpi=120, bbox_inches='tight')
plt.close(fig); gc.collect()
print(f"  Saved: {OUT_DIR}/train_block123_ncc_heatmap.png", flush=True)

# =====================================================
# Block-level NCC line plot
# =====================================================
print("Drawing NCC line plot...", flush=True)
fig, ax = plt.subplots(figsize=(11, 5))
phases = list(range(1, N_PHASES + 1))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for b_i, b_idx in enumerate(BLOCKS.keys()):
    block_mean = ncc_tensor[b_i].mean(axis=0)
    block_min  = ncc_tensor[b_i].min(axis=0)
    block_max  = ncc_tensor[b_i].max(axis=0)
    ax.plot(phases, block_mean, 'o-', color=colors[b_i], linewidth=2,
            markersize=8, label=f'Block {b_idx} (mean over 9 slices)')
    ax.fill_between(phases, block_min, block_max, color=colors[b_i], alpha=0.2,
                    label=f'Block {b_idx} [min, max]')

ax.set_xlabel('Phase', fontsize=10)
ax.set_ylabel('NCC(Fixed, Moving)', fontsize=10)
ax.set_title('NCC vs Phase: Training Set | Blocks 1, 2, 3 | Mean ± [min, max] over 9 slices',
             fontsize=11, fontweight='bold')
ax.set_xticks(phases)
ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/train_block123_ncc_lines.png', dpi=120, bbox_inches='tight')
plt.close(fig); gc.collect()
print(f"  Saved: {OUT_DIR}/train_block123_ncc_lines.png", flush=True)

# =====================================================
# Summary
# =====================================================
print('\n' + '=' * 80)
print('NCC SUMMARY (per block)', flush=True)
print('=' * 80)
for b_i, b_idx in enumerate(BLOCKS.keys()):
    print(f'\nBlock {b_idx} (fixed: {BLOCKS[b_idx]:03d}..{BLOCKS[b_idx]+170:03d}):')
    print(f'  Per-slice mean NCC:')
    for s in range(N_SLICES):
        fi = slice_idx[b_i, s]
        m = ncc_tensor[b_i, s].mean()
        mn_p = ncc_tensor[b_i, s].argmin() + 1
        mn = ncc_tensor[b_i, s].min()
        print(f'    S{s} /{fi:03d}: mean={m:.4f}  min@P{mn_p}={mn:.4f}')
    print(f'  Block-level: mean={ncc_tensor[b_i].mean():.4f}  '
          f'min={ncc_tensor[b_i].min():.4f}  max={ncc_tensor[b_i].max():.4f}')

print(f'\nAll outputs: {OUT_DIR}/')
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn)) / 1024 / 1024
    print(f'  {fn}  ({sz:.1f} MB)')