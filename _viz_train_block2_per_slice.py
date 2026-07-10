"""
Block 2 所有 slice 的 9 相位可视化
每个 slice 一张图: 9 rows x 3 cols (Fixed | Moving | Diff)
block 2 = indices 342..512, slice s -> fixed/{342+s:03d}, moving phase p -> {342+s+(p-1)*19:03d}
"""
import os, gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_ROOT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main'
FIXED_DIR  = os.path.join(DATA_ROOT, 'xcat_data/fixed/fixed')
MOVING_DIR = os.path.join(DATA_ROOT, 'xcat_data/moving/moving')
OUT_DIR    = os.path.join(DATA_ROOT, 'logs/raw_test_visualization/train_block2_per_slice_9phases')
os.makedirs(OUT_DIR, exist_ok=True)

BLOCK_START = 342  # block 2 starts at index 342
N_SLICES = 9

def ncc_local(a, b, stride=8):
    a_ds = a[::stride, ::stride].astype(np.float64)
    b_ds = b[::stride, ::stride].astype(np.float64)
    a_ds = (a_ds - a_ds.mean()) / (a_ds.std() + 1e-8)
    b_ds = (b_ds - b_ds.mean()) / (b_ds.std() + 1e-8)
    return float((a_ds * b_ds).mean())

# 每个 slice 一张图: 9 rows (phases) x 3 cols (Fixed / Moving / Diff)
for s in range(N_SLICES):
    gc.collect()
    fi = BLOCK_START + s
    f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))

    fig, axes = plt.subplots(9, 3, figsize=(11, 35))

    nccs = []
    for p in range(1, 10):
        mi = BLOCK_START + s + (p - 1) * 19
        m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
        d = np.abs(f.astype(np.float32) - m.astype(np.float32))
        nc = ncc_local(f, m)
        nccs.append(nc)

        # Fixed (only first row shows; rest just gray levels)
        ax = axes[p-1, 0]
        ax.imshow(f, cmap='gray', vmin=0, vmax=0.3)
        if p == 1:
            ax.set_title(f'Fixed\n/fixed/{fi:03d}', fontsize=8, fontweight='bold')
        else:
            ax.set_title('Fixed', fontsize=7)
        ax.axis('off')

        # Moving
        ax = axes[p-1, 1]
        ax.imshow(m, cmap='gray', vmin=0, vmax=0.3)
        ax.set_title(f'Moving P{p:02d}\n/moving/{mi:03d}', fontsize=8)
        ax.axis('off')

        # Diff
        ax = axes[p-1, 2]
        im = ax.imshow(d, cmap='hot', vmin=0, vmax=d.max())
        ax.set_title(f'|F-M|  NCC={nc:.3f}\nmax={d.max():.3f}  std={(f.astype(np.float64)-m.astype(np.float64)).std():.3f}',
                     fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        del m, d
        gc.collect()

    fig.suptitle(
        f'Training Set | Block 2 | Slice {s} | Fixed: /fixed/{fi:03d}\n'
        f'9 Phases: moving/{BLOCK_START+s:03d} (P1) .. moving/{BLOCK_START+s+(8)*19:03d} (P9)',
        fontsize=11, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = f'{OUT_DIR}/train_block2_slice{s:02d}_9phases.png'
    plt.savefig(out, dpi=80, bbox_inches='tight')
    plt.close(fig)
    del f
    gc.collect()
    print(f'Slice {s} done: NCCs={[round(n,3) for n in nccs]}')

# Summary
print('\n' + '='*60)
print('Block 2 NCC summary')
print('='*60)
for s in range(N_SLICES):
    fi = BLOCK_START + s
    f = np.load(os.path.join(FIXED_DIR, f'{fi:03d}.npy'))
    nccs = []
    for p in range(1, 10):
        mi = BLOCK_START + s + (p - 1) * 19
        m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
        nccs.append(ncc_local(f, m))
        del m
    del f
    print(f'S{s} (fixed/{fi:03d}): ' + ' '.join([f'P{p}={nccs[p-1]:.3f}' for p in range(1,10)]))

print(f'\nAll outputs: {OUT_DIR}/')
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn)) / 1024 / 1024
    print(f'  {fn}  ({sz:.1f} MB)')