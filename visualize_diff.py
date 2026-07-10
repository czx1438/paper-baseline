"""Visualize differences between fixed and moving images (first 3 files)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

fixed_dir = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/ruis_SEY_all_coronal/fixed'
moving_dir = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/ruis_SEY_all_coronal/moving'
out_path = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/diff_visualization_000_002.png'

fixed_files = sorted([f for f in os.listdir(fixed_dir) if f.endswith('.npy')])
moving_files = sorted([f for f in os.listdir(moving_dir) if f.endswith('.npy')])

fig, axes = plt.subplots(4, 3, figsize=(12, 14))
fig.suptitle('Fixed vs Moving (files 000, 001, 002) - Difference Visualization', fontsize=14)

for i in range(3):
    f = np.load(os.path.join(fixed_dir, fixed_files[i]))
    m = np.load(os.path.join(moving_dir, moving_files[i]))
    diff = m - f
    abs_diff = np.abs(diff)

    # Row 0: fixed
    im0 = axes[0, i].imshow(f, cmap='gray', vmin=0, vmax=0.05)
    axes[0, i].set_title(f'fixed[{i:03d}]')
    axes[0, i].axis('off')
    plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

    # Row 1: moving
    im1 = axes[1, i].imshow(m, cmap='gray', vmin=0, vmax=0.05)
    axes[1, i].set_title(f'moving[{i:03d}]')
    axes[1, i].axis('off')
    plt.colorbar(im1, ax=axes[1, i], fraction=0.046)

    # Row 2: raw diff (m - f)
    im2 = axes[2, i].imshow(diff, cmap='bwr', vmin=-0.02, vmax=0.02)
    axes[2, i].set_title(f'diff (mov - fix) [{i:03d}]')
    axes[2, i].axis('off')
    plt.colorbar(im2, ax=axes[2, i], fraction=0.046)

    # Row 3: abs diff
    im3 = axes[3, i].imshow(abs_diff, cmap='hot', vmin=0, vmax=0.02)
    axes[3, i].set_title(f'abs diff [{i:03d}]')
    axes[3, i].axis('off')
    plt.colorbar(im3, ax=axes[3, i], fraction=0.046)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_path, dpi=150)
plt.close()
print(f'Saved to {out_path}')
