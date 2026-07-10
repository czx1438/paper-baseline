"""直接可视化 fixed/346 + moving/346,365,384,403,422,441,460,479,498 (block2 slice4, 9 phases)"""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')

FIXED_DIR  = 'xcat_data/fixed/fixed'
MOVING_DIR = 'xcat_data/moving/moving'
OUT = './logs/raw_check/block2_slice4_phase0to9.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fixed_idx  = 346
moving_ids = [346, 365, 384, 403, 422, 441, 460, 479, 498]   # phase 1..9

fixed_img = np.load(os.path.join(FIXED_DIR, f'{fixed_idx:03d}.npy')).astype(np.float32)
moving_imgs = [np.load(os.path.join(MOVING_DIR, f'{i:03d}.npy')).astype(np.float32) for i in moving_ids]

# 2 rows: top=raw, bottom=|fixed - moving|
fig, axes = plt.subplots(2, 10, figsize=(28, 6))

vmin, vmax = 0.0, 0.30

# Row 0
axes[0, 0].imshow(fixed_img, cmap='gray', vmin=vmin, vmax=vmax)
axes[0, 0].set_title(f'fixed/346\nphase 0', fontsize=10, fontweight='bold', color='green')
for sp in axes[0, 0].spines.values(): sp.set_edgecolor('lime'); sp.set_linewidth(2)

for c, (idx, img) in enumerate(zip(moving_ids, moving_imgs), start=1):
    axes[0, c].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, c].set_title(f'moving/{idx}\nphase {c}', fontsize=10)

# Row 1: diff
diffs = [np.abs(fixed_img - m) for m in moving_imgs]
vmax_d = max(d.max() for d in diffs)
axes[1, 0].imshow(fixed_img, cmap='gray', vmin=vmin, vmax=vmax)
axes[1, 0].set_title('fixed (ref)', fontsize=9)
for sp in axes[1, 0].spines.values(): sp.set_edgecolor('lime'); sp.set_linewidth(2)

for c, (idx, d) in enumerate(zip(moving_ids, diffs), start=1):
    im = axes[1, c].imshow(d, cmap='hot', vmin=0, vmax=vmax_d)
    mae = d.mean()
    corr = np.corrcoef(fixed_img.flatten(), moving_imgs[c-1].flatten())[0, 1]
    axes[1, c].set_title(f'phase {c}\nr={corr:.3f}  mae={mae:.4f}', fontsize=9)

for ax in axes.flatten():
    ax.set_xticks([]); ax.set_yticks([])

fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.01, label='|fixed - moving|')
fig.suptitle('block=2, slice=4  |  phase 0=fixed + phase 1..9=moving  |  indices 346, 365..498',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches='tight')
print(f'Saved: {OUT}')

# 打印数值
print(f'\nblock2 slice4: fixed_idx=346')
for c, (idx, m) in enumerate(zip(moving_ids, moving_imgs), start=1):
    c_ = np.corrcoef(fixed_img.flatten(), m.flatten())[0, 1]
    mae = np.mean(np.abs(fixed_img - m))
    print(f'  phase {c} (moving/{idx}): corr={c_:.5f}  MAE={mae:.5f}  mean={m.mean():.4f}  std={m.std():.4f}')
