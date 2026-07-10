"""
验证集 3 slices × (phase0=fixed + phase1..9=moving) 全景可视化。
选 slice:
  - block4/slice04  (base_idx=80)  fixed_idx=688  moving_idx={688,707,726,745,764,783,802,821,840}
  - block4/slice10  (base_idx=86)  fixed_idx=694  moving_idx={694,713,732,751,770,789,808,827,846}
  - block5/slice00  (base_idx=95)  fixed_idx=855  moving_idx={855,874,893,912,931,950,969,988,1007}

raw_index 公式 (phase_id 0..8 for moving phases 1..9):
    fixed_idx  = block_id * 171 + 0 * 19 + slice_id
    moving_idx = block_id * 171 + phase_id * 19 + slice_id
"""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 8.5

FIXED_DIR  = 'xcat_data/fixed/fixed'
MOVING_DIR = 'xcat_data/moving/moving'
OUT_DIR    = './logs/raw_check'
os.makedirs(OUT_DIR, exist_ok=True)

SLICES = [
    (4,  4, 'block4_slice04'),
    (4, 10, 'block4_slice10'),
    (5,  0, 'block5_slice00'),
]

def load(block, phase_id, slice_id, pool='moving'):
    """
    phase_id: 0=fixed  → fixed pool, 索引 = block*171 + 0*19 + slice
              1..8=moving → moving pool, 索引 = block*171 + phase_id*19 + slice
    对应 moving phase (文件名) = phase_id + 1
    """
    idx = block * 171 + phase_id * 19 + slice_id
    path = os.path.join(FIXED_DIR if pool == 'fixed' else MOVING_DIR,
                        f'{idx:03d}.npy')
    return np.load(path).astype(np.float32)


# ================================================================
# Fig 1: 3 slices × 10 phases 灰度网格 (phase0=fixed + phase1..9=moving)
# ================================================================
n_phases = 10  # phase0(fixed) + phase1..9(moving)
n_slices = 3
fig, axes = plt.subplots(n_slices, n_phases, figsize=(30, 10))

for row, (b, s, lbl) in enumerate(SLICES):
    # phase 0 = fixed (pool='fixed')
    img = load(b, 0, s, pool='fixed')
    axes[row, 0].imshow(img, cmap='gray', vmin=0.0, vmax=0.25)
    axes[row, 0].set_ylabel(lbl, fontsize=9, fontweight='bold')
    axes[row, 0].set_xlabel('phase 0\n(fixed)', fontsize=8)
    axes[row, 0].tick_params(labelbottom=False, labelleft=False)
    for spine in axes[row, 0].spines.values():
        spine.set_edgecolor('lime'); spine.set_linewidth(2)

    # phase 1..9 = moving (phase_id 1..8)
    for p in range(1, 10):
        img = load(b, p, s, pool='moving')
        axes[row, p].imshow(img, cmap='gray', vmin=0.0, vmax=0.25)
        axes[row, p].set_xlabel(f'phase {p}', fontsize=8)
        axes[row, p].tick_params(labelbottom=False, labelleft=False)
        # mean/std 标注
        axes[row, p].text(0.98, 0.02,
                f'm={img.mean():.3f}\ns={img.std():.3f}',
                transform=axes[row, p].transAxes, fontsize=6.5,
                va='bottom', ha='right',
                bbox=dict(boxstyle='round', fc='white', alpha=0.7, ec='none'))

# title
for p_idx in range(n_phases):
    axes[0, p_idx].set_title(axes[0, p_idx].get_xlabel(), fontsize=9, fontweight='bold')

fig.text(0.5, 0.99,
         'Validation Raw: phase0=fixed (lime border) | phase1..9=moving',
         ha='center', va='top', fontsize=13, fontweight='bold')
plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.06,
                    hspace=0.35, wspace=0.15)
plt.savefig(f'{OUT_DIR}/inspect_val3slices_phases_raw.png', dpi=120, bbox_inches='tight')
print('Saved: inspect_val3slices_phases_raw.png')
plt.close()


# ================================================================
# Fig 2: |fixed - moving_phase| 热图 + corr/MAE 标注
# ================================================================
fig, axes = plt.subplots(3, 10, figsize=(30, 10))
vmax_diff = 0.0
for b, s, lbl in SLICES:
    f = load(b, 0, s, pool='fixed')
    for p in range(1, 10):
        m = load(b, p, s, pool='moving')
        vmax_diff = max(vmax_diff, np.abs(f - m).max())

for row, (b, s, lbl) in enumerate(SLICES):
    f = load(b, 0, s, pool='fixed')
    axes[row, 0].imshow(f, cmap='gray', vmin=0.0, vmax=0.25)
    axes[row, 0].set_ylabel(lbl, fontsize=9, fontweight='bold')
    axes[row, 0].set_xlabel('phase 0\n(fixed)', fontsize=8)
    axes[row, 0].tick_params(labelbottom=False, labelleft=False)
    for spine in axes[row, 0].spines.values():
        spine.set_edgecolor('lime'); spine.set_linewidth(2)

    for p in range(1, 10):
        m = load(b, p, s, pool='moving')
        diff = np.abs(f - m)
        axes[row, p].imshow(diff, cmap='hot', vmin=0, vmax=vmax_diff)
        axes[row, p].set_xlabel(f'phase {p}', fontsize=8)
        axes[row, p].tick_params(labelbottom=False, labelleft=False)
        c   = np.corrcoef(f.flatten(), m.flatten())[0, 1]
        mae = diff.mean()
        axes[row, p].text(0.98, 0.02,
                f'r={c:.3f}\nmae={mae:.4f}',
                transform=axes[row, p].transAxes, fontsize=6.5,
                va='bottom', ha='right',
                bbox=dict(boxstyle='round', fc='white', alpha=0.7, ec='none'))

for p_idx in range(10):
    axes[0, p_idx].set_title(axes[0, p_idx].get_xlabel(), fontsize=9, fontweight='bold')

fig.text(0.5, 0.99,
         'Validation: |fixed - moving_phase| (hot) with corr & MAE',
         ha='center', va='top', fontsize=13, fontweight='bold')
fig.colorbar(axes[0, 1].images[0], ax=axes[:, 1:].ravel().tolist(),
             fraction=0.02, pad=0.01, label='|diff|')
plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.06,
                    hspace=0.35, wspace=0.15)
plt.savefig(f'{OUT_DIR}/inspect_val3slices_phases_diff.png', dpi=120, bbox_inches='tight')
print('Saved: inspect_val3slices_phases_diff.png')
plt.close()


# ================================================================
# Fig 3: corr 和 MAE 随 phase 的变化曲线
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
phases = list(range(1, 10))

for row, (b, s, lbl) in enumerate(SLICES):
    f = load(b, 0, s, pool='fixed')
    corrs = []; maes = []
    for p in range(1, 10):
        m = load(b, p, s, pool='moving')
        corrs.append(np.corrcoef(f.flatten(), m.flatten())[0, 1])
        maes.append(np.mean(np.abs(f - m)))
    axes[0].plot(phases, corrs, marker='o', lw=2, ms=5, label=lbl, color=colors[row])
    axes[1].plot(phases, maes, marker='s', lw=2, ms=5, label=lbl, color=colors[row])

axes[0].set_xlabel('Moving Phase'); axes[0].set_ylabel('Correlation')
axes[0].set_title('corr(fixed, moving_phase) vs phase')
axes[0].set_xticks(phases); axes[0].set_ylim(0.82, 1.005)
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].axhline(0.99, color='red', ls='--', lw=0.8, alpha=0.5)

axes[1].set_xlabel('Moving Phase'); axes[1].set_ylabel('MAE')
axes[1].set_title('MAE(fixed, moving_phase) vs phase')
axes[1].set_xticks(phases); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('Validation 3 slices: corr & MAE vs moving phase', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/inspect_val3slices_corr_mae_curve.png', dpi=120, bbox_inches='tight')
print('Saved: inspect_val3slices_corr_mae_curve.png')
plt.close()

# ================================================================
# 打印数值
# ================================================================
print('\n=== detailed numbers ===')
for b, s, lbl in SLICES:
    f = load(b, 0, s, pool='fixed')
    fi = b*171 + s
    print(f'\n{lbl}: fixed_idx={fi}')
    for p in range(1, 10):
        m  = load(b, p, s, pool='moving')
        mi = b*171 + p*19 + s
        c  = np.corrcoef(f.flatten(), m.flatten())[0, 1]
        mae = np.mean(np.abs(f - m))
        print(f'  phase {p} (moving_idx={mi:4d}): corr={c:.5f}  MAE={mae:.5f}')

# 也打印 fixed 序列自相关 (block 内 slice 连续性)
print('\n=== fixed internal autocorr per block ===')
for b, s, lbl in SLICES:
    if s > 0:
        f0 = load(b, 0, s-1, pool='fixed')
        f1 = load(b, 0, s,   pool='fixed')
        c = np.corrcoef(f0.flatten(), f1.flatten())[0, 1]
        print(f'{lbl} vs slice-1: corr={c:.5f}')
    if s < 18:
        f0 = load(b, 0, s,   pool='fixed')
        f1 = load(b, 0, s+1, pool='fixed')
        c = np.corrcoef(f0.flatten(), f1.flatten())[0, 1]
        print(f'{lbl} vs slice+1: corr={c:.5f}')

print('\nDone.')
