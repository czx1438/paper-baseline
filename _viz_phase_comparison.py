"""
测试集每个 slice 的呼吸相位对比图：
- 每行是一个 phase (1~9)
- 每行包含: fixed | moving | diff

这样可以清晰看到:
1. fixed 同一张 (slice 不变)
2. moving 随 phase 逐渐变化
3. diff 图显示呼吸导致的形变随 phase 周期性变化
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

SAVE_DIR = './logs/raw_test_visualization/per_slice_phases'
os.makedirs(SAVE_DIR, exist_ok=True)

FIXED_DIR  = 'xcat_data/fixed/fixed'
MOVING_DIR = 'xcat_data/moving/moving'

# 测试集: block=5, slice=2..18, 每个 slice 9 phases
# fixed_idx = 855 + slice  (= 857..873)
# moving_idx = 855 + (phase-1)*19 + slice

# 加载所有数据
data = {}
for s in range(2, 19):          # slice 2..18
    data[s] = {}
    fixed_idx = 855 + s
    f = np.load(os.path.join(FIXED_DIR, f'{fixed_idx:03d}.npy'))
    data[s]['fixed'] = f
    data[s]['fixed_idx'] = fixed_idx
    data[s]['moving'] = []
    data[s]['moving_idx'] = []
    for p in range(1, 10):      # phase 1..9
        mi = 855 + (p-1)*19 + s
        m = np.load(os.path.join(MOVING_DIR, f'{mi:03d}.npy'))
        data[s]['moving'].append(m)
        data[s]['moving_idx'].append(mi)

# 全局 min/max for normalization
all_vals = [data[s]['fixed'] for s in data] + \
           [m for s in data for m in data[s]['moving']]
gmin = min(v.min() for v in all_vals)
gmax = max(v.max() for v in all_vals)
print(f'Global range: [{gmin:.4f}, {gmax:.4f}]')

def norm(arr):
    if gmax - gmin < 1e-6:
        return np.zeros_like(arr, dtype=float)
    return (arr - gmin) / (gmax - gmin)

def diff_map(a, b):
    """返回 |a - b| 差异图，0=相同, 1=最大差异"""
    d = np.abs(a.astype(float) - b.astype(float))
    return d / (d.max() + 1e-8)

print(f'\nDrawing per-slice phase comparison (17 slices × 9 phases)...')

for s in range(2, 19):
    f = data[s]['fixed']
    ms = data[s]['moving']

    fig, axes = plt.subplots(9, 3, figsize=(12, 28))
    fig.suptitle(
        f'Block 5 | Slice {s:02d} | Fixed: fixed/{data[s]["fixed_idx"]:03d}.npy\n'
        f'9 Phases: moving/{data[s]["moving_idx"][0]:03d} .. {data[s]["moving_idx"][-1]:03d}',
        fontsize=14, fontweight='bold', y=0.995
    )

    for row, p in enumerate(range(1, 10)):
        m = ms[row - 1]
        mi = data[s]['moving_idx'][row - 1]

        # Column 0: fixed
        ax = axes[row, 0]
        ax.imshow(norm(f), cmap='gray', vmin=0, vmax=1)
        phase_tag = '(=fixed)' if p == 1 else ''
        ax.set_title(f'Fixed\nfixed/{data[s]["fixed_idx"]:03d} {phase_tag}', fontsize=9)
        ax.axis('off')

        # Column 1: moving
        ax = axes[row, 1]
        ax.imshow(norm(m), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Moving Phase {p:02d}\nmoving/{mi:03d}', fontsize=9)
        ax.axis('off')

        # Column 2: diff
        ax = axes[row, 2]
        d = diff_map(f, m)
        ax.imshow(d, cmap='hot', vmin=0, vmax=d.max())
        ax.set_title(f'|Fixed - Mov| Phase {p:02d}\nmax={d.max():.3f} mean={d.mean():.3f}', fontsize=9)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(SAVE_DIR, f'slice_{s:02d}_phases.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f'  Slice {s:02d}: {out_path}')

print(f'\n✅ Saved to {SAVE_DIR}/')

# ---- 附加: 每个 slice 的相位 NCC 曲线 ----
print('\nDrawing NCC curves...')
import torch

def ncc(a, b):
    a = torch.from_numpy(a.astype(np.float32))[None, None]
    b = torch.from_numpy(b.astype(np.float32))[None, None]
    win = 15
    p = win // 2
    ap = torch.nn.functional.pad(a, [p, p, p, p], mode='reflect')
    bp = torch.nn.functional.pad(b, [p, p, p, p], mode='reflect')
    pa = ap.unfold(2, win, 1).unfold(3, win, 1).contiguous().view(*a.shape, -1)
    pb = bp.unfold(2, win, 1).unfold(3, win, 1).contiguous().view(*b.shape, -1)
    ma, mb = pa.mean(-1), pb.mean(-1)
    va = ((pa - ma.unsqueeze(-1)) ** 2).mean(-1)
    vb = ((pb - mb.unsqueeze(-1)) ** 2).mean(-1)
    cross = ((pa - ma.unsqueeze(-1)) * (pb - mb.unsqueeze(-1))).mean(-1)
    return (cross / (va.clamp(min=1e-8).sqrt() * vb.clamp(min=1e-8).sqrt() + 1e-8)).mean().item()

fig, axes = plt.subplots(4, 5, figsize=(25, 20))
axes = axes.flatten()

phases = list(range(1, 10))
for idx, s in enumerate(range(2, 19)):
    ax = axes[idx]
    f = data[s]['fixed']
    nccs = []
    for p in range(1, 10):
        m = data[s]['moving'][p-1]
        nccs.append(ncc(f, m))
    ax.plot(phases, nccs, 'bo-', linewidth=2, markersize=6)
    ax.fill_between(phases, nccs, alpha=0.2)
    ax.axhline(nccs[0], color='r', linestyle='--', alpha=0.5, label='phase1 NCC')
    ax.set_title(f'Slice {s:02d} (fixed/{data[s]["fixed_idx"]:03d})', fontsize=9)
    ax.set_xlabel('Phase')
    ax.set_ylabel('NCC')
    ax.set_xticks(phases)
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3)
    ax.text(5, nccs[4] + 0.02, f'min={min(nccs):.3f}', ha='center', fontsize=7)
    ax.text(1, nccs[0] + 0.02, f'{nccs[0]:.3f}', ha='center', fontsize=7)

# Hide last two
for i in range(17, 20):
    axes[i].axis('off')

plt.suptitle('Test Set: NCC(Fixed, Moving) vs Phase per Slice\n'
             'Phase 1=9 (end-exhale cycle back), Phase 4~5 (end-inhale = max deformation)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(SAVE_DIR, 'ncc_curves_all_slices.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'  NCC curves: {out_path}')
print(f'\n✅ All done!')
