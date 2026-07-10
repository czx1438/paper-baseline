"""可视化 fixed/003 与 moving/{003,022,041,060,079,098,117,136,145} 共 10 张."""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')

names = [
    ('fixed/003',  'xcat_data/fixed/fixed/003.npy'),
    ('moving/003', 'xcat_data/moving/moving/003.npy'),
    ('moving/022', 'xcat_data/moving/moving/022.npy'),
    ('moving/041', 'xcat_data/moving/moving/041.npy'),
    ('moving/060', 'xcat_data/moving/moving/060.npy'),
    ('moving/079', 'xcat_data/moving/moving/079.npy'),
    ('moving/098', 'xcat_data/moving/moving/098.npy'),
    ('moving/117', 'xcat_data/moving/moving/117.npy'),
    ('moving/136', 'xcat_data/moving/moving/136.npy'),
    ('moving/145', 'xcat_data/moving/moving/145.npy'),
]
arrs = {n: np.load(p) for n, p in names}

# 统一 vmin/vmax (排除 145 的极值影响)
vmax = 0.25
vmin = 0.0
diff_vmax = max(np.abs(arrs['fixed/003'] - a).max() for n, a in arrs.items() if n != 'fixed/003')

# ---- Fig 1: 2 行 x 5 列 展示 10 张 raw ----
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, (n, _) in zip(axes.flat, names):
    a = arrs[n]
    im = ax.imshow(a, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'{n}\nmean={a.mean():.4f} std={a.std():.4f}', fontsize=10)
    ax.axis('off')
fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label='intensity')
plt.suptitle('raw npy 对比: fixed/003  +  moving/{003,022,041,060,079,098,117,136,145}', fontsize=14)
plt.tight_layout()
out1 = './logs/raw_check/inspect_fixed003_vs_moving_seq.png'
plt.savefig(out1, dpi=110, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# ---- Fig 2: 与 fixed/003 的差异 (10 张差值图) ----
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
f3 = arrs['fixed/003']
for ax, (n, _) in zip(axes.flat, names):
    a = arrs[n]
    diff = np.abs(f3 - a)
    ax.imshow(diff, cmap='hot', vmax=diff_vmax)
    mae = diff.mean()
    ax.set_title(f'|fixed/003 - {n}|\nMAE={mae:.5f}', fontsize=10)
    ax.axis('off')
plt.suptitle(f'与 fixed/003 的 |diff| (vmax={diff_vmax:.4f})', fontsize=14)
plt.tight_layout()
out2 = './logs/raw_check/inspect_fixed003_vs_moving_seq_diff.png'
plt.savefig(out2, dpi=110, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()

# ---- Fig 3: 与 fixed/003 的相关系数柱状图 ----
fig, ax = plt.subplots(figsize=(11, 5))
labels = [n for n, _ in names if n != 'fixed/003']
corrs  = [np.corrcoef(f3.flatten(), arrs[n].flatten())[0,1] for n, _ in names if n != 'fixed/003']
maes   = [np.mean(np.abs(f3 - arrs[n])) for n, _ in names if n != 'fixed/003']
colors = ['tab:red' if c < 0.99 else 'tab:blue' for c in corrs]
bars = ax.bar(labels, corrs, color=colors)
ax.set_ylim(0.85, 1.001)
ax.set_ylabel('correlation with fixed/003')
ax.set_title('moving 序列与 fixed/003 的相关系数 (红=corr<0.99, 蓝=corr>=0.99)')
for b, c, m in zip(bars, corrs, maes):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{c:.4f}\nMAE={m:.5f}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=20)
plt.tight_layout()
out3 = './logs/raw_check/inspect_fixed003_vs_moving_seq_corr.png'
plt.savefig(out3, dpi=110, bbox_inches='tight')
print(f'Saved: {out3}')
plt.close()