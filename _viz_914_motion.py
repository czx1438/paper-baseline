"""可视化 moving/{876,895,914} 三张 + 与 fixed/914 的对比, 解释 fixed/moving 的相位关系."""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')

names = [
    ('fixed/857',   'xcat_data/fixed/fixed/857.npy',  'fixed: phase 0'),
    ('fixed/914',   'xcat_data/fixed/fixed/914.npy',  'fixed: phase 0 (=857)'),
    ('moving/876',  'xcat_data/moving/moving/876.npy', 'moving: phase ?'),
    ('moving/895',  'xcat_data/moving/moving/895.npy', 'moving: phase ?'),
    ('moving/914',  'xcat_data/moving/moving/914.npy', 'moving: phase ?'),
]
arrs = {n: np.load(p) for n, p, _ in names}
vmax = 0.25
vmin = 0.0

# ---- Fig 1: 5 张 raw ----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, (n, _, sub) in zip(axes.flat[:5], names):
    a = arrs[n]
    im = ax.imshow(a, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'{n}  ({sub})\nmean={a.mean():.4f} std={a.std():.4f}', fontsize=10)
    ax.axis('off')
axes.flat[5].axis('off')
fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label='intensity')
plt.suptitle('fixed/857 vs fixed/914 vs moving/{876,895,914}', fontsize=14)
plt.tight_layout()
out1 = './logs/raw_check/inspect_moving876_895_914.png'
plt.savefig(out1, dpi=110, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# ---- Fig 2: 与 fixed/914 的差值 (解释"为什么差别那么大") ----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
f914 = arrs['fixed/914']
diff_targets = [
    ('fixed/857',  arrs['fixed/857'],  'vs fixed/857  (= 0; 同一张)'),
    ('moving/876', arrs['moving/876'], 'vs moving/876 (差较小, 早期运动)'),
    ('moving/895', arrs['moving/895'], 'vs moving/895 (中等差异)'),
    ('moving/914', arrs['moving/914'], 'vs moving/914 (最大差异, 运动末)'),
]
diff_vmax = max(np.abs(f914 - arrs[n]).max() for n, _, _ in diff_targets)
for ax, (n, a, sub) in zip(axes.flat[:4], diff_targets):
    diff = np.abs(f914 - a)
    im = ax.imshow(diff, cmap='hot', vmax=diff_vmax)
    c = np.corrcoef(f914.flatten(), a.flatten())[0,1]
    mae = diff.mean()
    ax.set_title(f'|fixed/914 - {n}|  {sub}\nMAE={mae:.5f}  corr={c:.4f}', fontsize=10)
    ax.axis('off')
# 留 2 个空位
axes.flat[4].axis('off'); axes.flat[5].axis('off')
plt.suptitle(f'与 fixed/914 的 |diff|  -- fixed 是 phase0, moving 是运动后的相位 (vmax={diff_vmax:.4f})', fontsize=13)
plt.tight_layout()
out2 = './logs/raw_check/inspect_moving876_895_914_diff.png'
plt.savefig(out2, dpi=110, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()

# ---- Fig 3: moving 序列自身相关系数柱状图 ----
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左: 以 moving/876 为基准
ref = arrs['moving/876']
keys = ['moving/895', 'moving/914']
corrs = [np.corrcoef(ref.flatten(), arrs[k].flatten())[0,1] for k in keys]
maes  = [np.mean(np.abs(ref - arrs[k])) for k in keys]
bars = axes[0].bar(keys, corrs, color=['tab:orange','tab:red'])
axes[0].set_ylim(0.85, 1.005)
axes[0].set_title('Correlation with moving/876\n(moving 序列自身)')
axes[0].set_ylabel('corr')
for b, c, m in zip(bars, corrs, maes):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{c:.4f}\nMAE={m:.5f}', ha='center', va='bottom', fontsize=9)

# 右: 以 fixed/914 为基准 + moving 三张
ref = arrs['fixed/914']
keys = ['moving/876', 'moving/895', 'moving/914']
corrs = [np.corrcoef(ref.flatten(), arrs[k].flatten())[0,1] for k in keys]
maes  = [np.mean(np.abs(ref - arrs[k])) for k in keys]
colors = ['tab:green' if c > 0.95 else 'tab:orange' if c > 0.92 else 'tab:red' for c in corrs]
bars = axes[1].bar(keys, corrs, color=colors)
axes[1].set_ylim(0.85, 1.005)
axes[1].set_title('Correlation with fixed/914\n(fixed=phase0 vs moving=运动后)')
axes[1].set_ylabel('corr')
for b, c, m in zip(bars, corrs, maes):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{c:.4f}\nMAE={m:.5f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('moving 序列内部一致性  vs  fixed/moving 跨域一致性', fontsize=14)
plt.tight_layout()
out3 = './logs/raw_check/inspect_moving876_895_914_corr.png'
plt.savefig(out3, dpi=110, bbox_inches='tight')
print(f'Saved: {out3}')
plt.close()