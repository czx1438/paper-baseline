"""可视化 fixed/171, fixed/172 + moving/{172,191,210,229,248,267,286,305,324}."""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')

names = [
    ('fixed/171',  'xcat_data/fixed/fixed/171.npy'),
    ('fixed/172',  'xcat_data/fixed/fixed/172.npy'),
    ('moving/172', 'xcat_data/moving/moving/172.npy'),
    ('moving/191', 'xcat_data/moving/moving/191.npy'),
    ('moving/210', 'xcat_data/moving/moving/210.npy'),
    ('moving/229', 'xcat_data/moving/moving/229.npy'),
    ('moving/248', 'xcat_data/moving/moving/248.npy'),
    ('moving/267', 'xcat_data/moving/moving/267.npy'),
    ('moving/286', 'xcat_data/moving/moving/286.npy'),
    ('moving/305', 'xcat_data/moving/moving/305.npy'),
    ('moving/324', 'xcat_data/moving/moving/324.npy'),
]
arrs = {n: np.load(p) for n, p in names}
vmax = 0.25
vmin = 0.0

# ---- Fig 1: 3 行 x 4 列 (留 1 空) 展示 11 张 raw ----
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for ax, (n, _) in zip(axes.flat, names):
    a = arrs[n]
    im = ax.imshow(a, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'{n}\nmean={a.mean():.4f} std={a.std():.4f}', fontsize=10)
    ax.axis('off')
axes.flat[11].axis('off')
fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label='intensity')
plt.suptitle('raw npy 对比: fixed/171, fixed/172 + moving/{172,191,...,324}', fontsize=14)
plt.tight_layout()
out1 = './logs/raw_check/inspect_fixed171_172_vs_moving_seq.png'
plt.savefig(out1, dpi=110, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# ---- Fig 2: 与 fixed/172 的差值 (按 moving 索引顺序) ----
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
f172 = arrs['fixed/172']
# 用 moving 序列重新排序展示
moving_seq = ['moving/172','moving/191','moving/210','moving/229',
              'moving/248','moving/267','moving/286','moving/305','moving/324']
plot_list = [('fixed/172', f172)] + [(n, arrs[n]) for n in moving_seq]
diff_vmax = max(np.abs(f172 - arrs[n]).max() for n in moving_seq)
for ax, (n, a) in zip(axes.flat, plot_list):
    diff = np.abs(f172 - a)
    im = ax.imshow(diff, cmap='hot', vmax=diff_vmax)
    mae = diff.mean()
    c = np.corrcoef(f172.flatten(), a.flatten())[0,1] if n != 'fixed/172' else 1.0
    ax.set_title(f'|fixed/172 - {n}|\nMAE={mae:.5f} corr={c:.4f}', fontsize=10)
    ax.axis('off')
# 留 2 个空位
for idx in [10, 11]:
    axes.flat[idx].axis('off')
plt.suptitle(f'与 fixed/172 的 |diff| (vmax={diff_vmax:.4f}, hot)', fontsize=14)
plt.tight_layout()
out2 = './logs/raw_check/inspect_fixed172_vs_moving_seq_diff.png'
plt.savefig(out2, dpi=110, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()

# ---- Fig 3: 相关系数柱状图 (固定 171 vs 固定 172 双基准) ----
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
f171, f172 = arrs['fixed/171'], arrs['fixed/172']

# 左: 与 fixed/171
labels = [n for n, _ in names if n != 'fixed/171']
corrs1 = [np.corrcoef(f171.flatten(), arrs[n].flatten())[0,1] for n in labels]
maes1  = [np.mean(np.abs(f171 - arrs[n])) for n in labels]
colors1 = ['tab:blue' if c > 0.99 else 'tab:orange' if c > 0.98 else 'tab:red' for c in corrs1]
bars = axes[0].bar(labels, corrs1, color=colors1)
axes[0].set_ylim(0.85, 1.005)
axes[0].set_ylabel('correlation')
axes[0].set_title('Correlation with fixed/171')
axes[0].tick_params(axis='x', rotation=20)
for b, c, m in zip(bars, corrs1, maes1):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{c:.4f}\nMAE={m:.5f}', ha='center', va='bottom', fontsize=8)

# 右: 与 fixed/172
labels = [n for n, _ in names if n != 'fixed/172']
corrs2 = [np.corrcoef(f172.flatten(), arrs[n].flatten())[0,1] for n in labels]
maes2  = [np.mean(np.abs(f172 - arrs[n])) for n in labels]
colors2 = ['tab:blue' if c > 0.99 else 'tab:orange' if c > 0.98 else 'tab:red' for c in corrs2]
bars = axes[1].bar(labels, corrs2, color=colors2)
axes[1].set_ylim(0.85, 1.005)
axes[1].set_ylabel('correlation')
axes[1].set_title('Correlation with fixed/172')
axes[1].tick_params(axis='x', rotation=20)
for b, c, m in zip(bars, corrs2, maes2):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{c:.4f}\nMAE={m:.5f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('序列相关系数 (蓝 >0.99, 橙 0.98-0.99, 红 <0.98)', fontsize=14)
plt.tight_layout()
out3 = './logs/raw_check/inspect_fixed171_172_corr.png'
plt.savefig(out3, dpi=110, bbox_inches='tight')
print(f'Saved: {out3}')
plt.close()