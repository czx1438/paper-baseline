"""可视化 fixed/moving 900-1000 范围: 相关性曲线 + 典型样本对 + 跳变分析."""
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 9

# =============================================================
# 1. 加载数据
# =============================================================
print('Loading 900-1000 ...')
fixed  = {}
moving = {}
for i in range(900, 1001):
    fixed[i]  = np.load(f'xcat_data/fixed/fixed/{i}.npy')
    moving[i] = np.load(f'xcat_data/moving/moving/{i}.npy')

# =============================================================
# 2. 计算所有相关性
# =============================================================
pair_corr   = {i: np.corrcoef(fixed[i].flatten(), moving[i].flatten())[0,1] for i in range(900,1001)}
pair_mae    = {i: np.mean(np.abs(fixed[i] - moving[i])) for i in range(900,1001)}
fix_autoc   = {i: np.corrcoef(fixed[i-1].flatten(), fixed[i].flatten())[0,1] for i in range(901,1001)}
mov_autoc   = {i: np.corrcoef(moving[i-1].flatten(), moving[i].flatten())[0,1] for i in range(901,1001)}

xs = np.arange(900, 1001)

# =============================================================
# Fig 1: 相关性曲线 (配对 + fixed 自相关 + moving 自相关)
# =============================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

# 上: 配对相关性
ax = axes[0]
ax.plot(xs, [pair_corr[i] for i in xs], 'b-', lw=1.5, label='corr(fixed, moving)')
ax.axhline(0.99, color='green', ls='--', lw=1, alpha=0.6, label='corr=0.99 阈值')
ax.axhline(0.90, color='orange', ls='--', lw=1, alpha=0.6, label='corr=0.90 阈值')
ax.fill_between(xs, [pair_corr[i] for i in xs], 0.86,
                alpha=0.15, color='red', label='高运动幅度区 (corr<0.90)')
ax.set_ylabel('Correlation')
ax.set_title('900-1000 配对相关性 corr(fixed[i], moving[i])')
ax.legend(loc='lower right')
ax.set_ylim(0.84, 1.01)
ax.grid(alpha=0.3)
# 标注最低点
min_i = min(pair_corr, key=pair_corr.get)
ax.annotate(f'{min_i}\ncorr={pair_corr[min_i]:.3f}', xy=(min_i, pair_corr[min_i]),
            xytext=(min_i+8, pair_corr[min_i]-0.015),
            arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)

# 中: fixed 内部自相关
ax = axes[1]
ax.plot(np.arange(901,1001), [fix_autoc[i] for i in range(901,1001)], 'b-', lw=1.5, label='corr(fixed[i-1], fixed[i])')
ax.axhline(0.99, color='green', ls='--', lw=1, alpha=0.6)
ax.axhline(0.75, color='red', ls='--', lw=1.2, label='剧烈跳变阈值 0.75')
ax.fill_between(np.arange(901,1001), [fix_autoc[i] for i in range(901,1001)], 0.7,
                alpha=0.15, color='red', label='跳变区')
ax.set_ylabel('Correlation')
ax.set_title('fixed 内部自相关 corr(fixed[i-1], fixed[i])  -- 标识 patient/slice 边界')
ax.legend(loc='lower right')
ax.set_ylim(0.68, 1.01)
ax.grid(alpha=0.3)
# 标注跳变点
min_f = min(fix_autoc, key=fix_autoc.get)
ax.annotate(f'{min_f}\ncorr={fix_autoc[min_f]:.3f}', xy=(min_f, fix_autoc[min_f]),
            xytext=(min_f+10, fix_autoc[min_f]-0.04),
            arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)

# 下: moving 内部自相关
ax = axes[2]
ax.plot(np.arange(901,1001), [mov_autoc[i] for i in range(901,1001)], 'orange', lw=1.5, label='corr(moving[i-1], moving[i])')
ax.axhline(0.99, color='green', ls='--', lw=1, alpha=0.6)
ax.axhline(0.75, color='red', ls='--', lw=1.2, label='剧烈跳变阈值 0.75')
ax.fill_between(np.arange(901,1001), [mov_autoc[i] for i in range(901,1001)], 0.7,
                alpha=0.15, color='red')
ax.set_ylabel('Correlation')
ax.set_xlabel('Index')
ax.set_title('moving 内部自相关 corr(moving[i-1], moving[i])')
ax.legend(loc='lower right')
ax.set_ylim(0.68, 1.01)
ax.grid(alpha=0.3)
min_m = min(mov_autoc, key=mov_autoc.get)
ax.annotate(f'{min_m}\ncorr={mov_autoc[min_m]:.3f}', xy=(min_m, mov_autoc[min_m]),
            xytext=(min_m+10, mov_autoc[min_m]-0.04),
            arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)

plt.suptitle('fixed/moving 900-1000 范围: 相关性全景 (无重复, 真实运动)', fontsize=13)
plt.tight_layout()
plt.savefig('./logs/raw_check/inspect_900_1000_corr_curves.png', dpi=110, bbox_inches='tight')
print('Saved: inspect_900_1000_corr_curves.png')
plt.close()

# =============================================================
# Fig 2: 选取 5 个典型配对样本网格 (raw + diff)
# =============================================================
# 选点: 高corr(988) / 中高(970) / 中(940) / 低(930谷底) / 高corr(900)
selected = [988, 970, 940, 930, 900]
vmin, vmax = 0.0, 0.25

fig, axes = plt.subplots(3, 5, figsize=(22, 13))
for col, idx in enumerate(selected):
    f = fixed[idx]; m = moving[idx]; d = np.abs(f-m)
    im0 = axes[0, col].imshow(f, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'fixed/{idx}\ncorr={pair_corr[idx]:.4f}\nMAE={pair_mae[idx]:.5f}', fontsize=9)
    axes[0, col].axis('off')
    im1 = axes[1, col].imshow(m, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'moving/{idx}', fontsize=9)
    axes[1, col].axis('off')
    im2 = axes[2, col].imshow(d, cmap='hot')
    axes[2, col].set_title(f'|diff| max={d.max():.4f}', fontsize=9)
    axes[2, col].axis('off')
# colorbar
for ax in axes[0,:]:
    ax.images[0].set_extent(ax.images[0].get_extent())
fig.colorbar(im0, ax=axes[0,:].tolist() + axes[1,:].tolist(), fraction=0.02, pad=0.01, label='intensity')
fig.colorbar(im2, ax=axes[2,:].tolist(), fraction=0.02, pad=0.01, label='|diff|')

plt.suptitle('典型配对样本: 从高运动(左)到低运动(右)', fontsize=13)
plt.tight_layout()
plt.savefig('./logs/raw_check/inspect_900_1000_samples.png', dpi=110, bbox_inches='tight')
print('Saved: inspect_900_1000_samples.png')
plt.close()

# =============================================================
# Fig 3: 跳变分析: 925-935 逐张展开
# =============================================================
fig, axes = plt.subplots(3, 11, figsize=(26, 10))
for col, idx in enumerate(range(925, 936)):
    f = fixed[idx]; m = moving[idx]; d = np.abs(f-m)
    im0 = axes[0, col].imshow(f, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'F/{idx}', fontsize=8)
    axes[0, col].axis('off')
    im1 = axes[1, col].imshow(m, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'M/{idx}', fontsize=8)
    axes[1, col].axis('off')
    im2 = axes[2, col].imshow(d, cmap='hot')
    axes[2, col].set_title(f'|F-M|', fontsize=8)
    axes[2, col].axis('off')
    im0_master = im0; im2_master = im2
# colorbar
for ax in axes[0,:]:
    ax.images[0].set_extent(ax.images[0].get_extent())
fig.colorbar(im0_master, ax=axes[:2,:],  fraction=0.015, pad=0.01, label='intensity')
fig.colorbar(im2_master, ax=axes[2,:],   fraction=0.015, pad=0.01, label='|diff|')
plt.suptitle('跳变区间 925-935: 红框=931(剧烈跳变, fixed/moving自相关均跌至0.75)', fontsize=13)
plt.tight_layout()
plt.savefig('./logs/raw_check/inspect_900_1000_transition_925_935.png', dpi=110, bbox_inches='tight')
print('Saved: inspect_900_1000_transition_925_935.png')
plt.close()

# =============================================================
# Fig 4: MAE 曲线 + 分段标注
# =============================================================
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(xs, [pair_mae[i] for i in xs], 'b-', lw=1.5, label='MAE(fixed, moving)')
ax.set_xlabel('Index')
ax.set_ylabel('MAE')
ax.set_title('900-1000 配对 MAE 曲线')
ax.set_ylim(0.0015, 0.008)
ax.grid(alpha=0.3)
ax.legend()
# 分段着色
segments = [(900,931,'red','高运动区(930谷底)'),(931,950,'orange','恢复区'),(950,1001,'blue','低运动/静止区')]
for s,e,color,label in segments:
    ax.axvspan(s, e, alpha=0.08, color=color, label=label)
ax.legend()
plt.tight_layout()
plt.savefig('./logs/raw_check/inspect_900_1000_mae_curve.png', dpi=110, bbox_inches='tight')
print('Saved: inspect_900_1000_mae_curve.png')
plt.close()

print('Done.')
