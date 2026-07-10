"""可视化 fixed/880 vs moving/880: 同一索引配对下的真实运动位移."""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')

fixed  = np.load('xcat_data/fixed/fixed/880.npy')
moving = np.load('xcat_data/moving/moving/880.npy')
print(f'fixed/880 : shape={fixed.shape}  min={fixed.min():.4f}  max={fixed.max():.4f}  mean={fixed.mean():.4f}  std={fixed.std():.4f}')
print(f'moving/880: shape={moving.shape}  min={moving.min():.4f}  max={moving.max():.4f}  mean={moving.mean():.4f}  std={moving.std():.4f}')
print(f'corr={np.corrcoef(fixed.flatten(), moving.flatten())[0,1]:.6f}  MAE={np.mean(np.abs(fixed-moving)):.5f}')

vmin, vmax = 0.0, 0.25
diff = np.abs(fixed - moving)
diff_vmax = diff.max()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# 上排: raw + diff
im0 = axes[0,0].imshow(fixed,  cmap='gray', vmin=vmin, vmax=vmax)
axes[0,0].set_title(f'fixed/880  (phase 0 模板)\nmean={fixed.mean():.4f} std={fixed.std():.4f}', fontsize=11)
axes[0,0].axis('off')

im1 = axes[0,1].imshow(moving, cmap='gray', vmin=vmin, vmax=vmax)
axes[0,1].set_title(f'moving/880  (运动后相位)\nmean={moving.mean():.4f} std={moving.std():.4f}', fontsize=11)
axes[0,1].axis('off')

im2 = axes[0,2].imshow(diff, cmap='hot', vmax=diff_vmax)
axes[0,2].set_title(f'|fixed/880 - moving/880|\nMAE={diff.mean():.5f} max={diff_vmax:.4f}\n非零像素 { (diff>1e-6).sum()/diff.size*100:.1f}%', fontsize=11)
axes[0,2].axis('off')

fig.colorbar(im0, ax=axes[0,:2], fraction=0.025, pad=0.01, label='intensity')
fig.colorbar(im2, ax=axes[0,2],  fraction=0.025, pad=0.01, label='|diff|')

# 下排: 三张图的 RGB 叠加 + 像素级 scatter + 中心切片对比
# RGB 叠加: R=fixed, G=moving, B=0
overlay = np.stack([fixed/max(fixed.max(),1e-6),
                    moving/max(moving.max(),1e-6),
                    np.zeros_like(fixed)], axis=-1)
axes[1,0].imshow(overlay)
axes[1,0].set_title('R=fixed  G=moving  (黄=一致, 红/绿=位移)', fontsize=11)
axes[1,0].axis('off')

# scatter: 固定 5000 随机像素
rng = np.random.default_rng(0)
idx = rng.choice(fixed.size, 5000, replace=False)
axes[1,1].scatter(fixed.flatten()[idx], moving.flatten()[idx], s=2, alpha=0.4)
mx = max(fixed.max(), moving.max())
axes[1,1].plot([0, mx], [0, mx], 'r--', lw=1, label='y=x')
axes[1,1].set_xlabel('fixed pixel value')
axes[1,1].set_ylabel('moving pixel value')
axes[1,1].set_title('pixel-level scatter (5000 随机点)\n偏离 y=x 越多 = 形变越大', fontsize=11)
axes[1,1].legend()
axes[1,1].set_xlim(0, mx*1.05)
axes[1,1].set_ylim(0, mx*1.05)
axes[1,1].grid(alpha=0.3)

# 中心水平/垂直切片对比
cy, cx = 256, 256
axes[1,2].plot(fixed[cy, :],  'b-',  lw=1, alpha=0.8, label='fixed  (center row)')
axes[1,2].plot(moving[cy, :], 'r-',  lw=1, alpha=0.8, label='moving (center row)')
axes[1,2].set_xlabel('x')
axes[1,2].set_ylabel('intensity')
axes[1,2].set_title(f'中心行切片 y={cy}  (fixed vs moving)', fontsize=11)
axes[1,2].legend()
axes[1,2].grid(alpha=0.3)

plt.suptitle('fixed/880 vs moving/880: 同一个体/同一索引下的"配对训练样本"', fontsize=14)
plt.tight_layout()
out = './logs/raw_check/inspect_fixed880_vs_moving880.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()