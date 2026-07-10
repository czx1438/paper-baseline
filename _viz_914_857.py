"""可视化 fixed/857, fixed/914, moving/914 三个 raw npy."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

f857 = np.load('xcat_data/fixed/fixed/857.npy')
f914 = np.load('xcat_data/fixed/fixed/914.npy')
m914 = np.load('xcat_data/moving/moving/914.npy')

# 三者共享一个 vmax (按 fixed max)
vmax = float(max(f857.max(), f914.max(), m914.max()))
vmin = float(min(f857.min(), f914.min(), m914.min()))

# 三张差值图也用统一 vmax
diff_fm_vmax = float(np.abs(f914 - m914).max())
diff_ff_vmax = float(np.abs(f914 - f857).max())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

def show(ax, img, title, vmax=None, vmin=None, cmap='gray'):
    im = ax.imshow(img, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    return im

# Row 0: 三张 raw (统一 vmin/vmax)
im0 = show(axes[0, 0], f857,
           f'fixed/857.npy\nshape={f857.shape}  mean={f857.mean():.4f}  std={f857.std():.4f}\n[min,max]=[{f857.min():.4f}, {f857.max():.4f}]',
           vmax=vmax, vmin=vmin)
im1 = show(axes[0, 1], f914,
           f'fixed/914.npy\nshape={f914.shape}  mean={f914.mean():.4f}  std={f914.std():.4f}\n[min,max]=[{f914.min():.4f}, {f914.max():.4f}]',
           vmax=vmax, vmin=vmin)
im2 = show(axes[0, 2], m914,
           f'moving/914.npy\nshape={m914.shape}  mean={m914.mean():.4f}  std={m914.std():.4f}\n[min,max]=[{m914.min():.4f}, {m914.max():.4f}]',
           vmax=vmax, vmin=vmin)
fig.colorbar(im0, ax=axes[0, :], fraction=0.02, pad=0.01, label='intensity')

# Row 1: 差异 + 直方图
show(axes[1, 0], np.abs(f914 - f857), '|fixed/914 - fixed/857|\n(= 0; corr=1.000)', vmax=diff_ff_vmax, cmap='hot')
show(axes[1, 1], np.abs(f914 - m914), '|fixed/914 - moving/914|\n(corr=0.893)', vmax=diff_fm_vmax, cmap='hot')
show(axes[1, 2], np.abs(f857 - m914), '|fixed/857 - moving/914|\n(corr=0.893)', vmax=diff_fm_vmax, cmap='hot')

plt.suptitle('xcat raw npy 对比: fixed/857  vs  fixed/914  vs  moving/914', fontsize=14)
plt.tight_layout()
os.makedirs('./logs/raw_check', exist_ok=True)
out = './logs/raw_check/inspect_fixed857_fixed914_moving914.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved: {out}')