"""
Debug: 对比原始 npy 数据 vs prep npz 加载后的数据
"""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

# 配置
RAW_FIXED_DIR = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/ruis_SEY_all_coronal/fixed/'
RAW_MOVING_DIR = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/ruis_SEY_all_coronal/moving/'
PREP_DIR = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep/test/'
SAVE_DIR = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/debug_data_loading/'
os.makedirs(SAVE_DIR, exist_ok=True)

sys.path.insert(0, '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main')
from ldm.data.sey_registration import SEYRegistration

# 划分（与 preprocess_sey.py 一致）
n_fixed = 1518
n_train = round(n_fixed * 0.7)   # 1063
n_val = round(n_fixed * 0.15)    # 228
test_start_global = n_train + n_val  # 1291

print(f"划分: train={n_train}, val={n_val}, test={n_fixed - n_train - n_val}")
print(f"Test 从全局 index {test_start_global} 开始\n")

fx_all = sorted(glob.glob(os.path.join(RAW_FIXED_DIR, '*.npy')))
mv_all = sorted(glob.glob(os.path.join(RAW_MOVING_DIR, '*.npy')))

ds = SEYRegistration(
    data_root='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep',
    split='test', normalize=False
)

n_samples = 10
for local_i in range(n_samples):
    global_i = test_start_global + local_i

    # 原始 npy
    fix_raw = np.load(fx_all[global_i])
    mov_raw = np.load(mv_all[global_i])

    # prep npz
    prep = np.load(os.path.join(PREP_DIR, f'{local_i:04d}_pair.npz'))
    fix_prep = prep['img_large']
    mov_prep = prep['img_small']

    # DataLoader
    mov_dl, fix_dl, _, _, _ = ds[local_i]
    fix_dl = fix_dl.squeeze().numpy()
    mov_dl = mov_dl.squeeze().numpy()

    # 手动 joint_norm
    g_min = min(fix_raw.min(), mov_raw.min())
    g_max = max(fix_raw.max(), mov_raw.max())
    fix_jn = (fix_raw - g_min) / (g_max - g_min + 1e-8)
    mov_jn = (mov_raw - g_min) / (g_max - g_min + 1e-8)

    # 绘图：4行 x 4列
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'Test[{local_i:02d}] global={global_i} — raw vs prep vs DataLoader vs manual_joint_norm', y=0.99)

    row_data = [
        ('1. Raw (original .npy)', fix_raw, mov_raw),
        ('2. Prep (.npz, joint_norm)', fix_prep, mov_prep),
        ('3. DataLoader (normalize=False)', fix_dl, mov_dl),
        ('4. Manual JointNorm on raw', fix_jn, mov_jn),
    ]

    for r, (label, fix_arr, mov_arr) in enumerate(row_data):
        # Fixed
        axes[r, 0].imshow(fix_arr, cmap='gray')
        axes[r, 0].set_title(f'{label}\nFixed [{fix_arr.min():.4f}, {fix_arr.max():.4f}]', fontsize=9)
        axes[r, 0].axis('off')

        # Moving
        axes[r, 1].imshow(mov_arr, cmap='gray')
        axes[r, 1].set_title(f'{label}\nMoving [{mov_arr.min():.4f}, {mov_arr.max():.4f}]', fontsize=9)
        axes[r, 1].axis('off')

        # Overlay (G=fixed, R=moving) - 用共同 vmin/vmax 避免亮度不一致
        vmin = min(fix_arr.min(), mov_arr.min())
        vmax = max(fix_arr.max(), mov_arr.max())
        f_n = (fix_arr - vmin) / (vmax - vmin + 1e-8)
        m_n = (mov_arr - vmin) / (vmax - vmin + 1e-8)
        overlay = np.stack([m_n * 0.8, f_n * 0.8, np.zeros_like(f_n)], axis=-1)
        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title(f'Overlay\n(R=moving, G=fixed)', fontsize=9)
        axes[r, 2].axis('off')

        # |Fix - Mov| 差值
        diff = np.abs(fix_arr - mov_arr)
        axes[r, 3].imshow(diff, cmap='hot', vmin=diff.min(), vmax=diff.max())
        axes[r, 3].set_title(f'|Fix-Mov| diff\n[min={diff.min():.4f}, max={diff.max():.4f}]', fontsize=9)
        axes[r, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(SAVE_DIR, f'debug_sample_{local_i:04d}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [{local_i:02d}] global={global_i} -> saved: {out_path}')

print(f'\n全部完成！图片保存在: {SAVE_DIR}')
