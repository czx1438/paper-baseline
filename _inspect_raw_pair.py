"""集中可视化 4 个 raw npy + dataset 归一化版 + 配准结果."""
import sys
sys.argv = [
    'visualize_registration_xcat.py',
    '--no-xcat',
    '--datapath', 'xcat_data',
    '--ldm_config', 'configs/latent-diffusion/xcat_no_motion.yaml',
    '--ldm_checkpoint', 'logs/2026-06-02T18-07-16_xcat-ldm-vq16-64ch/last.ckpt',
    '--resume', 'logs/TransScorelm_Smooth_0.8_beta_0.8_7_15_15_707_base/NCCVal_0.8511_Epoch_5000.pth',
    '--n_samples', '1',
    '--split', 'test',
    '--start_idx', '3',
    '--save_dir', './logs/raw_check/',
]
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import visualize_registration_xcat as viz
from visualize_registration_xcat import body_mask, ncc_metric, ssim_metric

ds, _ = viz.build_dataset('test')
item = ds[3]
print(f'pairname: {item["pairname"]}')

# 关键 4 张 raw
f857  = np.load('xcat_data/fixed/fixed/857.npy')    # dataset.fixed
f914  = np.load('xcat_data/fixed/fixed/914.npy')    # 用户要求看
m914  = np.load('xcat_data/moving/moving/914.npy')  # dataset.moving
m855  = np.load('xcat_data/moving/moving/855.npy')  # 对照

# 2 张额外 raw 看 fixed/moving 分布
f000  = np.load('xcat_data/fixed/fixed/000.npy')    # phase 0 第 0 张
f171  = np.load('xcat_data/fixed/fixed/171.npy')
f873  = np.load('xcat_data/fixed/fixed/873.npy')    # 跨 block 边界
f855  = np.load('xcat_data/fixed/fixed/855.npy')    # block 5 起
f874  = np.load('xcat_data/fixed/fixed/874.npy')    # block 5 phase 1 起
m000  = np.load('xcat_data/moving/moving/000.npy')  # moving phase 0
m019  = np.load('xcat_data/moving/moving/019.npy')
m020  = np.load('xcat_data/moving/moving/020.npy')

# 配准结果
device = 'cuda'
X = item['moving'].unsqueeze(0).to(device)
Y = item['fixed'].unsqueeze(0).to(device)
ldm_model = viz.ldm_model; ldm_model.eval()
model = viz.model; model.eval()
with torch.no_grad():
    mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
    fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()
    noise = torch.randn_like(mov_z)
    t_enc = torch.tensor([viz.opt.t_enc]).cuda()
    x_noisy = ldm_model.q_sample(x_start=mov_z, t=t_enc, noise=noise)
    y_noisy = ldm_model.q_sample(x_start=fix_z, t=t_enc, noise=noise)
    outx = ldm_model.apply_model(x_noisy, t=t_enc, cond=None, return_ids=True)
    outy = ldm_model.apply_model(y_noisy, t=t_enc, cond=None, return_ids=True)
    s0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
    s1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
    s2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
    s3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]), dim=1)
    D_f_xy, _ = model(X, Y, s0, s1, s2, s3, phase_id=None)
    _, warped_X = viz.transform(X, D_f_xy.permute(0, 2, 3, 1))

fg = body_mask(Y)
ncc_b = float(ncc_metric(Y, X, mask=fg))
ncc_a = float(ncc_metric(Y, warped_X, mask=fg))
ssim_b = float(ssim_metric(Y, X, mask=fg))
ssim_a = float(ssim_metric(Y, warped_X, mask=fg))

# ===== 画 4x4 大图 =====
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

# Row 0: 跟 block5_slice02_phase04 直接相关的 4 张
def show(ax, img, title, vmax=None):
    ax.imshow(img, cmap='gray', vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

show(axes[0,0], f857, f'RAW fixed/857.npy\n(block5 phase0 slice2)\nmean={f857.mean():.4f}')
show(axes[0,1], f914, f'RAW fixed/914.npy\n(block5 phase3 slice0)\nmean={f914.mean():.4f}\n  ← 你要看的', vmax=0.1)
show(axes[0,2], m855, f'RAW moving/855.npy\n(block5 phase0 slice0)\nmean={m855.mean():.4f}')
show(axes[0,3], m914, f'RAW moving/914.npy\n(block5 phase3 slice0)\nmean={m914.mean():.4f}\n= dataset.moving', vmax=0.1)

# Row 1: 跨 block 边界看 fixed 复制模式
show(axes[1,0], f000, f'fixed/000.npy\n(block0 phase0 slice0)\nmean={f000.mean():.4f}')
show(axes[1,1], f171, f'fixed/171.npy\n(block1 phase0 slice0)\nmean={f171.mean():.4f}')
show(axes[1,2], f855, f'fixed/855.npy\n(block5 phase0 slice0)\nmean={f855.mean():.4f}')
show(axes[1,3], f874, f'fixed/874.npy\n(block5 phase1 slice0)\nmean={f874.mean():.4f}')

# Row 2: moving 跨 block / 跨 phase
show(axes[2,0], m000, f'moving/000.npy\n(block0 phase0 slice0)\nmean={m000.mean():.4f}')
show(axes[2,1], m019, f'moving/019.npy\n(block0 phase0 slice18)\nmean={m019.mean():.4f}')
show(axes[2,2], m020, f'moving/020.npy\n(block0 phase1 slice0)\nmean={m020.mean():.4f}')
show(axes[2,3], f873, f'fixed/873.npy\n(block5 phase0 slice18)\nmean={f873.mean():.4f}\n(= phase0 末尾, 后续复制起点)')

# Row 3: dataset item + 配准结果
ncc_diff = ncc_a - ncc_b
ssim_diff = ssim_a - ssim_b
title_diff = (f'  NCC Δ={ncc_diff:+.4f}  SSIM Δ={ssim_diff:+.4f}\n'
              f'  before NCC={ncc_b:.4f} SSIM={ssim_b:.4f}\n  after  NCC={ncc_a:.4f} SSIM={ssim_a:.4f}')
show(axes[3,0], item['moving'].squeeze().cpu().numpy(), 'dataset.moving (归一化)')
show(axes[3,1], item['fixed'].squeeze().cpu().numpy(), 'dataset.fixed (归一化)')
show(axes[3,2], warped_X.squeeze().cpu().numpy(), f'Warped X (配准后)\n{title_diff}')
axes[3,3].imshow(np.abs(warped_X.squeeze().cpu().numpy() - Y.squeeze().cpu().numpy()),
                 cmap='hot', vmax=0.5)
axes[3,3].set_title('|Warped − Fixed| 残差', fontsize=10)
axes[3,3].axis('off')

# Row labels
for r, label in enumerate(['block5_slice02_phase04 关键 4 张', 'fixed 跨 block 复制模式', 'moving 跨 phase 分布', 'dataset 配准结果']):
    axes[r, 0].annotate(label, xy=(-0.18, 0.5), xycoords='axes fraction',
                         rotation=90, ha='right', va='center', fontsize=12, fontweight='bold')

plt.suptitle(f'block5_slice02_phase04  数据诊断图  (pairname={item["pairname"]}, phase_id={item["phase_id"]})',
             fontsize=14, y=0.995)
plt.tight_layout()
import os
os.makedirs('./logs/raw_check', exist_ok=True)
out = './logs/raw_check/inspect_block5_slice02_phase04.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f'Saved: {out}')