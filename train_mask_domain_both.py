"""
train_mask_domain_both.py - 双域适应配准训练（心脏 + 肝脏）

核心思路：
    - 双域训练：XCAT 心脏 + SEY 肝脏 交替训练
    - 互补掩码 + 一致性损失：
        - 掩码A ∪ 掩码B = 全集，掩码A ∩ 掩码B = ∅
        - 强制网络从"部分A"和"部分B"都能推断完整结构
        - L_mask_consist = ||disp_A - disp_B||₁（前景区域）
    - 掩码率设计：浅层 15%, 30% / 深层 45%, 55%（块状掩码）
    - 域对抗：GRL + DANN，迫使特征变成域不变的
    - 自监督：NCC + MSE_latent，无 GT DVF

特征使用策略：
    - 配准损失（NCC + MSE_latent）：使用原始完整特征 → 信号强、训练稳
    - 一致性损失：使用掩码A vs 掩码B 的变形场
    - 域对抗：使用掩码A+B 融合特征

Loss = NCC(心脏) + NCC(肝脏)
     + MSE_latent(心脏) + MSE_latent(肝脏)
     + smooth(A+B) + bending(A+B) + jac_det(A+B)
     + 0.5 * L_mask_consist                    # 掩码一致性
     + λ * L_domain_adv                        # 域对抗
"""
import os
import glob
import json
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from utils.utils import jacobian_determinant_vxm
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv

from utils.utils import Dataset_epoch_with_name
from omegaconf import OmegaConf

import TransModels.LDMMorph as LDMMorph
import TransModels.DomainDiscriminator as DANN
from ldm.util import instantiate_from_config, default
from ldm.data.xcat_npz import XCATNPZRegistration
from ldm.models.diffusion.ddim import DDIMSampler

# ===================== 命令行参数 =====================
parser = ArgumentParser()
parser.add_argument("--resume", type=str,
                    dest="resume", default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-30T21-02-35_xcat-motion-ldm/checkpoints/last.ckpt',
                    help="pretrained model")
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--bs", type=int, dest="bs", default=1)
parser.add_argument("--iteration", type=int, dest="iteration", default=24001)
parser.add_argument("--smth_labda", type=float, dest="smth_labda", default=0.4)
parser.add_argument("--checkpoint", type=int, dest="checkpoint", default=5000)
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data')
parser.add_argument("--beta", type=float, dest="beta", default=0.8)

# 双域数据路径
parser.add_argument("--xcat_path", type=str,
                    dest="xcat_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="XCAT heart data root")
parser.add_argument("--sey_path", type=str,
                    dest="sey_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep',
                    help="SEY liver data root (contains train/, val/, test/)")

# 域适应参数
parser.add_argument("--w_domain", type=float, dest="w_domain", default=0.1,
                    help="Domain adversarial loss weight")
parser.add_argument("--grl_warmup_iters", type=int, dest="grl_warmup_iters", default=2000,
                    help="GRL warmup iterations")

# 掩码策略
parser.add_argument("--mask_ratio", type=float, dest="mask_ratio", default=0.5,
                    help="Feature mask ratio (0.0-1.0), default 0.5 means 50%% features masked")

# LDM 配置
parser.add_argument("--ldm_config", type=str,
                    dest="ldm_config",
                    default=None,
                    help="LDM config file path")
parser.add_argument("--no_ldm", action="store_true", dest="no_ldm")

# 正则化
parser.add_argument("--bending_w", type=float, dest="bending_w", default=0.0)
parser.add_argument("--jacdet_w", type=float, dest="jacdet_w", default=0.0)

# loss type
parser.add_argument("--loss_type", type=str, default='ncc', choices=['ncc', 'mse'])

opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
beta = opt.beta
t_enc = 1
w_domain = opt.w_domain
grl_warmup = opt.grl_warmup_iters
mask_ratio = opt.mask_ratio

opt, unknown = parser.parse_known_args()
ckpt = None
if opt.ldm_config:
    configs = [opt.ldm_config]
else:
    configs = ['/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_no_motion.yaml']
opt.ldm = configs
print(f"LDM resume: {opt.resume}")


# ===================== LDM 加载 =====================
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_lgm():
    configs_list = [OmegaConf.load(cfg) for cfg in opt.ldm]
    cli = OmegaConf.from_dotlist(unknown)
    configs = OmegaConf.merge(*configs_list, cli)
    if opt.resume:
        pl_sd = torch.load(opt.resume, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}
    model = load_model_from_config(configs.model, pl_sd["state_dict"])
    print(f"LDM loaded from {opt.resume}")
    return model


# ===================== 损失函数 =====================
def ncc_loss(fixed, moving, win_size=15, mask=None):
    """Local Normalized Cross-Correlation loss."""
    assert fixed.shape == moving.shape
    b, c, h, w = fixed.shape
    pad = win_size // 2
    fixed_pad = F.pad(fixed, [pad, pad, pad, pad], mode='reflect')
    moving_pad = F.pad(moving, [pad, pad, pad, pad], mode='reflect')
    patches_fix = fixed_pad.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    patches_mov = moving_pad.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    mean_fix = patches_fix.mean(dim=-1)
    mean_mov = patches_mov.mean(dim=-1)
    centered_fix = patches_fix - mean_fix.unsqueeze(-1)
    centered_mov = patches_mov - mean_mov.unsqueeze(-1)
    var_fix = (centered_fix ** 2).mean(dim=-1)
    var_mov = (centered_mov ** 2).mean(dim=-1)
    cross = (centered_fix * centered_mov).mean(dim=-1)
    eps = 1e-8
    ncc = cross / (torch.sqrt(var_fix.clamp(min=eps)) * torch.sqrt(var_mov.clamp(min=eps)) + eps)
    if mask is not None:
        m = mask.to(ncc.dtype)
        denom = m.sum().clamp(min=1.0)
        ncc_mean = (ncc * m).sum() / denom
    else:
        ncc_mean = ncc.mean()
    return 1.0 - ncc_mean

def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom

def bending_energy_loss(y_pred):
    """二阶弯曲能量正则."""
    h2, w2 = y_pred.shape[-2:]
    dy = (y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dx = (y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    dyy = (dy[:, :, 1:, :] - dy[:, :, :-1, :]) / 2 * h2
    dxx = (dx[:, :, :, 1:] - dx[:, :, :, :-1]) / 2 * w2
    dxy = (dx[:, :, 1:, :] - dx[:, :, :-1, :]) / 2 * h2
    return (torch.mean(dyy * dyy) + torch.mean(dxx * dxx) + 2.0 * torch.mean(dxy * dxy)) / 4.0

def jacobian_neg_loss(y_pred):
    """Jacobian 负值惩罚."""
    h2, w2 = y_pred.shape[-2:]
    disp = torch.stack([y_pred[:, 0] * h2 / 2.0, y_pred[:, 1] * w2 / 2.0], dim=1)
    dfx_dy = disp[:, 0, 1:, :] - disp[:, 0, :-1, :]
    dfx_dx = disp[:, 0, :, 1:] - disp[:, 0, :, :-1]
    dfy_dy = disp[:, 1, 1:, :] - disp[:, 1, :-1, :]
    dfy_dx = disp[:, 1, :, 1:] - disp[:, 1, :, :-1]
    dfx_dy = dfx_dy[:, :, :-1]
    dfy_dy = dfy_dy[:, :, :-1]
    dfx_dx = dfx_dx[:, :-1, :]
    dfy_dx = dfy_dx[:, :-1, :]
    j11 = 1.0 + dfx_dy
    j12 = dfx_dx
    j21 = dfy_dy
    j22 = 1.0 + dfy_dx
    detJ = j11 * j22 - j12 * j21
    return torch.relu(-detJ).mean()

def body_mask(img_tensor, auto_percentile=False, pct=30):
    """前景/人体 mask（percentile-based）。

    auto_percentile=True 时：
      - XCAT（dark border）：自动用 p15 → ~90% body 覆盖亮器官区域
      - SEY（bright border）：自动用 p30 → ~70% body 覆盖肝脏区域

    auto_percentile=False 时：用固定 pct。
    """
    from scipy.ndimage import binary_fill_holes, label
    arr = img_tensor.detach().cpu().numpy()
    b, c, h, w = arr.shape
    out = np.zeros((b, c, h, w), dtype=np.float32)
    for bi in range(b):
        img = arr[bi, 0]
        border_vals = [img[h // 2, 0], img[h // 2, w - 1],
                       img[0, w // 2], img[h - 1, w // 2]]
        border_mean = np.mean(border_vals)
        if auto_percentile:
            pct_used = 15 if border_mean < 0.08 else 30
        else:
            pct_used = pct
        thr = np.percentile(img, pct_used)
        m = img > thr
        m = binary_fill_holes(m)
        lab, n = label(m)
        if n > 0:
            sizes = [(np.sum(lab == i), i) for i in range(1, n + 1)]
            sizes.sort(reverse=True)
            m = (lab == sizes[0][1])
        out[bi, 0] = m.astype(np.float32)
    return torch.from_numpy(out).to(img_tensor.device)

def jac_stats(D_f_xy):
    """Jacobian 统计."""
    dvf = D_f_xy[0].detach().cpu().numpy()
    _, h, w = dvf.shape
    dvf_px = dvf.copy()
    dvf_px[0] = dvf_px[0] * h / 2.0
    dvf_px[1] = dvf_px[1] * w / 2.0
    jd = jacobian_determinant_vxm(dvf_px)
    n_fold = int(np.sum(jd < 0))
    neg_ratio = float(n_fold / jd.size)
    return neg_ratio, float(jd.min()), float(jd.max()), n_fold


def dvf_stats(dvf):
    """dvf: [B,2,H,W] or [2,H,W]"""
    dv = dvf.detach().cpu().numpy()

    if dv.ndim == 4:
        dv = dv[0]   # [2,H,W]

    ux, vy = dv[0], dv[1]
    disp_mag = np.sqrt(ux**2 + vy**2)

    _, h, w = dv.shape
    phys_scale = max(h, w) / 2.0

    return {
        'dvf_x_mean': float(ux.mean()),
        'dvf_x_std': float(ux.std()),
        'dvf_x_max': float(np.abs(ux).max()),
        'dvf_y_mean': float(vy.mean()),
        'dvf_y_std': float(vy.std()),
        'dvf_y_max': float(np.abs(vy).max()),
        'disp_total_max': float(disp_mag.max()),
        'disp_phys_max': float(disp_mag.max() * phys_scale),
    }


# ===================== SEY 数据加载（归一化） =====================
class SEYRegistration(Data.Dataset):
    """SEY 肝脏数据集，加载 fixed/moving 图像对."""
    def __init__(self, data_root, split='train', normalize=True):
        super().__init__()
        self.split = split
        self.normalize = normalize
        split_dir = os.path.join(data_root, split)
        self.files = natsorted(glob.glob(os.path.join(split_dir, '*.npz')))
        print(f"  [SEY {split}] loaded {len(self.files)} files from {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        arr = np.load(self.files[index])
        mov = arr["img_small"].astype(np.float32)
        fix = arr["img_large"].astype(np.float32)

        # 归一化：仅在数据未归一化时启用；SEY npz 来自 preprocess_sey.py（已 joint_norm）
        if self.normalize:
            minv, maxv = fix.min(), fix.max()
            if (maxv - minv) > 1e-6:
                fix = (fix - minv) / (maxv - minv)
                mov = (mov - minv) / (maxv - minv)

        # 随机水平翻转
        if self.split == 'train' and np.random.rand() > 0.5:
            fix = np.flip(fix, axis=-1).copy()
            mov = np.flip(mov, axis=-1).copy()

        # 返回格式：[1, H, W]
        mov = torch.from_numpy(mov).unsqueeze(0)
        fix = torch.from_numpy(fix).unsqueeze(0)
        movlab = torch.zeros_like(mov)
        tarlab = torch.zeros_like(fix)
        name = os.path.basename(self.files[index])
        return mov, fix, movlab, tarlab, name


# ===================== 双域互补掩码策略 =====================
def dual_complementary_mask(score_list, seed, mask_ratio=0.5):
    """
    生成两个互补的块状掩码

    核心思想：
        - mask_A ∪ mask_B = 全集（合起来包含完整信息）
        - mask_A ∩ mask_B = ∅（互斥）
        - 各自保留约 (1-mask_ratio) 的信息

    物理意义：强制网络从"部分A"和"部分B"都能推断完整结构

    参数:
        score_list: [score0, score1, score2, score3]
        seed: 随机种子
        mask_ratio: 掩码比例（0-1），0=不用掩码，1=全部掩码

    返回:
        mask_a_scores: 掩码A后的特征列表（保留 mask 位置）
        mask_b_scores: 掩码B后的特征列表（保留 1-mask 位置，即互补位置）
    """
    if mask_ratio <= 0:
        return score_list, score_list

    # 掩码率随深度递增：浅层少掩码，深层多掩码
    mask_ratios = [mask_ratio * r for r in [0.3, 0.6, 0.9, 1.2]]
    mask_ratios = [min(r, 0.7) for r in mask_ratios]  # 上限 70%

    mask_a_list = []
    mask_b_list = []

    for idx, score in enumerate(score_list):
        b, c, h, w = score.shape
        ratio = mask_ratios[idx]

        # 块大小随深度增加：浅层小块(2x2)，深层大块(16x16)
        patch_size = 2 ** (idx + 1)
        ph, pw = h // patch_size, w // patch_size
        if ph < 1: ph = 1
        if pw < 1: pw = 1

        # 用给定种子在 patch 级别生成掩码
        gen = torch.Generator(device=score.device)
        gen.manual_seed(seed + idx)
        patch_mask = torch.rand(b, 1, ph, pw, device=score.device, generator=gen) > ratio
        mask = F.interpolate(patch_mask.float(), size=(h, w), mode='nearest')

        # mask_A = 保留 mask 为 True 的位置
        # mask_B = 保留 mask 为 False 的位置（互补）
        mask_a = mask
        mask_b = 1 - mask

        mask_a_list.append(score * mask_a)
        mask_b_list.append(score * mask_b)

    return mask_a_list, mask_b_list


def mask_consistency_loss(disp_a, disp_b, fg_mask=None):
    """
    掩码一致性损失：两种掩码版本产生的变形场应该一致

    物理意义：不管看到部分A还是部分B，网络都应该输出相同的变形场

    参数:
        disp_a: 掩码A产生的变形场 [B, 2, H, W]
        disp_b: 掩码B产生的变形场 [B, 2, H, W]
        fg_mask: 前景掩码 [B, 1, H, W] 或 None

    返回:
        L1 正则化的一致性损失
    """
    # 计算变形场差异
    diff = torch.abs(disp_a - disp_b)

    if fg_mask is not None:
        # 只在前景区域计算损失
        if fg_mask.dim() == 3:
            fg_mask = fg_mask.unsqueeze(1)
        if fg_mask.shape[1] != disp_a.shape[1]:
            fg_mask = fg_mask.expand_as(diff)
        loss = (diff * fg_mask).sum()
        denom = fg_mask.sum() + 1e-6
        return loss / denom
    else:
        return diff.mean()


# ===================== GRL Lambda Schedule =====================
def get_grl_lambda(step):
    return min(1.0, step / grl_warmup)


# ===================== LDM 特征提取 =====================
def get_ldm_scores_pair(ldm_model, img_mov, img_fix, t_enc):
    """提取配对的 LDM 特征."""
    if opt.no_ldm:
        return None, None, None, None
    mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(img_mov)).detach()
    fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(img_fix)).detach()
    noise = torch.randn_like(mov_z)
    x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
    y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
    outx = ldm_model.apply_model(x_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
    outy = ldm_model.apply_model(y_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
    score0 = torch.cat((outx[1][0][0], outx[1][0][2], outy[1][0][0], outy[1][0][2]), dim=1)
    score1 = torch.cat((outx[1][0][3], outx[1][0][5], outy[1][0][3], outy[1][0][5]), dim=1)
    score2 = torch.cat((outx[1][0][6], outx[1][0][8], outy[1][0][6], outy[1][0][8]), dim=1)
    score3 = torch.cat((outx[1][0][9], outx[1][0][11], outy[1][0][9], outy[1][0][11]), dim=1)
    return score0, score1, score2, score3


# ===================== 保存检查点 =====================
def save_checkpoint(state, save_dir, save_filename, max_model_num=50):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))


# ===================== 可视化 =====================
def vis_sample(step, mov, fix, warped, disp, domain_tag, ncc_before, ncc_after, save_path, fg_mask=None):
    mov_np = mov[0, 0].cpu().numpy()
    fix_np = fix[0, 0].cpu().numpy()
    warp_np = warped[0, 0].cpu().numpy()
    D_disp = disp[0].cpu().numpy()
    D_disp_px = D_disp.copy()
    _, h, w = D_disp_px.shape
    D_disp_px[0] = D_disp_px[0] * h / 2.0
    D_disp_px[1] = D_disp_px[1] * w / 2.0
    jac = jacobian_determinant_vxm(D_disp_px)
    n_fold = int((jac < 0).sum())
    min_jac = float(jac.min())
    diff_before = np.abs(mov_np - fix_np)
    diff_after = np.abs(warp_np - fix_np)
    dvf_mag = np.sqrt(D_disp[0]**2 + D_disp[1]**2)

    # 如果有 mask，也转为 numpy
    mask_np = fg_mask[0, 0].cpu().numpy() if fg_mask is not None else None

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    imgs = [mov_np, fix_np, warp_np, diff_before,
            diff_after, jac, dvf_mag,
            np.abs(mov_np - fix_np) - np.abs(warp_np - fix_np)]
    titles = [
        f'Moving [{domain_tag}]',
        'Fixed (Ref)',
        f'Warped NCC: {ncc_before:.4f}->{ncc_after:.4f}',
        'Abs Diff Before',
        'Abs Diff After',
        f'Jac Det [folds={n_fold}, min={min_jac:.3f}]',
        'DVF Magnitude',
        'Diff Reduction'
    ]
    for ax, img, title in zip(axes.flat, imgs, titles):
        if 'Jac' in title or 'DVF' in title:
            ax.imshow(img, cmap='RdBu', vmin=-0.5, vmax=1.5)
        elif 'Reduction' in title:
            ax.imshow(img, cmap='hot', vmin=0, vmax=None)
        else:
            vmax = max(mov_np.max(), fix_np.max(), warp_np.max())
            ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    fig.suptitle(f'[Step {step}] {domain_tag} | NCC: {ncc_before:.4f}->{ncc_after:.4f}', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # 单独保存 mask 可视化（如果有）
    if mask_np is not None:
        mask_save = save_path.replace('.png', '_mask.png')
        fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
        ax2[0].imshow(fix_np, cmap='gray', vmin=0, vmax=fix_np.max())
        ax2[0].set_title('Fixed Image', fontsize=10)
        ax2[0].axis('off')

        ax2[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        ax2[1].set_title(f'Foreground Mask (area={mask_np.sum():.0f}px)', fontsize=10)
        ax2[1].axis('off')

        # 叠加显示
        overlay = np.zeros((*mask_np.shape, 3))
        overlay[:, :, 0] = fix_np / (fix_np.max() + 1e-8)  # R: 原图
        overlay[:, :, 2] = mask_np  # B: mask
        ax2[2].imshow(overlay)
        ax2[2].set_title('Overlay (R=Image, B=Mask)', fontsize=10)
        ax2[2].axis('off')

        fig2.suptitle(f'[Step {step}] {domain_tag} Foreground Mask Visualization', fontsize=12)
        plt.tight_layout()
        plt.savefig(mask_save, dpi=100, bbox_inches='tight')
        plt.close(fig2)


# ===================== 训练函数 =====================
def train():
    print("=" * 60)
    print("双域适应配准训练：心脏(XCAT) + 肝脏(SEY)")
    print("=" * 60)

    # 加载 LDM
    ldm_model = load_lgm()

    # 数据加载器
    print("\n[Data] 加载 XCAT 心脏数据...")
    xcat_train = XCATNPZRegistration(data_root=opt.xcat_path, split='train', flip_p=0.5, normalize=False)
    xcat_val   = XCATNPZRegistration(data_root=opt.xcat_path, split='val',   flip_p=0.0, normalize=False)
    xcat_test  = XCATNPZRegistration(data_root=opt.xcat_path, split='test',  flip_p=0.0, normalize=False)
    xcat_train_loader = Data.DataLoader(xcat_train, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    xcat_val_loader = Data.DataLoader(xcat_val, batch_size=bs, shuffle=False, num_workers=0)
    xcat_test_loader = Data.DataLoader(xcat_test, batch_size=bs, shuffle=False, num_workers=0)
    print(f"  XCAT: train={len(xcat_train)}, val={len(xcat_val)}, test={len(xcat_test)}")

    print("\n[Data] 加载 SEY 肝脏数据...")
    sey_train = SEYRegistration(data_root=opt.sey_path, split='train', normalize=False)
    sey_val = SEYRegistration(data_root=opt.sey_path, split='val', normalize=False)
    sey_test = SEYRegistration(data_root=opt.sey_path, split='test', normalize=False)
    sey_train_loader = Data.DataLoader(sey_train, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    sey_val_loader = Data.DataLoader(sey_val, batch_size=bs, shuffle=False, num_workers=0)
    sey_test_loader = Data.DataLoader(sey_test, batch_size=bs, shuffle=False, num_workers=0)
    print(f"  SEY: train={len(sey_train)}, val={len(sey_val)}, test={len(sey_test)}")

    # 模型
    model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2, use_ldm=not opt.no_ldm)
    model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print(f"\n[Model] 配准网络参数: {total/1e6:.2f}M")
    #改成swin_8_feature
    domain_disc = DANN.DomainAdversarialModule(in_channels=128)
    domain_disc.cuda()
    total_dom = sum([param.nelement() for param in domain_disc.parameters()])
    print(f"[Model] 域判别器参数: {total_dom/1e6:.2f}M")

    transform = SpatialTransform().cuda()
    for param in transform.parameters():
        param.requires_grad = False

    optimizer_reg = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_dom = torch.optim.Adam(domain_disc.parameters(), lr=lr * 0.5)

    # 保存目录
    model_dir = f'./logs/DA_Both_XCAT_SEY_wd{w_domain}_smooth{smooth}_mask{mask_ratio}_grl{grl_warmup}_702/'
    csv_name = model_dir + 'training_log.csv'
    os.makedirs(model_dir, exist_ok=True)

    # CSV
    f = open(csv_name, 'w')
    with f:
        fnames = ['Step', 'Loss_Total', 'NCC_XCAT', 'NCC_SEY', 'Smooth_XCAT', 'Smooth_SEY',
                  'Bending_XCAT', 'Bending_SEY', 'Jac_XCAT', 'Jac_SEY',
                  'Loss_Domain', 'GRL_Lambda', 'Disp_XCAT', 'Disp_SEY',
                  'NCC_Val_XCAT_Bef', 'NCC_Val_XCAT_Aft', 'NCC_Val_SEY_Bef', 'NCC_Val_SEY_Aft',
                  'DVF_XCAT_Max', 'DVF_XCAT_PhysMax', 'DVF_SEY_Max', 'DVF_SEY_PhysMax',
                  'DVF_XCAT_StdX', 'DVF_XCAT_StdY', 'DVF_SEY_StdX', 'DVF_SEY_StdY']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    f.close()

    # 数据迭代器
    xcat_iter = iter(xcat_train_loader)
    sey_iter = iter(sey_train_loader)

    lossall = np.zeros((3, iteration+1))
    step = 1

    print(f"\n[Training] 掩码比例: {mask_ratio*100:.0f}% | 域对抗权重: {w_domain} | GRL 热身: {grl_warmup} 步")
    print(f"[Training] 正则化: smooth={smooth}, bending_w={opt.bending_w}, jacdet_w={opt.jacdet_w}")
    print(f"[Training] 保存目录: {model_dir}")
    print("=" * 60)

    while step <= iteration:
        model.train()
        domain_disc.train()

        # 获取 XCAT 心脏批次
        try:
            x_xcat, y_xcat, _, _, _ = next(xcat_iter)
        except StopIteration:
            xcat_iter = iter(xcat_train_loader)
            x_xcat, y_xcat, _, _, _ = next(xcat_iter)
        x_xcat = x_xcat.cuda().float()
        y_xcat = y_xcat.cuda().float()

        # 获取 SEY 肝脏批次
        try:
            x_sey, y_sey, _, _, _ = next(sey_iter)
        except StopIteration:
            sey_iter = iter(sey_train_loader)
            x_sey, y_sey, _, _, _ = next(sey_iter)
        x_sey = x_sey.cuda().float()
        y_sey = y_sey.cuda().float()

        # GRL lambda
        lam = get_grl_lambda(step)
        domain_disc.set_lambda(lam)

        # 前景 mask
        fg_xcat = body_mask(y_xcat)
        fg_sey = body_mask(y_sey)

        # LDM 特征
        s0_x, s1_x, s2_x, s3_x = get_ldm_scores_pair(ldm_model, x_xcat, y_xcat, t_enc)
        s0_s, s1_s, s2_s, s3_s = get_ldm_scores_pair(ldm_model, x_sey, y_sey, t_enc)

        # ========== 互补掩码：生成 A、B 两个版本 ==========
        # 源域（XCAT 心脏）
        if mask_ratio > 0:
            scores_xcat_a, scores_xcat_b = dual_complementary_mask(
                [s0_x, s1_x, s2_x, s3_x], seed=step,
                mask_ratio=mask_ratio
            )
            # 目标域（SEY 肝脏）
            scores_sey_a, scores_sey_b = dual_complementary_mask(
                [s0_s, s1_s, s2_s, s3_s], seed=step + 1000000,
                mask_ratio=mask_ratio
            )
        else:
            # 不使用掩码时，A、B 版本相同
            scores_xcat_a, scores_xcat_b = [s0_x, s1_x, s2_x, s3_x], [s0_x, s1_x, s2_x, s3_x]
            scores_sey_a, scores_sey_b = [s0_s, s1_s, s2_s, s3_s], [s0_s, s1_s, s2_s, s3_s]

        # ========== 前向：完整特征（用于配准损失） ==========
        # XCAT 心脏 - 原始特征
        disp_xcat_full, feat_xcat_full = model(x_xcat, y_xcat, s0_x, s1_x, s2_x, s3_x)
        _, warped_xcat_full = transform(x_xcat, disp_xcat_full.permute(0, 2, 3, 1))

        # SEY 肝脏 - 原始特征
        disp_sey_full, feat_sey_full = model(x_sey, y_sey, s0_s, s1_s, s2_s, s3_s)
        _, warped_sey_full = transform(x_sey, disp_sey_full.permute(0, 2, 3, 1))

        # ========== 前向：掩码A 版本（用于一致性损失 + 域对抗） ==========
        # XCAT 心脏 - 掩码A
        disp_xcat_a, feat_xcat_a = model(x_xcat, y_xcat, *scores_xcat_a)
        _, warped_xcat_a = transform(x_xcat, disp_xcat_a.permute(0, 2, 3, 1))

        # SEY 肝脏 - 掩码A
        disp_sey_a, feat_sey_a = model(x_sey, y_sey, *scores_sey_a)
        _, warped_sey_a = transform(x_sey, disp_sey_a.permute(0, 2, 3, 1))

        # ========== 前向：掩码B 版本 ==========
        # XCAT 心脏 - 掩码B
        disp_xcat_b, feat_xcat_b = model(x_xcat, y_xcat, *scores_xcat_b)
        _, warped_xcat_b = transform(x_xcat, disp_xcat_b.permute(0, 2, 3, 1))

        # SEY 肝脏 - 掩码B
        disp_sey_b, feat_sey_b = model(x_sey, y_sey, *scores_sey_b)
        _, warped_sey_b = transform(x_sey, disp_sey_b.permute(0, 2, 3, 1))

        # ========== 掩码一致性损失 ==========
        # 物理意义：不管看到部分A还是部分B，网络都应该输出相同的变形场
        loss_mask_consist = 0.0
        if mask_ratio > 0:
            loss_mask_consist += mask_consistency_loss(disp_xcat_a, disp_xcat_b, fg_xcat)
            loss_mask_consist += mask_consistency_loss(disp_sey_a, disp_sey_b, fg_sey)

        # ========== 潜空间 MSE Loss（使用原始特征） ==========
        # XCAT latent（使用完整特征的变形结果）
        mov_z_xcat = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(x_xcat)).detach()
        fix_z_xcat = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(y_xcat)).detach()
        warped_z_xcat = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(warped_xcat_full))#去掉detach
        mse_latent_xcat = ((warped_z_xcat - fix_z_xcat) ** 2).mean()

        # SEY latent（使用完整特征的变形结果）
        mov_z_sey = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(x_sey)).detach()
        fix_z_sey = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(y_sey)).detach()
        warped_z_sey = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(warped_sey_full))#去掉detach
        mse_latent_sey = ((warped_z_sey - fix_z_sey) ** 2).mean()

        # 图像域损失（使用原始特征的变形结果）
        if opt.loss_type == 'ncc':
            loss_image_xcat = ncc_loss(warped_xcat_full, y_xcat, mask=fg_xcat)
            loss_image_sey = ncc_loss(warped_sey_full, y_sey, mask=fg_sey)
        else:
            loss_image_xcat = masked_mse(warped_xcat_full, y_xcat, fg_xcat)
            loss_image_sey = masked_mse(warped_sey_full, y_sey, fg_sey)

        # 平滑正则化（A、B 两个版本的平均）
        sm_xcat = smoothloss(disp_xcat_full)
        sm_sey = smoothloss(disp_sey_full)

        # 二阶弯曲能量（A、B 两个版本的平均）
        bend_xcat = torch.tensor(0.0, device=disp_xcat_a.device)
        bend_sey = torch.tensor(0.0, device=disp_sey_a.device)
        if opt.bending_w > 0:
            bend_xcat = bending_energy_loss(disp_xcat_full)
            bend_sey = bending_energy_loss(disp_sey_full)

        # Jacobian 负值惩罚（A、B 两个版本的平均）
        jac_xcat = torch.tensor(0.0, device=disp_xcat_a.device)
        jac_sey = torch.tensor(0.0, device=disp_sey_a.device)
        if opt.jacdet_w > 0:
            #直接约束心脏和肝脏的变形场
            jac_xcat = jacobian_neg_loss(disp_xcat_full)
            jac_sey = jacobian_neg_loss(disp_sey_full)

        # ========== 配准损失 ==========
        # beta*NCC + (1-beta)*MSE_latent + smooth
        loss1_xcat = beta * loss_image_xcat + (1 - beta) * mse_latent_xcat
        loss1_sey = beta * loss_image_sey + (1 - beta) * mse_latent_sey

        loss_reg_xcat = loss1_xcat + smooth * sm_xcat
        loss_reg_sey = loss1_sey + smooth * sm_sey

        loss_reg = loss_reg_xcat + loss_reg_sey + \
                   opt.bending_w * (bend_xcat + bend_sey) + \
                   opt.jacdet_w * (jac_xcat + jac_sey)

        # ===================== 域对抗训练 =====================
        # 融合 A、B 两个版本的特征（取平均）
        feat_xcat = (feat_xcat_a + feat_xcat_b) / 2
        feat_sey = (feat_sey_a + feat_sey_b) / 2

        # Step 1: 判别器更新（正常梯度）
        optimizer_dom.zero_grad()
        feat_xcat_d = feat_xcat.detach()
        feat_xcat_d.requires_grad_(True)
        feat_sey_d = feat_sey.detach()
        feat_sey_d.requires_grad_(True)
        logits_xcat_d = domain_disc.disc_forward(feat_xcat_d)
        logits_sey_d = domain_disc.disc_forward(feat_sey_d)
        d_src = torch.zeros_like(logits_xcat_d)
        d_tgt = torch.ones_like(logits_sey_d)
        loss_dom_disc = (F.binary_cross_entropy_with_logits(logits_xcat_d, d_src) +
                         F.binary_cross_entropy_with_logits(logits_sey_d, d_tgt)) / 2
        loss_dom_disc.backward()
        optimizer_dom.step()

        # Step 2: 配准网络更新（GRL 反转梯度）
        optimizer_reg.zero_grad()
        logits_xcat_r = domain_disc.reg_forward(feat_xcat)
        logits_sey_r = domain_disc.reg_forward(feat_sey)
        loss_dom_reg = (F.binary_cross_entropy_with_logits(logits_xcat_r, d_src) +
                        F.binary_cross_entropy_with_logits(logits_sey_r, d_tgt)) / 2

        # 掩码一致性损失权重
        w_mask_consist = 0.05  # 可调参数

        loss_total = loss_reg + w_domain * loss_dom_reg + w_mask_consist * loss_mask_consist
        loss_total.backward()
        optimizer_reg.step()

        # DVF 范围监控
        dvf_xcat = dvf_stats(disp_xcat_full)
        dvf_sey = dvf_stats(disp_sey_full)
        neg_xcat, minj_xcat, maxj_xcat, folds_xcat = jac_stats(disp_xcat_full)
        neg_sey,  minj_sey,  maxj_sey,  folds_sey  = jac_stats(disp_sey_full)

        # 记录
        lossall[:, step] = np.array([loss_total.item(), loss_reg.item(), loss_dom_disc.item()])

        sys.stdout.write(
            "\r[Step {0}] L={1:.4f} IMG_XCAT={2:.4f} IMG_SEY={3:.4f} "
            "Sm_XCAT={4:.4f} Sm_SEY={5:.4f} L_dom={6:.4f} L_mask={7:.4f} lambda={8:.3f} "
            "DVF_XCAT={9:.2f}px DVF_SEY={10:.2f}px "
            "Jac_XCAT negR={11:.3f}% minJ={12:.3f} maxJ={13:.3f} "
            "Jac_SEY negR={14:.3f}% minJ={15:.3f} maxJ={16:.3f}".format(
                step, loss_total.item(), loss_image_xcat.item(), loss_image_sey.item(),
                sm_xcat.item(), sm_sey.item(), loss_dom_disc.item(), loss_mask_consist.item(), lam,
                dvf_xcat['disp_phys_max'], dvf_sey['disp_phys_max'],
                neg_xcat * 100, minj_xcat, maxj_xcat,
                neg_sey * 100, minj_sey, maxj_sey
            )
)
        sys.stdout.flush()

        # 可视化（使用原始特征的变形结果）
        if step % 50 == 0:
            with torch.no_grad():
                ncc_bef_xcat = 1 - ncc_loss(y_xcat, x_xcat, mask=fg_xcat).item()
                ncc_aft_xcat = 1 - ncc_loss(y_xcat, warped_xcat_full, mask=fg_xcat).item()
                ncc_bef_sey = 1 - ncc_loss(y_sey, x_sey, mask=fg_sey).item()
                ncc_aft_sey = 1 - ncc_loss(y_sey, warped_sey_full, mask=fg_sey).item()

                vis_sample(step, x_xcat, y_xcat, warped_xcat_full, disp_xcat_full, 'XCAT',
                           ncc_bef_xcat, ncc_aft_xcat,
                           f'{model_dir}vis_xcat_{step:06d}.png', fg_mask=fg_xcat)
                vis_sample(step, x_sey, y_sey, warped_sey_full, disp_sey_full, 'SEY',
                           ncc_bef_sey, ncc_aft_sey,
                           f'{model_dir}vis_sey_{step:06d}.png', fg_mask=fg_sey)
                print(f"\n  [Visualization] 已保存")

        # 验证
        if step % n_checkpoint == 0:
            model.eval()
            domain_disc.eval()
            with torch.no_grad():
                # XCAT 验证
                ncc_xcat_bef = []
                ncc_xcat_aft = []
                for xv, yv, _, _, _ in xcat_val_loader:
                    xv, yv = xv.cuda().float(), yv.cuda().float()
                    fg_v = body_mask(yv)
                    ncc_b = 1 - ncc_loss(yv, xv, mask=fg_v).item()
                    s0, s1, s2, s3 = get_ldm_scores_pair(ldm_model, xv, yv, t_enc)
                    d, _ = model(xv, yv, s0, s1, s2, s3)
                    _, w = transform(xv, d.permute(0, 2, 3, 1))
                    ncc_a = 1 - ncc_loss(yv, w, mask=fg_v).item()
                    ncc_xcat_bef.append(ncc_b)
                    ncc_xcat_aft.append(ncc_a)

                # SEY 验证
                ncc_sey_bef = []
                ncc_sey_aft = []
                for xv, yv, _, _, _ in sey_val_loader:
                    xv, yv = xv.cuda().float(), yv.cuda().float()
                    fg_v = body_mask(yv)
                    ncc_b = 1 - ncc_loss(yv, xv, mask=fg_v).item()
                    s0, s1, s2, s3 = get_ldm_scores_pair(ldm_model, xv, yv, t_enc)
                    d, _ = model(xv, yv, s0, s1, s2, s3)
                    _, w = transform(xv, d.permute(0, 2, 3, 1))
                    ncc_a = 1 - ncc_loss(yv, w, mask=fg_v).item()
                    ncc_sey_bef.append(ncc_b)
                    ncc_sey_aft.append(ncc_a)

                mean_xcat_bef = np.mean(ncc_xcat_bef)
                mean_xcat_aft = np.mean(ncc_xcat_aft)
                mean_sey_bef = np.mean(ncc_sey_bef)
                mean_sey_aft = np.mean(ncc_sey_aft)

            print(f"\n  [Val @ Step {step}]")
            print(f"    XCAT val  NCC: {mean_xcat_bef:.4f} -> {mean_xcat_aft:.4f} (Δ {mean_xcat_aft - mean_xcat_bef:+.4f})")
            print(f"    SEY  val  NCC: {mean_sey_bef:.4f} -> {mean_sey_aft:.4f} (Δ {mean_sey_aft - mean_sey_bef:+.4f})")
            print(f"    XCAT DVF   Max={dvf_xcat['disp_phys_max']:.2f}px  Std_X={dvf_xcat['dvf_x_std']:.4f}  Std_Y={dvf_xcat['dvf_y_std']:.4f}")
            print(f"    SEY  DVF   Max={dvf_sey['disp_phys_max']:.2f}px  Std_X={dvf_sey['dvf_x_std']:.4f}  Std_Y={dvf_sey['dvf_y_std']:.4f}")
            print(f"    GRL lambda: {lam:.3f}")

            # 保存检查点
            modelname = f'NCC_XCAT_{mean_xcat_aft:.4f}_Step_{step:06d}.pth'
            save_checkpoint(model.state_dict(), model_dir, modelname)
            np.save(model_dir + 'Loss.npy', lossall)

            # CSV
            f = open(csv_name, 'a')
            with f:
                writer = csv.writer(f)
                writer.writerow([step, loss_total.item(), mean_xcat_aft, mean_sey_aft,
                                 sm_xcat.item(), sm_sey.item(),
                                 opt.bending_w * bend_xcat.item(), opt.bending_w * bend_sey.item(),
                                 opt.jacdet_w * jac_xcat.item(), opt.jacdet_w * jac_sey.item(),
                                 loss_dom_disc.item(), lam,
                                 disp_xcat_full.abs().mean().item(), disp_sey_full.abs().mean().item(),
                                 mean_xcat_bef, mean_xcat_aft, mean_sey_bef, mean_sey_aft,
                                 dvf_xcat['disp_total_max'], dvf_xcat['disp_phys_max'],
                                 dvf_sey['disp_total_max'], dvf_sey['disp_phys_max'],
                                 dvf_xcat['dvf_x_std'], dvf_xcat['dvf_y_std'],
                                 dvf_sey['dvf_x_std'], dvf_sey['dvf_y_std']])
            f.close()

            model.train()
            domain_disc.train()

        step += 1
        if step > iteration:
            break
    print("=" * 60)
    print(f"训练完成！日志: {csv_name}")
    print(f"模型保存: {model_dir}")
    print("=" * 60)

    # 测试
    print("\n[测试] 在 XCAT 测试集上评估...")
    model.eval()
    with torch.no_grad():
        ncc_test_bef = []
        ncc_test_aft = []
        for xk, yk, _, _, _ in xcat_test_loader:
            xk, yk = xk.cuda().float(), yk.cuda().float()
            fg_k = body_mask(yk)
            ncc_b = 1 - ncc_loss(yk, xk, mask=fg_k).item()
            s0, s1, s2, s3 = get_ldm_scores_pair(ldm_model, xk, yk, t_enc)
            d, _ = model(xk, yk, s0, s1, s2, s3)
            _, w = transform(xk, d.permute(0, 2, 3, 1))
            ncc_a = 1 - ncc_loss(yk, w, mask=fg_k).item()
            ncc_test_bef.append(ncc_b)
            ncc_test_aft.append(ncc_a)
        print(f"  XCAT 测试 NCC: {np.mean(ncc_test_bef):.4f} -> {np.mean(ncc_test_aft):.4f} "
              f"(Δ {np.mean(ncc_test_aft) - np.mean(ncc_test_bef):+.4f})")

    print("\n[测试] 在 SEY 测试集上评估...")
    model.eval()
    with torch.no_grad():
        ncc_test_bef = []
        ncc_test_aft = []
        for xv, yv, _, _, _ in sey_test_loader:
            xv, yv = xv.cuda().float(), yv.cuda().float()
            fg_v = body_mask(yv)
            ncc_b = 1 - ncc_loss(yv, xv, mask=fg_v).item()
            s0, s1, s2, s3 = get_ldm_scores_pair(ldm_model, xv, yv, t_enc)
            d, _ = model(xv, yv, s0, s1, s2, s3)
            _, w = transform(xv, d.permute(0, 2, 3, 1))
            ncc_a = 1 - ncc_loss(yv, w, mask=fg_v).item()
            ncc_test_bef.append(ncc_b)
            ncc_test_aft.append(ncc_a)
        print(f"  SEY 测试 NCC: {np.mean(ncc_test_bef):.4f} -> {np.mean(ncc_test_aft):.4f} "
              f"(Δ {np.mean(ncc_test_aft) - np.mean(ncc_test_bef):+.4f})")


if __name__ == '__main__':
    train()
