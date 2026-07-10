"""
SEY 肝脏配准可视化（用于肝脏可视化测试）
- 数据集：与肝脏训练同源（ldm/data/sey_registration.py: SEYRegistration）
- 指标 / 绘图：完全复用 visualize_registration_xcat.py 的 4×5 布局逻辑
- 不计算 Dice；叠加 before/after 的 weighted_red_overlay 额外子图

用法:
  python visualize_registration_sey.py --resume <reg.pth> --sey_path <datasets/SEY/prep> --ldm_config <yaml>
"""
import os
import glob
import json
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import SpatialTransform, jacobian_determinant_vxm
from ldm.data.sey_registration import SEYRegistration
import TransModels.LDMMorph as LDMMorph
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


# ======================== 参数配置 ========================
parser = argparse.ArgumentParser()

# 配准网络 checkpoint
parser.add_argument("--resume", type=str, dest="resume", default='',
                    help="配准网络 checkpoint 路径")

# 训练脚本同源参数
parser.add_argument("--sey_path", type=str,
                    dest="sey_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep',
                    help="SEY 数据根目录（包含 train/val/test 子目录）")
parser.add_argument("--ldm_config", type=str, dest="ldm_config", default=None,
                    help="LDM 配置文件路径（默认 ./configs/latent-diffusion/sey-ldm-vq16-64ch.yaml）")
parser.add_argument("--ldm_checkpoint", type=str, dest="ldm_checkpoint", default=None,
                    help="LDM checkpoint 路径（默认自动查找）")
parser.add_argument("--smooth", type=float, default=0.4)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--t_enc", type=int, default=1)
parser.add_argument("--no_ldm", action="store_true", dest="no_ldm",
                    help="Disable LDM features (match --no_ldm from training)")

# 可视化范围
parser.add_argument("--n_samples", type=int, default=10,
                    help="可视化的样本数量")
parser.add_argument("--split", type=str, default='test',
                    choices=['train', 'val', 'test'],
                    help="使用哪个划分的可视化")
parser.add_argument("--start_idx", type=int, default=0,
                    help="从第几个样本开始（0-indexed）")
parser.add_argument("--save_dir", type=str, dest="save_dir", default=None,
                    help="输出保存目录（默认自动生成）")
opt, unknown = parser.parse_known_args()

# 默认 LDM 配置：与 SEY 训练同源
if opt.ldm_config is None:
    opt.ldm_config = './configs/latent-diffusion/sey-ldm-vq16-64ch.yaml'

# 默认保存目录
if opt.save_dir is None:
    ckpt_name = os.path.basename(opt.resume).replace('.pth', '').replace('.ckpt', '')
    opt.save_dir = f'./logs/visualization_sey_{ckpt_name}_{opt.split}_{opt.start_idx}_{opt.start_idx + opt.n_samples}/'

print(f"\n{'='*60}")
print(f"Mode: SEY Liver Registration Visualization (test only)")
print(f"Resume: {opt.resume}")
print(f"LDM Config: {opt.ldm_config}")
print(f"Data Path: {opt.sey_path}/{opt.split}/")
print(f"Save Dir: {opt.save_dir}")
print(f"{'='*60}\n")


# ======================== LDM 加载 ========================
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


configs_list = [OmegaConf.load(opt.ldm_config)]
cli = OmegaConf.from_dotlist(unknown)
configs = OmegaConf.merge(*configs_list, cli)

ldm_model = load_model_from_config(configs.model, {"state_dict": None})

if opt.ldm_checkpoint:
    ldm_path = opt.ldm_checkpoint
    pl_sd = torch.load(ldm_path, map_location="cpu")
    ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
    print(f"LDM loaded: {ldm_path}")
elif opt.resume and os.path.dirname(opt.resume):
    candidates = glob.glob(os.path.join(os.path.dirname(opt.resume), '..', '..', '..', 'checkpoints', 'last.ckpt'))
    if candidates:
        ldm_path = candidates[0]
        pl_sd = torch.load(ldm_path, map_location="cpu")
        ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
        print(f"LDM loaded: {ldm_path}")
    else:
        print("WARNING: No LDM checkpoint found, using randomly initialized LDM")
else:
    print("WARNING: No LDM checkpoint specified, using randomly initialized LDM")


# ======================== 配准网络加载 ========================
model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2, use_ldm=not opt.no_ldm).cuda()
ckpt_path = opt.resume
if os.path.isfile(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    print(f"Registration model loaded: {ckpt_path}")
else:
    print(f"WARNING: checkpoint not found: {ckpt_path}, using random init")

model.eval()
transform = SpatialTransform().cuda()
for param in transform.parameters():
    param.requires_grad = False


# ======================== 辅助函数：创建网格图像 ========================
def mk_grid_img(grid_step, line_thickness=1, grid_sz=(512, 512)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i + line_thickness - 1] = 1
    return grid_img


# ======================== 数据加载（与训练脚本同源） ========================
os.makedirs(opt.save_dir, exist_ok=True)
dataset = SEYRegistration(data_root=opt.sey_path, split=opt.split, normalize=False)
print(f"Dataset: SEY split={opt.split} size={len(dataset)} (loaded via SEYRegistration normalize=False, 与 train 同源：prep npz 已 joint_norm)")

start = opt.start_idx
if start >= len(dataset):
    print(f"  WARNING: start_idx={start} >= dataset size {len(dataset)}, clipping to 0")
    start = 0
end = min(start + opt.n_samples, len(dataset))
indices_to_vis = list(range(start, end))
print(f"Visualizing: index {start} to {end-1} ({len(indices_to_vis)} samples)")


# ======================== 指标函数（完全复用 XCAT 版） ========================
def body_mask(img_tensor, thr=0.05):
    """从图像生成前景/人体 mask，排除黑色背景。"""
    arr = img_tensor.detach().cpu().numpy()
    b, c, h, w = arr.shape
    out = np.zeros((b, c, h, w), dtype=bool)
    for bi in range(b):
        m = arr[bi, 0] > thr
        m = binary_fill_holes(m)
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = (lab == int(sizes.argmax()))
        out[bi, 0] = m
    return torch.from_numpy(out).to(img_tensor.device)


def ncc_metric(fixed, moving, win_size=15, mask=None):
    """计算滑动窗口局部 NCC。"""
    assert fixed.shape == moving.shape
    b, c, h, w = fixed.shape
    pad = win_size // 2
    fixed_pad = F.pad(fixed, [pad, pad, pad, pad], mode='reflect')
    moving_pad = F.pad(moving, [pad, pad, pad, pad], mode='reflect')
    patches_fix = fixed_pad.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    patches_mov = moving_pad.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    mean_fix = patches_fix.mean(dim=-1)
    mean_mov = patches_mov.mean(dim=-1)
    cf = patches_fix - mean_fix.unsqueeze(-1)
    cm = patches_mov - mean_mov.unsqueeze(-1)
    var_fix = (cf ** 2).mean(dim=-1)
    var_mov = (cm ** 2).mean(dim=-1)
    cross = (cf * cm).mean(dim=-1)
    eps = 1e-8
    ncc = cross / (torch.sqrt(var_fix.clamp(min=eps)) * torch.sqrt(var_mov.clamp(min=eps)) + eps)
    if mask is not None:
        m = mask.to(ncc.device).bool()
        if m.shape != ncc.shape:
            m = m.expand_as(ncc)
        return ncc[m].mean().item()
    return ncc.mean().item()


def ssim_metric(fixed, moving, window_size=11, size_average=True, L=1.0, mask=None):
    """计算 SSIM。"""
    from torch.nn.functional import avg_pool2d
    b, c, h, w = fixed.shape
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    mu1 = avg_pool2d(fixed, window_size, stride=1, padding=pad)
    mu2 = avg_pool2d(moving, window_size, stride=1, padding=pad)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = avg_pool2d(fixed.pow(2), window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = avg_pool2d(moving.pow(2), window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = avg_pool2d(fixed * moving, window_size, stride=1, padding=pad) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mask is not None:
        m = mask.to(ssim_map.device).bool()
        if m.shape != ssim_map.shape:
            m = m.expand_as(ssim_map)
        return ssim_map[m].mean().item()
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map


# ======================== 肝脏 ROI（SEY 肝脏在右腹部）=======================
DEFAULT_ROI = {
    'liver': (150, 250, 300, 570),   # x1, y1, x2, y2
}
print(f"[Mask Configuration]")
print(f"  Liver ROI: x=[{DEFAULT_ROI['liver'][0]}, {DEFAULT_ROI['liver'][2]}], y=[{DEFAULT_ROI['liver'][1]}, {DEFAULT_ROI['liver'][3]}]")
print(f"{'='*60}\n")


# ======================== 叠加图函数 ========================
def red_overlay(fixed, moving, alpha=0.6):
    """红色叠加视图（XCAT 同款）：背景=Fixed 灰度，前景=Moving 红色 alpha 混合。"""
    f_min, f_max = fixed.min(), fixed.max()
    f = (fixed - f_min) / (f_max - f_min + 1e-8)
    m = (moving - f_min) / (f_max - f_min + 1e-8)
    f = np.clip(f, 0, 1)
    m = np.clip(m, 0, 1)
    rgb = np.stack([f, f, f], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] * (1 - alpha) + m * alpha, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] * (1 - alpha), 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] * (1 - alpha), 0, 1)
    return rgb


def weighted_red_overlay(fixed, moving, moving_weight=1.4, fixed_weight=0.8):
    """亮度一致版红色叠加（旧 SEY 可视化风格）。"""
    f_min, f_max = fixed.min(), fixed.max()
    f = (fixed - f_min) / (f_max - f_min + 1e-8)
    m = (moving - f_min) / (f_max - f_min + 1e-8)
    f = np.clip(f, 0, 1)
    m = np.clip(m, 0, 1)
    rgb = np.zeros((*f.shape, 3))
    rgb[..., 0] = m * moving_weight
    rgb[..., 1] = f * fixed_weight
    rgb[..., 2] = f * fixed_weight
    return np.clip(rgb, 0, 1)


def _make_weighted_overlay_pair(fix_np, mov_np, warp_np):
    """生成一张 1x2 子图：before / after 的 weighted_red_overlay，方便嵌到一个 axes。"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(weighted_red_overlay(fix_np, mov_np))
    ax[0].set_title("Weighted Overlay BEFORE", fontsize=10)
    ax[0].axis('off')
    ax[1].imshow(weighted_red_overlay(fix_np, warp_np))
    ax[1].set_title("Weighted Overlay AFTER", fontsize=10)
    ax[1].axis('off')
    plt.tight_layout()
    return fig


# ======================== Jacobian Sanity Check ========================
def _sanity_check_jacobian():
    print("\n" + "="*60 + "\n[Jacobian Sanity Check]\n" + "="*60)
    # LDMMorph 实际输出 DVF 形状为 [B, 2, H, W]，与下面第一行 sanity 一致。
    arr = np.zeros((2, 512, 512), np.float32)
    try:
        j = jacobian_determinant_vxm(arr)
        print(f"  identity [2,H,W] (与 LDMMorph 输出形状一致) -> shape={j.shape}, "
              f"mean={j.mean():.4f}, min={j.min():.4f}, max={j.max():.4f}   (期望 mean=1.0)")
    except Exception as e:
        print(f"  identity [2,H,W] -> ERROR: {e}")
    print("="*60 + "\n")
_sanity_check_jacobian()


# ======================== 可视化主循环 ========================
splits_to_vis = [opt.split]

for split in splits_to_vis:
    split_save_dir = opt.save_dir
    os.makedirs(split_save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing split: {split}  |  Dataset size: {len(dataset)}")
    print(f"  Visualizing: index {start} to {end-1} ({len(indices_to_vis)} samples)")
    print(f"{'='*60}")

    all_ncc_before = []
    all_ncc_after = []
    all_ncc_liver_roi_before = []
    all_ncc_liver_roi_after = []
    all_ssim_before = []
    all_ssim_after = []
    all_ssim_liver_roi_before = []
    all_ssim_liver_roi_after = []
    all_min_jac = []
    all_max_jac = []
    all_n_foldings = []
    all_jac_neg_ratio = []

    for i, idx in enumerate(indices_to_vis):
        X, Y, segx, segy, pairname = dataset[idx]

        if X.dim() == 2:
            X = X.unsqueeze(0).float().cuda()
            Y = Y.unsqueeze(0).float().cuda()
        else:
            X = X.float().cuda()
            Y = Y.float().cuda()

        if X.dim() == 3:
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)

        print(f"\n[{i+1}/{len(indices_to_vis)}] idx={idx} pairname={pairname}")

        mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
        fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()

        noise = torch.randn_like(mov_z)
        x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([opt.t_enc]).cuda(), noise=noise)
        y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([opt.t_enc]).cuda(), noise=noise)

        outx = ldm_model.apply_model(x_noisy, t=torch.tensor([opt.t_enc]).cuda(), cond=None, return_ids=True)
        outy = ldm_model.apply_model(y_noisy, t=torch.tensor([opt.t_enc]).cuda(), cond=None, return_ids=True)

        score0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
        score1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
        score2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
        score3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]), dim=1)

        img_h, img_w = Y.shape[2], Y.shape[3]

        with torch.no_grad():
            D_f_xy, _ = model(X, Y, score0, score1, score2, score3)
            _, warped_X = transform(X, D_f_xy.permute(0, 2, 3, 1))

            grid_img_np = mk_grid_img(grid_step=24, line_thickness=2, grid_sz=(img_h, img_w))
            grid_img_tensor = torch.from_numpy(grid_img_np[np.newaxis, np.newaxis, ...]).cuda().float()
            _, warped_grid = transform(grid_img_tensor, D_f_xy.permute(0, 2, 3, 1))
            warped_grid_np = warped_grid.squeeze().cpu().numpy()

        # 前景（人体）mask
        fg = body_mask(Y)

        ncc_before = ncc_metric(Y, X, mask=fg)
        ncc_after  = ncc_metric(Y, warped_X, mask=fg)

        ssim_before = ssim_metric(Y, X, mask=fg)
        ssim_after  = ssim_metric(Y, warped_X, mask=fg)

        # 肝脏 ROI NCC（XCAT 版相同写法）
        x1_l, y1_l, x2_l, y2_l = DEFAULT_ROI['liver']
        f_liver_roi = Y[..., y1_l:y2_l, x1_l:x2_l]
        m_liver_roi = X[..., y1_l:y2_l, x1_l:x2_l]
        w_liver_roi = warped_X[..., y1_l:y2_l, x1_l:x2_l]
        f_mean, m_mean, w_mean = f_liver_roi.mean(), m_liver_roi.mean(), w_liver_roi.mean()
        ncc_liver_roi_b = ((f_liver_roi - f_mean) * (m_liver_roi - m_mean)).sum() / \
                          (torch.sqrt(((f_liver_roi - f_mean)**2).sum()) * torch.sqrt(((m_liver_roi - m_mean)**2).sum()) + 1e-8)
        ncc_liver_roi_a = ((f_liver_roi - f_mean) * (w_liver_roi - w_mean)).sum() / \
                          (torch.sqrt(((f_liver_roi - f_mean)**2).sum()) * torch.sqrt(((w_liver_roi - w_mean)**2).sum()) + 1e-8)
        ncc_liver_roi_before = ncc_liver_roi_b.item()
        ncc_liver_roi_after  = ncc_liver_roi_a.item()

        ssim_liver_roi_before = ssim_metric(f_liver_roi, m_liver_roi, window_size=7)
        ssim_liver_roi_after  = ssim_metric(f_liver_roi, w_liver_roi, window_size=7)

        # Jacobian
        mov_np = X.squeeze().cpu().numpy()
        fix_np = Y.squeeze().cpu().numpy()
        warp_np = warped_X.squeeze().cpu().numpy()
        dvf_np = D_f_xy.squeeze().cpu().numpy()
        dvf_mag = np.sqrt(dvf_np[0]**2 + dvf_np[1]**2)

        # [Jacobian] LDMMorph 输出 DVF 是 normalized grid 域 (Softsign ∈ [-1,1])，
        # 而 utils.jacobian_determinant_vxm 内部使用 pystrum.pynd.ndutils.volsize2ndgrid，
        # 其返回的是 **像素坐标 grid** (step=1)，不是 normalized grid。
        # 因此要先把 normalized disp 换算成像素 disp: dvf_px = dvf_norm * (size/2)。
        # 这与 infer.py:199-200 论文原版口径完全一致 (XCAT 版保留同一逻辑)。
        dvf_px = dvf_np.copy()
        dvf_px[0] *= img_h / 2.0
        dvf_px[1] *= img_w / 2.0
        jac_det = jacobian_determinant_vxm(dvf_px)

        if i == 0:
            print(f"    [DVF range] raw=[{dvf_np.min():.4f}, {dvf_np.max():.4f}]  (normalized)")
            print(f"    [Jac px    ] mean={jac_det.mean():.4f} min={jac_det.min():.4f} max={jac_det.max():.4f}  <- 像素域 Jacobian")

        n_foldings = int(np.sum(jac_det < 0))
        min_jac = float(jac_det.min())
        max_jac = float(jac_det.max())
        jac_neg_ratio = float(np.sum(jac_det < 0) / jac_det.size)

        all_ncc_before.append(ncc_before)
        all_ncc_after.append(ncc_after)
        all_ncc_liver_roi_before.append(ncc_liver_roi_before)
        all_ncc_liver_roi_after.append(ncc_liver_roi_after)
        all_ssim_before.append(ssim_before)
        all_ssim_after.append(ssim_after)
        all_ssim_liver_roi_before.append(ssim_liver_roi_before)
        all_ssim_liver_roi_after.append(ssim_liver_roi_after)
        all_min_jac.append(min_jac)
        all_max_jac.append(max_jac)
        all_n_foldings.append(n_foldings)
        all_jac_neg_ratio.append(jac_neg_ratio)

        print(f"    NCC (Full)       before: {ncc_before:.4f}  after: {ncc_after:.4f}  Δ: {ncc_after - ncc_before:+.4f}")
        print(f"    SSIM (Full)      before: {ssim_before:.4f}  after: {ssim_after:.4f}  Δ: {ssim_after - ssim_before:+.4f}")
        print(f"    NCC (Liver ROI)  before: {ncc_liver_roi_before:.4f}  after: {ncc_liver_roi_after:.4f}  Δ: {ncc_liver_roi_after - ncc_liver_roi_before:+.4f}")
        print(f"    SSIM (Liver ROI) before: {ssim_liver_roi_before:.4f}  after: {ssim_liver_roi_after:.4f}  Δ: {ssim_liver_roi_after - ssim_liver_roi_before:+.4f}")
        print(f"    Jacobian: min={min_jac:.4f}, folds={n_foldings}, neg_ratio={jac_neg_ratio*100:.2f}%")

        fix_norm = np.clip(fix_np, 0, 1)
        mov_norm = np.clip(mov_np, 0, 1)
        warp_norm = np.clip(warp_np, 0, 1)

        overlay_before = red_overlay(fix_norm, mov_norm)
        overlay_after  = red_overlay(fix_norm, warp_norm)

        diff_before = np.abs(fix_np - mov_np)
        diff_after  = np.abs(fix_np - warp_np)

        dvf_x = dvf_np[0]
        dvf_y = dvf_np[1]
        dvf_max = float(dvf_mag.max())
        vmax_x = max(abs(dvf_x.min()), abs(dvf_x.max()))
        vmax_y = max(abs(dvf_y.min()), abs(dvf_y.max()))
        vmax_x = max(vmax_x, 0.01)
        vmax_y = max(vmax_y, 0.01)

        # ======================== 4×5 布局（与 XCAT 完全一致）====================
        fig, axes = plt.subplots(4, 5, figsize=(30, 24))
        fig.suptitle(
            f"SEY Liver | {split.upper()} | [{i+1}/{len(indices_to_vis)}] | {pairname}\n"
            f"NCC: {ncc_before:.4f}->{ncc_after:.4f} ({ncc_after - ncc_before:+.4f}) | "
            f"SSIM: {ssim_before:.4f}->{ssim_after:.4f} ({ssim_after - ssim_before:+.4f}) | "
            f"Neg Ratio: {jac_neg_ratio*100:.2f}%",
            fontsize=11, fontweight='bold', y=0.99
        )

        def _contour_overlay(base_gray, *masks_and_styles):
            from skimage import measure
            base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)
            colors = ['red', 'yellow', 'cyan', 'magenta']
            for idx2, m in enumerate(masks_and_styles):
                if m is None or m.sum() < 4:
                    continue
                contours = measure.find_contours(m, 0.5)
                color = colors[idx2 % len(colors)]
                for c in contours:
                    plt.plot(c[:, 1], c[:, 0], color=color, linewidth=1.8, alpha=0.95)
            return base_rgb

        from skimage import measure

        # ---------- 第1行：配准前 ----------
        axes[0, 0].imshow(mov_np, cmap='gray')
        axes[0, 0].set_title(f"Moving (X)\n{pairname}", fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(fix_np, cmap='gray')
        axes[0, 1].set_title("Fixed (Y)", fontsize=11)
        axes[0, 1].axis('off')

        # 肝脏 ROI 边框（XCAT 同款）
        from matplotlib.patches import Rectangle
        roi_x1, roi_y1, roi_x2, roi_y2 = DEFAULT_ROI['liver']
        rect = Rectangle((roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
                         linewidth=2.2, edgecolor='#FFD700', facecolor='none',
                         linestyle='-', alpha=0.95)
        axes[0, 1].add_patch(rect)
        L = max(8, int(0.08 * max(roi_x2 - roi_x1, roi_y2 - roi_y1)))
        for cx, cy, dx, dy in [
            (roi_x1, roi_y1,  1,  1), (roi_x2, roi_y1, -1,  1),
            (roi_x1, roi_y2,  1, -1), (roi_x2, roi_y2, -1, -1),
        ]:
            axes[0, 1].plot([cx, cx + dx * L], [cy, cy],
                            color='#FFD700', linewidth=2.8, solid_capstyle='butt')
            axes[0, 1].plot([cx, cx], [cy, cy + dy * L],
                            color='#FFD700', linewidth=2.8, solid_capstyle='butt')
        axes[0, 1].text(roi_x1, max(0, roi_y1 - 4), 'liver ROI (NCC)',
                        color='#FFD700', fontsize=9, fontweight='bold')

        axes[0, 2].imshow(diff_before, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 2].set_title(f"Abs Diff Before\nNCC: {ncc_before:.4f}", fontsize=11)
        axes[0, 2].axis('off')

        axes[0, 3].imshow(overlay_before)
        axes[0, 3].set_title("Overlay Before\n(Red=Moving)", fontsize=11)
        axes[0, 3].axis('off')

        # 第 1 行第 5 格：不计算 Dice（仅肝脏 ROI 边框在 fixed 上示意）
        axes[0, 4].imshow(_contour_overlay(fix_norm))
        axes[0, 4].set_title("Fixed + Liver ROI\n(无 Dice)", fontsize=10)
        axes[0, 4].axis('off')

        # ---------- 第2行：配准后 ----------
        axes[1, 0].imshow(warp_np, cmap='gray')
        axes[1, 0].set_title(f"Warped (X->Y)\nNCC: {ncc_after:.4f}", fontsize=11)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(fix_np, cmap='gray')
        axes[1, 1].set_title("Fixed (Y)", fontsize=11)
        axes[1, 1].axis('off')
        rect2 = Rectangle((roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
                          linewidth=2.2, edgecolor='#FFD700', facecolor='none',
                          linestyle='-', alpha=0.95)
        axes[1, 1].add_patch(rect2)
        for cx, cy, dx, dy in [
            (roi_x1, roi_y1,  1,  1), (roi_x2, roi_y1, -1,  1),
            (roi_x1, roi_y2,  1, -1), (roi_x2, roi_y2, -1, -1),
        ]:
            axes[1, 1].plot([cx, cx + dx * L], [cy, cy],
                            color='#FFD700', linewidth=2.8, solid_capstyle='butt')
            axes[1, 1].plot([cx, cx], [cy, cy + dy * L],
                            color='#FFD700', linewidth=2.8, solid_capstyle='butt')
        axes[1, 1].text(roi_x1, max(0, roi_y1 - 4), 'liver ROI',
                        color='#FFD700', fontsize=9, fontweight='bold')

        axes[1, 2].imshow(diff_after, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 2].set_title(f"Abs Diff After\n({ncc_after - ncc_before:+.4f})", fontsize=11)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(overlay_after)
        axes[1, 3].set_title("Overlay After\n(Red=Warped)", fontsize=11)
        axes[1, 3].axis('off')

        # 第 2 行第 5 格：固定占位（XCAT 此处是 Heart mask，本数据集不做 Dice，留空显示文字）
        axes[1, 4].imshow(_contour_overlay(fix_norm))
        axes[1, 4].set_title("Fixed (post)\n(无 Dice 计算)", fontsize=10)
        axes[1, 4].axis('off')

        # ---------- 第3行：变形场 ----------
        axes[2, 0].imshow(warp_np, cmap='gray')
        axes[2, 0].imshow(warped_grid_np, cmap='gray', alpha=0.8)
        axes[2, 0].set_title("Warped Grid\non Image", fontsize=11)
        axes[2, 0].axis('off')

        # DVF magnitude: 论文式固定 vmax=0.1，0位移显示为深蓝
        dvf_display = np.sqrt(dvf_x**2 + dvf_y**2)
        DVF_MAG_VMAX = 0.1  # 论文常用范围：归一化像素位移 (0~10% 像素宽度)
        im_dvf = axes[2, 1].imshow(dvf_display, cmap='jet', vmin=0, vmax=DVF_MAG_VMAX)
        axes[2, 1].set_title(f"DVF Magnitude\n(jet, vmax={DVF_MAG_VMAX})", fontsize=11)
        axes[2, 1].axis('off')
        plt.colorbar(im_dvf, ax=axes[2, 1], fraction=0.046, pad=0.04)

        # DVF RGB: 论文式 ±thr 截断，无位移显示为中性色 (R=0.5, G=0.5, B=0)
        # thr=0.05 → |dvf|<0.05 像素的位置保持 R=G=0.5（黄），向右下偏移时 R 偏 G 偏
        DVF_THR = 0.05
        dvf_rgb = np.zeros((dvf_x.shape[0], dvf_x.shape[1], 3), dtype=np.float32)
        dvf_rgb[..., 0] = np.clip((dvf_x + DVF_THR) / (2 * DVF_THR), 0, 1)  # R = dvf_x
        dvf_rgb[..., 1] = np.clip((dvf_y + DVF_THR) / (2 * DVF_THR), 0, 1)  # G = dvf_y
        dvf_rgb[..., 2] = 0.0  # B = 0
        axes[2, 2].imshow(dvf_rgb)
        axes[2, 2].set_title(f"DVF RGB\n(R=X, G=Y, B=0, thr=±{DVF_THR})", fontsize=11)
        axes[2, 2].axis('off')

        im_dvf_x = axes[2, 3].imshow(dvf_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
        axes[2, 3].set_title(f"DVF X (L-R)\n(vmax={vmax_x:.3f})", fontsize=11)
        axes[2, 3].axis('off')
        plt.colorbar(im_dvf_x, ax=axes[2, 3], fraction=0.046, pad=0.04)

        # 第 3 行第 5 格：weighted_red_overlay 的 before/after 对比（额外子图）
        fig_weighted = _make_weighted_overlay_pair(fix_np, mov_np, warp_np)
        weighted_path = os.path.join(split_save_dir, f"_weighted_pair_{i:03d}_{pairname}.png")
        fig_weighted.savefig(weighted_path, dpi=100, bbox_inches='tight')
        plt.close(fig_weighted)
        weighted_img = plt.imread(weighted_path)
        axes[2, 4].imshow(weighted_img)
        axes[2, 4].set_title("Weighted Overlay\nBEFORE | AFTER", fontsize=9)
        axes[2, 4].axis('off')

        # ---------- 第4行：原始灰度图对比 ----------
        h, w = mov_np.shape

        axes[3, 0].imshow(mov_np, cmap='gray')
        axes[3, 0].set_title(f"Moving (X)\n{pairname}", fontsize=11)
        axes[3, 0].axis('off')

        axes[3, 1].imshow(fix_np, cmap='gray')
        axes[3, 1].set_title("Fixed (Y)", fontsize=11)
        axes[3, 1].axis('off')

        axes[3, 2].imshow(warp_np, cmap='gray')
        axes[3, 2].set_title("Warped (X->Y)", fontsize=11)
        axes[3, 2].axis('off')

        axes[3, 3].imshow(np.concatenate([mov_np, fix_np, warp_np], axis=1), cmap='gray')
        axes[3, 3].set_title("M vs F vs W", fontsize=11)
        axes[3, 3].axis('off')
        axes[3, 3].axvline(x=w - 0.5, color='white', linewidth=2)
        axes[3, 3].axvline(x=2*w - 0.5, color='white', linewidth=2)
        axes[3, 3].text(w//2, h + 15, 'M', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
        axes[3, 3].text(3*w//2, h + 15, 'F', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
        axes[3, 3].text(5*w//2, h + 15, 'W', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

        diff_combined = np.concatenate([diff_before, diff_after], axis=1)
        im_diff = axes[3, 4].imshow(diff_combined, cmap='hot', vmin=0, vmax=0.3)
        axes[3, 4].set_title("Diff: Before | After", fontsize=11)
        axes[3, 4].axis('off')
        plt.colorbar(im_diff, ax=axes[3, 4], fraction=0.046, pad=0.04)
        axes[3, 4].axvline(x=w - 0.5, color='cyan', linewidth=2)
        axes[3, 4].text(w//2, -15, 'Before', ha='center', va='top', color='cyan', fontsize=10)
        axes[3, 4].text(3*w//2, -15, 'After', ha='center', va='top', color='cyan', fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(split_save_dir, f"sample_{i:03d}_{pairname}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {out_path}")

    # ======================== 统计摘要（不含 Dice） ========================
    if len(all_ncc_before) == 0:
        print(f"  WARNING: No samples processed in {split.upper()}, skipping stats.")
    else:
        print(f"\n{'='*60}")
        print(f"[{split.upper()}] Statistics ({len(all_ncc_before)}/{len(dataset)} samples visualized)")
        print(f"{'='*60}")
        print(f"  NCC (Full)      : {np.mean(all_ncc_before):.4f} -> {np.mean(all_ncc_after):.4f}  "
              f"({np.mean(np.array(all_ncc_after) - np.array(all_ncc_before)):+.4f})")
        print(f"  SSIM (Full)     : {np.mean(all_ssim_before):.4f} -> {np.mean(all_ssim_after):.4f}  "
              f"({np.mean(np.array(all_ssim_after) - np.array(all_ssim_before)):+.4f})")
        print(f"  NCC (Liver ROI) : {np.mean(all_ncc_liver_roi_before):.4f} -> {np.mean(all_ncc_liver_roi_after):.4f}  "
              f"({np.mean(np.array(all_ncc_liver_roi_after) - np.array(all_ncc_liver_roi_before)):+.4f})")
        print(f"  SSIM (Liver ROI): {np.mean(all_ssim_liver_roi_before):.4f} -> {np.mean(all_ssim_liver_roi_after):.4f}  "
              f"({np.mean(np.array(all_ssim_liver_roi_after) - np.array(all_ssim_liver_roi_before)):+.4f})")
        print(f"  Min / Max NCC: {np.min(all_ncc_after):.4f} / {np.max(all_ncc_after):.4f}")
        print(f"  Jacobian Det  : min={np.min(all_min_jac):.4f}  max={np.max(all_max_jac):.4f}  "
              f"total_foldings={np.sum(all_n_foldings)}  neg_ratio={np.mean(all_jac_neg_ratio)*100:.2f}%")

    # 保存统计到 CSV（不含 Dice）
    stats_csv = os.path.join(split_save_dir, 'stats.csv')
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Pairname',
                         'NCC_Before', 'NCC_After', 'NCC_Delta',
                         'SSIM_Before', 'SSIM_After', 'SSIM_Delta',
                         'NCC_LiverROI_Before', 'NCC_LiverROI_After', 'NCC_LiverROI_Delta',
                         'SSIM_LiverROI_Before', 'SSIM_LiverROI_After', 'SSIM_LiverROI_Delta',
                         'Min_Jac', 'N_Foldings', 'Jac_Neg_Ratio'])
        for i, idx in enumerate(indices_to_vis):
            _, _, _, _, pairname = dataset[idx]
            writer.writerow([
                idx,
                pairname,
                f"{all_ncc_before[i]:.4f}",
                f"{all_ncc_after[i]:.4f}",
                f"{all_ncc_after[i] - all_ncc_before[i]:+.4f}",
                f"{all_ssim_before[i]:.4f}",
                f"{all_ssim_after[i]:.4f}",
                f"{all_ssim_after[i] - all_ssim_before[i]:+.4f}",
                f"{all_ncc_liver_roi_before[i]:.4f}",
                f"{all_ncc_liver_roi_after[i]:.4f}",
                f"{all_ncc_liver_roi_after[i] - all_ncc_liver_roi_before[i]:+.4f}",
                f"{all_ssim_liver_roi_before[i]:.4f}",
                f"{all_ssim_liver_roi_after[i]:.4f}",
                f"{all_ssim_liver_roi_after[i] - all_ssim_liver_roi_before[i]:+.4f}",
                f"{all_min_jac[i]:.4f}",
                all_n_foldings[i],
                f"{all_jac_neg_ratio[i]*100:.2f}",
            ])
    print(f"  Stats saved: {stats_csv}")

print(f"\nAll done! Results saved to: {opt.save_dir}")
