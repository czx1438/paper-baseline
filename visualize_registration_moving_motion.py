"""
配准可视化脚本（支持 XCAT 运动增强版、XCAT moving_motion 版、NPZ 模式）
用法:
  XCAT 几何形变模式:  python visualize_registration_motion.py --xcat --resume <pth> --xcat_path <path>
  moving_motion 模式:  python visualize_registration_motion.py --moving_motion --resume <pth> --moving_motion_path <path>
  fixed_motion 模式:   python visualize_registration_motion.py --fixed_motion --resume <pth> --fixed_motion_path <path>
  NPZ 模式:            python visualize_registration_motion.py --npz --resume <pth> --datapath <path> --ldm_config <yaml>
"""
import os
import glob
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import Dataset_XCAT_Registration, Dataset_epoch_with_name, SpatialTransform, jacobian_determinant_vxm
import TransModels.LDMMorph as LDMMorph
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


# ======================== 参数配置 ========================
parser = argparse.ArgumentParser()

# 配准网络 checkpoint
parser.add_argument("--resume", type=str,
                    dest="resume",
                    default='',
                    help="配准网络 checkpoint 路径")

# 模式切换（互斥）
parser.add_argument("--xcat", action="store_true",
                    dest="xcat",
                    help="XCAT 几何形变模式（4 种形变：identity/rotate10/scale05/warp）")
parser.add_argument("--moving_motion", action="store_true",
                    dest="moving_motion",
                    help="moving_motion 模式（XCAT 物理仿真 8 帧形变序列，1 fixed + 9 moving 变体）")
parser.add_argument("--fixed_motion", action="store_true",
                    dest="fixed_motion",
                    help="fixed_motion 模式（同 moving_motion 物理形变）")
parser.add_argument("--npz", action="store_true",
                    dest="npz",
                    help="NPZ 标准模式（train/val/test 目录结构）")

# 各模式数据路径
parser.add_argument("--xcat_path", type=str,
                    dest="xcat_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="XCAT 几何形变数据根目录")
parser.add_argument("--moving_motion_path", type=str,
                    dest="moving_motion_path",
                    default='/home/b109/Desktop/czx/LDM-Morph-main_heart/datasets/xcat_data',
                    help="moving_motion 数据根目录")
parser.add_argument("--fixed_motion_path", type=str,
                    dest="fixed_motion_path",
                    default='/home/b109/Desktop/czx/LDM-Morph-main_heart/datasets/xcat_data',
                    help="fixed_motion 数据根目录")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/b109/Desktop/czx/LDM-Morph-main_heart/datasets/XCAT/prep',
                    help="NPZ 数据根目录")

# LDM 配置（XCAT 默认运动增强版，moving_motion/fixed_motion 默认 vq16-64ch，NPZ 默认标准版）
parser.add_argument("--ldm_config", type=str,
                    dest="ldm_config",
                    default=None,
                    help="LDM 配置文件路径（默认根据模式自动选择）")
parser.add_argument("--ldm_checkpoint", type=str,
                    dest="ldm_checkpoint",
                    default=None,
                    help="LDM checkpoint 路径（默认从 --resume 所在目录自动查找）")

# 训练相关参数
parser.add_argument("--smooth", type=float, default=0.1)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--t_enc", type=int, default=1)

# 可视化范围
parser.add_argument("--n_samples", type=int, default=154,
                    help="每个 split 可视化的样本数量")
parser.add_argument("--split", type=str, default='test',
                    choices=['train', 'val', 'test'],
                    help="使用哪个划分的可视化")
parser.add_argument("--start_idx", type=int, default=0,
                    help="从第几个样本开始（0-indexed）")

# XCAT 几何模式过滤
parser.add_argument("--motion_type", type=str, default='identity',
                    choices=['identity', 'rotate10', 'scale05', 'warp', None],
                    help="XCAT 模式：仅可视化特定运动类型，None 表示所有类型")

# moving_motion/fixed_motion 模式过滤
parser.add_argument("--source_type", type=str, default='original',
                    choices=['original', 'seq', 'all'],
                    help="moving_motion/fixed_motion 模式：仅可视化 original（原始 moving）或 seq（形变帧）或 all（全部）")

parser.add_argument("--save_dir", type=str,
                    dest="save_dir",
                    default=None,
                    help="输出保存目录（默认根据模式自动生成）")
opt, unknown = parser.parse_known_args()

# 模式决策（互斥：优先级 moving_motion > fixed_motion > xcat > npz）
if opt.moving_motion:
    mode = 'moving_motion'
elif opt.fixed_motion:
    mode = 'fixed_motion'
elif opt.xcat:
    mode = 'xcat'
elif opt.npz:
    mode = 'npz'
else:
    # 默认 moving_motion（与用户当前训练模式一致）
    mode = 'moving_motion'
    opt.moving_motion = True

# 默认 LDM 配置（按模式）
if opt.ldm_config is None:
    if mode == 'xcat':
        opt.ldm_config = './configs/latent-diffusion/xcat_motion-ldm.yaml'
    elif mode == 'moving_motion':
        opt.ldm_config = './configs/latent-diffusion/xcat-seq-ldm-vq16-64ch.yaml'
    elif mode == 'fixed_motion':
        opt.ldm_config = './configs/latent-diffusion/xcat-seq-ldm-vq16-64ch.yaml'
    else:  # npz
        opt.ldm_config = './configs/latent-diffusion/xcat-ldm-vq16-64ch.yaml'

# 默认保存目录（按模式）
if opt.save_dir is None:
    if mode == 'moving_motion':
        opt.save_dir = f'./logs/visualization_moving_motion_{opt.split}_{opt.source_type}/'
    elif mode == 'fixed_motion':
        opt.save_dir = f'./logs/visualization_fixed_motion_{opt.split}_{opt.source_type}/'
    elif mode == 'xcat':
        opt.save_dir = f'./logs/visualization_xcat_motion_{opt.split}_{opt.motion_type or "all"}/'
    else:
        opt.save_dir = f'./logs/visualization_npz_{opt.split}/'

print(f"\n{'='*60}")
print(f"Mode: {mode}")
print(f"Resume: {opt.resume}")
print(f"LDM Config: {opt.ldm_config}")
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

    # 调试：打印 first_stage_model 的实际结构
    fsm = ldm_model.first_stage_model
    print(f"[DEBUG] first_stage_model type: {type(fsm).__name__}")
    enc = fsm.encoder if hasattr(fsm, 'encoder') else None
    if enc is not None:
        print(f"[DEBUG] encoder type: {type(enc).__name__}")
        if hasattr(enc, 'down'):
            print(f"[DEBUG] encoder.down len: {len(enc.down)}")
            for i, d in enumerate(enc.down):
                print(f"  down[{i}] has {len(d.block)} blocks, block[0] norm1.weight.shape: {d.block[0].norm1.weight.shape}")
        print(f"[DEBUG] encoder.down[0].block[0].norm1.weight.shape: {enc.down[0].block[0].norm1.weight.shape}")
        print(f"[DEBUG] encoder.down[1].block[0].norm1.weight.shape: {enc.down[1].block[0].norm1.weight.shape}")
        # 测试一下能不能跑
        test_x = torch.randn(1, 1, 512, 512).cuda()
        with torch.no_grad():
            z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(test_x))
        print(f"[DEBUG] encode test OK, z.shape = {z.shape}")
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
model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2).cuda()
ckpt_path = opt.resume
if os.path.isfile(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict)
    print(f"Registration model loaded: {ckpt_path}")
else:
    print(f"WARNING: checkpoint not found: {ckpt_path}, using random init")

model.eval()
transform = SpatialTransform().cuda()
for param in transform.parameters():
    param.requires_grad = False


# ======================== 辅助函数：创建网格图像 ========================
def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128, 128)):
    """创建网格图像用于可视化形变场（与infer.py一致）"""
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i+line_thickness-1] = 1
    return grid_img


# ======================== 数据加载 ========================
motion_types_map = {
    'train': ['identity', 'rotate10', 'scale05', 'warp'],
    'val':   ['identity'],
    'test':  ['identity'],
}
flip_p_map = {'train': 0.5, 'val': 0.0, 'test': 0.0}

os.makedirs(opt.save_dir, exist_ok=True)


def build_dataset(split):
    """根据模式构建数据集

    各模式返回的张量 shape:
      XCAT (Dataset_XCAT_Registration)    : moving/fixed = [H, W]
      moving_motion (XCATSeqRegistration) : moving/fixed = [1, H, W]   (含 channel)
      fixed_motion (XCATSeqRegistration)  : moving/fixed = [1, H, W]
      NPZ (Dataset_epoch_with_name)       : moving/fixed = [1, H, W]
    """
    if mode == 'xcat':
        motion_types = [opt.motion_type] if opt.motion_type else motion_types_map[split]
        flip_p = flip_p_map[split]
        ds = Dataset_XCAT_Registration(
            data_root=opt.xcat_path,
            split=split,
            motion_types=motion_types,
            flip_p=flip_p,
        )
        return ds, f"XCAT (geometry) motion_types={motion_types}"

    elif mode in ('moving_motion', 'fixed_motion'):
        from ldm.data.xcat_Motion_Seq import XCATSeqRegistration
        data_root = opt.moving_motion_path if mode == 'moving_motion' else opt.fixed_motion_path
        ds = XCATSeqRegistration(
            data_root=data_root,
            split=split,
            flip_p=0.0,   # 可视化时不做随机翻转
        )
        # 按 source_type 过滤：只保留 'original' / 'seq' / 'all'
        if opt.source_type in ('original', 'seq'):
            kept = [(p, mp, st, fi) for (p, mp, st, fi) in ds.samples
                    if (opt.source_type == 'original' and st == 'original')
                    or (opt.source_type == 'seq' and st == 'seq')]
            ds.samples = kept
        # 重新计算长度
        ds._length = len(ds.samples)
        return ds, f"{mode} source_type={opt.source_type} n={len(ds.samples)}"

    else:  # npz
        base_dir = opt.datapath.rstrip('/')
        # 优先尝试 split 子目录结构
        split_dir = os.path.join(base_dir, split)
        npz_files = sorted(glob.glob(os.path.join(split_dir, '*_pair.npz')))
        # flat 目录结构：所有文件在同一目录，按比例划分
        if len(npz_files) == 0:
            all_files = sorted(glob.glob(os.path.join(base_dir, '*_pair.npz')))
            if len(all_files) == 0:
                raise FileNotFoundError(f"No *_pair.npz files found in {base_dir}/ or {split_dir}/")
            n = len(all_files)
            train_end = int(n * 0.70)
            val_end   = int(n * 0.85)
            if split == 'train':
                npz_files = all_files[:train_end]
            elif split == 'val':
                npz_files = all_files[train_end:val_end]
            else:
                npz_files = all_files[val_end:]
            ds = Dataset_epoch_with_name(npz_files)
            return ds, f"NPZ flat {split}={len(npz_files)}/{n} (last 15%)"
        ds = Dataset_epoch_with_name(npz_files)
        return ds, f"NPZ split {split}={len(npz_files)}"


# ======================== 辅助函数 ========================
def ncc_metric(fixed, moving, win_size=9):
    """计算局部 NCC（全图平均）"""
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
    return ncc.mean().item()


def create_mask(img, img_size=(512, 512)):
    """创建心脏区域掩码"""
    h, w = img_size
    y_start = int(h * 0.30)
    y_end   = int(h * 0.75)
    x_start = int(w * 0.35)
    x_end   = int(w * 0.75)
    mask = torch.zeros_like(img)
    mask[..., y_start:y_end, x_start:x_end] = 1.0
    return mask


def red_overlay(fixed, moving, alpha=0.6):
    """
    红色叠加视图：
    - 背景：Fixed 灰度
    - 前景：Moving 红色（alpha 混合）
    完全对齐的区域被 Moving 覆盖呈灰色，对齐不好的区域呈红色
    """
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


def weighted_red_overlay(fixed, moving):
    return red_overlay(fixed, moving, alpha=0.6)


# ======================== Dice计算 (基于亮度阈值mask) ========================
# mask设置与 analyze_xcat_mask.py 统一
DEFAULT_ROI = {
    'heart': (250, 180, 340, 370),    # x1, y1, x2, y2
    'liver': (150, 250, 300, 520),   # x1, y1, x2, y2
}

# 亮度阈值百分位（与analyze_xcat_mask.py一致）
THRESHOLD_PERCENTILE = 25

# 打印 mask 配置
print(f"[Mask Configuration - 与 analyze_xcat_mask.py 一致]")
print(f"  Heart ROI: x=[{DEFAULT_ROI['heart'][0]}, {DEFAULT_ROI['heart'][2]}], y=[{DEFAULT_ROI['heart'][1]}, {DEFAULT_ROI['heart'][3]}]")
print(f"  Liver ROI: x=[{DEFAULT_ROI['liver'][0]}, {DEFAULT_ROI['liver'][2]}], y=[{DEFAULT_ROI['liver'][1]}, {DEFAULT_ROI['liver'][3]}]")
print(f"  Threshold Percentile: {THRESHOLD_PERCENTILE}")
print(f"{'='*60}\n")


def create_brightness_mask(img_np, organ_name):
    """
    基于亮度阈值创建器官mask（与analyze_xcat_mask.py一致）

    Args:
        img_np: 归一化后的numpy图像 [H, W]
        organ_name: 'heart' 或 'liver'

    Returns:
        mask: 二值mask [H, W]
    """
    mask = np.zeros_like(img_np)
    x1, y1, x2, y2 = DEFAULT_ROI[organ_name]

    # 在ROI区域内使用百分位阈值
    roi = img_np[y1:y2, x1:x2]
    threshold = np.percentile(roi, THRESHOLD_PERCENTILE)
    mask[y1:y2, x1:x2] = (roi > threshold).astype(np.float32)

    return mask


def dice_score(pred, target):
    """计算Dice系数"""
    intersection = np.sum(pred * target)
    return 2.0 * intersection / (np.sum(pred) + np.sum(target) + 1e-8)


def compute_dice_with_mask(img_fixed, img_moving, img_warped):
    """
    使用亮度阈值mask计算Dice系数
    
    Args:
        img_fixed: 固定图像 [H, W]
        img_moving: 移动图像 [H, W]
        img_warped: 变形后的移动图像 [H, W]
    
    Returns:
        dice_dict: 各器官的Dice系数
    """
    dice_dict = {}
    for organ in ['heart', 'liver']:
        mask_fixed = create_brightness_mask(img_fixed, organ)
        mask_moving = create_brightness_mask(img_moving, organ)
        mask_warped = create_brightness_mask(img_warped, organ)
        
        # 计算moving->fixed的Dice (配准前)
        dice_before = dice_score(mask_fixed, mask_moving)
        # 计算warped->fixed的Dice (配准后)
        dice_after = dice_score(mask_fixed, mask_warped)
        
        dice_dict[organ] = {
            'before': dice_before,
            'after': dice_after,
            'delta': dice_after - dice_before,
        }
    
    return dice_dict


# ======================== 可视化逻辑 ========================
splits_to_vis = [opt.split]

for split in splits_to_vis:
    dataset, dataset_info = build_dataset(split)

    # 取样范围
    start = opt.start_idx
    if start >= len(dataset):
        print(f"  WARNING: start_idx={start} >= dataset size {len(dataset)}, clipping to 0")
        start = 0
    end = min(start + opt.n_samples, len(dataset))
    indices_to_vis = list(range(start, end))

    split_save_dir = os.path.join(opt.save_dir, split)
    os.makedirs(split_save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"Dataset: {dataset_info}")
    print(f"Dataset size: {len(dataset)} (total in {split})")
    print(f"Visualizing: index {start} to {end-1} ({len(indices_to_vis)} samples)")
    print(f"{'='*60}")

    # 全局统计
    all_ncc_before = []
    all_ncc_after = []
    all_ncc_roi_before = []
    all_ncc_roi_after = []
    all_dice_heart_before = []
    all_dice_heart_after = []
    all_dice_liver_before = []
    all_dice_liver_after = []
    all_min_jac = []
    all_n_foldings = []
    all_jac_neg_ratio = []

    for i, idx in enumerate(indices_to_vis):
        X, Y, segx, segy, pairname = dataset[idx]

        # 维度归一化到 [B, 1, H, W] (B 通常为 1)
        # XCAT 模式    : X, Y = [H, W]
        # 其余三模式  : X, Y = [1, H, W]
        if X.dim() == 2:
            X = X.unsqueeze(0).float().cuda()      # [H, W] -> [1, H, W]
            Y = Y.unsqueeze(0).float().cuda()
        elif X.dim() == 3:
            X = X.float().cuda()
            Y = Y.float().cuda()
        else:
            X = X.float().cuda()
            Y = Y.float().cuda()
        # 加通道维 -> [B, 1, H, W]
        if X.dim() == 3:
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)

        print(f"\n[{i+1}/{len(indices_to_vis)}] idx={idx} pairname={pairname}  X.shape={tuple(X.shape)}")

        # LDM encode
        mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
        fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()

        # 加噪 & score 提取（Moving 和 Fixed 使用同一份 noise）
        noise = torch.randn_like(mov_z)
        x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([opt.t_enc]).cuda(), noise=noise)
        y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([opt.t_enc]).cuda(), noise=noise)

        outx = ldm_model.apply_model(x_noisy, t=torch.tensor([opt.t_enc]).cuda(), cond=None, return_ids=True)
        outy = ldm_model.apply_model(y_noisy, t=torch.tensor([opt.t_enc]).cuda(), cond=None, return_ids=True)

        score0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
        score1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
        score2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
        score3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]), dim=1)

        # 获取图像尺寸
        img_h, img_w = Y.shape[2], Y.shape[3]

        # 配准
        with torch.no_grad():
            D_f_xy = model(X, Y, score0, score1, score2, score3)
            _, warped_X = transform(X, D_f_xy.permute(0, 2, 3, 1))

            # 创建并变形网格图像（与infer.py一致：512图像用step≈24保持和128图像step=6类似的网格密度）
            grid_img_np = mk_grid_img(grid_step=24, line_thickness=2, grid_sz=(img_h, img_w))
            grid_img_tensor = torch.from_numpy(grid_img_np[np.newaxis, np.newaxis, ...]).cuda().float()
            _, warped_grid = transform(grid_img_tensor, D_f_xy.permute(0, 2, 3, 1))
            warped_grid_np = warped_grid.squeeze().cpu().numpy()

        # NCC 指标
        ncc_before = ncc_metric(Y, X)
        ncc_after  = ncc_metric(Y, warped_X)

        # ROI NCC（仅当图像尺寸合适时计算）
        if img_h >= 256 and img_w >= 256:
            roi_mask = create_mask(Y, img_size=(img_w, img_h)).cuda()
            f_roi = Y[roi_mask == 1]
            m_roi = X[roi_mask == 1]
            w_roi = warped_X[roi_mask == 1]
            f_mean, m_mean, w_mean = f_roi.mean(), m_roi.mean(), w_roi.mean()
            ncc_roi_before = ((f_roi - f_mean) * (m_roi - m_mean)).sum() / \
                             (torch.sqrt(((f_roi - f_mean)**2).sum()) * torch.sqrt(((m_roi - m_mean)**2).sum()) + 1e-8)
            ncc_roi_after = ((f_roi - f_mean) * (w_roi - w_mean)).sum() / \
                            (torch.sqrt(((f_roi - f_mean)**2).sum()) * torch.sqrt(((w_roi - w_mean)**2).sum()) + 1e-8)
            ncc_roi_before = ncc_roi_before.item()
            ncc_roi_after  = ncc_roi_after.item()
        else:
            ncc_roi_before = ncc_before
            ncc_roi_after  = ncc_after

        all_ncc_before.append(ncc_before)
        all_ncc_after.append(ncc_after)
        all_ncc_roi_before.append(ncc_roi_before)
        all_ncc_roi_after.append(ncc_roi_after)

        # 转为 numpy
        mov_np = X.squeeze().cpu().numpy()
        fix_np = Y.squeeze().cpu().numpy()
        warp_np = warped_X.squeeze().cpu().numpy()
        dvf_np  = D_f_xy.squeeze().cpu().numpy()
        dvf_mag = np.sqrt(dvf_np[0]**2 + dvf_np[1]**2)

        # 计算雅可比行列式 (修复：正确的2D雅可比公式)
        # 对于形变场 φ(x,y) = (x+u, y+v)，雅可比矩阵为
        # J = | 1+∂u/∂x  ∂u/∂y |   = | 1+dx  dy  |
        #     | ∂v/∂x    1+∂v/∂y |     | dxx  1+dyy |
        # 计算雅可比行列式（仿照 infer.py：对DVF缩放后使用 jacobian_determinant_vxm）
        # DVF 缩放到像素坐标（原infer.py第198-200行）
        h, w = dvf_np.shape[1], dvf_np.shape[2]  # [2, H, W]
        dvf_scaled = dvf_np.copy()
        dvf_scaled[0] = dvf_scaled[0] * h / 2  # x方向位移
        dvf_scaled[1] = dvf_scaled[1] * w / 2  # y方向位移
        
        # 使用 jacobian_determinant_vxm 计算雅可比（它会自动处理 grid + disp）
        jac_det = jacobian_determinant_vxm(dvf_scaled)  # 返回 [H, W]
        
        # 调试：打印雅可比行列式的分布
        print(f"    [DEBUG] Jac: min={jac_det.min():.4f}, max={jac_det.max():.4f}, mean={jac_det.mean():.4f}")
        
        n_foldings = int(np.sum(jac_det < 0))
        min_jac = float(jac_det.min())
        jac_neg_ratio = float(np.sum(jac_det <= 0) / jac_det.size)  # 非正雅可比例子

        # Dice计算 (基于亮度阈值mask)
        dice_dict = compute_dice_with_mask(fix_np, mov_np, warp_np)
        all_dice_heart_before.append(dice_dict['heart']['before'])
        all_dice_heart_after.append(dice_dict['heart']['after'])
        all_dice_liver_before.append(dice_dict['liver']['before'])
        all_dice_liver_after.append(dice_dict['liver']['after'])
        all_min_jac.append(min_jac)
        all_n_foldings.append(n_foldings)
        all_jac_neg_ratio.append(jac_neg_ratio)

        print(f"    NCC (全图)  before: {ncc_before:.4f}  after: {ncc_after:.4f}  Δ: {ncc_after - ncc_before:+.4f}")
        print(f"    NCC (ROI)   before: {ncc_roi_before:.4f}  after: {ncc_roi_after:.4f}  Δ: {ncc_roi_after - ncc_roi_before:+.4f}")
        print(f"    Dice (Heart): {dice_dict['heart']['before']:.4f} -> {dice_dict['heart']['after']:.4f}  Δ: {dice_dict['heart']['delta']:+.4f}")
        print(f"    Dice (Liver): {dice_dict['liver']['before']:.4f} -> {dice_dict['liver']['after']:.4f}  Δ: {dice_dict['liver']['delta']:+.4f}")
        print(f"    Jacobian: min={min_jac:.4f}, folds={n_foldings}, neg_ratio={jac_neg_ratio*100:.2f}%")

        # 归一化（数据已是 [0, 1]）
        fix_norm = np.clip(fix_np, 0, 1)
        mov_norm = np.clip(mov_np, 0, 1)
        warp_norm = np.clip(warp_np, 0, 1)

        # 叠加图
        overlay_before = red_overlay(fix_norm, mov_norm)
        overlay_after  = red_overlay(fix_norm, warp_norm)

        # 差值图
        diff_before = np.abs(fix_np - mov_np)
        diff_after  = np.abs(fix_np - warp_np)

        # DVF 显示范围自适应
        dvf_x = dvf_np[0]
        dvf_y = dvf_np[1]
        dvf_max = float(dvf_mag.max())
        # 使用DVF分量的实际最大绝对值作为颜色范围
        vmax_x = max(abs(dvf_x.min()), abs(dvf_x.max()))
        vmax_y = max(abs(dvf_y.min()), abs(dvf_y.max()))
        # 确保最小值避免除零
        vmax_x = max(vmax_x, 0.01)
        vmax_y = max(vmax_y, 0.01)

        # ==================== 绘图（3行4列） ====================
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        mode_label = {
            'xcat':           'XCAT Geometry',
            'moving_motion':  'XCAT MovingMotion (Physical)',
            'fixed_motion':   'XCAT FixedMotion (Physical)',
            'npz':            'NPZ Standard',
        }[mode]
        fig.suptitle(
            f"{mode_label} | {split.upper()} | [{i+1}/{len(indices_to_vis)}] | {pairname}\n"
            f"NCC: {ncc_before:.4f} -> {ncc_after:.4f} ({ncc_after - ncc_before:+.4f}) | "
            f"ROI: {ncc_roi_before:.4f} -> {ncc_roi_after:.4f} ({ncc_roi_after - ncc_roi_before:+.4f})\n"
            f"Heart Dice: {dice_dict['heart']['before']:.4f} -> {dice_dict['heart']['after']:.4f} | "
            f"Liver Dice: {dice_dict['liver']['before']:.4f} -> {dice_dict['liver']['after']:.4f} | "
            f"Neg Ratio: {jac_neg_ratio*100:.2f}%",
            fontsize=12, fontweight='bold', y=0.99
        )

        # ---------- 第1行：配准前 ----------
        axes[0, 0].imshow(mov_np, cmap='gray')
        axes[0, 0].set_title(f"Moving (X)\n{pairname}", fontsize=11)
        axes[0, 0].axis('off')
        # 绘制ROI边界框（只显示框，不显示半透明mask）
        for name, coords in DEFAULT_ROI.items():
            x1, y1, x2, y2 = coords
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                edgecolor='lime' if name == 'heart' else 'cyan',
                                linewidth=2.5, linestyle='--')
            axes[0, 0].add_patch(rect)
            axes[0, 0].text(x1+5, y1+15, f"{name.upper()} Dice", 
                           color='lime' if name == 'heart' else 'cyan',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        axes[0, 1].imshow(fix_np, cmap='gray')
        axes[0, 1].set_title("Fixed (Y)", fontsize=11)
        axes[0, 1].axis('off')
        # 绘制ROI边界框（只显示框，不显示半透明mask）
        for name, coords in DEFAULT_ROI.items():
            x1, y1, x2, y2 = coords
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                edgecolor='lime' if name == 'heart' else 'cyan',
                                linewidth=2.5, linestyle='--')
            axes[0, 1].add_patch(rect)
            axes[0, 1].text(x1+5, y1+15, f"{name.upper()} Dice", 
                           color='lime' if name == 'heart' else 'cyan',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        axes[0, 2].imshow(diff_before, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 2].set_title(f"Abs Diff Before\nNCC: {ncc_before:.4f}", fontsize=11)
        axes[0, 2].axis('off')

        axes[0, 3].imshow(overlay_before)
        axes[0, 3].set_title("Overlay Before\n(Red=Moving)", fontsize=11)
        axes[0, 3].axis('off')

        # ---------- 第2行：配准后 ----------
        axes[1, 0].imshow(warp_np, cmap='gray')
        axes[1, 0].set_title(f"Warped (X->Y)\nNCC: {ncc_after:.4f}", fontsize=11)
        axes[1, 0].axis('off')
        # 绘制ROI边界框（只显示框，不显示半透明mask）
        for name, coords in DEFAULT_ROI.items():
            x1, y1, x2, y2 = coords
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                edgecolor='lime' if name == 'heart' else 'cyan',
                                linewidth=2.5, linestyle='--')
            axes[1, 0].add_patch(rect)
            axes[1, 0].text(x1+5, y1+15, f"{name.upper()} Dice", 
                           color='lime' if name == 'heart' else 'cyan',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        axes[1, 1].imshow(fix_np, cmap='gray')
        axes[1, 1].set_title("Fixed (Y)", fontsize=11)
        axes[1, 1].axis('off')
        # 绘制ROI边界框（只显示框，不显示半透明mask）
        for name, coords in DEFAULT_ROI.items():
            x1, y1, x2, y2 = coords
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                edgecolor='lime' if name == 'heart' else 'cyan',
                                linewidth=2.5, linestyle='--')
            axes[1, 1].add_patch(rect)
            axes[1, 1].text(x1+5, y1+15, f"{name.upper()} Dice", 
                           color='lime' if name == 'heart' else 'cyan',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        axes[1, 2].imshow(diff_after, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 2].set_title(f"Abs Diff After\n({ncc_after - ncc_before:+.4f})", fontsize=11)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(overlay_after)
        axes[1, 3].set_title("Overlay After\n(Red=Warped)", fontsize=11)
        axes[1, 3].axis('off')

        # ---------- 第3行：变形场 ----------
        # 变形网格叠加在灰度图上（与infer.py一致：白色线条+黑色背景）
        axes[2, 0].imshow(warp_np, cmap='gray')
        axes[2, 0].imshow(warped_grid_np, cmap='gray', alpha=0.8)
        axes[2, 0].set_title("Warped Grid\non Image", fontsize=11)
        axes[2, 0].axis('off')

        # 形变场 magnitude（位移幅度）
        dvf_display = np.sqrt(dvf_x**2 + dvf_y**2)
        im_dvf = axes[2, 1].imshow(dvf_display, cmap='jet')
        axes[2, 1].set_title("DVF Magnitude", fontsize=11)
        axes[2, 1].axis('off')
        plt.colorbar(im_dvf, ax=axes[2, 1], fraction=0.046, pad=0.04)

        # 形变场 RGB 可视化（X->R, Y->G, B=0）
        dvf_x_norm = (dvf_x - dvf_x.min()) / (dvf_x.max() - dvf_x.min() + 1e-8)
        dvf_y_norm = (dvf_y - dvf_y.min()) / (dvf_y.max() - dvf_y.min() + 1e-8)
        dvf_rgb = np.stack([dvf_x_norm, dvf_y_norm, np.zeros_like(dvf_x)], axis=-1)
        axes[2, 2].imshow(dvf_rgb)
        axes[2, 2].set_title("DVF RGB\n(R=X, G=Y, B=0)", fontsize=11)
        axes[2, 2].axis('off')

        # 形变场 X方向分量
        im_dvf_x = axes[2, 3].imshow(dvf_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
        axes[2, 3].set_title("DVF X (L-R)", fontsize=11)
        axes[2, 3].axis('off')
        plt.colorbar(im_dvf_x, ax=axes[2, 3], fraction=0.046, pad=0.04)

        # 计算mask覆盖比例
        # heart_ratio_fix = np.sum(heart_mask_fix) / heart_mask_fix.size * 100
        # liver_ratio_fix = np.sum(liver_mask_fix) / liver_mask_fix.size * 100

        # axes[2, 3].text(0.5, 0.5,
        #                 f"Summary\n"
        #                 f"NCC (Full): {ncc_before:.4f} -> {ncc_after:.4f}\n"
        #                 f"Delta: {ncc_after - ncc_before:+.4f}\n"
        #                 f"NCC (ROI): {ncc_roi_before:.4f} -> {ncc_roi_after:.4f}\n"
        #                 f"Delta: {ncc_roi_after - ncc_roi_before:+.4f}\n"
        #                 # f"Heart Dice: {dice_dict['heart']['before']:.4f} -> {dice_dict['heart']['after']:.4f}\n"
        #                 # f"Liver Dice: {dice_dict['liver']['before']:.4f} -> {dice_dict['liver']['after']:.4f}\n"
        #                 # f"Heart Mask: {heart_ratio_fix:.2f}% | Liver Mask: {liver_ratio_fix:.2f}%\n"
        #                 f"DVF max: {dvf_max:.2f}",
        #                 ha='center', va='center', fontsize=10,
        #                 transform=axes[2, 3].transAxes,
        #                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        # axes[2, 3].axis('off')

        plt.tight_layout()
        out_path = os.path.join(split_save_dir, f"sample_{i:03d}_{pairname}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {out_path}")

    # ======================== 统计摘要 ========================
    if len(all_ncc_before) == 0:
        print(f"  WARNING: No samples processed in {split.upper()}, skipping stats.")
    else:
        print(f"\n{'='*60}")
        print(f"[{split.upper()}] Statistics ({len(all_ncc_before)}/{len(dataset)} samples visualized)")
        print(f"{'='*60}")
        print(f"  NCC (Full)  : {np.mean(all_ncc_before):.4f} -> {np.mean(all_ncc_after):.4f}  "
              f"({np.mean(np.array(all_ncc_after) - np.array(all_ncc_before)):+.4f})")
        print(f"  NCC (ROI)   : {np.mean(all_ncc_roi_before):.4f} -> {np.mean(all_ncc_roi_after):.4f}  "
              f"({np.mean(np.array(all_ncc_roi_after) - np.array(all_ncc_roi_before)):+.4f})")
        print(f"  Dice (Heart): {np.mean(all_dice_heart_before):.4f} -> {np.mean(all_dice_heart_after):.4f}  "
              f"({np.mean(np.array(all_dice_heart_after) - np.array(all_dice_heart_before)):+.4f})")
        print(f"  Dice (Liver): {np.mean(all_dice_liver_before):.4f} -> {np.mean(all_dice_liver_after):.4f}  "
              f"({np.mean(np.array(all_dice_liver_after) - np.array(all_dice_liver_before)):+.4f})")
        print(f"  Min / Max NCC: {np.min(all_ncc_after):.4f} / {np.max(all_ncc_after):.4f}")
        print(f"  Jacobian Det  : min={np.min(all_min_jac):.4f}  max={np.max(all_min_jac):.4f}  "
              f"total_foldings={np.sum(all_n_foldings)}  neg_ratio={np.mean(all_jac_neg_ratio)*100:.2f}%")

    # 保存统计到 CSV
    stats_csv = os.path.join(split_save_dir, f'stats_{mode}.csv')
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Pairname', 'NCC_Before', 'NCC_After', 'Delta',
                         'ROI_Before', 'ROI_After', 'ROI_Delta',
                         'Dice_Heart_Before', 'Dice_Heart_After', 'Dice_Heart_Delta',
                         'Dice_Liver_Before', 'Dice_Liver_After', 'Dice_Liver_Delta',
                         'Min_Jac', 'N_Foldings', 'Jac_Neg_Ratio'])
        for i, idx in enumerate(indices_to_vis):
            writer.writerow([
                idx,
                dataset[idx][4],
                f"{all_ncc_before[i]:.4f}",
                f"{all_ncc_after[i]:.4f}",
                f"{all_ncc_after[i] - all_ncc_before[i]:+.4f}",
                f"{all_ncc_roi_before[i]:.4f}",
                f"{all_ncc_roi_after[i]:.4f}",
                f"{all_ncc_roi_after[i] - all_ncc_roi_before[i]:.4f}",
                f"{all_dice_heart_before[i]:.4f}",
                f"{all_dice_heart_after[i]:.4f}",
                f"{all_dice_heart_after[i] - all_dice_heart_before[i]:.4f}",
                f"{all_dice_liver_before[i]:.4f}",
                f"{all_dice_liver_after[i]:.4f}",
                f"{all_dice_liver_after[i] - all_dice_liver_before[i]:.4f}",
                f"{all_min_jac[i]:.4f}",
                all_n_foldings[i],
                f"{all_jac_neg_ratio[i]*100:.2f}",
            ])
    print(f"  Stats saved: {stats_csv}")

print(f"\nAll done! Results saved to: {opt.save_dir}")
