"""
配准可视化脚本（支持 XCAT 运动增强版 / XCAT 序列版 / 普通 NPZ 模式）
用法:
  XCAT 模式:     python visualize_registration_xcat.py --xcat --resume <pth> --xcat_path <path>
  XCAT Seq 模式: python visualize_registration_xcat.py --xcat_seq --resume <pth> --xcat_path <path>
  NPZ 模式:      python visualize_registration_xcat.py --resume <pth> --datapath <path> --ldm_config <yaml>

NPZ 模式说明：
  - test 集划分直接读取 datapath 下的 split_indices.json 的 registration.split.test，
    与训练（XCATNPZRegistration）完全同源；索引相对 sorted(glob('*_pair.npz')) 的位置。
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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import Dataset_XCAT_Registration, Dataset_epoch_with_name, SpatialTransform, jacobian_determinant_vxm
from ldm.data.xcat_Motion_Seq import XCATSeqRegistration, XCATOriginalRegistration
from ldm.data.xcat_npz import XCATNPZRegistration, XCATNPZRegistrationFromNPY
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

# 模式切换
parser.add_argument("--xcat", action="store_true",
                    dest="xcat",
                    help="使用 XCAT 运动增强数据集（默认开启）")
parser.add_argument("--no-xcat", action="store_true",
                    dest="no_xcat",
                    help="使用普通 NPZ 数据集（会覆盖 --xcat）")
parser.add_argument("--xcat_seq", action="store_true",
                    dest="xcat_seq",
                    help="使用 XCAT 序列运动数据集（XCATSeqRegistration，优先级高于 --xcat）")

# XCAT 模式参数
parser.add_argument("--xcat_path", type=str,
                    dest="xcat_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="XCAT 数据根目录（包含 fixed/, moving/ 子目录）")

# NPZ 模式参数
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep',
                    help="NPZ 数据根目录（包含 *_pair.npz 与 split_indices.json）")

# LDM 配置（XCAT 默认运动增强版，NPZ 默认标准版）
parser.add_argument("--ldm_config", type=str,
                    dest="ldm_config",
                    default=None,
                    help="LDM 配置文件路径（默认根据模式自动选择）")
parser.add_argument("--ldm_checkpoint", type=str,
                    dest="ldm_checkpoint",
                    default=None,
                    help="LDM checkpoint 路径（默认自动查找）")

# 训练相关参数
parser.add_argument("--smooth", type=float, default=0.1)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--t_enc", type=int, default=1)
parser.add_argument("--no_ldm", action="store_true",
                    dest="no_ldm",
                    help="Disable LDM features (match --no_ldm from training)")
parser.add_argument("--use_motion_film", action="store_true",
                    dest="use_motion_film",
                    help="Load model with use_motion_film=True (for motion_film checkpoint ablation)")
parser.add_argument("--use_phase_film", action="store_true",
                    dest="use_phase_film",
                    help="Forward pass with phase_id=phase (for phase_film checkpoint). "
                         "Leave OFF for baseline (FiLM disabled in forward).")

# 可视化范围
parser.add_argument("--n_samples", type=int, default=10,
                    help="每个 split 可视化的样本数量")
parser.add_argument("--split", type=str, default='test',
                    choices=['train', 'val', 'test'],
                    help="使用哪个划分的可视化")
parser.add_argument("--start_idx", type=int, default=40,
                    help="从第几个样本开始（0-indexed）")
parser.add_argument("--stride", type=int, default=None,
                    help="样本步长. NPZ 9-phase dataset 中相邻 idx 是同 base 不同 phase; "
                         "设 stride=9 可让 n_samples 个 sample 分散到 n_samples 个不同 base.")
parser.add_argument("--motion_type", type=str, default='identity',
                    choices=['identity', 'rotate10', 'scale05', 'warp', None],
                    help=" 模式：仅可视化特定运动类型，None 表示所有类型")
parser.add_argument("--save_dir", type=str,
                    dest="save_dir",
                    default=None,
                    help="输出保存目录（默认根据模式自动生成）")
opt, unknown = parser.parse_known_args()

# 模式决策：--xcat_seq 优先于 --xcat，--no-xcat 会覆盖 --xcat
use_xcat = opt.xcat and not opt.no_xcat and not opt.xcat_seq
use_xcat_seq = opt.xcat_seq and not opt.no_xcat

# 默认 LDM 配置
if opt.ldm_config is None:
    if use_xcat_seq:
        opt.ldm_config = './configs/latent-diffusion/xcat-seq-ldm-vq16-64ch.yaml'
    elif use_xcat:
        opt.ldm_config = './configs/latent-diffusion/xcat_motion-ldm.yaml'
    else:
        opt.ldm_config = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_no_motion.yaml'

# 默认保存目录
if opt.save_dir is None:
    mode_suffix = ''
    if opt.use_motion_film:
        mode_suffix = '_motionfilm'
    elif opt.use_phase_film:
        mode_suffix = '_phase'
    else:
        mode_suffix = '_baseline'
    if use_xcat_seq:
        opt.save_dir = f'./logs/visualization_xcat_seq_{opt.start_idx}_{opt.start_idx + opt.n_samples}_627_win_15_jacdet2.0_bending0.01_smth0.2{mode_suffix}/'
    elif use_xcat:
        opt.save_dir = f'./logs/visualization_xcat_motion_{opt.start_idx}_{opt.start_idx + opt.n_samples}{mode_suffix}/'
    else:
        opt.save_dir = f'./logs/visualization_npz_{opt.start_idx}_{opt.start_idx + opt.n_samples}_917{mode_suffix}/'

print(f"\n{'='*60}")
if use_xcat_seq:
    mode_label = 'XCAT Seq Motion'
elif use_xcat:
    mode_label = 'XCAT Motion-Augmented'
else:
    mode_label = 'NPZ Standard'
print(f"Mode: {mode_label}")
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
# 与 train_mask.py 对齐：根据 --use_motion_film 决定 LDMMorph 内部是否启用 motion FiLM
# phase FiLM 走前向时的 phase_id 注入控制（不传 = skip, 传 = enable)
model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2,
                          use_ldm=not opt.no_ldm,
                          use_motion_film=opt.use_motion_film).cuda()
print(f"[LDMMorph] use_motion_film={opt.use_motion_film}, use_phase_film={opt.use_phase_film}")
ckpt_path = opt.resume
if os.path.isfile(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict,strict=False)
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
    """根据模式构建数据集。

    NPZ 模式：
      - 优先读取 datapath 下 split_indices.json 的 registration.split[split]（与训练同源），
        索引相对 sorted(glob('*_pair.npz')) 的位置，区间为 [start, end] 含两端。
      - 若没有 json，回退到旧的 70/85% 切片。
    """
    if use_xcat:
        motion_types = [opt.motion_type] if opt.motion_type else motion_types_map[split]
        flip_p = flip_p_map[split]
        ds = Dataset_XCAT_Registration(
            data_root=opt.xcat_path,
            split=split,
            motion_types=motion_types,
            flip_p=flip_p,
        )
        return ds, f"XCAT motion_types={motion_types}"
    elif use_xcat_seq:
        ds = XCATSeqRegistration(
            data_root=opt.xcat_path,
            split=split,
            flip_p=0.0 if split in ('val', 'test') else 0.5,
        )
        return ds, f"XCATSeqRegistration split={split}"
    else:
        # NPZ 模式：与 train_mask.py 严格同源 - 使用 XCATNPZRegistration
        # 关键点：要让 visualize 的样本与 train 的 test split 完全一致，
        #        并能拿到 phase_id（0..8），便于 --use_phase_film 严格还原 phase_film 推理路径
        ds = XCATNPZRegistration(
            data_root=opt.xcat_path,
            split=split,
            flip_p=0.0,                # val/test 不翻
            normalize=True,
        )
        return ds, f"XCATNPZRegistration split={split} (same as train_mask.py NPZ mode)"


# ======================== 辅助函数 ========================
def body_mask(img_tensor, thr=0.05):
    """从图像生成前景/人体 mask，用于排除黑色背景。

    流程：阈值二值化 -> 填充内部孔洞 -> 保留最大连通块。
    这样得到完整的人体轮廓（内部的暗器官/肺也算前景），只把
    画面外圈的纯黑空气背景排除掉，避免全图 NCC/SSIM 被背景污染。

    Args:
        img_tensor: torch.Tensor, shape [B, 1, H, W]，值域 [0, 1]
        thr: 前景阈值（大于该亮度算前景）

    Returns:
        torch.BoolTensor, shape [B, 1, H, W]，与输入同 device
    """
    from scipy.ndimage import binary_fill_holes, label
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
    """计算滑动窗口局部 NCC（全图逐像素平均）。

    mask: 可选 [B,1,H,W] 布尔张量；提供时只在 mask=True 的像素上求平均
          （用于排除黑色背景，避免背景把全图 NCC 稀释/拉低）。
    """
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
    """
    计算 SSIM（Structural Similarity Index）。

    Args:
        fixed, moving: torch.Tensor, shape [B, C, H, W]，值域 [0, 1]
        window_size: 高斯窗口大小（奇数）
        size_average: True -> 返回全图平均 SSIM；False -> 返回逐像素 SSIM
        L: 动态范围，图像值域为 [0,1] 时 L=1.0
        mask: 可选 [B,1,H,W] 布尔张量；提供时只在 mask=True 的像素上求平均
    """
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


def create_mask(img, img_size=(512, 512)):
    """创建心脏区域掩码（基于固定百分比，已被 DEFAULT_ROI 替代，仅保留兼容性）"""
    h, w = img_size
    y_start = int(h * 0.30)
    y_end   = int(h * 0.75)
    x_start = int(w * 0.35)
    x_end   = int(w * 0.75)
    mask = torch.zeros_like(img)
    mask[..., y_start:y_end, x_start:x_end] = 1.0
    return mask


def create_roi_mask_from_default(img_tensor, organ_name):
    """使用与 Dice 计算相同的 DEFAULT_ROI 创建掩码。"""
    x1, y1, x2, y2 = DEFAULT_ROI[organ_name]
    if img_tensor.dim() == 4:
        mask = torch.zeros_like(img_tensor)
        mask[..., y1:y2, x1:x2] = 1.0
    else:
        mask = torch.zeros_like(img_tensor)
        mask[..., y1:y2, x1:x2] = 1.0
    return mask


def red_overlay(fixed, moving, alpha=0.6):
    """红色叠加视图：背景=Fixed 灰度，前景=Moving 红色（alpha 混合）。"""
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
DEFAULT_ROI = {
    'heart': (280, 160, 320, 340),    # x1, y1, x2, y2
    'liver': (150, 250, 300, 570),   # x1, y1, x2, y2
}

THRESHOLD_PERCENTILE_HEART = 70
THRESHOLD_PERCENTILE_LIVER = 45
THRESHOLD_PERCENTILE = THRESHOLD_PERCENTILE_HEART

print(f"[Mask Configuration]")
print(f"  Heart ROI: x=[{DEFAULT_ROI['heart'][0]}, {DEFAULT_ROI['heart'][2]}], y=[{DEFAULT_ROI['heart'][1]}, {DEFAULT_ROI['heart'][3]}]")
print(f"  Liver ROI: x=[{DEFAULT_ROI['liver'][0]}, {DEFAULT_ROI['liver'][2]}], y=[{DEFAULT_ROI['liver'][1]}, {DEFAULT_ROI['liver'][3]}]")
print(f"  Threshold Percentile: {THRESHOLD_PERCENTILE}")
print(f"{'='*60}\n")


def create_brightness_mask(img_np, organ_name, percentile=THRESHOLD_PERCENTILE):
    mask = np.zeros_like(img_np)
    x1, y1, x2, y2 = DEFAULT_ROI[organ_name]
    roi = img_np[y1:y2, x1:x2]
    threshold = np.percentile(roi, percentile)
    mask[y1:y2, x1:x2] = (roi > threshold).astype(np.float32)
    return mask


def dice_score(pred, target):
    intersection = np.sum(pred * target)
    return 2.0 * intersection / (np.sum(pred) + np.sum(target) + 1e-8)


from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing


def draw_liver_roi_box(ax, mask, color='#FFD700', label='liver ROI',
                       pad_px=12, line_w=2.2):
    from matplotlib.patches import Rectangle
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H, W = mask.shape
    y0 = max(0, y0 - pad_px); y1 = min(H - 1, y1 + pad_px)
    x0 = max(0, x0 - pad_px); x1 = min(W - 1, x1 + pad_px)
    w = x1 - x0; h = y1 - y0
    rect = Rectangle((x0, y0), w, h, linewidth=line_w,
                     edgecolor=color, facecolor='none', linestyle='-', alpha=0.95)
    ax.add_patch(rect)
    L = max(8, int(0.08 * max(w, h)))
    for cx, cy, dx, dy in [
        (x0, y0,  1,  1), (x1, y0, -1,  1),
        (x0, y1,  1, -1), (x1, y1, -1, -1),
    ]:
        ax.plot([cx, cx + dx * L], [cy, cy], color=color, linewidth=line_w + 0.6, solid_capstyle='butt')
        ax.plot([cx, cx], [cy, cy + dy * L], color=color, linewidth=line_w + 0.6, solid_capstyle='butt')
    ax.text(x0, max(0, y0 - 4), label, color=color, fontsize=9,
            ha='left', va='bottom',
            bbox=dict(facecolor='black', edgecolor='none', alpha=0.55, pad=1.5))


def warp_mask_np(mask_np, D_f_xy, smooth_sigma=1.5):
    mt = torch.from_numpy(mask_np)[None, None].cuda().float()
    _, mw = transform(mt, D_f_xy.permute(0, 2, 3, 1))
    warped = mw.squeeze().cpu().numpy()
    if smooth_sigma > 0:
        warped = gaussian_filter(warped, sigma=smooth_sigma)
    return (warped > 0.5).astype(np.float32)


def _adaptive_roi_for_organ(img_np, organ_name, pad=20):
    from scipy.ndimage import label
    x1, y1, x2, y2 = DEFAULT_ROI[organ_name]
    roi = img_np[y1:y2, x1:x2]
    threshold = np.percentile(roi, THRESHOLD_PERCENTILE)
    seg = (roi > threshold).astype(np.float32)
    if seg.sum() < 10:
        return DEFAULT_ROI[organ_name]
    lab, n = label(seg)
    if n == 0:
        return DEFAULT_ROI[organ_name]
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main_id = int(np.argmax(sizes))
    ys, xs = np.where(lab == main_id)
    cy0, cy1 = ys.min(), ys.max()
    cx0, cx1 = xs.min(), xs.max()
    nx1 = max(0, x1 + cx0 - pad)
    ny1 = max(0, y1 + cy0 - pad)
    nx2 = min(img_np.shape[1], x1 + cx1 + 1 + pad)
    ny2 = min(img_np.shape[0], y1 + cy1 + 1 + pad)
    return (nx1, ny1, nx2, ny2)


def create_brightness_mask_stable(img_np, organ_name, percentile=None,
                                  min_size=20, do_fill_holes=True, do_close=True,
                                  adaptive_roi=True):
    from scipy.ndimage import label, binary_opening
    mask = np.zeros_like(img_np)
    if adaptive_roi:
        x1, y1, x2, y2 = _adaptive_roi_for_organ(img_np, organ_name)
    else:
        x1, y1, x2, y2 = DEFAULT_ROI[organ_name]
    if percentile is None:
        percentile = THRESHOLD_PERCENTILE_LIVER if organ_name == 'liver' else THRESHOLD_PERCENTILE_HEART

    roi = img_np[y1:y2, x1:x2]
    threshold = np.percentile(roi, percentile)
    seg = (roi > threshold).astype(np.float32)

    roi_area = max(1, seg.size)
    if seg.sum() < 50 or seg.sum() > 0.8 * roi_area:
        threshold = float(roi.mean()) * 0.8
        seg = (roi > threshold).astype(np.float32)

    if min_size > 0:
        lab, n = label(seg)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = sizes >= min_size
            seg = keep[lab].astype(np.float32)

    if do_close:
        seg = binary_closing(seg, structure=np.ones((3, 3))).astype(np.float32)

    if do_fill_holes:
        seg = binary_fill_holes(seg).astype(np.float32)

    mask[y1:y2, x1:x2] = seg
    return mask


def _make_organ_mask_liver(img_np, percentile=None):
    from scipy.ndimage import label, binary_opening, binary_dilation
    if percentile is None:
        percentile = THRESHOLD_PERCENTILE_LIVER
    base = create_brightness_mask_stable(
        img_np, 'liver', percentile,
        min_size=5, do_fill_holes=True, do_close=True, adaptive_roi=True,
    )
    base = binary_dilation(base, iterations=2).astype(np.float32)
    return base


def compute_dice_with_mask(img_fixed, img_moving, D_f_xy, percentile=None):
    dice_dict = {}
    masks_dict = {}
    for organ in ['heart', 'liver']:
        if organ == 'liver':
            mask_fixed  = _make_organ_mask_liver(img_fixed,  percentile)
            mask_moving = _make_organ_mask_liver(img_moving, percentile)
        else:
            mask_fixed  = create_brightness_mask_stable(img_fixed,  organ, percentile)
            mask_moving = create_brightness_mask_stable(img_moving,  organ, percentile)

        mask_warped = warp_mask_np(mask_moving, D_f_xy, smooth_sigma=1.5)

        dice_before = dice_score(mask_fixed, mask_moving)
        dice_after  = dice_score(mask_fixed, mask_warped)

        fixed_sum  = float(mask_fixed.sum())
        warped_sum = float(mask_warped.sum())
        intersection = float((mask_fixed * mask_warped).sum())
        coverage = warped_sum / (fixed_sum + 1e-8)
        recall   = intersection / (fixed_sum + 1e-8)
        iou      = intersection / (fixed_sum + warped_sum - intersection + 1e-8)

        dice_dict[organ] = {
            'before':     dice_before,
            'after':      dice_after,
            'delta':      dice_after - dice_before,
            'fixed_mask_sum':      fixed_sum,
            'moving_mask_sum':     float(mask_moving.sum()),
            'warped_mask_sum':     warped_sum,
            'coverage':   coverage,
            'recall':     recall,
            'iou':        iou,
        }
        masks_dict[organ] = {
            'fixed':  mask_fixed,
            'moving': mask_moving,
            'warped': mask_warped,
        }

    return dice_dict, masks_dict


def _sanity_check_jacobian():
    import numpy as np
    H = W = 512
    print("\n" + "="*60 + "\n[Jacobian Sanity Check]\n" + "="*60)

    for fmt, arr in [('[2,H,W]', np.zeros((2, H, W), np.float32)),
                     ('[H,W,2]', np.zeros((H, W, 2), np.float32))]:
        try:
            j = jacobian_determinant_vxm(arr)
            print(f"  identity {fmt} -> shape={j.shape}, mean={j.mean():.4f}, "
                  f"min={j.min():.4f}, max={j.max():.4f}   (期望 ~1.0)")
        except Exception as e:
            print(f"  identity {fmt} -> ERROR: {e}")

    s = 1.1
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    for fmt in ['[H,W,2]', '[2,H,W]']:
        if fmt == '[H,W,2]':
            disp = np.zeros((H, W, 2), np.float32); disp[...,0]=(s-1)*yy; disp[...,1]=(s-1)*xx
        else:
            disp = np.zeros((2, H, W), np.float32); disp[0]=(s-1)*yy; disp[1]=(s-1)*xx
        try:
            j = jacobian_determinant_vxm(disp)
            print(f"  scale s={s} {fmt} -> mean={j.mean():.4f}   (期望 ≈ {s*s:.3f})")
        except Exception as e:
            print(f"  scale s={s} {fmt} -> ERROR: {e}")
    print("="*60 + "\n")

_sanity_check_jacobian()
# ======================== 可视化逻辑 ========================
splits_to_vis = [opt.split]

for split in splits_to_vis:
    dataset, dataset_info = build_dataset(split)

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

    all_ncc_before = []
    all_ncc_after = []
    all_ncc_heart_roi_before = []
    all_ncc_heart_roi_after = []
    all_ncc_liver_roi_before = []
    all_ncc_liver_roi_after = []
    all_dice_heart_before = []
    all_dice_heart_after = []
    all_dice_liver_before = []
    all_dice_liver_after = []
    all_dice_heart_coverage = []
    all_dice_heart_recall = []
    all_dice_heart_iou = []
    all_dice_liver_coverage = []
    all_dice_liver_recall = []
    all_dice_liver_iou = []
    all_min_jac = []
    all_max_jac = []
    all_n_foldings = []
    all_jac_neg_ratio = []
    all_ssim_before = []
    all_ssim_after = []
    all_ssim_heart_roi_before = []
    all_ssim_heart_roi_after = []
    all_ssim_liver_roi_before = []
    all_ssim_liver_roi_after = []

    for i, idx in enumerate(indices_to_vis):
        # 兼容两种 dataset:
        #   - Dataset_XCAT_Registration (XCAT mode) / XCATOriginalRegistration / Dataset_epoch_with_name
        #     → 5-tuple (X, Y, segx, segy, pairname)
        #   - XCATNPZRegistration (NPZ mode) → dict
        #     {moving, fixed, phase, phase_id, moving_idx, fixed_idx, pairname}
        item = dataset[idx]
        if isinstance(item, dict):
            X       = item["moving"]
            Y       = item["fixed"]
            segx    = torch.zeros_like(X)
            segy    = torch.zeros_like(Y)
            pairname = item.get("pairname", f"sample_{idx}")
            phase_id_v = int(item.get("phase_id", 0))   # 0..8
        else:
            X, Y, segx, segy, pairname = item
            phase_id_v = 0  # 非 NPZ 模式没有 phase_id, fallback to 0

        if X.dim() == 2:
            X = X.unsqueeze(0).float().cuda()
            Y = Y.unsqueeze(0).float().cuda()
        else:
            X = X.float().cuda()
            Y = Y.float().cuda()

        if X.dim() == 3:
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)

        print(f"\n[{i+1}/{len(indices_to_vis)}] idx={idx} pairname={pairname}  ({i+1}/{len(dataset)} total)")

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
        score0 = torch.zeros_like(score0)
        score1 = torch.zeros_like(score1)
        score2 = torch.zeros_like(score2)
        #score3 = torch.zeros_like(score3)

        with torch.no_grad():
            # 严格还原 train 时的 forward 路径:
            #   - --use_motion_film  → model(..., phase_id=None)  (model 内部自动用 motion encoder)
            #   - --use_phase_film   → model(..., phase_id=LongTensor)  (走 phase FiLM)
            #   - 都不加              → model(..., phase_id=None)  (skip FiLM = baseline)
            # NOTE: LDMMorph.phase_embedding 要求 phase_id 为 [B] LongTensor, 不是 int
            if opt.use_phase_film:
                fwd_phase_id = torch.tensor([phase_id_v], dtype=torch.long).cuda()
            else:
                fwd_phase_id = None
            
            out = model(X, Y, score0, score1, score2, score3, phase_id=fwd_phase_id)

            if isinstance(out, (tuple, list)) and len(out) == 3:
                D_f_xy, _, motion_code = out
            else:
                D_f_xy, _ = out
                motion_code = None
            _, warped_X = transform(X, D_f_xy.permute(0, 2, 3, 1))

            grid_img_np = mk_grid_img(grid_step=24, line_thickness=2, grid_sz=(img_h, img_w))
            grid_img_tensor = torch.from_numpy(grid_img_np[np.newaxis, np.newaxis, ...]).cuda().float()
            _, warped_grid = transform(grid_img_tensor, D_f_xy.permute(0, 2, 3, 1))
            warped_grid_np = warped_grid.squeeze().cpu().numpy()

        # 前景（人体）mask：基于 fixed(Y) 估计
        fg = body_mask(Y)

        ncc_before = ncc_metric(Y, X, mask=fg)
        ncc_after  = ncc_metric(Y, warped_X, mask=fg)

        ssim_before = ssim_metric(Y, X, mask=fg)
        ssim_after  = ssim_metric(Y, warped_X, mask=fg)

        ncc_heart_roi_before, ncc_heart_roi_after = 0.0, 0.0
        ncc_liver_roi_before, ncc_liver_roi_after = 0.0, 0.0
        for organ in ['heart', 'liver']:
            x1, y1, x2, y2 = DEFAULT_ROI[organ]
            f_roi = Y[..., y1:y2, x1:x2]
            m_roi = X[..., y1:y2, x1:x2]
            w_roi = warped_X[..., y1:y2, x1:x2]
            f_mean, m_mean, w_mean = f_roi.mean(), m_roi.mean(), w_roi.mean()
            ncc_roi_b = ((f_roi - f_mean) * (m_roi - m_mean)).sum() / \
                        (torch.sqrt(((f_roi - f_mean)**2).sum()) * torch.sqrt(((m_roi - m_mean)**2).sum()) + 1e-8)
            ncc_roi_a = ((f_roi - f_mean) * (w_roi - w_mean)).sum() / \
                        (torch.sqrt(((f_roi - f_mean)**2).sum()) * torch.sqrt(((w_roi - w_mean)**2).sum()) + 1e-8)
            if organ == 'heart':
                ncc_heart_roi_before = ncc_roi_b.item()
                ncc_heart_roi_after  = ncc_roi_a.item()
            else:
                ncc_liver_roi_before = ncc_roi_b.item()
                ncc_liver_roi_after  = ncc_roi_a.item()

        x1_h, y1_h, x2_h, y2_h = DEFAULT_ROI['heart']
        x1_l, y1_l, x2_l, y2_l = DEFAULT_ROI['liver']
        f_heart_roi = Y[..., y1_h:y2_h, x1_h:x2_h]
        m_heart_roi = X[..., y1_h:y2_h, x1_h:x2_h]
        w_heart_roi = warped_X[..., y1_h:y2_h, x1_h:x2_h]
        ssim_heart_roi_before = ssim_metric(f_heart_roi, m_heart_roi, window_size=7)
        ssim_heart_roi_after  = ssim_metric(f_heart_roi, w_heart_roi, window_size=7)

        f_liver_roi = Y[..., y1_l:y2_l, x1_l:x2_l]
        m_liver_roi = X[..., y1_l:y2_l, x1_l:x2_l]
        w_liver_roi = warped_X[..., y1_l:y2_l, x1_l:x2_l]
        ssim_liver_roi_before = ssim_metric(f_liver_roi, m_liver_roi, window_size=7)
        ssim_liver_roi_after  = ssim_metric(f_liver_roi, w_liver_roi, window_size=7)

        all_ncc_before.append(ncc_before)
        all_ncc_after.append(ncc_after)
        all_ncc_heart_roi_before.append(ncc_heart_roi_before)
        all_ncc_heart_roi_after.append(ncc_heart_roi_after)
        all_ncc_liver_roi_before.append(ncc_liver_roi_before)
        all_ncc_liver_roi_after.append(ncc_liver_roi_after)
        all_ssim_before.append(ssim_before)
        all_ssim_after.append(ssim_after)
        all_ssim_heart_roi_before.append(ssim_heart_roi_before)
        all_ssim_heart_roi_after.append(ssim_heart_roi_after)
        all_ssim_liver_roi_before.append(ssim_liver_roi_before)
        all_ssim_liver_roi_after.append(ssim_liver_roi_after)

        mov_np = X.squeeze().cpu().numpy()
        fix_np = Y.squeeze().cpu().numpy()
        warp_np = warped_X.squeeze().cpu().numpy()
        dvf_np  = D_f_xy.squeeze().cpu().numpy()
        dvf_mag = np.sqrt(dvf_np[0]**2 + dvf_np[1]**2)

        # [Jacobian] LDMMorph 输出 DVF 是 normalized (Softsign ∈ [-1,1])，
        # pystrum.jacobian_determinant_vxm 内部 volsize2ndgrid 返回的是像素 grid
        # (step=1)，故先把 normalized disp 换算到像素: ×(size/2)。
        # 与 infer.py:199-200 论文原版口径一致。
        dvf_px = dvf_np.copy()
        dvf_px[0] = dvf_px[0] * img_h / 2.0
        dvf_px[1] = dvf_px[1] * img_w / 2.0
        jac_det = jacobian_determinant_vxm(dvf_px)

        if i == 0:
            print(f"    [DVF range] raw=[{dvf_np.min():.4f}, {dvf_np.max():.4f}]  (normalized)")
            print(f"    [Jac px    ] mean={jac_det.mean():.4f} min={jac_det.min():.4f} max={jac_det.max():.4f}  <- 像素域 Jacobian")

        print(f"    [DEBUG] Jac: min={jac_det.min():.4f}, max={jac_det.max():.4f}, mean={jac_det.mean():.4f}")

        n_foldings = int(np.sum(jac_det < 0))
        min_jac = float(jac_det.min())
        max_jac = float(jac_det.max())
        jac_neg_ratio = float(np.sum(jac_det < 0) / jac_det.size)

        dice_dict, masks_dict = compute_dice_with_mask(fix_np, mov_np, D_f_xy)
        all_dice_heart_before.append(dice_dict['heart']['before'])
        all_dice_heart_after.append(dice_dict['heart']['after'])
        all_dice_liver_before.append(dice_dict['liver']['before'])
        all_dice_liver_after.append(dice_dict['liver']['after'])
        all_dice_heart_coverage.append(dice_dict['heart']['coverage'])
        all_dice_heart_recall.append(dice_dict['heart']['recall'])
        all_dice_heart_iou.append(dice_dict['heart']['iou'])
        all_dice_liver_coverage.append(dice_dict['liver']['coverage'])
        all_dice_liver_recall.append(dice_dict['liver']['recall'])
        all_dice_liver_iou.append(dice_dict['liver']['iou'])
        all_min_jac.append(min_jac)
        all_max_jac.append(max_jac)
        all_n_foldings.append(n_foldings)
        all_jac_neg_ratio.append(jac_neg_ratio)

        _mask_heart_fixed  = masks_dict['heart']['fixed']
        _mask_liver_fixed  = masks_dict['liver']['fixed']
        _mask_heart_moving = masks_dict['heart']['moving']
        _mask_liver_moving = masks_dict['liver']['moving']
        _mask_heart_warped = masks_dict['heart']['warped']
        _mask_liver_warped = masks_dict['liver']['warped']

        print(f"    NCC (Full)      before: {ncc_before:.4f}  after: {ncc_after:.4f}  Δ: {ncc_after - ncc_before:+.4f}")
        print(f"    SSIM (Full)     before: {ssim_before:.4f}  after: {ssim_after:.4f}  Δ: {ssim_after - ssim_before:+.4f}")
        print(f"    NCC (Heart ROI) before: {ncc_heart_roi_before:.4f}  after: {ncc_heart_roi_after:.4f}  Δ: {ncc_heart_roi_after - ncc_heart_roi_before:+.4f}")
        print(f"    SSIM (Heart ROI)before: {ssim_heart_roi_before:.4f}  after: {ssim_heart_roi_after:.4f}  Δ: {ssim_heart_roi_after - ssim_heart_roi_before:+.4f}")
        print(f"    NCC (Liver ROI) before: {ncc_liver_roi_before:.4f}  after: {ncc_liver_roi_after:.4f}  Δ: {ncc_liver_roi_after - ncc_liver_roi_before:+.4f}")
        print(f"    SSIM (Liver ROI)before: {ssim_liver_roi_before:.4f}  after: {ssim_liver_roi_after:.4f}  Δ: {ssim_liver_roi_after - ssim_liver_roi_before:+.4f}")
        print(f"    Dice (Heart): {dice_dict['heart']['before']:.4f} -> {dice_dict['heart']['after']:.4f}  Δ: {dice_dict['heart']['delta']:+.4f}"
              f"  | fixed/warp/mov px = {int(dice_dict['heart']['fixed_mask_sum'])}/"
              f"{int(dice_dict['heart']['warped_mask_sum'])}/{int(dice_dict['heart']['moving_mask_sum'])}"
              f"  | cov={dice_dict['heart']['coverage']:.2f} rec={dice_dict['heart']['recall']:.2f}"
              f"  iou={dice_dict['heart']['iou']:.2f}")
        print(f"    Dice (Liver): {dice_dict['liver']['before']:.4f} -> {dice_dict['liver']['after']:.4f}  Δ: {dice_dict['liver']['delta']:+.4f}"
              f"  | fixed/warp/mov px = {int(dice_dict['liver']['fixed_mask_sum'])}/"
              f"{int(dice_dict['liver']['warped_mask_sum'])}/{int(dice_dict['liver']['moving_mask_sum'])}"
              f"  | cov={dice_dict['liver']['coverage']:.2f} rec={dice_dict['liver']['recall']:.2f}"
              f"  iou={dice_dict['liver']['iou']:.2f}")
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

        fig, axes = plt.subplots(4, 5, figsize=(30, 24))
        mode_label = 'XCAT Seq Motion' if use_xcat_seq else ('XCAT Motion' if use_xcat else 'NPZ Standard')
        fig.suptitle(
            f"{mode_label} | {split.upper()} | [{i+1}/{len(indices_to_vis)}] | {pairname}\n"
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
        for c in measure.find_contours(_mask_heart_moving, 0.5):
            axes[0, 0].plot(c[:, 1], c[:, 0], color='yellow', linewidth=1.2, linestyle='--', alpha=0.9)
        for c in measure.find_contours(_mask_liver_moving, 0.5):
            axes[0, 0].plot(c[:, 1], c[:, 0], color='magenta', linewidth=1.2, linestyle='--', alpha=0.9)

        axes[0, 1].imshow(fix_np, cmap='gray')
        axes[0, 1].set_title("Fixed (Y)", fontsize=11)
        axes[0, 1].axis('off')
        for c in measure.find_contours(_mask_heart_fixed, 0.5):
            axes[0, 1].plot(c[:, 1], c[:, 0], color='red', linewidth=1.6, alpha=0.95)
        for c in measure.find_contours(_mask_liver_fixed, 0.5):
            axes[0, 1].plot(c[:, 1], c[:, 0], color='magenta', linewidth=1.6, alpha=0.95)
        draw_liver_roi_box(axes[0, 1], _mask_liver_fixed, color='#FFD700',
                            label=f"liver ROI (NCC)", pad_px=12, line_w=2.2)

        axes[0, 2].imshow(diff_before, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 2].set_title(f"Abs Diff Before\nNCC: {ncc_before:.4f}", fontsize=11)
        axes[0, 2].axis('off')

        axes[0, 3].imshow(overlay_before)
        axes[0, 3].set_title("Overlay Before\n(Red=Moving)", fontsize=11)
        axes[0, 3].axis('off')

        axes[0, 4].imshow(_contour_overlay(fix_norm, _mask_heart_fixed, _mask_heart_moving))
        axes[0, 4].set_title(
            f"Mask Heart BEFORE  (red=fixed, yellow=moving)\nDice={dice_dict['heart']['before']:.3f}",
            fontsize=10
        )
        axes[0, 4].axis('off')

        # ---------- 第2行：配准后 ----------
        axes[1, 0].imshow(warp_np, cmap='gray')
        axes[1, 0].set_title(f"Warped (X->Y)\nNCC: {ncc_after:.4f}", fontsize=11)
        axes[1, 0].axis('off')
        for c in measure.find_contours(_mask_heart_warped, 0.5):
            axes[1, 0].plot(c[:, 1], c[:, 0], color='cyan', linewidth=1.2, linestyle='--', alpha=0.9)
        for c in measure.find_contours(_mask_liver_warped, 0.5):
            axes[1, 0].plot(c[:, 1], c[:, 0], color='magenta', linewidth=1.2, linestyle='--', alpha=0.9)

        axes[1, 1].imshow(fix_np, cmap='gray')
        axes[1, 1].set_title("Fixed (Y)", fontsize=11)
        axes[1, 1].axis('off')
        for c in measure.find_contours(_mask_heart_fixed, 0.5):
            axes[1, 1].plot(c[:, 1], c[:, 0], color='red', linewidth=1.6, alpha=0.95)
        for c in measure.find_contours(_mask_liver_fixed, 0.5):
            axes[1, 1].plot(c[:, 1], c[:, 0], color='magenta', linewidth=1.6, alpha=0.95)
        draw_liver_roi_box(axes[1, 1], _mask_liver_fixed, color='#FFD700',
                            label="liver ROI", pad_px=12, line_w=2.2)

        axes[1, 2].imshow(diff_after, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 2].set_title(f"Abs Diff After\n({ncc_after - ncc_before:+.4f})", fontsize=11)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(overlay_after)
        axes[1, 3].set_title("Overlay After\n(Red=Warped)", fontsize=11)
        axes[1, 3].axis('off')

        axes[1, 4].imshow(_contour_overlay(fix_norm, _mask_heart_fixed, _mask_heart_warped))
        axes[1, 4].set_title(
            f"Mask Heart AFTER  (red=fixed, cyan=warp(moving))\n"
            f"Dice={dice_dict['heart']['after']:.3f}  cov={dice_dict['heart']['coverage']:.2f} rec={dice_dict['heart']['recall']:.2f}",
            fontsize=9
        )
        axes[1, 4].axis('off')

        # ---------- 第3行：变形场 + Liver mask ----------
        axes[2, 0].imshow(warp_np, cmap='gray')
        axes[2, 0].imshow(warped_grid_np, cmap='gray', alpha=0.8)
        axes[2, 0].set_title("Warped Grid\non Image", fontsize=11)
        axes[2, 0].axis('off')

        dvf_display = np.sqrt(dvf_x**2 + dvf_y**2)
        im_dvf = axes[2, 1].imshow(dvf_display, cmap='jet')
        axes[2, 1].set_title("DVF Magnitude", fontsize=11)
        axes[2, 1].axis('off')
        plt.colorbar(im_dvf, ax=axes[2, 1], fraction=0.046, pad=0.04)

        dvf_x_norm = (dvf_x - dvf_x.min()) / (dvf_x.max() - dvf_x.min() + 1e-8)
        dvf_y_norm = (dvf_y - dvf_y.min()) / (dvf_y.max() - dvf_y.min() + 1e-8)
        dvf_rgb = np.stack([dvf_x_norm, dvf_y_norm, np.zeros_like(dvf_x)], axis=-1)
        axes[2, 2].imshow(dvf_rgb)
        axes[2, 2].set_title("DVF RGB\n(R=X, G=Y, B=0)", fontsize=11)
        axes[2, 2].axis('off')

        im_dvf_x = axes[2, 3].imshow(dvf_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
        axes[2, 3].set_title("DVF X (L-R)", fontsize=11)
        axes[2, 3].axis('off')
        plt.colorbar(im_dvf_x, ax=axes[2, 3], fraction=0.046, pad=0.04)

        axes[2, 4].imshow(_contour_overlay(fix_norm, _mask_liver_fixed, _mask_liver_warped))
        axes[2, 4].set_title(
            f"Mask Liver AFTER  (red=fixed, cyan=warp(moving))\n"
            f"Dice={dice_dict['liver']['after']:.3f}  cov={dice_dict['liver']['coverage']:.2f} rec={dice_dict['liver']['recall']:.2f}",
            fontsize=9
        )
        axes[2, 4].axis('off')

        # ---------- 第4行：原始灰度图对比 (Moving, Fixed, Warped) ----------
        # Moving
        axes[3, 0].imshow(mov_np, cmap='gray')
        axes[3, 0].set_title(f"Moving (X)\n{pairname}", fontsize=11)
        axes[3, 0].axis('off')

        # Fixed
        axes[3, 1].imshow(fix_np, cmap='gray')
        axes[3, 1].set_title(f"Fixed (Y)", fontsize=11)
        axes[3, 1].axis('off')

        # Warped
        axes[3, 2].imshow(warp_np, cmap='gray')
        axes[3, 2].set_title(f"Warped (X->Y)", fontsize=11)
        axes[3, 2].axis('off')

        # 三图对比 (Moving vs Fixed vs Warped)
        axes[3, 3].imshow(np.concatenate([mov_np, fix_np, warp_np], axis=1), cmap='gray')
        axes[3, 3].set_title("M vs F vs W", fontsize=11)
        axes[3, 3].axis('off')
        # 添加分界线
        h, w = mov_np.shape
        axes[3, 3].axvline(x=w-0.5, color='white', linewidth=2)
        axes[3, 3].axvline(x=2*w-0.5, color='white', linewidth=2)
        axes[3, 3].text(w//2, h+15, 'M', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
        axes[3, 3].text(3*w//2, h+15, 'F', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
        axes[3, 3].text(5*w//2, h+15, 'W', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

        # Diff 对比 (Before vs After)
        diff_combined = np.concatenate([diff_before, diff_after], axis=1)
        im_diff = axes[3, 4].imshow(diff_combined, cmap='hot', vmin=0, vmax=0.3)
        axes[3, 4].set_title(f"Diff: Before | After", fontsize=11)
        axes[3, 4].axis('off')
        plt.colorbar(im_diff, ax=axes[3, 4], fraction=0.046, pad=0.04)
        axes[3, 4].axvline(x=w-0.5, color='cyan', linewidth=2)
        axes[3, 4].text(w//2, -15, 'Before', ha='center', va='top', color='cyan', fontsize=10)
        axes[3, 4].text(3*w//2, -15, 'After', ha='center', va='top', color='cyan', fontsize=10)

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
        print(f"  NCC (Full)      : {np.mean(all_ncc_before):.4f} -> {np.mean(all_ncc_after):.4f}  "
              f"({np.mean(np.array(all_ncc_after) - np.array(all_ncc_before)):+.4f})")
        print(f"  SSIM (Full)     : {np.mean(all_ssim_before):.4f} -> {np.mean(all_ssim_after):.4f}  "
              f"({np.mean(np.array(all_ssim_after) - np.array(all_ssim_before)):+.4f})")
        print(f"  NCC (Heart ROI) : {np.mean(all_ncc_heart_roi_before):.4f} -> {np.mean(all_ncc_heart_roi_after):.4f}  "
              f"({np.mean(np.array(all_ncc_heart_roi_after) - np.array(all_ncc_heart_roi_before)):+.4f})")
        print(f"  SSIM (Heart ROI): {np.mean(all_ssim_heart_roi_before):.4f} -> {np.mean(all_ssim_heart_roi_after):.4f}  "
              f"({np.mean(np.array(all_ssim_heart_roi_after) - np.array(all_ssim_heart_roi_before)):+.4f})")
        print(f"  NCC (Liver ROI) : {np.mean(all_ncc_liver_roi_before):.4f} -> {np.mean(all_ncc_liver_roi_after):.4f}  "
              f"({np.mean(np.array(all_ncc_liver_roi_after) - np.array(all_ncc_liver_roi_before)):+.4f})")
        print(f"  SSIM (Liver ROI): {np.mean(all_ssim_liver_roi_before):.4f} -> {np.mean(all_ssim_liver_roi_after):.4f}  "
              f"({np.mean(np.array(all_ssim_liver_roi_after) - np.array(all_ssim_liver_roi_before)):+.4f})")
        print(f"  Dice (Heart)     : {np.mean(all_dice_heart_before):.4f} -> {np.mean(all_dice_heart_after):.4f}  "
              f"({np.mean(np.array(all_dice_heart_after) - np.array(all_dice_heart_before)):+.4f})"
              f"   | cov={np.mean(all_dice_heart_coverage):.2f} rec={np.mean(all_dice_heart_recall):.2f} iou={np.mean(all_dice_heart_iou):.3f}")
        print(f"  Dice (Liver)     : {np.mean(all_dice_liver_before):.4f} -> {np.mean(all_dice_liver_after):.4f}  "
              f"({np.mean(np.array(all_dice_liver_after) - np.array(all_dice_liver_before)):+.4f})"
              f"   | cov={np.mean(all_dice_liver_coverage):.2f} rec={np.mean(all_dice_liver_recall):.2f} iou={np.mean(all_dice_liver_iou):.3f}")
        print(f"  Min / Max NCC: {np.min(all_ncc_after):.4f} / {np.max(all_ncc_after):.4f}")
        print(f"  Jacobian Det  : min={np.min(all_min_jac):.4f}  max={np.max(all_max_jac):.4f}  "
              f"total_foldings={np.sum(all_n_foldings)}  neg_ratio={np.mean(all_jac_neg_ratio)*100:.2f}%")

    # 保存统计到 CSV
    stats_csv = os.path.join(split_save_dir, 'stats.csv')
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Pairname', 'NCC_Before', 'NCC_After', 'NCC_Delta',
                         'SSIM_Before', 'SSIM_After', 'SSIM_Delta',
                         'NCC_HeartROI_Before', 'NCC_HeartROI_After', 'NCC_HeartROI_Delta',
                         'SSIM_HeartROI_Before', 'SSIM_HeartROI_After', 'SSIM_HeartROI_Delta',
                         'NCC_LiverROI_Before', 'NCC_LiverROI_After', 'NCC_LiverROI_Delta',
                         'SSIM_LiverROI_Before', 'SSIM_LiverROI_After', 'SSIM_LiverROI_Delta',
                         'Dice_Heart_Before', 'Dice_Heart_After', 'Dice_Heart_Delta',
                         'Dice_Heart_Cov', 'Dice_Heart_Rec', 'Dice_Heart_IoU',
                         'Dice_Liver_Before', 'Dice_Liver_After', 'Dice_Liver_Delta',
                         'Dice_Liver_Cov', 'Dice_Liver_Rec', 'Dice_Liver_IoU',
                         'Min_Jac', 'N_Foldings', 'Jac_Neg_Ratio'])
        for i, idx in enumerate(indices_to_vis):
            item_i = dataset[idx]
            if isinstance(item_i, dict):
                pairname_i = item_i.get("pairname", f"sample_{idx}")
            else:
                pairname_i = item_i[4]
            writer.writerow([
                idx,
                pairname_i,
                f"{all_ncc_before[i]:.4f}",
                f"{all_ncc_after[i]:.4f}",
                f"{all_ncc_after[i] - all_ncc_before[i]:+.4f}",
                f"{all_ssim_before[i]:.4f}",
                f"{all_ssim_after[i]:.4f}",
                f"{all_ssim_after[i] - all_ssim_before[i]:+.4f}",
                f"{all_ncc_heart_roi_before[i]:.4f}",
                f"{all_ncc_heart_roi_after[i]:.4f}",
                f"{all_ncc_heart_roi_after[i] - all_ncc_heart_roi_before[i]:.4f}",
                f"{all_ssim_heart_roi_before[i]:.4f}",
                f"{all_ssim_heart_roi_after[i]:.4f}",
                f"{all_ssim_heart_roi_after[i] - all_ssim_heart_roi_before[i]:.4f}",
                f"{all_ncc_liver_roi_before[i]:.4f}",
                f"{all_ncc_liver_roi_after[i]:.4f}",
                f"{all_ncc_liver_roi_after[i] - all_ncc_liver_roi_before[i]:.4f}",
                f"{all_ssim_liver_roi_before[i]:.4f}",
                f"{all_ssim_liver_roi_after[i]:.4f}",
                f"{all_ssim_liver_roi_after[i] - all_ssim_liver_roi_before[i]:.4f}",
                f"{all_dice_heart_before[i]:.4f}",
                f"{all_dice_heart_after[i]:.4f}",
                f"{all_dice_heart_after[i] - all_dice_heart_before[i]:.4f}",
                f"{all_dice_heart_coverage[i]:.3f}",
                f"{all_dice_heart_recall[i]:.3f}",
                f"{all_dice_heart_iou[i]:.3f}",
                f"{all_dice_liver_before[i]:.4f}",
                f"{all_dice_liver_after[i]:.4f}",
                f"{all_dice_liver_after[i] - all_dice_liver_before[i]:.4f}",
                f"{all_dice_liver_coverage[i]:.3f}",
                f"{all_dice_liver_recall[i]:.3f}",
                f"{all_dice_liver_iou[i]:.3f}",
                f"{all_min_jac[i]:.4f}",
                all_n_foldings[i],
                f"{all_jac_neg_ratio[i]*100:.2f}",
            ])
    print(f"  Stats saved: {stats_csv}")

print(f"\nAll done! Results saved to: {opt.save_dir}")