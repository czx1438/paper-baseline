"""
配准可视化 / 评估脚本（支持 XCAT 运动增强版 和 普通 NPZ 模式）

相对你上一版的修改（都用 [FIX] 标注）:
  [FIX-JAC-1] 启动时跑一次雅可比 sanity check：恒等→1.0、10%缩放→1.21，
              用的是 jacobian_determinant_vxm 的正确分量约定（分量0=行/y, 分量1=列/x）。
  [FIX-JAC-2] 新增 --dvf_already_pixel 开关 + 第一个样本打印“原始DVF范围 + 两种缩放的雅可比统计”，
              用来确认 DVF 单位（归一化 vs 像素）。默认按归一化处理（×size/2）。
  [FIX-JAC-3] 折叠统一用 < 0（n_foldings 和 neg_ratio 一致；论文若用 ≤0 自行改）。
  [FIX-LDM ]  加载 LDM 后打印 scale_factor，≈1.0 时报警（说明没加载到 scale_by_std 的缩放）。
  [FIX-DICE]  Dice 改成 warp moving 的 mask（再阈值），而不是对 warped 图像重新分割；
              阈值百分位改成可配 --dice_percentile。
  [FIX-MASK]  create_mask 的 (h,w) 顺序修正。

用法:
  XCAT 模式:  python visualize_registration_xcat.py --xcat --resume <pth> --xcat_path <path> \
                  --ldm_config <带scale_by_std的yaml> --ldm_checkpoint <修好的LDM.ckpt>
  NPZ 模式:   python visualize_registration_xcat.py --no-xcat --resume <pth> --datapath <path> \
                  --ldm_config <yaml> --ldm_checkpoint <ckpt>
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

from utils.utils import (Dataset_XCAT_Registration, Dataset_epoch_with_name,
                         SpatialTransform, jacobian_determinant_vxm)
import TransModels.LDMMorph as LDMMorph
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


# ======================== 参数配置 ========================
parser = argparse.ArgumentParser()

parser.add_argument("--resume", type=str, dest="resume", default='',
                    help="配准网络 checkpoint 路径")

# 模式切换
parser.add_argument("--xcat", action="store_true", dest="xcat",
                    help="使用 XCAT 运动增强数据集")
parser.add_argument("--no-xcat", action="store_true", dest="no_xcat",
                    help="使用普通 NPZ 数据集（会覆盖 --xcat）")

# XCAT 模式参数
parser.add_argument("--xcat_path", type=str, dest="xcat_path",
                    default='/home/b109/Desktop/czx/LDM-Morph-main_heart/datasets/XCAT',
                    help="XCAT 数据根目录")

# NPZ 模式参数
parser.add_argument("--datapath", type=str, dest="datapath",
                    default='/home/b109/Desktop/czx/LDM-Morph-main_heart/datasets/XCAT/prep',
                    help="NPZ 数据根目录（含 *_pair.npz）")

# LDM 配置 / checkpoint —— 必须和训练配准时用的完全一致（带 scale_by_std）
parser.add_argument("--ldm_config", type=str, dest="ldm_config", default=None,
                    help="LDM 配置文件路径（务必是带 scale_by_std 的那个）")
parser.add_argument("--ldm_checkpoint", type=str, dest="ldm_checkpoint", default=None,
                    help="LDM checkpoint 路径（修好的那个，必填以保证特征正确）")

# 训练相关参数
parser.add_argument("--smooth", type=float, default=0.1)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--t_enc", type=int, default=1)

# [FIX-JAC-2] DVF 单位开关：默认认为 DVF 是归一化坐标（需 ×size/2 转像素）；
#             若你的 SpatialTransform 直接吃像素位移，加 --dvf_already_pixel 跳过缩放
parser.add_argument("--dvf_already_pixel", action="store_true",
                    help="DVF 已是像素单位，跳过 ×size/2 缩放")

# [FIX-DICE] 亮度 mask 阈值百分位（保留“比该百分位亮”的像素）
parser.add_argument("--dice_percentile", type=float, default=25.0,
                    help="器官亮度 mask 的百分位阈值。越高 mask 越只圈最亮的器官")

# 可视化范围
parser.add_argument("--n_samples", type=int, default=154)
parser.add_argument("--split", type=str, default='test', choices=['train', 'val', 'test'])
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--motion_type", type=str, default='identity',
                    choices=['identity', 'rotate10', 'scale05', 'warp', None])
parser.add_argument("--save_dir", type=str, dest="save_dir", default=None)
opt, unknown = parser.parse_known_args()

use_xcat = opt.xcat and not opt.no_xcat

if opt.ldm_config is None:
    if use_xcat:
        opt.ldm_config = './configs/latent-diffusion/xcat_motion-ldm.yaml'
    else:
        opt.ldm_config = './configs/latent-diffusion/xcat_no_motion.yaml'

if opt.save_dir is None:
    if use_xcat:
        opt.save_dir = f'./logs/visualization_xcat_motion_{opt.start_idx}_{opt.start_idx + opt.n_samples}/'
    else:
        opt.save_dir = './logs/visualization_npz/'

print(f"\n{'='*60}")
print(f"Mode: {'XCAT Motion-Augmented' if use_xcat else 'NPZ Standard'}")
print(f"Resume: {opt.resume}")
print(f"LDM Config: {opt.ldm_config}")
print(f"DVF units: {'PIXEL (no scaling)' if opt.dvf_already_pixel else 'NORMALIZED (x size/2)'}")
print(f"Save Dir: {opt.save_dir}")
print(f"{'='*60}\n")


# ======================== [FIX-JAC-1] 雅可比 sanity check ========================
def sanity_check_jacobian():
    """验证 jacobian_determinant_vxm：恒等→1.0，10%均匀扩张→1.21。
    注意函数约定：分量0 ↔ 行(y/axis0)，分量1 ↔ 列(x/axis1)。"""
    H = W = 256
    print(f"{'='*60}\n[Jacobian Sanity Check]\n{'='*60}")

    # 恒等形变（零位移）→ |J| 必须处处 = 1.0
    j0 = jacobian_determinant_vxm(np.zeros((2, H, W), np.float32))
    print(f"  identity        -> mean={j0.mean():.4f} min={j0.min():.4f} max={j0.max():.4f}  (期望 1.0)")

    # 10% 均匀扩张（像素单位，按函数约定：分量0沿行、分量1沿列）→ |J| ≈ 1.21
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    t = np.zeros((2, H, W), np.float32)
    t[0] = 0.1 * yy   # 分量0 = 行(y)位移
    t[1] = 0.1 * xx   # 分量1 = 列(x)位移
    j1 = jacobian_determinant_vxm(t)
    ok_id = abs(j0.mean() - 1.0) < 1e-2
    ok_sc = abs(j1.mean() - 1.21) < 5e-2
    print(f"  scale s=1.1     -> mean={j1.mean():.4f}                       (期望 1.21)")
    print(f"  => {'PASS ✅ 函数正确' if (ok_id and ok_sc) else 'FAIL ❌ 检查格式/约定'}")
    print(f"{'='*60}\n")


sanity_check_jacobian()


def compute_jac_det(dvf_np, h, w, already_pixel=False):
    """[FIX-JAC-2/3] 统一的雅可比计算。
    dvf_np: [2, H, W]。函数约定 分量0↔行(高h)，分量1↔列(宽w)。
    归一化坐标需 ×size/2 转像素再喂给 jacobian_determinant_vxm。"""
    d = dvf_np.copy()
    if not already_pixel:
        d[0] = d[0] * h / 2.0   # 分量0 = 行/y → 高 h
        d[1] = d[1] * w / 2.0   # 分量1 = 列/x → 宽 w
    jac = jacobian_determinant_vxm(d)            # [H, W]
    n_fold = int(np.sum(jac < 0))                # 统一用 < 0
    neg_ratio = float(np.sum(jac < 0) / jac.size)
    return jac, n_fold, neg_ratio


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
    pl_sd = torch.load(opt.ldm_checkpoint, map_location="cpu")
    ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
    print(f"LDM loaded: {opt.ldm_checkpoint}")
elif opt.resume and os.path.dirname(opt.resume):
    candidates = glob.glob(os.path.join(os.path.dirname(opt.resume), '..', '..', '..', 'checkpoints', 'last.ckpt'))
    if candidates:
        pl_sd = torch.load(candidates[0], map_location="cpu")
        ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
        print(f"LDM loaded (auto): {candidates[0]}")
    else:
        print("WARNING: No LDM checkpoint found, using RANDOM LDM (结果无意义!)")
else:
    print("WARNING: No LDM checkpoint specified, using RANDOM LDM (结果无意义!)")

# [FIX-LDM] 确认 scale_factor —— 必须 ≈ 训练时的值(例如 ~41.7)，≈1.0 说明没加载到缩放
try:
    sf = float(ldm_model.scale_factor)
    print(f"[LDM] scale_factor = {sf:.4f}")
    if abs(sf - 1.0) < 0.1:
        print("  ⚠️  scale_factor ≈ 1.0：很可能没加载到 scale_by_std 的缩放！")
        print("      检查 --ldm_config 是不是带 scale_by_std 的那个、且 checkpoint 里有该 buffer。")
        print("      此时抽出的特征会和训练配准时不一致，评估不可信。")
except Exception as e:
    print(f"[LDM] 读取 scale_factor 失败: {e}")


# ======================== 配准网络加载 ========================
model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2).cuda()
if os.path.isfile(opt.resume):
    model.load_state_dict(torch.load(opt.resume, map_location="cuda"))
    print(f"Registration model loaded: {opt.resume}")
else:
    print(f"WARNING: checkpoint not found: {opt.resume}, using random init")

model.eval()
transform = SpatialTransform().cuda()
for param in transform.parameters():
    param.requires_grad = False


# ======================== 辅助函数 ========================
def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i+line_thickness-1] = 1
    return grid_img


def ncc_metric(fixed, moving, win_size=9):
    """局部 NCC（全图平均）"""
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


def create_mask(img, img_size):
    """心脏区域 ROI 掩码。 [FIX-MASK] img_size=(H, W) 顺序统一。"""
    H, W = img_size
    y_start, y_end = int(H * 0.30), int(H * 0.75)
    x_start, x_end = int(W * 0.35), int(W * 0.75)
    mask = torch.zeros_like(img)
    mask[..., y_start:y_end, x_start:x_end] = 1.0
    return mask


def red_overlay(fixed, moving, alpha=0.6):
    f_min, f_max = fixed.min(), fixed.max()
    f = np.clip((fixed - f_min) / (f_max - f_min + 1e-8), 0, 1)
    m = np.clip((moving - f_min) / (f_max - f_min + 1e-8), 0, 1)
    rgb = np.stack([f, f, f], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] * (1 - alpha) + m * alpha, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] * (1 - alpha), 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] * (1 - alpha), 0, 1)
    return rgb


# ======================== Dice（亮度阈值 mask，无真值时的近似） ========================
DEFAULT_ROI = {
    'heart': (250, 180, 340, 370),   # x1, y1, x2, y2
    'liver': (150, 250, 300, 520),
}


def create_brightness_mask(img_np, organ_name, percentile):
    mask = np.zeros_like(img_np)
    x1, y1, x2, y2 = DEFAULT_ROI[organ_name]
    roi = img_np[y1:y2, x1:x2]
    threshold = np.percentile(roi, percentile)
    mask[y1:y2, x1:x2] = (roi > threshold).astype(np.float32)
    return mask


def dice_score(pred, target):
    inter = np.sum(pred * target)
    return 2.0 * inter / (np.sum(pred) + np.sum(target) + 1e-8)


def warp_mask_np(mask_np, D_f_xy):
    """[FIX-DICE] 用形变场 warp 一个 numpy mask（双线性后阈值 0.5）。"""
    mt = torch.from_numpy(mask_np)[None, None].cuda().float()
    _, mw = transform(mt, D_f_xy.permute(0, 2, 3, 1))
    return (mw.squeeze().cpu().numpy() > 0.5).astype(np.float32)


def compute_dice(fix_np, mov_np, D_f_xy, percentile):
    """[FIX-DICE] 配准后 Dice = dice(fixed_mask, warp(moving_mask))，
    而不是对 warped 图像重新分割。"""
    dice_dict = {}
    for organ in ['heart', 'liver']:
        mask_fixed = create_brightness_mask(fix_np, organ, percentile)
        mask_moving = create_brightness_mask(mov_np, organ, percentile)
        mask_warped = warp_mask_np(mask_moving, D_f_xy)
        dice_before = dice_score(mask_fixed, mask_moving)
        dice_after = dice_score(mask_fixed, mask_warped)
        dice_dict[organ] = {'before': dice_before, 'after': dice_after,
                            'delta': dice_after - dice_before}
    return dice_dict


# ======================== 数据加载 ========================
def build_dataset(split):
    if use_xcat:
        motion_types = [opt.motion_type] if opt.motion_type else ['identity']
        flip_p = 0.5 if split == 'train' else 0.0
        ds = Dataset_XCAT_Registration(data_root=opt.xcat_path, split=split,
                                       motion_types=motion_types, flip_p=flip_p)
        return ds, f"XCAT motion_types={motion_types}"
    else:
        base_dir = opt.datapath.rstrip('/')
        split_dir = os.path.join(base_dir, split)
        npz_files = sorted(glob.glob(os.path.join(split_dir, '*_pair.npz')))
        if len(npz_files) == 0:
            all_files = sorted(glob.glob(os.path.join(base_dir, '*_pair.npz')))
            if len(all_files) == 0:
                raise FileNotFoundError(f"No *_pair.npz in {base_dir}/ or {split_dir}/")
            n = len(all_files)
            train_end, val_end = int(n * 0.70), int(n * 0.85)
            if split == 'train':
                npz_files = all_files[:train_end]
            elif split == 'val':
                npz_files = all_files[train_end:val_end]
            else:
                npz_files = all_files[val_end:]
            return Dataset_epoch_with_name(npz_files), f"NPZ flat {split}={len(npz_files)}/{n}"
        return Dataset_epoch_with_name(npz_files), f"NPZ split {split}={len(npz_files)}"


# ======================== 主循环 ========================
print(f"[Mask] percentile={opt.dice_percentile}  (保留框内最亮的 {100-opt.dice_percentile:.0f}%)")
os.makedirs(opt.save_dir, exist_ok=True)

for split in [opt.split]:
    dataset, dataset_info = build_dataset(split)

    start = opt.start_idx if opt.start_idx < len(dataset) else 0
    end = min(start + opt.n_samples, len(dataset))
    indices_to_vis = list(range(start, end))

    split_save_dir = os.path.join(opt.save_dir, split)
    os.makedirs(split_save_dir, exist_ok=True)

    print(f"\n{'='*60}\nProcessing split: {split}\nDataset: {dataset_info}")
    print(f"Visualizing index {start}..{end-1} ({len(indices_to_vis)} samples)\n{'='*60}")

    all_ncc_before, all_ncc_after = [], []
    all_ncc_roi_before, all_ncc_roi_after = [], []
    all_dice_heart_before, all_dice_heart_after = [], []
    all_dice_liver_before, all_dice_liver_after = [], []
    all_min_jac, all_n_foldings, all_jac_neg_ratio = [], [], []

    for i, idx in enumerate(indices_to_vis):
        X, Y, segx, segy, pairname = dataset[idx]

        if X.dim() == 2:
            X = X.unsqueeze(0).float().cuda(); Y = Y.unsqueeze(0).float().cuda()
        else:
            X = X.float().cuda(); Y = Y.float().cuda()
        if X.dim() == 3:
            X = X.unsqueeze(1); Y = Y.unsqueeze(1)

        print(f"\n[{i+1}/{len(indices_to_vis)}] idx={idx} pairname={pairname}")

        # LDM 特征
        mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
        fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()
        noise = torch.randn_like(mov_z)
        x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([opt.t_enc]).cuda(), noise=noise)
        y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([opt.t_enc]).cuda(), noise=noise)
        outx = ldm_model.apply_model(x_noisy, t=torch.tensor([opt.t_enc]).cuda(), cond=None, return_ids=True)
        outy = ldm_model.apply_model(y_noisy, t=torch.tensor([opt.t_enc]).cuda(), cond=None, return_ids=True)
        score0 = torch.cat((outx[1][0][0], outx[1][0][2], outy[1][0][0], outy[1][0][2]), dim=1)
        score1 = torch.cat((outx[1][0][3], outx[1][0][5], outy[1][0][3], outy[1][0][5]), dim=1)
        score2 = torch.cat((outx[1][0][6], outx[1][0][8], outy[1][0][6], outy[1][0][8]), dim=1)
        score3 = torch.cat((outx[1][0][9], outx[1][0][11], outy[1][0][9], outy[1][0][11]), dim=1)

        img_h, img_w = Y.shape[2], Y.shape[3]

        with torch.no_grad():
            D_f_xy = model(X, Y, score0, score1, score2, score3)
            _, warped_X = transform(X, D_f_xy.permute(0, 2, 3, 1))
            grid_img_np = mk_grid_img(grid_step=24, line_thickness=2, grid_sz=(img_h, img_w))
            grid_img_tensor = torch.from_numpy(grid_img_np[np.newaxis, np.newaxis, ...]).cuda().float()
            _, warped_grid = transform(grid_img_tensor, D_f_xy.permute(0, 2, 3, 1))
            warped_grid_np = warped_grid.squeeze().cpu().numpy()

        # NCC
        ncc_before = ncc_metric(Y, X)
        ncc_after = ncc_metric(Y, warped_X)
        if img_h >= 256 and img_w >= 256:
            roi_mask = create_mask(Y, img_size=(img_h, img_w)).cuda()
            f_roi, m_roi, w_roi = Y[roi_mask == 1], X[roi_mask == 1], warped_X[roi_mask == 1]
            fm, mm, wm = f_roi.mean(), m_roi.mean(), w_roi.mean()
            ncc_roi_before = (((f_roi-fm)*(m_roi-mm)).sum() /
                              (torch.sqrt(((f_roi-fm)**2).sum())*torch.sqrt(((m_roi-mm)**2).sum())+1e-8)).item()
            ncc_roi_after = (((f_roi-fm)*(w_roi-wm)).sum() /
                             (torch.sqrt(((f_roi-fm)**2).sum())*torch.sqrt(((w_roi-wm)**2).sum())+1e-8)).item()
        else:
            ncc_roi_before, ncc_roi_after = ncc_before, ncc_after

        all_ncc_before.append(ncc_before); all_ncc_after.append(ncc_after)
        all_ncc_roi_before.append(ncc_roi_before); all_ncc_roi_after.append(ncc_roi_after)

        mov_np = X.squeeze().cpu().numpy()
        fix_np = Y.squeeze().cpu().numpy()
        warp_np = warped_X.squeeze().cpu().numpy()
        dvf_np = D_f_xy.squeeze().cpu().numpy()
        dvf_x, dvf_y = dvf_np[0], dvf_np[1]

        # [FIX-JAC-2] 第一个样本：打印原始 DVF 范围 + 两种缩放的雅可比统计，确认单位
        if i == 0:
            print(f"    [DVF range] raw min={dvf_np.min():.4f} max={dvf_np.max():.4f} "
                  f"absmax={np.abs(dvf_np).max():.4f}  (≲1→归一化; ≳10→已是像素)")
            for nm, ap in [('raw(归一化?)', True), ('×size/2(像素)', False)]:
                jj, _, _ = compute_jac_det(dvf_np, img_h, img_w, already_pixel=ap)
                print(f"    [Jac试算-{nm}] mean={jj.mean():.4f} min={jj.min():.4f} max={jj.max():.4f}  "
                      f"(正确应 mean≈1, max>1)")

        # 雅可比（按 --dvf_already_pixel 选择）
        jac_det, n_foldings, jac_neg_ratio = compute_jac_det(
            dvf_np, img_h, img_w, already_pixel=opt.dvf_already_pixel)
        min_jac = float(jac_det.min())

        # Dice（warp moving mask）
        dice_dict = compute_dice(fix_np, mov_np, D_f_xy, opt.dice_percentile)
        all_dice_heart_before.append(dice_dict['heart']['before'])
        all_dice_heart_after.append(dice_dict['heart']['after'])
        all_dice_liver_before.append(dice_dict['liver']['before'])
        all_dice_liver_after.append(dice_dict['liver']['after'])
        all_min_jac.append(min_jac); all_n_foldings.append(n_foldings)
        all_jac_neg_ratio.append(jac_neg_ratio)

        print(f"    NCC full {ncc_before:.4f}->{ncc_after:.4f} | ROI {ncc_roi_before:.4f}->{ncc_roi_after:.4f}")
        print(f"    Dice heart {dice_dict['heart']['before']:.4f}->{dice_dict['heart']['after']:.4f} | "
              f"liver {dice_dict['liver']['before']:.4f}->{dice_dict['liver']['after']:.4f}")
        print(f"    Jac: min={min_jac:.4f}  folds={n_foldings}  neg_ratio={jac_neg_ratio*100:.3f}%")

        # ---------- 绘图 ----------
        fix_norm, mov_norm, warp_norm = np.clip(fix_np, 0, 1), np.clip(mov_np, 0, 1), np.clip(warp_np, 0, 1)
        overlay_before, overlay_after = red_overlay(fix_norm, mov_norm), red_overlay(fix_norm, warp_norm)
        diff_before, diff_after = np.abs(fix_np - mov_np), np.abs(fix_np - warp_np)
        vmax_x = max(abs(dvf_x.min()), abs(dvf_x.max()), 0.01)

        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        mode_label = 'XCAT Motion' if use_xcat else 'NPZ Standard'
        fig.suptitle(
            f"{mode_label} | {split.upper()} | [{i+1}/{len(indices_to_vis)}] | {pairname}\n"
            f"NCC: {ncc_before:.4f}->{ncc_after:.4f} | ROI: {ncc_roi_before:.4f}->{ncc_roi_after:.4f}\n"
            f"Heart Dice: {dice_dict['heart']['before']:.4f}->{dice_dict['heart']['after']:.4f} | "
            f"Liver Dice: {dice_dict['liver']['before']:.4f}->{dice_dict['liver']['after']:.4f} | "
            f"Neg Ratio: {jac_neg_ratio*100:.3f}% (min J={min_jac:.3f})",
            fontsize=12, fontweight='bold', y=0.99)

        def draw_roi(ax):
            for name, coords in DEFAULT_ROI.items():
                x1, y1, x2, y2 = coords
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                             edgecolor='lime' if name == 'heart' else 'cyan',
                             linewidth=2.5, linestyle='--'))

        axes[0, 0].imshow(mov_np, cmap='gray'); axes[0, 0].set_title(f"Moving (X)\n{pairname}"); draw_roi(axes[0, 0])
        axes[0, 1].imshow(fix_np, cmap='gray'); axes[0, 1].set_title("Fixed (Y)"); draw_roi(axes[0, 1])
        axes[0, 2].imshow(diff_before, cmap='hot', vmin=0, vmax=0.3); axes[0, 2].set_title(f"Abs Diff Before\nNCC {ncc_before:.4f}")
        axes[0, 3].imshow(overlay_before); axes[0, 3].set_title("Overlay Before (Red=Moving)")

        axes[1, 0].imshow(warp_np, cmap='gray'); axes[1, 0].set_title(f"Warped (X->Y)\nNCC {ncc_after:.4f}"); draw_roi(axes[1, 0])
        axes[1, 1].imshow(fix_np, cmap='gray'); axes[1, 1].set_title("Fixed (Y)"); draw_roi(axes[1, 1])
        axes[1, 2].imshow(diff_after, cmap='hot', vmin=0, vmax=0.3); axes[1, 2].set_title(f"Abs Diff After\n({ncc_after-ncc_before:+.4f})")
        axes[1, 3].imshow(overlay_after); axes[1, 3].set_title("Overlay After (Red=Warped)")

        axes[2, 0].imshow(warp_np, cmap='gray'); axes[2, 0].imshow(warped_grid_np, cmap='gray', alpha=0.8)
        axes[2, 0].set_title("Warped Grid on Image")
        im1 = axes[2, 1].imshow(np.sqrt(dvf_x**2 + dvf_y**2), cmap='jet'); axes[2, 1].set_title("DVF Magnitude")
        plt.colorbar(im1, ax=axes[2, 1], fraction=0.046, pad=0.04)
        # 雅可比图（folding 用红色高亮）
        im2 = axes[2, 2].imshow(jac_det, cmap='RdBu_r', vmin=-1, vmax=2)
        axes[2, 2].set_title(f"Jacobian Det\n(red<0 = fold)")
        plt.colorbar(im2, ax=axes[2, 2], fraction=0.046, pad=0.04)
        im3 = axes[2, 3].imshow(dvf_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x); axes[2, 3].set_title("DVF X")
        plt.colorbar(im3, ax=axes[2, 3], fraction=0.046, pad=0.04)
        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        out_path = os.path.join(split_save_dir, f"sample_{i:03d}_{pairname}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()

    # ---------- 统计摘要 ----------
    if len(all_ncc_before) > 0:
        print(f"\n{'='*60}\n[{split.upper()}] Statistics ({len(all_ncc_before)} samples)\n{'='*60}")
        print(f"  NCC (Full)  : {np.mean(all_ncc_before):.4f} -> {np.mean(all_ncc_after):.4f}  "
              f"({np.mean(np.array(all_ncc_after)-np.array(all_ncc_before)):+.4f})")
        print(f"  NCC (ROI)   : {np.mean(all_ncc_roi_before):.4f} -> {np.mean(all_ncc_roi_after):.4f}  "
              f"({np.mean(np.array(all_ncc_roi_after)-np.array(all_ncc_roi_before)):+.4f})")
        print(f"  Dice (Heart): {np.mean(all_dice_heart_before):.4f} -> {np.mean(all_dice_heart_after):.4f}  "
              f"({np.mean(np.array(all_dice_heart_after)-np.array(all_dice_heart_before)):+.4f})")
        print(f"  Dice (Liver): {np.mean(all_dice_liver_before):.4f} -> {np.mean(all_dice_liver_after):.4f}  "
              f"({np.mean(np.array(all_dice_liver_after)-np.array(all_dice_liver_before)):+.4f})")
        print(f"  Jacobian Det: min={np.min(all_min_jac):.4f}  "
              f"total_foldings={int(np.sum(all_n_foldings))}  neg_ratio={np.mean(all_jac_neg_ratio)*100:.3f}%")

    # ---------- CSV ----------
    stats_csv = os.path.join(split_save_dir, 'stats.csv')
    with open(stats_csv, 'w', newline='') as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(['Index', 'Pairname', 'NCC_Before', 'NCC_After',
                     'ROI_Before', 'ROI_After',
                     'Dice_Heart_Before', 'Dice_Heart_After',
                     'Dice_Liver_Before', 'Dice_Liver_After',
                     'Min_Jac', 'N_Foldings', 'Jac_Neg_Ratio(%)'])
        for i, idx in enumerate(indices_to_vis):
            wr.writerow([idx, dataset[idx][4],
                         f"{all_ncc_before[i]:.4f}", f"{all_ncc_after[i]:.4f}",
                         f"{all_ncc_roi_before[i]:.4f}", f"{all_ncc_roi_after[i]:.4f}",
                         f"{all_dice_heart_before[i]:.4f}", f"{all_dice_heart_after[i]:.4f}",
                         f"{all_dice_liver_before[i]:.4f}", f"{all_dice_liver_after[i]:.4f}",
                         f"{all_min_jac[i]:.4f}", all_n_foldings[i], f"{all_jac_neg_ratio[i]*100:.3f}"])
    print(f"  Stats saved: {stats_csv}")

print(f"\nAll done! Results saved to: {opt.save_dir}")