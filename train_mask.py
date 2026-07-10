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
from utils.utils import jacobian_determinant_vxm  # 与可视化脚本一致的 Jacobian 计算
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import random

# =========================
# Reproducibility (用于严谨的消融对比)
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# 不开 benchmark: 保证 cuDNN 算法选择是确定性的
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import glob
import warnings
import torch
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data
from natsort import natsorted
import TransModels.LDMMorph as LDMMorph 

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default
from ldm.data.xcat_npz import XCATNPZRegistration, collate_registration_dicts
from omegaconf import OmegaConf
from torch.autograd import Variable
#用于xcat运动增强版本的训练
#原本是通过npz文件进行训练的，现在通过xcat_Motion.py进行训练
parser = ArgumentParser()
parser.add_argument("--resume", type=str,
                    dest="resume", default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-30T21-02-35_xcat-motion-ldm/checkpoints/last.ckpt',
                    help="pretrained model")
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=24001,
                    help="number of total iterations")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.4, 
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="data path for datasets (contains train/, val/, test/ subdirs)") 
parser.add_argument("--beta", type=float,
                    dest="beta",
                    default=0.8,
                    help="beta loss: range from 0.1 to 1.0")
parser.add_argument("--xcat", action="store_true",
                    dest="xcat",
                    help="Use XCAT dataset with motion augmentation (no npz required)")
parser.add_argument("--xcat_path", type=str,
                    dest="xcat_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="Data root for XCAT dataset")
parser.add_argument("--fixed_motion", action="store_true",
                    dest="fixed_motion",
                    help="Use XCAT fixed+motion dataset (fixed image + moving sequence frames)")
parser.add_argument("--fixed_motion_path", type=str,
                    dest="fixed_motion_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="Data root for fixed_motion dataset")
parser.add_argument("--loss_type", type=str, default='ncc',
                    choices=['ncc', 'mse'],
                    dest="loss_type",
                    help="Image domain loss type: ncc (default) or mse. "
                         "注意：--fixed_motion 模式会强制使用 mse（原论文设置）。")
parser.add_argument("--no_ldm", action="store_true",
                    dest="no_ldm",
                    help="Disable LDM features, use learnable placeholders instead")
parser.add_argument("--no_phase_cond", action="store_true",
                    dest="no_phase_cond",
                    help="Disable phase-aware FiLM conditioning (default ON). "
                         "When set, LDMMorph.forward(phase_id=None) → 原行为完全一致.")
parser.add_argument("--use_motion_film", action="store_true",
                    dest="use_motion_film",
                    help="Use Image-conditioned Motion Embedding + FiLM "
                         "(替代 phase-aware FiLM). "
                         "优先级高于 --no_phase_cond.")
parser.add_argument("--ldm_config", type=str,
                    dest="ldm_config",
                    default=None,
                    help="LDM config file path")
parser.add_argument("--fg_thr", type=float,
                    dest="fg_thr", default=0.05,
                    help="前景(人体)mask 的亮度阈值，用于遮住黑色背景。"
                         "若身体边缘被切掉调小(如0.02)，黑边没排干净调大。")
# ===================== 新增：二阶弯曲能量 + Jacobian 折叠惩罚 =====================
parser.add_argument("--bending_w", type=float,
                    dest="bending_w", default=0.0,
                    help="二阶 bending energy(弯曲能量)正则权重。>0 时叠加到一阶 smooth 上。"
                         "惩罚位移场的曲率(急转弯)，但允许平滑过渡的大位移，"
                         "因此能压住心肝交界的折叠而不误伤合理大位移。"
                         "建议从 0.1~0.5 起试，与一阶 smth_labda 同量级。")
parser.add_argument("--jacdet_w", type=float,
                    dest="jacdet_w", default=0.0,
                    help="Jacobian 负值(折叠)惩罚权重。>0 时只在 det(J)<=0 的像素加压，"
                         "空间自适应，其它区域 loss=0，不影响欠配的心脏区。"
                         "建议从 1.0~10.0 起试；太大会压制合理形变、降低配准精度。")
opt = parser.parse_args()


lr = opt.lr
bs = opt.bs
iteration = opt.iteration
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
beta = opt.beta
t_enc = 1 

opt, unknown = parser.parse_known_args()
ckpt = None
if opt.ldm_config:
    configs = [opt.ldm_config]
else:
    # 默认使用非运动增强版本的LDM配置，如需更改请修改此处
    configs = ['/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_motion-ldm.yaml']
opt.ldm = configs
print(opt.resume)

# 说明：所有模式（xcat / fixed_motion / npz）都按命令行 --loss_type 选择图像域 loss，
# 不再对 fixed_motion 强制 MSE。想用哪种就显式传 --loss_type ncc / mse。


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval() 
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def dice(pred1, truth1):
    if datapath=='acdc':
        VOI_lbls = [2,3]
    else:
        VOI_lbls = [1]
    dice_all=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        
        dice_all[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_all)


# ===================== 前景(人体)mask =====================
def body_mask(img_tensor, thr=0.05):
    """从 fixed 图像生成前景/人体 mask，用于遮住黑色背景。

    流程：阈值二值化 -> 填充内部孔洞 -> 保留最大连通块。
    得到完整人体轮廓（内部暗器官/肺也算前景），只把外圈纯黑空气背景排除。

    Args:
        img_tensor: torch.Tensor, [B,1,H,W]，值域 [0,1]（用 fixed Y 传入）
        thr: 前景亮度阈值
    Returns:
        torch.FloatTensor, [B,1,H,W]，前景=1.0 背景=0.0，与输入同 device
    """
    from scipy.ndimage import binary_fill_holes, label
    arr = img_tensor.detach().cpu().numpy()
    b, c, h, w = arr.shape
    out = np.zeros((b, c, h, w), dtype=np.float32)
    for bi in range(b):
        m = arr[bi, 0] > thr
        m = binary_fill_holes(m)
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = (lab == int(sizes.argmax()))
        out[bi, 0] = m.astype(np.float32)
    return torch.from_numpy(out).to(img_tensor.device)


# ===================== 前景内 MSE =====================
def masked_mse(pred, target, mask):
    """只在前景 mask=1 的像素上计算 MSE（去掉黑背景对 MSE 的稀释）。

    全图 MSE 会把大量 0 vs 0 的背景像素平均进去，导致 MSE 一开始就极小、
    梯度极小、网络几乎不更新。这里改为只在前景上平均。

    Args:
        pred, target: [B,1,H,W]
        mask: [B,1,H,W]，前景=1 背景=0
    """
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


def ncc_loss(fixed, moving, win_size=15, mask=None):
    """Local Normalized Cross-Correlation loss - single pooling operation.

    mask: 可选 [B,1,H,W]（前景=1 背景=0）。提供时只在前景像素上平均逐像素 NCC，
          注意不是把图像背景刷成 0 再算（那样会制造假边界、污染窗口），
          而是先算出逐像素 NCC map，再只在前景上取均值。
    """
    assert fixed.shape == moving.shape
    assert win_size % 2 == 1

    b, c, h, w = fixed.shape
    pad = win_size // 2

    fixed_pad = F.pad(fixed, [pad, pad, pad, pad], mode='reflect')
    moving_pad = F.pad(moving, [pad, pad, pad, pad], mode='reflect')

    patches_fix = fixed_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)
    patches_mov = moving_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)
    patches_fix = patches_fix.contiguous().view(b, c, h, w, -1)
    patches_mov = patches_mov.contiguous().view(b, c, h, w, -1)

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


# ===================== 新增：二阶弯曲能量正则 =====================
def bending_energy_loss(y_pred):
    """二阶弯曲能量(bending energy)正则：惩罚位移场的二阶导(曲率)，
    而非一阶导(大小变化)。与一阶 smoothloss 用同样的像素换算口径(×size/2)，
    因此两项尺度可比，可直接加权叠加。

    适用场景：心脏 / 肝脏反向大位移的交界带会产生"急转弯"(高曲率)→ 折叠，
    bending energy 专门压这种弯折，但允许平滑过渡的大位移(不惩罚平缓大位移)，
    所以不会像一阶 smooth 那样把"合理大位移"也一起压住。

    y_pred: [B, 2, H, W] 位移场（归一化坐标 [-1,1]）
    """
    h2, w2 = y_pred.shape[-2:]
    # 一阶差分（换算到像素单位，口径与 smoothloss 一致）
    dy = (y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2   # ∂/∂行
    dx = (y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2   # ∂/∂列

    # 二阶差分
    dyy = (dy[:, :, 1:, :] - dy[:, :, :-1, :]) / 2 * h2          # ∂²/∂行²
    dxx = (dx[:, :, :, 1:] - dx[:, :, :, :-1]) / 2 * w2          # ∂²/∂列²
    # 交叉二阶项 ∂²/∂行∂列（对 dx 再沿行方向差分）
    dxy = (dx[:, :, 1:, :] - dx[:, :, :-1, :]) / 2 * h2

    # bending energy = dyy² + dxx² + 2·dxy²（薄板样条能量的标准形式）
    return (torch.mean(dyy * dyy)
            + torch.mean(dxx * dxx)
            + 2.0 * torch.mean(dxy * dxy)) / 4.0


# ===================== 新增：Jacobian 负值(折叠)惩罚 =====================
def jacobian_neg_loss(y_pred):
    """空间自适应折叠惩罚：只惩罚 Jacobian 行列式 <=0 的像素(发生折叠/翻转处)，
    其它地方 loss=0，完全不影响（包括欠配但未折叠的心脏区）。可微，直接进 loss。

    口径与可视化 / jac_stats 一致：DVF ×size/2 转像素后算 det(J)，
    判定折叠用 det(J) < 0。这里用可微的 relu(-det(J)) 作为惩罚。
    注意：训练惩罚用前向差分近似 det(J)（快、可微）；
    评测仍用 jacobian_determinant_vxm（np.gradient 中心差分）那套，二者各司其职，
    所以 CSV / 日志里的 neg_ratio 口径保持不变、前后可比。

    y_pred: [B, 2, H, W] 位移场（归一化坐标 [-1,1]）
    返回每像素 relu(-detJ) 的均值。
    """
    h2, w2 = y_pred.shape[-2:]
    # 换算到像素位移（与 jac_stats 口径一致）：分量0=行×H/2，分量1=列×W/2
    disp = torch.stack([y_pred[:, 0] * h2 / 2.0,
                        y_pred[:, 1] * w2 / 2.0], dim=1)  # [B, 2, H, W]

    # 位移分量对行 / 列方向的前向差分
    dfx_dy = disp[:, 0, 1:, :] - disp[:, 0, :-1, :]   # ∂(行位移)/∂行  -> [B, H-1, W]
    dfx_dx = disp[:, 0, :, 1:] - disp[:, 0, :, :-1]   # ∂(行位移)/∂列  -> [B, H, W-1]
    dfy_dy = disp[:, 1, 1:, :] - disp[:, 1, :-1, :]   # ∂(列位移)/∂行  -> [B, H-1, W]
    dfy_dx = disp[:, 1, :, 1:] - disp[:, 1, :, :-1]   # ∂(列位移)/∂列  -> [B, H, W-1]

    # 对齐到公共尺寸 [B, H-1, W-1]
    dfx_dy = dfx_dy[:, :, :-1]
    dfy_dy = dfy_dy[:, :, :-1]
    dfx_dx = dfx_dx[:, :-1, :]
    dfy_dx = dfy_dx[:, :-1, :]

    # 形变映射 φ = id + disp 的 Jacobian 行列式（恒等映射的 +1 加在对角）
    j11 = 1.0 + dfx_dy   # ∂φ_行/∂行
    j12 = dfx_dx         # ∂φ_行/∂列
    j21 = dfy_dy         # ∂φ_列/∂行
    j22 = 1.0 + dfy_dx   # ∂φ_列/∂列
    detJ = j11 * j22 - j12 * j21

    return torch.relu(-detJ).mean()


def jac_stats(D_f_xy):
    """计算形变场的 Jacobian 统计，口径与可视化脚本完全一致：
       - DVF ×size/2 转成像素单位（分量0=行×H/2，分量1=列×W/2）
       - 用 jacobian_determinant_vxm 计算 |J|
       - 折叠判定用 < 0（与可视化的 neg_ratio 一致）

    返回 (neg_ratio, min_jac, max_jac, n_foldings)，方便和原论文 neg_ratio(如 0.24%) 对比。
    """
    dvf = D_f_xy[0].detach().cpu().numpy()  # [2, H, W]
    _, h, w = dvf.shape
    dvf_px = dvf.copy()
    dvf_px[0] = dvf_px[0] * h / 2.0
    dvf_px[1] = dvf_px[1] * w / 2.0
    jd = jacobian_determinant_vxm(dvf_px)
    n_fold = int(np.sum(jd < 0))
    neg_ratio = float(n_fold / jd.size)
    return neg_ratio, float(jd.min()), float(jd.max()), n_fold


def save_checkpoint(state, save_dir, save_filename, max_model_num=50):
    torch.save(state, save_dir + save_filename)
    # 只清理 .pth 文件，不影响可视化图片等其他文件
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))
    
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))

class FlipWrapper(Data.Dataset):
    """在 Dataset_epoch_with_name 外面套一层 (1) fixed-minmax 归一化 + (2) 随机水平翻转。

    归一化：与 XCATSeqRegistration 完全一致——
        minv, maxv = fixed.min(), fixed.max()       # fixed = tar = Y
        fixed  = (fixed  - minv) / (maxv - minv)     # 若 maxv-minv > 1e-6
        moving = (moving - minv) / (maxv - minv)     # moving 用 fixed 的 min/max
    这样 NPZ 模式与 fixed_motion 模式在输入预处理上逐位对齐，
    消除两者 NCC_Before 基线不一致的问题（之前 NPZ 不做任何归一化）。
    顺序：先归一化，再随机翻转（与 XCATSeqRegistration 一致）。

    翻转：水平翻转、fixed/moving/label 同步翻、flip_p=0.5（仅 train）；flip_p=0 即不翻（val/test）。
    返回结构与底层数据集一致：(mov, tar, movlab, tarlab, name)，均为 [1,H,W] tensor。

    normalize=False 时退化为纯翻转（保留旧行为，便于对照实验）。
    """
    def __init__(self, base, flip_p=0.0, normalize=True):
        self.base = base
        self.flip_p = flip_p
        self.normalize = normalize

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        mov, tar, movlab, tarlab, name = self.base[i]   # 均为 [1,H,W] tensor

        # (1) fixed-minmax 归一化（用 tar=fixed=Y 的 min/max 同时缩放 mov 和 tar）
        if self.normalize:
            mov = mov.float()
            tar = tar.float()
            minv = tar.min()
            maxv = tar.max()
            if (maxv - minv) > 1e-6:
                tar = (tar - minv) / (maxv - minv)
                mov = (mov - minv) / (maxv - minv)

        # (2) 随机水平翻转（宽度方向；fixed/moving/label 同步翻，保持配对）
        if self.flip_p > 0 and torch.rand(1).item() < self.flip_p:
            mov    = torch.flip(mov,    dims=[-1])
            tar    = torch.flip(tar,    dims=[-1])
            movlab = torch.flip(movlab, dims=[-1])
            tarlab = torch.flip(tarlab, dims=[-1])
        return mov, tar, movlab, tarlab, name


def build_npz_dataset_from_json(base_dir, split):
    """NPZ 模式：按 split_indices.json 的 registration 段直接切分 train/val/test。

    与可视化脚本完全同源：
      - 文件列表 = sorted(glob('<base_dir>/*_pair.npz'))
      - 划分区间 = split_indices.json -> registration.split[split] = [start, end]（含两端）
      - 索引相对 sorted 列表的位置；区间会安全裁剪到实际文件数
      - 用 Dataset_epoch_with_name 包装（和可视化一致）

    注意：本函数只负责按 json 切分并用 Dataset_epoch_with_name 包装，不做增强；
    train 端的随机水平翻转由外层 FlipWrapper(flip_p=0.5) 提供。
    """
    base_dir = base_dir.rstrip('/')
    all_files = sorted(glob.glob(os.path.join(base_dir, '*_pair.npz')))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No *_pair.npz files found in {base_dir}/")
    n = len(all_files)

    split_json = os.path.join(base_dir, 'split_indices.json')
    if not os.path.exists(split_json):
        raise FileNotFoundError(
            f"split_indices.json not found in {base_dir}/ ; NPZ 模式现在依赖该文件做 70/15/15 划分")
    with open(split_json) as jf:
        cfg = json.load(jf)
    reg_split = cfg['registration']['split']
    if split not in reg_split:
        raise KeyError(f"split '{split}' not in split_indices.json registration.split")
    lo, hi = reg_split[split]          # [start, end]，含两端
    lo = max(0, lo)
    hi = min(n - 1, hi)
    files = all_files[lo:hi + 1]
    print(f"  [NPZ split] {split}: indices [{lo},{hi}] -> {len(files)} files (total {n})")
    return Dataset_epoch_with_name(files)


def train():
    global opt, datapath
    print(opt.resume)
    ckpt = opt.resume
    
    configs_list = [OmegaConf.load(cfg) for cfg in opt.ldm]
    cli = OmegaConf.from_dotlist(unknown)
    configs = OmegaConf.merge(*configs_list, cli)

    gpu = True
    eval_mode = True

    ldm_model, global_step = load_model(configs, ckpt, gpu, eval_mode)
    print(f"VQ autoencoder loaded from {configs_list[0].model.params.first_stage_config.params.ckpt_path}")
    print(f"[INFO] 图像域 loss = {opt.loss_type.upper()}（前景遮罩已开启，fg_thr={opt.fg_thr}）")
    print(f"[INFO] 正则项：一阶 smooth(smth_labda)={smooth} | bending_w={opt.bending_w} | jacdet_w={opt.jacdet_w}")
    #-------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    if opt.xcat:
        from utils.utils import Dataset_XCAT_Registration
        train_loader = Data.DataLoader(
            Dataset_XCAT_Registration(
                data_root=opt.xcat_path, split='train',
                motion_types=['identity', 'rotate10', 'scale05', 'warp'],  # 添加 scale05
                flip_p=0.5,
            ),
            batch_size=bs, shuffle=True, num_workers=0
        )
        val_loader = Data.DataLoader(
            Dataset_XCAT_Registration(
                data_root=opt.xcat_path, split='val',
                motion_types=['identity'],
                flip_p=0.0,
            ),
            batch_size=bs, shuffle=False, num_workers=0
        )
        test_loader = Data.DataLoader(
            Dataset_XCAT_Registration(
                data_root=opt.xcat_path, split='test',
                motion_types=['identity'],
                flip_p=0.0,
            ),
            batch_size=bs, shuffle=False, num_workers=0
        )
        print(f"XCAT mode: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    elif opt.fixed_motion:
        from ldm.data.xcat_Motion_Seq import XCATSeqRegistration
        train_loader = Data.DataLoader(
            XCATSeqRegistration(data_root=opt.fixed_motion_path, split='train', flip_p=0.5),
            batch_size=bs, shuffle=True, num_workers=0
        )
        val_loader = Data.DataLoader(
            XCATSeqRegistration(data_root=opt.fixed_motion_path, split='val', flip_p=0.0),
            batch_size=bs, shuffle=False, num_workers=0
        )
        test_loader = Data.DataLoader(
            XCATSeqRegistration(data_root=opt.fixed_motion_path, split='test', flip_p=0.0),
            batch_size=bs, shuffle=False, num_workers=0
        )
        print(f"FixedMotion mode: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    else:
        # NPZ 模式：使用 XCATNPZRegistration（9 相位 base-sample 划分 + phase_id 输出）
        xcat_path = opt.datapath
        train_loader = Data.DataLoader(
            XCATNPZRegistration(data_root=xcat_path, split='train', flip_p=0.5, normalize=True),
            batch_size=bs, shuffle=True, num_workers=0,
            collate_fn=collate_registration_dicts,
        )
        val_loader = Data.DataLoader(
            XCATNPZRegistration(data_root=xcat_path, split='val', flip_p=0.0, normalize=True),
            batch_size=bs, shuffle=False, num_workers=0,
            collate_fn=collate_registration_dicts,
        )
        test_loader = Data.DataLoader(
            XCATNPZRegistration(data_root=xcat_path, split='test', flip_p=0.0, normalize=True),
            batch_size=bs, shuffle=False, num_workers=0,
            collate_fn=collate_registration_dicts,
        )
        print(f"NPZ mode (9-phase base-sample split): train={len(train_loader.dataset)}, "
              f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model = LDMMorph.LDMMorph(128*2,192*2,320*2,448*2,
                              use_ldm=not opt.no_ldm,
                              use_motion_film=getattr(opt, 'use_motion_film', False))
    model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    loss_similarity_ncc = ncc_loss
    loss_similarity_mse = MSE().loss
    loss_smooth = smoothloss

    transform = SpatialTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if opt.xcat:
        model_dir = f'./logs/XCAT_TransMorph_Smooth_{smooth}_beta_{beta}_xcat_motion_604/'
        csv_name  = f'./logs/XCAT_TransMorph_Smooth_{smooth}_beta_{beta}_xcat_motion_604.csv'
    elif opt.fixed_motion:
        model_dir = f'./logs/FixedMotion_TransMorph_Smooth_{smooth}_beta_{beta}_627_jac2.0_bending0.01/'
        csv_name  = f'./logs/FixedMotion_TransMorph_Smooth_{smooth}_beta_{beta}_627_jac2.0_bending0.01.csv'
    else:
        # 9-phase 消融实验: 用目录后缀区分三种条件方式
        #   - motion_film  : use_motion_film=True  (--use_motion_film)
        #   - phase_film   : 默认 (phase-aware FiLM, ON)
        #   - baseline_9phase: --no_phase_cond
        if getattr(opt, 'use_motion_film', False):
            cond_tag = 'motion_film'
        elif getattr(opt, 'no_phase_cond', False):
            cond_tag = 'baseline_9phase'
        else:
            cond_tag = 'phase_film'
        model_dir = f'./logs/TransScorelm_Smooth_0.8_beta_0.8_7_15_15_707_{cond_tag}/'
        csv_name  = f'./logs/TransScorelm_Smooth_0.8_beta_0.8_7_15_15_707_{cond_tag}.csv'
        print(f'[Train] conditioning tag = {cond_tag}')

    # CSV表头：根据 loss_type 显示相应列s
    f = open(csv_name, 'w')
    with f:
        if opt.loss_type == 'ncc':
            fnames = ['Index', 'NCC_Val_S', 'OrgNCC_Val_S', 'NCC_Test', 'OrgNCC_Test']
        else:
            fnames = ['Index', 'MSE_Val_S', 'NCC_Val_S', 'MSE_Test', 'NCC_Test']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((3, iteration+1))
    step = 1
    epoch = 0
    csv_dice = 0
    while step <= iteration:
        for batch in train_loader:
            if opt.fixed_motion or opt.xcat:
                # legacy loads (XCATSeqRegistration / Dataset_XCAT_Registration) → 5-tuple
                X, Y, segx, segy, _ = batch
                phase = None
                phase_id = None
            else:
                # NPZ 9-phase dataset → 9-tuple:
                #   (moving, fixed, segx, segy, pairname,
                #    phase 1..9, phase_id 0..8, moving_idx, fixed_idx)
                X, Y, segx, segy, _, phase, phase_id, _, _ = batch
                if step == 1 and isinstance(phase, torch.Tensor):
                    print(f"[NPZ 9-phase] phase(1..9)    ={phase.tolist()}  "
                          f"(moving 文件名约定)")
                    print(f"               phase_id(0..8) ={phase_id.tolist()}  "
                          f"(embedding lookup)")

            X = X.cuda().float()
            Y = Y.cuda().float()

            # 前景(人体)mask：用 fixed(Y) 估计，遮住黑色背景
            fg = body_mask(Y, thr=opt.fg_thr)

            # 第 10 步单独可视化一次前景 mask，确认背景是否被正确遮掉
            if step == 10:
                with torch.no_grad():
                    import matplotlib.pyplot as plt
                    Yv = Y[0, 0].detach().cpu().numpy()
                    fgv = fg[0, 0].detach().cpu().numpy()
                    overlay = np.stack([Yv, Yv, Yv], axis=-1)
                    overlay = np.clip(overlay, 0, 1)
                    overlay[..., 0] = np.clip(overlay[..., 0] + 0.4 * fgv, 0, 1)  # 前景叠红
                    figm, axm = plt.subplots(1, 3, figsize=(15, 5))
                    axm[0].imshow(Yv, cmap='gray'); axm[0].set_title('Fixed (Y)'); axm[0].axis('off')
                    axm[1].imshow(fgv, cmap='gray')
                    axm[1].set_title(f'Foreground mask (fg_ratio={fgv.mean():.3f}, thr={opt.fg_thr})')
                    axm[1].axis('off')
                    axm[2].imshow(overlay); axm[2].set_title('Overlay (red = foreground)'); axm[2].axis('off')
                    figm.suptitle(f'[Step {step}] Background-mask check', fontsize=12)
                    plt.tight_layout()
                    mask_vis_path = f'{model_dir}mask_check_step{step:06d}.png'
                    figm.savefig(mask_vis_path, dpi=100, bbox_inches='tight')
                    plt.close(figm)
                    print(f'\n[Mask Check] Saved foreground-mask visualization to {mask_vis_path}')

            # 调试信息：输入数据范围
            if step == 1 or step % 50 == 0:
                print(f'\n[Step {step}] X range: [{X.min():.4f}, {X.max():.4f}], Y range: [{Y.min():.4f}, {Y.max():.4f}]'
                      f'  fg_ratio: {fg.mean().item():.3f}')

            # =========================================================
            # [Ablation] LDM 特征生成：--no_ldm 时仍然计算（供 latent loss 使用），
            # 但模型内部会忽略它们（用 CNN 特征替代）
            # =========================================================
            mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
            fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()

            noise = None
            noise = default(noise, lambda: torch.randn_like(mov_z))
            x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
            y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)

            outx = ldm_model.apply_model(x_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
            outy = ldm_model.apply_model(y_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

            score0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
            score1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
            score2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
            score3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]),  dim=1)
            if step == 1:
                print('score0:', score0.shape)
                print('score1:', score1.shape)
                print('score2:', score2.shape)
                print('score3:', score3.shape)
            # ----- Conditioning → FiLM -----
            # 三种模式 (优先级: use_motion_film > no_phase_cond):
            #   - use_motion_film=True  → 传 None (模型用 motion encoder, 忽略 phase_id)
            #   - no_phase_cond=True    → 传 None (完全跳过 FiLM, baseline)
            #   - 否则                   → 传 phase_id (phase-aware FiLM)
            if getattr(opt, 'use_motion_film', False) or getattr(opt, 'no_phase_cond', False):
                pi = None
            else:
                pi = phase_id
            if pi is not None:
                # collate 返回的是 CPU long tensor，搬到与 LDMMorph 同设备，避免
                # phase_embedding(weight on cuda) vs index on cpu 的 device mismatch
                pi = pi.to(X.device)
            if step == 1:
                if getattr(opt, 'use_motion_film', False):
                    print('[PhaseCond] use_motion_film=True → motion_encoder(moving,fixed), '
                          'phase_id ignored.')
                elif getattr(opt, 'no_phase_cond', False):
                    print('[PhaseCond] no_phase_cond=True → FiLM disabled (baseline).')
                else:
                    print(f'[PhaseCond] phase_id fed to LDMMorph = '
                          f'{None if pi is None else pi.tolist()}')
            D_f_xy,score_output = model(X, Y, score0, score1, score2, score3,
                                        phase_id=pi)
            # print('score_output:', score_output.shape)
            _, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1))
            
            # 调试信息：形变场范围
            if step == 1 or step % 50 == 0:
                print(f'[Step {step}] D_f_xy range: [{D_f_xy.min():.6f}, {D_f_xy.max():.6f}], D_f_xy mean: {D_f_xy.mean():.6f}, D_f_xy std: {D_f_xy.std():.6f}')

            # [Ablation] 潜空间 MSE loss：--no_ldm 时仍然计算（LDM encoder 始终存在）
            # 注意：latent 在 LDM 潜空间，没有"黑背景"问题，也没有对应的前景 mask，保持原样不遮罩。
            mov_z_warped = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X_Y))
            loss_mse_latent = loss_similarity_mse(mov_z_warped, fix_z)

            # 图像域 loss：在前景(人体)上计算，遮住黑色背景
            if opt.loss_type == 'ncc':
                loss_image = ncc_loss(X_Y, Y, mask=fg)
            else:
                loss_image = masked_mse(X_Y, Y, fg)

            loss1 = beta * loss_image + (1 - beta) * loss_mse_latent
            # print('beta:', beta)
            loss2 = loss_smooth(D_f_xy)         # 一阶 smooth（原样保留）
            # print('smooth:', smooth)
            loss = loss1 + smooth * loss2

            # ============ 新增正则项（默认权重 0，不传参时行为与旧版完全一致）============
            # 二阶 bending energy：压心肝交界的"急转弯"折叠，不惩罚平缓大位移
            loss_bend = torch.tensor(0.0, device=loss.device)
            if opt.bending_w > 0:
                loss_bend = bending_energy_loss(D_f_xy)
                loss = loss + opt.bending_w * loss_bend

            # Jacobian 负值惩罚：空间自适应，只压 det(J)<0 的折叠像素
            loss_jac = torch.tensor(0.0, device=loss.device)
            if opt.jacdet_w > 0:
                loss_jac = jacobian_neg_loss(D_f_xy)
                loss = loss + opt.jacdet_w * loss_jac
            # ===========================================================================

            # 可视化：每 1000 步生成一张图（同一进程内的下一帧）— jac_det / Disp 用 99-percentile 自适应 clamp
            if step % 1000 == 0:
                with torch.no_grad():
                    import matplotlib.pyplot as plt
                    X_cpu = X[0, 0].cpu().numpy()
                    Y_cpu = Y[0, 0].cpu().numpy()
                    XY_cpu = X_Y[0, 0].cpu().numpy()
                    D_cpu = D_f_xy[0].cpu().numpy()

                    # ----- 抽一张 val 样本做对比（与 train 同 step 同一进程，零额外 I/O）-----
                    val_XY_cpu = None; val_Y_cpu = None; val_diff_after = None; val_jac_det = None
                    try:
                        val_iter = iter(val_loader)
                        vb = next(val_iter)
                        # XCATNPZRegistration 的 collate 返回 dict-list 格式
                        # 形如 {'image':[N,...], 'seg':..., 'name':..., 'phase':..., 'phase_id':..., 'moving_idx':..., 'fixed_idx':...}
                        if isinstance(vb, (list, tuple)):
                            # 兼容 (X, Y, segx, segy, ...) 形式
                            vx, vy = vb[0], vb[1]
                        else:
                            vx, vy = vb.get('image'), vb.get('fixed_image')
                            if vx is None:
                                vx = vb.get('moving_image')
                                vy = vb.get('fixed_image')
                        vx = vx.to(device); vy = vy.to(device)
                        vpi = None
                        # 优先级: use_motion_film > no_phase_cond > phase-aware FiLM
                        if getattr(opt, 'use_motion_film', False) or getattr(opt, 'no_phase_cond', False):
                            vpi = None
                        elif isinstance(vb, dict):
                            vpi = vb.get('phase_id')
                            if vpi is not None:
                                vpi = vpi.to(vx.device)
                        vD, _ = model(vx, vy, score0.detach(), score1.detach(), score2.detach(), score3.detach(), phase_id=vpi)
                        _, vX_Y = transform(vx, vD.permute(0, 2, 3, 1))
                        val_XY_cpu = vX_Y[0, 0].cpu().numpy()
                        val_Y_cpu  = vy[0, 0].cpu().numpy()
                        val_diff_after = np.abs(val_XY_cpu - val_Y_cpu)
                        vD_disp = vD[0].cpu().numpy(); hh, ww = vD_disp.shape[1], vD_disp.shape[2]
                        vD_disp_px = vD_disp.copy()
                        vD_disp_px[0] = vD_disp_px[0] * hh / 2.0
                        vD_disp_px[1] = vD_disp_px[1] * ww / 2.0
                        val_jac_det = jacobian_determinant_vxm(vD_disp_px)
                    except Exception as _e:
                        val_XY_cpu = None  # val 抽不到也不影响 train 图
                    # -------------------------------------------------------------------

                    # 计算雅可比行列式（与可视化脚本口径一致：×size/2 + jacobian_determinant_vxm）
                    D_disp = D_f_xy[0].cpu().numpy()        # [2, H, W]
                    _hh, _ww = D_disp.shape[1], D_disp.shape[2]
                    D_disp_px = D_disp.copy()
                    D_disp_px[0] = D_disp_px[0] * _hh / 2.0
                    D_disp_px[1] = D_disp_px[1] * _ww / 2.0
                    jac_det = jacobian_determinant_vxm(D_disp_px)

                    n_foldings = np.sum(jac_det < 0)
                    min_jac = jac_det.min()

                    # 自适应色标：jac_det / Disp 用 99-percentile，避免把 >1.5 的折叠峰值 clip 成同色
                    jac_v = float(np.percentile(np.abs(jac_det), 99)); jac_vmax = max(jac_v, 0.5)
                    disp_vx = float(np.percentile(np.abs(D_cpu[0]), 99))
                    disp_vy = float(np.percentile(np.abs(D_cpu[1]), 99))
                    disp_vmax = max(disp_vx, disp_vy, 0.05)

                    diff_before = np.abs(X_cpu - Y_cpu)
                    diff_after = np.abs(XY_cpu - Y_cpu)

                    # 前景版参考指标（与训练口径一致）
                    ncc_standard = 1.0 - ncc_loss(X_Y, Y, mask=fg).item()
                    loss_image_val = loss_image.item()
                    mse_standard = masked_mse(X_Y, Y, fg).item()

                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    titles = [
                        'Moving (X)', 'Fixed (Y)', 'Warped (X→Y)', 'Diff Before',
                        'Diff After',
                        'Jac Det (fold={}, min={:.3f}, ±clamp={:.2f})'.format(n_foldings, min_jac, jac_vmax),
                        'Disp Field-X (clamp=±{:.2f})'.format(disp_vmax),
                        'Disp Field-Y (clamp=±{:.2f})'.format(disp_vmax),
                    ] + ([] if val_jac_det is None else [
                        'VAL Fixed (Y)', 'VAL Warped', 'VAL Diff After',
                        'VAL Jac Det (fold={}, min={:.3f})'.format(int(np.sum(val_jac_det < 0)), float(val_jac_det.min()))
                    ])
                    imgs = [X_cpu, Y_cpu, XY_cpu, diff_before,
                            diff_after, jac_det, D_cpu[0], D_cpu[1]]
                    if val_XY_cpu is not None:
                        imgs += [val_Y_cpu, val_XY_cpu, val_diff_after, val_jac_det]

                    # 图像域统一用当前 batch 三张图的最大值固定 vmax，避免灰度值跳
                    img_vmax = max(float(X_cpu.max()), float(Y_cpu.max()), float(XY_cpu.max()), 1e-8)
                    if val_XY_cpu is not None:
                        img_vmax = max(img_vmax, float(val_Y_cpu.max()), float(val_XY_cpu.max()))
                    for ax, img, title in zip(axes.flat, imgs, titles):
                        if title.startswith('Jac Det'):
                            ax.imshow(img, cmap='RdBu_r', vmin=-jac_vmax, vmax=jac_vmax)
                        elif title.startswith('VAL Jac Det'):
                            v = float(np.percentile(np.abs(img), 99)); vmax = max(v, 0.5)
                            ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                        elif title.startswith('Disp Field-X'):
                            ax.imshow(img, cmap='RdBu_r', vmin=-disp_vmax, vmax=disp_vmax)
                        elif title.startswith('Disp Field-Y'):
                            ax.imshow(img, cmap='RdBu_r', vmin=-disp_vmax, vmax=disp_vmax)
                        else:
                            ax.imshow(img, cmap='gray', vmin=0, vmax=img_vmax)
                        ax.set_title(title, fontsize=10)
                        ax.axis('off')

                    if opt.loss_type == 'ncc':
                        suptitle_str = (f'[Step {step}] loss={loss.item():.4f}  NCC_train(fg)={loss_image_val:.4f}  '
                                        f'MSE_z={loss_mse_latent.item():.4f}  '
                                        f'bend={opt.bending_w * loss_bend.item():.4f}  jac={opt.jacdet_w * loss_jac.item():.4f}')
                    else:
                        suptitle_str = (f'[Step {step}] loss={loss.item():.4f}  MSE_train(fg)={loss_image_val:.4f}  '
                                        f'MSE_z={loss_mse_latent.item():.4f}  NCC_ref(fg)={ncc_standard:.4f}  '
                                        f'bend={opt.bending_w * loss_bend.item():.4f}  jac={opt.jacdet_w * loss_jac.item():.4f}')
                    fig.suptitle(suptitle_str, fontsize=12)
                    plt.tight_layout()
                    fig.savefig(f'{model_dir}vis_step_{step:06d}.png', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    print(f'\n[Visualization] Saved to {model_dir}vis_step_{step:06d}.png')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lossall[:,step] = np.array([loss.item(), loss1.item(), loss2.item()])
            loss_name = f'{opt.loss_type.upper()}'
            weighted_latent_loss = (1-beta) * loss_mse_latent.item()
            weighted_ncc_loss = beta * loss_image.item()
            # 实时 Jacobian 统计（与可视化脚本/原论文同口径：×size/2, |J|<0 计折叠）
            neg_ratio_step, minj_step, maxj_step, nfold_step = jac_stats(D_f_xy)
            sys.stdout.write("\r" + 'step "{0}" -> train loss "{1:.4f}" - {3} "{2:.4f}" - w_MSE_z "{4:.4f}" - w_img "{5:.4f}" - smh "{6:.4f}" - bend "{10:.4f}" - jac "{11:.4f}" - negR "{7:.3f}%" minJ "{8:.3f}" maxJ "{9:.3f}"'.format(
                step, loss.item(), loss_image.item(), loss_name, weighted_latent_loss, weighted_ncc_loss, loss2.item(),
                neg_ratio_step * 100, minj_step, maxj_step,
                opt.bending_w * loss_bend.item(), opt.jacdet_w * loss_jac.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    # 验证集：根据 loss_type 计算相应的 loss（均为前景版）
                    Val_Loss_List = []
                    NCCs_Val_NCC = []
                    NCCs_Val_Org = []
                    Val_NegRatio = []
                    Val_MinJac = []
                    Val_MaxJac = []
                    Val_Folds = []
                    
                    val_iter = iter(val_loader)
                    vb = next(val_iter)
                    if opt.fixed_motion or opt.xcat:
                        # legacy 5-tuple: (moving, fixed, segx, segy, pairname)
                        unpack = lambda b: (b[0], b[1], b[2], b[3], b[4])
                    else:
                        # NPZ 9-phase 9-tuple → 截取前 5 个:
                        #   (moving, fixed, segx, segy, pairname,
                        #    phase 1..9, phase_id 0..8, moving_idx, fixed_idx)
                        unpack = lambda b: (b[0], b[1], b[2], b[3], b[4])
                    xv, yv, xv_seg, yv_seg, _ = unpack(vb)
                    while True:
                        xv, yv, xv_seg, yv_seg = xv.to(device), yv.to(device), xv_seg.to(device), yv_seg.to(device)
                        
                        model.eval()

                        # 前景 mask（用 fixed yv 估计）
                        fg_v = body_mask(yv, thr=opt.fg_thr)

                        # =========================================================
                        # [Ablation] 验证集：LDM 特征始终计算（latent loss 仍生效）
                        # =========================================================
                        vmov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(xv)).detach()
                        vfix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(yv)).detach()

                        noise_v = torch.randn_like(vmov_z)
                        vx_noisy = ldm_model.q_sample(x_start=vmov_z, t=torch.tensor([t_enc]).cuda(), noise=noise_v)
                        vy_noisy = ldm_model.q_sample(x_start=vfix_z, t=torch.tensor([t_enc]).cuda(), noise=noise_v)

                        voutx = ldm_model.apply_model(vx_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                        vouty = ldm_model.apply_model(vy_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

                        vscore0 = torch.cat((voutx[1][0][0],  voutx[1][0][2], vouty[1][0][0],  vouty[1][0][2]),  dim=1)
                        vscore1 = torch.cat((voutx[1][0][3],  voutx[1][0][5], vouty[1][0][3],  vouty[1][0][5]),  dim=1)
                        vscore2 = torch.cat((voutx[1][0][6],  voutx[1][0][8], vouty[1][0][6],  vouty[1][0][8]),  dim=1)
                        vscore3 = torch.cat((voutx[1][0][9],  voutx[1][0][11], vouty[1][0][9],  vouty[1][0][11]),  dim=1)

                        Dv_f_xy,score_output = model(xv, yv, vscore0, vscore1, vscore2, vscore3)
                        _, warped_xv = transform(xv, Dv_f_xy.permute(0, 2, 3, 1))

                        # Jacobian 统计（与可视化/原论文同口径）
                        v_negr, v_minj, v_maxj, v_fold = jac_stats(Dv_f_xy)
                        Val_NegRatio.append(v_negr)
                        Val_MinJac.append(v_minj)
                        Val_MaxJac.append(v_maxj)
                        Val_Folds.append(v_fold)

                        for bs_index in range(xv.shape[0]):
                            yv_i = yv[bs_index,...].unsqueeze(0)
                            wv_i = warped_xv[bs_index,...].unsqueeze(0).detach()
                            xv_i = xv[bs_index,...].unsqueeze(0).detach()
                            fgv_i = fg_v[bs_index,...].unsqueeze(0)

                            # 根据 loss_type 计算相应的图像域 loss（前景版）
                            if opt.loss_type == 'ncc':
                                loss_val = 1.0 - ncc_loss(yv_i, wv_i, mask=fgv_i).item()
                            else:
                                loss_val = masked_mse(yv_i, wv_i, fgv_i).item()

                            # 配准质量 NCC（前景版，配准后）
                            ncc_s = 1.0 - ncc_loss(yv_i, wv_i, mask=fgv_i).item()
                            # 配准质量 NCC（前景版，配准前）
                            ncc_org_s = 1.0 - ncc_loss(yv_i, xv_i, mask=fgv_i).item()

                            Val_Loss_List.append(loss_val)
                            NCCs_Val_NCC.append(ncc_s)
                            NCCs_Val_Org.append(ncc_org_s)

                        try:
                            vb = next(val_iter)
                            xv, yv, xv_seg, yv_seg, _ = unpack(vb)
                        except StopIteration:
                            break

                    # 计算平均值
                    csv_loss_s = np.mean(Val_Loss_List)
                    csv_ncc_s = np.mean(NCCs_Val_NCC)
                    csv_ncc_org_s = np.mean(NCCs_Val_Org)

                    modelname = 'NCCVal_{:.4f}_Epoch_{:04d}.pth'.format(csv_ncc_s, step)
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)

                    print(f'\n    [Validation] {opt.loss_type.upper()}_S(fg): {csv_loss_s:.4f}  '
                          f'NCC_S(fg): {csv_ncc_s:.4f}  OrgNCC_S(fg): {csv_ncc_org_s:.4f}')
                    print(f'    Delta_S: {csv_ncc_s - csv_ncc_org_s:+.4f}')
                    print(f'    [Jacobian on val] neg_ratio: {np.mean(Val_NegRatio)*100:.3f}%  '
                          f'min|J|: {np.min(Val_MinJac):.3f}  max|J|: {np.max(Val_MaxJac):.3f}  '
                          f'avg_folds/img: {np.mean(Val_Folds):.1f}   (原论文 CAMUS 参考 ~0.24%)')

                    # CSV 记录
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        if opt.loss_type == 'ncc':
                            writer.writerow([step, csv_loss_s, csv_ncc_org_s, -1, -1])
                        else:
                            writer.writerow([step, csv_loss_s, csv_ncc_s, -1, -1])

                    model.train()

            step += 1

            if step > iteration:
                break
        print("one epoch pass")

    np.save(model_dir + '/Loss.npy', lossall)

    # ==================== 训练结束后评估测试集 ====================
    if test_loader is not None:
        print("\n" + "="*60)
        print("Final Test Set Evaluation (after training, no leakage)")
        print("="*60)
        model.eval()
        Test_Loss_List = []
        NCCs_Test = []
        NCCs_Test_org = []
        Test_NegRatio = []
        Test_MinJac = []
        Test_MaxJac = []
        Test_Folds = []
        with torch.no_grad():
            # XCATNPZRegistration (NPZ mode) 的 collate 返回 9 元组:
            #   (moving, fixed, segx, segy, name, phase, phase_id, moving_idx, fixed_idx)
            # 这里只用到前两维 (xt, yt)；其它维度一律丢弃.
            for batch_t in test_loader:
                if len(batch_t) == 9:
                    xt, yt = batch_t[0], batch_t[1]
                else:
                    xt, yt = batch_t[0], batch_t[1]
                xt, yt = xt.to(device), yt.to(device)

                # 前景 mask（用 fixed yt 估计）
                fg_t = body_mask(yt, thr=opt.fg_thr)

                # =========================================================
                # [Ablation] 测试集：LDM 特征始终计算（latent loss 仍生效）
                # =========================================================
                tmov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(xt)).detach()
                tfix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(yt)).detach()
                tx_noisy = ldm_model.q_sample(x_start=tmov_z, t=torch.tensor([t_enc]).cuda(), noise=torch.randn_like(tmov_z))
                ty_noisy = ldm_model.q_sample(x_start=tfix_z, t=torch.tensor([t_enc]).cuda(), noise=torch.randn_like(tfix_z))
                toutx = ldm_model.apply_model(tx_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                touty = ldm_model.apply_model(ty_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                tscore0 = torch.cat((toutx[1][0][0], toutx[1][0][2], touty[1][0][0], touty[1][0][2]), dim=1)
                tscore1 = torch.cat((toutx[1][0][3], toutx[1][0][5], touty[1][0][3], touty[1][0][5]), dim=1)
                tscore2 = torch.cat((toutx[1][0][6], toutx[1][0][8], touty[1][0][6], touty[1][0][8]), dim=1)
                tscore3 = torch.cat((toutx[1][0][9], toutx[1][0][11], touty[1][0][9], touty[1][0][11]), dim=1)

                # LDMMorph.forward() 始终返回 (D_f_xy, score_output); 解包拿 flow
                _out = model(xt, yt, tscore0, tscore1, tscore2, tscore3)
                Dt_f_xy = _out[0] if isinstance(_out, tuple) else _out
                _, warped_xt = transform(xt, Dt_f_xy.permute(0, 2, 3, 1))

                # Jacobian 统计（与可视化/原论文同口径）
                t_negr, t_minj, t_maxj, t_fold = jac_stats(Dt_f_xy)
                Test_NegRatio.append(t_negr)
                Test_MinJac.append(t_minj)
                Test_MaxJac.append(t_maxj)
                Test_Folds.append(t_fold)
                for bs_index in range(xt.shape[0]):
                    yt_i = yt[bs_index, ...].unsqueeze(0)
                    wt_i = warped_xt[bs_index, ...].unsqueeze(0).detach()
                    xt_i = xt[bs_index, ...].unsqueeze(0).detach()
                    fgt_i = fg_t[bs_index, ...].unsqueeze(0)

                    # 图像域 loss（前景版）
                    if opt.loss_type == 'ncc':
                        loss_t = 1.0 - ncc_loss(yt_i, wt_i, mask=fgt_i).item()
                    else:
                        loss_t = masked_mse(yt_i, wt_i, fgt_i).item()

                    # 配准质量 NCC（前景版）
                    ncc_t = 1.0 - ncc_loss(yt_i, wt_i, mask=fgt_i).item()
                    ncc_t_org = 1.0 - ncc_loss(yt_i, xt_i, mask=fgt_i).item()
                    Test_Loss_List.append(loss_t)
                    NCCs_Test.append(ncc_t)
                    NCCs_Test_org.append(ncc_t_org)
        print(f"\n    [Test] {opt.loss_type.upper()}(fg): {np.mean(Test_Loss_List):.4f}  NCC(fg): {np.mean(NCCs_Test):.4f}  OrgNCC(fg): {np.mean(NCCs_Test_org):.4f}  Delta: {np.mean(NCCs_Test) - np.mean(NCCs_Test_org):+.4f}")
        print(f"    [Jacobian on test] neg_ratio: {np.mean(Test_NegRatio)*100:.3f}%  "
              f"min|J|: {np.min(Test_MinJac):.3f}  max|J|: {np.max(Test_MaxJac):.3f}  "
              f"avg_folds/img: {np.mean(Test_Folds):.1f}   (原论文 CAMUS 参考 ~0.24%)")
        print(f"    Test samples: {len(NCCs_Test)}")

        # 将测试结果追加到 CSV 最后一行
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if opt.loss_type == 'ncc':
                writer.writerow(['FINAL_TEST', -1, -1, np.mean(NCCs_Test), np.mean(NCCs_Test_org)])
            else:
                writer.writerow(['FINAL_TEST', np.mean(Test_Loss_List), -1, np.mean(NCCs_Test), np.mean(NCCs_Test_org)])
        print(f"\nTest results appended to: {csv_name}")

train()