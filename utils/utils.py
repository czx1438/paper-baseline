'''
Jiong Wu 
University of Florida
jiongwu.application@ufl.edu

Thanks to 
Junyu Chen
Johns Hopkins Unversity
jchen245@jhmi.edu
'''

import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import torch.utils.data as Data
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0
  
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        h2, w2 = mov_image.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        #（B,H,W）
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        #（B，H，W）
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        #（H，W）
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True,padding_mode="border")#原本是border，现在改成zeros
        #（B，C，H，W）
        return sample_grid, warped


def smoothloss(y_pred):
    h2, w2 = y_pred.shape[-2:]
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0

def bending_energy_loss(y_pred):
    """二阶弯曲能量(bending energy)正则:惩罚位移场的二阶导(曲率),
    而非一阶导(大小变化)。与一阶 smoothloss 用同样的像素换算口径(×size/2),
    因此两项尺度可比,可直接加权叠加。

    适用场景:心脏/肝脏反向大位移的交界带会产生"急转弯"(高曲率)→ 折叠,
    bending energy 专门压这种弯折,但允许平滑过渡的大位移(不惩罚平缓的大位移),
    所以不会像一阶 smooth 那样把"合理大位移"也一起压住。

    y_pred: [B, 2, H, W] 位移场(归一化坐标 [-1,1])
    """
    h2, w2 = y_pred.shape[-2:]
    # 一阶差分(换算到像素单位,口径与 smoothloss 一致)
    dy = (y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2   # ∂/∂行
    dx = (y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2   # ∂/∂列

    # 二阶差分
    dyy = (dy[:, :, 1:, :] - dy[:, :, :-1, :]) / 2 * h2          # ∂²/∂行²
    dxx = (dx[:, :, :, 1:] - dx[:, :, :, :-1]) / 2 * w2          # ∂²/∂列²
    # 交叉二阶项 ∂²/∂行∂列(对 dx 再沿行方向差分)
    dxy = (dx[:, :, 1:, :] - dx[:, :, :-1, :]) / 2 * h2

    # bending energy = dyy² + dxx² + 2·dxy²(薄板样条能量的标准形式)
    return (torch.mean(dyy * dyy)
            + torch.mean(dxx * dxx)
            + 2.0 * torch.mean(dxy * dxy)) / 4.0


def jacobian_neg_loss(y_pred):
    """空间自适应折叠惩罚:只惩罚 Jacobian 行列式 ≤0 的像素(发生折叠/翻转处),
    其它地方 loss=0,完全不影响。可微,直接进 loss。

    口径与可视化/jac_stats 一致:DVF ×size/2 转像素后算 det(J),
    判定折叠用 det(J) < 0。这里用可微的 relu(-det) 作为惩罚。

    用途:bending energy 仍压不住的顽固折叠样本(如心肝交界),
    这个项精准地只在折叠像素加压,不碰其它区域(包括欠配的心脏)。

    y_pred: [B, 2, H, W] 位移场(归一化坐标 [-1,1])
    返回每像素 relu(-detJ) 的均值。
    """
    h2, w2 = y_pred.shape[-2:]
    # 换算到像素位移(与 jac_stats 口径一致)
    disp = y_pred.clone()
    disp = torch.stack([disp[:, 0] * h2 / 2.0, disp[:, 1] * w2 / 2.0], dim=1)

    # 位移 + 恒等网格的梯度 = 形变映射的 Jacobian
    # ∂/∂行, ∂/∂列(中心差分用前向差分近似,和 np.gradient 的内部点略有差异,
    # 但作为可微惩罚足够;评测仍用 jacobian_determinant_vxm 的口径)
    dfx_dy = disp[:, 0, 1:, :] - disp[:, 0, :-1, :]   # ∂(行位移)/∂行
    dfx_dx = disp[:, 0, :, 1:] - disp[:, 0, :, :-1]   # ∂(行位移)/∂列
    dfy_dy = disp[:, 1, 1:, :] - disp[:, 1, :-1, :]   # ∂(列位移)/∂行
    dfy_dx = disp[:, 1, :, 1:] - disp[:, 1, :, :-1]   # ∂(列位移)/∂列

    # 对齐到公共尺寸 [B, H-1, W-1]
    dfx_dy = dfx_dy[:, :, :-1]
    dfy_dy = dfy_dy[:, :, :-1]
    dfx_dx = dfx_dx[:, :-1, :]
    dfy_dx = dfy_dx[:, :-1, :]

    # 形变映射 φ = id + disp 的 Jacobian 行列式
    # J = [[1+dfx_dy_... ]] —— 注意恒等映射的 +1 加在对角:
    # 行位移对行求导 +1, 列位移对列求导 +1
    j11 = 1.0 + dfx_dy   # ∂φ_行/∂行
    j12 = dfx_dx         # ∂φ_行/∂列
    j21 = dfy_dy         # ∂φ_列/∂行
    j22 = 1.0 + dfy_dx   # ∂φ_列/∂列
    detJ = j11 * j22 - j12 * j21

    return torch.relu(-detJ).mean()

def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff


class MSE:
    """
    Mean squared error loss.
    """
 
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def crop_center(img, cropx, cropy, cropz):
    x, y, z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2
    return img[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]


def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm

def loadnpz(npzpath):
    features=np.load(npzpath, allow_pickle=True)
    f_all = features['arr_0'].item()
    imglist = f_all['imglist']
    movimg = imglist[0,:,:]
    movlab = imglist[1,:,:]
    tarimg = imglist[2,:,:]
    tarlab = imglist[3,:,:]
    return movimg, movlab, tarimg, tarlab

class Dataset_epoch_with_name(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names):
        'Initialization'
        super(Dataset_epoch_with_name, self).__init__()
        self.names = names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, index):
        'Generates one sample of data'
        arr = np.load(self.names[index])

        movimg = arr["img_small"].astype(np.float32)
        tarimg = arr["img_large"].astype(np.float32)
        
        # Handle both labeled and image-only data
        if "mask_small" in arr and "mask_large" in arr:
            movlab = arr["mask_small"]
            tarlab = arr["mask_large"]
        else:
            movlab = np.zeros_like(arr["img_small"])
            tarlab = np.zeros_like(arr["img_large"])
        
        movimg = torch.from_numpy(movimg).float()
        tarimg = torch.from_numpy(tarimg).float()
        movlab = torch.from_numpy(movlab).float()
        tarlab = torch.from_numpy(tarlab).float()

        pairname = self.names[index].split('/')[-1].split('.npz')[0]

        return movimg.unsqueeze(0), tarimg.unsqueeze(0), movlab.unsqueeze(0), tarlab.unsqueeze(0), pairname


# =============================================
# Wrapper for XCATMotionAugmented so it matches
# the 5-value interface of Dataset_epoch_with_name:
#   X, Y, segx, segy, name
# XCAT has no segmentation masks -> segx/segy are zero tensors.
# =============================================
class Dataset_XCAT_Registration:
    """
    Thin wrapper around ldm.data.xcat_Motion.XCATMotionAugmented.

    Usage in train.py:
        from utils.utils import Dataset_XCAT_Registration
        train_ds = Dataset_XCAT_Registration(
            data_root='/path/to/xcat_data',
            split='train',
            motion_types=['identity', 'rotate10', 'scale05', 'warp'],
        )
        train_loader = Data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    """
    def __init__(self, data_root, split='train', motion_types=None,
                 return_condition=True, flip_p=0.5):
        from ldm.data.xcat_Motion import XCATMotionAugmented
        #接受XCATMotionAugmented类，并初始化
        self._ds = XCATMotionAugmented(
            data_root=data_root,
            split=split,
            motion_types=motion_types,
            return_condition=return_condition,
            flip_p=flip_p,
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        #调用XCATMotionAugmented类的__getitem__方法，获取样本
        example = self._ds[idx]
        moving_t  = example["image"]       # [H, W]
        fixed_t  = example["condition"]    # [H, W]

        # Add channel & batch dims, then strip them to keep shape [H, W]
        H, W = fixed_t.shape
        moving_t  = moving_t.unsqueeze(0)  # [1, H, W]
        fixed_t   = fixed_t.unsqueeze(0)   # [1, H, W]

        # segx/segy: all-zeros (XCAT has no segmentation)
        segx = torch.zeros(1, H, W)
        segy = torch.zeros(1, H, W)

        # name encodes motion type so you can trace which augmentation was used
        #获取样本索引i的样本路径、是否为移动图像、运动类型
        sample = self._ds.samples[idx]
        motion_type = sample[1]
        #构建样本名称fixed0007_rotate10记录该样本来自哪对fixed和moving以及moving施加了什么操作
        pairname = f"fixed{sample[0]:04d}_{motion_type}"
        return moving_t, fixed_t, segx, segy, pairname


class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names):
        'Initialization'
        super(Dataset_epoch, self).__init__()
        self.names = names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, index):
        'Generates one sample of data'
        npzpath = self.names[index]
        movimg, movlab, tarimg, tarlab = loadnpz(npzpath)

        movimg = torch.from_numpy(movimg).float()
        tarimg = torch.from_numpy(tarimg).float()
        movlab = torch.from_numpy(movlab).float()
        tarlab = torch.from_numpy(tarlab).float()

        return movimg.unsqueeze(0), tarimg.unsqueeze(0), movlab.unsqueeze(0), tarlab.unsqueeze(0)


# =========================================================
# ROI-Based Loss 工具函数
# =========================================================

def generate_roi_mask(img, mode='liver', threshold=None, high_percentile=75, low_percentile=15):
    """
    基于图像强度自动生成 ROI 分割掩膜（三区域分类）。

    适用于 PET/CT：自适应找背景/软组织/心脏边界。

    原理：
    - 心脏（最亮软组织）：强度 >= high_percentile 百分位阈值 → label=2
    - 软组织（中等亮度）：强度在 low_percentile ~ high_percentile 之间 → label=1
    - 背景（空气/低密度）：强度 < low_percentile 百分位 → label=0

    Args:
        img: [B, 1, H, W] 或 [H, W] 的图像张量
        mode: 'liver' (最高亮度区域) 或 'cardiac' (中等阈值)
        threshold: 手动指定高阈值（覆盖 high_percentile 计算）
        high_percentile: 用于自动计算高阈值的百分位（默认 75，即最亮的 25% 为心脏）
        low_percentile: 用于自动计算低阈值的百分位（默认 15，即最低的 15% 为背景）

    Returns:
        seg_mask: [B, 1, H, W] 的分割掩膜张量，值为 0(背景)/1(软组织)/2(心脏)
        info: dict，包含 low_thresh, high_thresh
    """
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(1)
    img = img.float()

    flat = img.flatten()
    N = flat.numel()

    # 高阈值：区分心脏和普通软组织（从高往低数 high_percentile%）
    if threshold is not None:
        high_thresh = threshold
    else:
        k_high = max(1, min(N, int(N * high_percentile / 100)))
        high_thresh = torch.kthvalue(flat, k_high)[0].item()

    # 低阈值：区分背景和软组织（从低往高数 low_percentile%）
    k_low = max(1, min(N, int(N * low_percentile / 100)))
    low_thresh = torch.kthvalue(flat, k_low)[0].item()

    # 三区域分割
    seg_mask = torch.zeros_like(img, dtype=torch.float32)
    seg_mask[(img >= low_thresh) & (img < high_thresh)] = 1.0   # 软组织
    seg_mask[img >= high_thresh] = 2.0                          # 心脏/肝脏

    info = {'low_thresh': float(low_thresh), 'high_thresh': float(high_thresh)}
    return seg_mask, info


def ncc_loss_weighted(fixed, moving, seg_mask, cardiac_weight=3.0, soft_weight=0.3, bg_weight=0.0, win_size=9):
    """
    加权局部 NCC Loss——让心脏区域主导梯度，背景几乎不贡献梯度。

    对每个像素的 NCC 值乘以对应区域的权重，然后加权平均。

    Args:
        fixed: 固定图像 [B, 1, H, W]
        moving: 移动图像 [B, 1, H, W]
        seg_mask: 分割掩膜 [B, 1, H, W]，值为 0(背景)/1(软组织)/2(心脏)
        cardiac_weight: 心脏/肝脏区域权重（默认 3.0）
        soft_weight: 软组织区域权重（默认 0.3）
        bg_weight: 背景权重（默认 0.0）
        win_size: NCC 窗口大小（默认 9）
    """
    weight_mask = torch.where(seg_mask >= 2.0, cardiac_weight,
                     torch.where(seg_mask >= 1.0, soft_weight, bg_weight))
    assert fixed.shape == moving.shape == weight_mask.shape
    b, c, h, w = fixed.shape
    pad = win_size // 2

    # 填充
    fixed_pad = F.pad(fixed, [pad, pad, pad, pad], mode='reflect')
    moving_pad = F.pad(moving, [pad, pad, pad, pad], mode='reflect')
    weight_pad = F.pad(weight_mask, [pad, pad, pad, pad], mode='constant', value=0)

    # 提取 patch
    patches_fix = fixed_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)
    patches_mov = moving_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)
    patches_wgt = weight_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)

    patches_fix = patches_fix.contiguous().view(b, c, h, w, -1)
    patches_mov = patches_mov.contiguous().view(b, c, h, w, -1)
    patches_wgt = patches_wgt.contiguous().view(b, c, h, w, -1)

    # 局部 NCC
    mean_fix = patches_fix.mean(dim=-1)
    mean_mov = patches_mov.mean(dim=-1)
    centered_fix = patches_fix - mean_fix.unsqueeze(-1)
    centered_mov = patches_mov - mean_mov.unsqueeze(-1)
    var_fix = (centered_fix ** 2).mean(dim=-1)
    var_mov = (centered_mov ** 2).mean(dim=-1)
    cross = (centered_fix * centered_mov).mean(dim=-1)

    eps = 1e-8
    ncc_local = cross / (torch.sqrt(var_fix.clamp(min=eps)) * torch.sqrt(var_mov.clamp(min=eps)) + eps)

    # 加权平均
    weight_sum = patches_wgt.sum(dim=-1)
    weighted_ncc = (ncc_local * weight_mask * win_size * win_size).sum() / (weight_sum.sum() + eps)

    return 1.0 - weighted_ncc


def mse_loss_weighted(pred, target, seg_mask, cardiac_weight=3.0, soft_weight=0.3, bg_weight=0.0):
    """
    加权 MSE Loss——心脏区域主导，背景几乎不贡献。

    Args:
        pred: 预测图像 [B, 1, H, W]
        target: 目标图像 [B, 1, H, W]
        seg_mask: 分割掩膜 [B, 1, H, W]，值为 0(背景)/1(软组织)/2(心脏)
        cardiac_weight: 心脏/肝脏区域权重（默认 3.0）
        soft_weight: 软组织区域权重（默认 0.3）
        bg_weight: 背景权重（默认 0.0）
    """
    weight_mask = torch.where(seg_mask >= 2.0, cardiac_weight,
                     torch.where(seg_mask >= 1.0, soft_weight, bg_weight))
    diff = (pred - target) ** 2
    weighted_diff = diff * weight_mask
    return weighted_diff.sum() / (weight_mask.sum() + 1e-8)


def ncc_loss_global_weighted(fixed, moving, weight_mask, win_size=9):
    """
    全局加权 NCC Loss：
    - 先对每个像素计算局部 NCC
    - 再用 weight_mask 进行加权平均（而非简单平均）
    """
    assert fixed.shape == moving.shape == weight_mask.shape
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
    ncc_local = cross / (torch.sqrt(var_fix.clamp(min=eps)) * torch.sqrt(var_mov.clamp(min=eps)) + eps)

    # 加权平均
    w = weight_mask / (weight_mask.sum() + eps)
    weighted_ncc = (ncc_local * w).sum()
    return 1.0 - weighted_ncc


