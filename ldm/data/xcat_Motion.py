import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
import json
from scipy.ndimage import affine_transform, map_coordinates
from scipy.interpolate import RegularGridInterpolator


# =============================================
# VQ / LDM 训练用：单图模式（fixed + moving 混合，各自 70%）
# 与 ldm/data/camus.py 中的 XCATBase 设计完全一致
# =============================================
#这个类用于ldm和vq的训练
class XCATBase(Dataset):
    """
    XCAT 医学图像数据集基类（用于 VQ / LDM 训练）。

    数据目录结构：
        data_root/
            fixed/fixed/*.npy    # 固定图像
            moving/moving/*.npy  # 移动图像（运动增强后的 phase1）

    训练时：将 fixed 和 moving 合并为一个大池子，
    各取前 70% 作为训练集，混合打乱。
    每次返回单张图像（fixed 或 moving），VQ 学习单图重建。

    VQ 输入 = moving_t（单图像）
    VQ 目标 = 重建 moving_t

    数据增强（仅训练）：
        - 随机水平翻转（fixed 和 moving 独立翻转，非配对）
        - 运动模拟（仅对 moving 图像施加额外变形）

    与 ldm/data/camus.py 中的 XCATBase 完全一致的设计思路。
    """
    def __init__(self,
                 data_root,
                 split='train',     # 'train' | 'val' | 'test'
                 flip_p=0.5,
                 split_file=None,
                 motion_types=None   # 训练时对 moving 图像施加的运动类型
                 ):
        self.split = split
        self.data_root = data_root
        self.flip_p = flip_p if split == 'train' else 0.0
        self.motion_types = motion_types if motion_types and split == 'train' else ['identity']

        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_dir = os.path.join(data_root, 'moving', 'moving')
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.npy')))

        n_fixed = len(self.fixed_paths)
        n_moving = len(self.moving_paths)

        # 划分：各取 70% 训练、15% 验证、15% 测试（fixed 和 moving 独立划分）
        n_f = n_fixed
        n_m = n_moving
        if split == 'train':
            self.fixed_idx_range = list(range(0, int(n_f * 0.7)))
            self.moving_idx_range = list(range(0, int(n_m * 0.7)))
        elif split == 'val':
            self.fixed_idx_range = list(range(int(n_f * 0.7), int(n_f * 0.85)))
            self.moving_idx_range = list(range(int(n_m * 0.7), int(n_m * 0.85)))
        else:  # test
            self.fixed_idx_range = list(range(int(n_f * 0.85), n_f))
            self.moving_idx_range = list(range(int(n_m * 0.85), n_m))

        # 构建样本池：固定图像样本 + 移动图像样本，混合在一起
        # 每个样本 = (路径, 是否为 moving 图像, 运动类型)
        self.samples = []
        for idx in self.fixed_idx_range:
            self.samples.append((self.fixed_paths[idx], False, 'identity'))
        for idx in self.moving_idx_range:
            for mt in self.motion_types:
                # 构建的是：moving0+identity, moving0+rotate10, moving0+warp
                self.samples.append((self.moving_paths[idx], True, mt))

        # 训练时打乱，移动图像由于多运动类型会重复出现
        if split == 'train':
            import random
            random.seed(42)
            random.shuffle(self.samples)

        self._length = len(self.samples)
        print(f"[XCATBase] split={split}, fixed_paths={len(self.fixed_idx_range)}, "
              f"moving_paths={len(self.moving_idx_range)}, total_samples={self._length}")

    def _apply_motion(self, img, motion_type):
        """对单张 2D 图像施加运动变形（仅 moving 图像）"""
        h, w = img.shape
        if motion_type == 'identity':
            return img.copy()
        elif motion_type == 'rotate10':
            angle = 5 * np.pi / 180  # 从 10 度改成 5 度
            center = np.array([h / 2, w / 2])
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            offset = center - rot @ center
            return affine_transform(img, rot, offset=offset, order=3, mode='constant', cval=0)
        elif motion_type == 'warp':
            dx = np.random.uniform(-2, 2, size=(4, 4))  # 从 -5,5 改成 -2,2
            dy = np.random.uniform(-2, 2, size=(4, 4))
            x_grid = np.linspace(0, w - 1, 4)
            y_grid = np.linspace(0, h - 1, 4)
            interp_x = RegularGridInterpolator((y_grid, x_grid), dx, bounds_error=False, fill_value=0)
            interp_y = RegularGridInterpolator((y_grid, x_grid), dy, bounds_error=False, fill_value=0)
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=-1)
            x_new = x_coords + interp_x(coords).reshape(h, w)
            y_new = y_coords + interp_y(coords).reshape(h, w)
            return map_coordinates(img, [y_new, x_new], order=3, mode='constant', cval=0)
        elif motion_type == 'scale05':
            scale_factors = (1.0, 0.5)
            matrix = np.diag(scale_factors)
            center = np.array([h / 2, w / 2])
            offset = center - matrix @ center
            return affine_transform(img, matrix, offset=offset, order=3, mode='constant', cval=0)
        else:
            return img.copy()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        #获取样本索引i的样本路径、是否为移动图像、运动类型
        path, is_moving, motion_type = self.samples[i]

        # 加载图像
        image = np.load(path).astype(np.float32)

        # 运动模拟（仅 moving 图像）
        if is_moving and motion_type != 'identity':
            image = self._apply_motion(image, motion_type)

        # Min-Max 归一化到 [0, 1]
        minv = image.min()
        maxv = image.max()
        if maxv - minv > 1e-6:
            image = (image - minv) / (maxv - minv)

        # 随机水平翻转（独立翻转，非配对翻转）
        if np.random.rand() < self.flip_p:
            image = np.fliplr(image).copy()

        # 转为 tensor，形状 [H, W]
        image_t = torch.from_numpy(image).float()

        # VQ 训练只需 image，与 ldm/data/camus.py 中的 XCATBase 一致
        return {"image": image_t}


class XCATTrain(XCATBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='train',
            flip_p=0.5,
            motion_types=['identity', 'rotate10', 'warp'],#去掉scale05
            **kwargs
        )


class XCATValidation(XCATBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='val',
            **kwargs
        )


class XCATTest(XCATBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='test',
            **kwargs
        )


# =============================================
# 配准训练用：保持原有的配对模式（fixed + moving 配对输入）
# 训练时每对数据 × 3 种运动类型
# =============================================
#这个类用于配准网络的训练
class XCATMotionAugmented(Dataset):
    """
    基于 XCAT 配对数据（fixed/moving 独立文件夹）的运动模拟数据增强 Dataset。
    用于配准网络训练（需要 fixed + moving 配对）。

    数据目录结构：
        data_root/
            fixed/fixed/*.npy    # 固定图像 (phase0)
            moving/moving/*.npy  # 移动图像 (phase1)，将对其施加额外运动模拟

    增强逻辑：
        fixed  = fixed/*.npy[对应索引]        （phase0，原封不动）
        moving = 对 moving/*.npy[对应索引] 施加运动：
            identity  → 移动图像 = phase1 原始
            rotate10  → 移动图像 = phase1 + 旋转
            warp      → 移动图像 = phase1 + 弹性形变

    训练时：每对数据 × 3 种运动类型 = N × 3 样本
    验证/测试：仅 identity 配对

    Args:
        data_root (str): 数据根目录（包含 fixed/ 和 moving/ 子目录）
        split (str): 'train' / 'val' / 'test'
        motion_types (list): 运动类型列表
        return_condition (bool): 是否返回条件图像（固定图像）
        flip_p (float): 随机水平翻转概率
        split_file (str): 可选的划分文件（JSON）
    """
    def __init__(self,
                 data_root,
                 split='train',
                 motion_types=['identity', 'rotate10', 'warp'],  # 去掉 scale05
                 return_condition=True,
                 flip_p=0.5,
                 split_file=None):
        self.data_root = data_root
        self.split = split
        self.motion_types = motion_types
        self.return_condition = return_condition
        self.flip_p = flip_p if split == 'train' else 0.0

        # 加载 fixed 和 moving 文件夹下的 .npy 文件（一一对应）
        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_dir = os.path.join(data_root, 'moving', 'moving')
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.npy')))

        assert len(self.fixed_paths) == len(self.moving_paths), \
            f"Fixed ({len(self.fixed_paths)}) and Moving ({len(self.moving_paths)}) file count mismatch"

        # 划分数据集
        n = len(self.fixed_paths)
        if split_file is None:
            split_file = os.path.join(data_root, 'split_indices.json')
        if os.path.exists(split_file):
            with open(split_file) as f:
                cfg = json.load(f)
            info = cfg.get('vq_ldm', cfg.get('registration', {}))
            split_conf = info.get('split', {})
            idx_range = split_conf.get(split, [0, n - 1])
            self.pair_indices = list(range(idx_range[0], idx_range[1] + 1))
            print(f"[XCATMotionAug] split={split}, loaded {len(self.pair_indices)} pairs from split_indices.json")
        else:
            if split == 'train':
                self.pair_indices = list(range(0, int(n * 0.7)))
            elif split == 'val':
                self.pair_indices = list(range(int(n * 0.7), int(n * 0.85)))
            else:
                self.pair_indices = list(range(int(n * 0.85), n))
            print(f"[XCATMotionAug] split={split}, loaded {len(self.pair_indices)} pairs by ratio")

        # 训练时：扩展样本列表，每对 × 多种运动类型
        if split == 'train':
            self.samples = []
            for idx in self.pair_indices:
                for mt in motion_types:
                    self.samples.append((idx, mt))
        else:
            self.samples = [(idx, 'identity') for idx in self.pair_indices]

        print(f"[XCATMotionAug] split={split}, total samples (after expansion): {len(self.samples)}")

    def _apply_motion(self, img, motion_type):
        """对单张 2D 图像施加运动变形"""
        h, w = img.shape
        if motion_type == 'identity':
            return img.copy()
        elif motion_type == 'rotate10':
            angle = 10 * np.pi / 180  # 原本是10度改成5度
            center = np.array([h / 2, w / 2])
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            offset = center - rot @ center
            return affine_transform(img, rot, offset=offset, order=3, mode='constant', cval=0)
        elif motion_type == 'warp':
            dx = np.random.uniform(-5, 5, size=(4, 4))  # 从 -5,5 改成 -2,2
            dy = np.random.uniform(-5, 5, size=(4, 4))
            x_grid = np.linspace(0, w - 1, 4)
            y_grid = np.linspace(0, h - 1, 4)
            interp_x = RegularGridInterpolator((y_grid, x_grid), dx, bounds_error=False, fill_value=0)
            interp_y = RegularGridInterpolator((y_grid, x_grid), dy, bounds_error=False, fill_value=0)
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=-1)
            x_new = x_coords + interp_x(coords).reshape(h, w)
            y_new = y_coords + interp_y(coords).reshape(h, w)
            return map_coordinates(img, [y_new, x_new], order=3, mode='constant', cval=0)
        elif motion_type == 'scale05':
            scale_factors = (1.0, 0.5)
            matrix = np.diag(scale_factors)
            center = np.array([h / 2, w / 2])
            offset = center - matrix @ center
            return affine_transform(img, matrix, offset=offset, order=3, mode='constant', cval=0)
        else:
            return img.copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair_idx, motion_type = self.samples[idx]

        # 加载配对数据
        fixed = np.load(self.fixed_paths[pair_idx]).astype(np.float32)
        moving = np.load(self.moving_paths[pair_idx]).astype(np.float32)

        # 对移动图像（phase1）施加额外运动模拟
        moving = self._apply_motion(moving, motion_type)

        # Min-Max 归一化到 [0, 1]（fixed 和 moving 使用相同的 min/max）
        minv = fixed.min()
        maxv = fixed.max()
        if maxv - minv > 1e-6:
            fixed = (fixed - minv) / (maxv - minv)
            moving = (moving - minv) / (maxv - minv)

        # 随机翻转（fixed 和 moving 同时翻转，保持配对关系）
        if np.random.rand() < self.flip_p:
            fixed = np.fliplr(fixed).copy()
            moving = np.fliplr(moving).copy()

        # 转为 tensor
        fixed_t = torch.from_numpy(fixed).float()
        moving_t = torch.from_numpy(moving).float()

        # 配准训练：返回固定图像和移动图像
        example = {"image": moving_t}
        if self.return_condition:
            example["condition"] = fixed_t
        return example
