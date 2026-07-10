import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
import random


# =============================================
# NPZ 模式 VQ / LDM 训练用：单图模式
# 与 ldm/data/xcat_Motion_Seq.py 设计完全一致
# =============================================
class XCATNPZBase(Dataset):
    """
    XCAT NPZ 模式单图数据集（用于 VQ / LDM 训练）。

    数据目录结构：
        data_root/
            fixed/fixed/*.npy       # 固定图像 (H, W)
            moving/moving/*.npy     # 原始移动图像 phase1 (H, W)
            moving_seq/*.npy        # 8 帧形变序列 (8, H, W)

    训练逻辑：
        - 将 fixed 图像 + 原始 moving 图像 + moving_seq 8帧全部合并为一个大池
        - 混合打乱，作为单图输入送入 VQ/LDM 进行重建训练
        - 无额外运动增强，仅保留随机水平翻转

    数据增强（仅训练）：
        - 随机水平翻转（独立翻转）
    """

    def __init__(self,
                 data_root,
                 split='train',
                 flip_p=0.5,
                 split_file=None,
                 **kwargs):
        self.split = split
        self.data_root = data_root
        self.flip_p = flip_p if split == 'train' else 0.0

        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_dir = os.path.join(data_root, 'moving', 'moving')
        moving_seq_dir = os.path.join(data_root, 'moving_seq')
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.npy')))
        self.moving_seq_paths = sorted(glob.glob(os.path.join(moving_seq_dir, '*.npy')))

        n_fixed = len(self.fixed_paths)
        n_moving = len(self.moving_paths)
        n_seq = len(self.moving_seq_paths)

        # 划分：各取 70% 训练、15% 验证、15% 测试（三者独立划分）
        if split == 'train':
            fixed_idx_list = list(range(0, int(n_fixed * 0.7)))
            moving_idx_list = list(range(0, int(n_moving * 0.7)))
            seq_idx_list = list(range(0, int(n_seq * 0.7)))
        elif split == 'val':
            fixed_idx_list = list(range(int(n_fixed * 0.7), int(n_fixed * 0.85)))
            moving_idx_list = list(range(int(n_moving * 0.7), int(n_moving * 0.85)))
            seq_idx_list = list(range(int(n_seq * 0.7), int(n_seq * 0.85)))
        else:  # test
            fixed_idx_list = list(range(int(n_fixed * 0.85), n_fixed))
            moving_idx_list = list(range(int(n_moving * 0.85), n_moving))
            seq_idx_list = list(range(int(n_seq * 0.85), n_seq))

        # 构建样本池：fixed + original moving + moving_seq 8帧展开
        # sample_type: 'fixed' | 'moving' | 'seq'
        self.samples = []

        # fixed 图像
        for idx in fixed_idx_list:
            self.samples.append((self.fixed_paths[idx], 'fixed', None))

        # 原始 moving 图像
        for idx in moving_idx_list:
            self.samples.append((self.moving_paths[idx], 'moving', None))

        # moving_seq 序列展开为 8 个独立帧
        for idx in seq_idx_list:
            for frame_idx in range(8):
                self.samples.append((self.moving_seq_paths[idx], 'seq', frame_idx))

        # 训练时打乱
        if split == 'train':
            random.seed(42)
            random.shuffle(self.samples)

        self._length = len(self.samples)

        print(f"[XCATNPZBase] split={split}, "
              f"fixed={len(fixed_idx_list)}, "
              f"moving={len(moving_idx_list)}, "
              f"seq_files={len(seq_idx_list)}, "
              f"total_samples={self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path, sample_type, frame_idx = self.samples[i]

        image = np.load(path).astype(np.float32)

        # moving_seq 文件：从 (8, H, W) 中取对应帧
        if sample_type == 'seq' and frame_idx is not None:
            image = image[frame_idx]

        # Min-Max 归一化到 [0, 1]
        minv = image.min()
        maxv = image.max()
        if maxv - minv > 1e-6:
            image = (image - minv) / (maxv - minv)

        # 随机水平翻转（独立翻转，非配对）
        if random.random() < self.flip_p:
            image = np.fliplr(image).copy()

        # 转为 tensor，形状 [H, W]
        image_t = torch.from_numpy(image).float()

        # VQ/LDM 训练只需 image
        return {"image": image_t}


class XCATNPZTrain(XCATNPZBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='train',
            flip_p=0.5,
            **kwargs
        )


class XCATNPZValidation(XCATNPZBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='val',
            flip_p=0.0,
            **kwargs
        )


class XCATNPZTest(XCATNPZBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='test',
            flip_p=0.0,
            **kwargs
        )
