import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
import json


# ===============================================================
# VQ / LDM 训练用：单图模式（所有 fixed + 所有 moving 混合大池）
# ===============================================================
class XCATSeqBase(Dataset):
    """
    XCAT 心脏序列单图数据集（用于 VQ / LDM 训练）。

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
        moving_dir = os.path.join(data_root, 'moving','moving')
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
            import random
            random.seed(42)
            random.shuffle(self.samples)

        self._length = len(self.samples)

        print(f"[XCATSeqBase] split={split}, "
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
        if np.random.rand() < self.flip_p:
            image = np.fliplr(image).copy()

        # 转为 tensor，形状 [H, W]
        image_t = torch.from_numpy(image).float()

        # VQ/LDM 训练只需 image
        return {"image": image_t}


class XCATSeqTrain(XCATSeqBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='train',
            flip_p=0.5,
            **kwargs
        )


class XCATSeqValidation(XCATSeqBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='val',
            flip_p=0.0,
            **kwargs
        )


class XCATSeqTest(XCATSeqBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='test',
            flip_p=0.0,
            **kwargs
        )


# ===============================================================
# 配准网络训练用：fixed + moving 配对模式
# 每个 fixed 对应：1 个原始 moving + 8 个形变帧 = 9 个 moving 变体
# ===============================================================
class XCATSeqRegistration(Dataset):
    """
    XCAT 心脏序列配对数据集（用于配准网络训练）。

    数据目录结构：
        data_root/
            fixed/fixed/*.npy       # 固定图像 (H, W)
            moving/moving/*.npy     # 原始移动图像 phase1 (H, W)
            moving_seq/*.npy        # 8 帧形变序列 (8, H, W)

    配准逻辑：
        - fixed = fixed/*.npy[对应索引]
        - moving 变体（共 9 种）：
            1. 原始 moving/*.npy[对应索引]（phase1 原图）
            2-9. moving_seq/*.npy[对应索引] 的 8 帧形变
        - 训练时 9 种变体全部展开为独立样本（确保全部用到）
        - 验证/测试时仅使用原始 moving（phase1）

    返回格式：与 Dataset_XCAT_Registration 完全一致
        (moving_t, fixed_t, segx, segy, pairname)
    """

    def __init__(self,
                 data_root,
                 split='train',
                 flip_p=0.5,
                 split_file=None):
        self.data_root = data_root
        self.split = split
        self.flip_p = flip_p if split == 'train' else 0.0

        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_dir = os.path.join(data_root, 'moving','moving')
        moving_seq_dir = os.path.join(data_root, 'moving_seq')
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.npy')))
        self.moving_seq_paths = sorted(glob.glob(os.path.join(moving_seq_dir, '*.npy')))

        assert len(self.fixed_paths) == len(self.moving_paths), \
            f"Fixed ({len(self.fixed_paths)}) and Moving ({len(self.moving_paths)}) file count mismatch"
        assert len(self.fixed_paths) == len(self.moving_seq_paths), \
            f"Fixed ({len(self.fixed_paths)}) and MovingSeq ({len(self.moving_seq_paths)}) file count mismatch"

        # 划分数据集（fixed / moving / moving_seq 三者一一对应），固定 70:15:15
        n = len(self.fixed_paths)
        if split == 'train':
            self.pair_indices = list(range(0, int(n * 0.7)))
        elif split == 'val':
            self.pair_indices = list(range(int(n * 0.7), int(n * 0.85)))
        else:
            self.pair_indices = list(range(int(n * 0.85), n))
        print(f"[XCATSeqRegistration] split={split}, loaded {len(self.pair_indices)} pairs (70:15:15 ratio)")

        # 构建样本池：(fixed_path, moving_path_or_seq, source_type, frame_idx_or_None)
        # source_type: 'original' | 'seq'
        # 训练时：每个 pair × (1 original + 8 seq frames) = 9 变体
        # 验证/测试时：每个 pair 仅 original
        self.samples = []
        for pair_idx in self.pair_indices:
            if split == 'train':
                # 原始 moving
                self.samples.append((
                    self.fixed_paths[pair_idx],
                    self.moving_paths[pair_idx],
                    'original',
                    None
                ))
                # 8 帧形变序列
                for frame_idx in range(8):
                    self.samples.append((
                        self.fixed_paths[pair_idx],
                        self.moving_seq_paths[pair_idx],
                        'seq',
                        frame_idx
                    ))
            else:
                # 验证/测试：仅原始 moving
                self.samples.append((
                    self.fixed_paths[pair_idx],
                    self.moving_paths[pair_idx],
                    'original',
                    None
                ))

        self._length = len(self.samples)

        print(f"[XCATSeqRegistration] split={split}, pairs={len(self.pair_indices)}, "
              f"variants_per_pair={'9' if split == 'train' else '1'}, total_samples={self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        fixed_path, moving_path, source_type, frame_idx = self.samples[idx]

        # 加载 fixed
        fixed = np.load(fixed_path).astype(np.float32)  # (H, W)

        # 加载 moving（原始 or 序列帧）
        if source_type == 'original':
            moving = np.load(moving_path).astype(np.float32)  # (H, W)
        else:  # 'seq'
            moving_seq = np.load(moving_path).astype(np.float32)  # (8, H, W)
            moving = moving_seq[frame_idx]

        # Min-Max 归一化（fixed 和 moving 使用相同的 min/max）
        minv = fixed.min()
        maxv = fixed.max()
        if maxv - minv > 1e-6:
            fixed = (fixed - minv) / (maxv - minv)
            moving = (moving - minv) / (maxv - minv)

        # 随机翻转（fixed 和 moving 同时翻转，保持配对关系）
        if np.random.rand() < self.flip_p:
            fixed = np.fliplr(fixed).copy()
            moving = np.fliplr(moving).copy()

        # 转为 tensor，添加通道维度 (H, W) -> (1, H, W)
        fixed_t = torch.from_numpy(fixed).float().unsqueeze(0)
        moving_t = torch.from_numpy(moving).float().unsqueeze(0)

        H, W = fixed_t.shape[1], fixed_t.shape[2]
        segx = torch.zeros(1, 1, H, W)
        segy = torch.zeros(1, 1, H, W)

        # 与老的 XCATMotionAugmented (utils/utils.py:406) 保持一致:
        # 老版本 pairname = f"fixed{pair_idx:04d}_{motion_type}", 这里的 pair_idx 就是 self.pair_indices 在配对层级的索引
        # moving_motion 可视化时只显示 sample 编号 + 配对号 (即配对层级的 pair_indices 里的字典序索引, 对应实际的 mps[pair_idx] 文件)
        # 注意 self.samples 展开后 train 模式每对 9 个样本 (1 original + 8 seq), val/test 每对 1 个样本,
        # 所以配对层级索引 = idx // 9 (train) 或 idx (val/test).
        # 用 source_type='original' 时拼 pairname = pair_indices[pair_idx] (字典序索引 = 实际 mps[pair_idx] 的位置)
        # 用 source_type='seq' 时拼 pairname = f"{basename}_seq{frame_idx}" (老版本未使用, 保留)
        if source_type == 'original':
            pair_idx_in_pair_indices = idx // 9 if len(self.pair_indices) * 9 == len(self.samples) else idx
            pairname = f"{self.pair_indices[pair_idx_in_pair_indices]}_original"
        else:
            basename = os.path.splitext(os.path.basename(fixed_path))[0]
            pairname = f"{basename}_seq{frame_idx}"

        return moving_t, fixed_t, segx, segy, pairname


class XCATOriginalRegistration(XCATSeqRegistration):
    """XCAT 原始配准模式：只用原始 moving 配准到 fixed（不含 8 帧运动形变序列）。

    与 XCATSeqRegistration 使用相同的数据来源、预处理和划分逻辑，
    唯一区别：训练/验证/测试都只用 original moving，不展开 8 帧形变序列。
    """
    def __init__(self, data_root, split='train', flip_p=0.5, split_file=None):
        self.data_root = data_root
        self.split = split
        self.flip_p = flip_p if split == 'train' else 0.0

        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_dir = os.path.join(data_root, 'moving', 'moving')
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.npy')))

        assert len(self.fixed_paths) == len(self.moving_paths), \
            f"Fixed ({len(self.fixed_paths)}) and Moving ({len(self.moving_paths)}) file count mismatch"

        # 划分数据集（与 XCATSeqRegistration 完全一致）
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
            print(f"[XCATOriginalRegistration] split={split}, loaded {len(self.pair_indices)} pairs from split_indices.json")
        else:
            if split == 'train':
                self.pair_indices = list(range(0, int(n * 0.7)))
            elif split == 'val':
                self.pair_indices = list(range(int(n * 0.7), int(n * 0.85)))
            else:
                self.pair_indices = list(range(int(n * 0.85), n))
            print(f"[XCATOriginalRegistration] split={split}, loaded {len(self.pair_indices)} pairs by ratio")

        # 构建样本池：所有 split 都只用 original（不展开 8 帧序列）
        self.samples = []
        for pair_idx in self.pair_indices:
            self.samples.append((
                self.fixed_paths[pair_idx],
                self.moving_paths[pair_idx],
                'original',
                None
            ))
        self._length = len(self.samples)
        print(f"[XCATOriginalRegistration] split={split}, pairs={len(self.pair_indices)}, "
              f"variants_per_pair=1 (original only, NO seq), total_samples={self._length}")

