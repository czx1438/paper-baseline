"""
XCAT 心脏序列数据集 - 解耦版（按被试成组）
==============================================================
改动点（相对于 xcat_Motion_Seq.py 的 XCATSeqBase）：
  - 不再把 fixed / moving / seq 打散混合，每个被试返回一组 9 张图（fixed + 8 seq frames）
  - __getitem__ 返回 dict（keys: images, phases, subject_id）
  - collate_fn 把同一 batch 内同一被试的多相位合并为一个 batch 维度
  - 假设：fixed/fixed/i.npy, moving_seq/i.npy 的 i 对应同一被试

⚠️ 假设：fixed_paths[i] 与 moving_seq_paths[i] 对应同一被试（心脏仿真数据）
⚠️ 假设：moving_seq/*.npy 形状为 (8, H, W)，每帧是一个相位
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from scipy.ndimage import zoom as np_zoom


class XCATSeqGroupedBase(Dataset):
    """
    按被试成组返回多相位的数据集。

    每个被试 i 提供：
        - 1 张 fixed 图（相位 0）
        - 1 个 moving_seq (8, H, W)，包含相位 1~8
    共 9 张图 / 被试。

    数据目录结构（与 XCATSeqBase 完全一致）：
        data_root/
            fixed/fixed/*.npy       # 固定图像 (H, W)
            moving/moving/*.npy     # 原始移动图像 phase1 (H, W)  [本类不直接使用]
            moving_seq/*.npy        # 8 帧形变序列 (8, H, W)
    """

    def __init__(self,
                 data_root,
                 split='train',
                 flip_p=0.5,
                 num_frames=9,
                 **kwargs):
        self.split = split
        self.data_root = data_root
        self.flip_p = flip_p if split == 'train' else 0.0
        # ⚠️ num_frames: 每个被试的相位数量（fixed + seq frames）
        # XCAT 心脏：1 fixed + 8 seq = 9
        self.num_frames = num_frames

        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_seq_dir = os.path.join(data_root, 'moving_seq')

        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.moving_seq_paths = sorted(glob.glob(os.path.join(moving_seq_dir, '*.npy')))

        n = len(self.fixed_paths)
        # ⚠️ 假设 fixed 与 moving_seq 一一对应，文件顺序即为被试顺序
        assert n == len(self.moving_seq_paths), \
            f"Fixed ({n}) and MovingSeq ({len(self.moving_seq_paths)}) count mismatch"

        # 70:15:15 划分
        if split == 'train':
            self.subject_indices = list(range(0, int(n * 0.7)))
        elif split == 'val':
            self.subject_indices = list(range(int(n * 0.7), int(n * 0.85)))
        else:
            self.subject_indices = list(range(int(n * 0.85), n))

        self._length = len(self.subject_indices)
        print(f"[XCATSeqGroupedBase] split={split}, "
              f"subjects={self._length}, frames_per_subject={self.num_frames}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        subject_idx = self.subject_indices[i]

        # 加载 fixed (H, W)
        fixed = np.load(self.fixed_paths[subject_idx]).astype(np.float32)

        # 加载 moving_seq (8, H, W)
        seq = np.load(self.moving_seq_paths[subject_idx]).astype(np.float32)
        # seq: (8, H, W) → 8 张独立帧

        # resize 到 256×256（降低 VQ 显存占用，只缩空间维度，不缩帧数）
        TARGET_SIZE = 256
        orig_H, orig_W = fixed.shape
        if (orig_H, orig_W) != (TARGET_SIZE, TARGET_SIZE):
            zf_2d = (TARGET_SIZE / orig_H, TARGET_SIZE / orig_W)
            zf_3d = (1.0, TARGET_SIZE / orig_H, TARGET_SIZE / orig_W)
            fixed = np_zoom(fixed, zf_2d, order=1)
            seq = np_zoom(seq, zf_3d, order=1)

        # 归一化：每个被试内用 fixed 的 min/max
        minv = fixed.min()
        maxv = fixed.max()
        if maxv - minv > 1e-6:
            fixed = (fixed - minv) / (maxv - minv)
            seq = (seq - minv) / (maxv - minv)

        # 随机水平翻转（fixed 和 seq 同时翻转，保持配对）
        if np.random.rand() < self.flip_p:
            fixed = np.fliplr(fixed).copy()
            seq = np.fliplr(seq).copy()

        # 组合：frame[0] = fixed, frame[1..8] = seq[0..7]
        images = [fixed] + [seq[f] for f in range(8)]
        # images: list of 9 × (H, W)

        # 相位标签：0=fixed, 1..8=seq frames
        phases = list(range(9))

        # subject_id 用于 collate 校验
        subject_id = str(subject_idx)

        return {
            "images": images,          # list of 9 × (H, W) numpy
            "phases": phases,         # list of 9 ints [0,1,2,...,8]
            "subject_id": subject_id,  # str
        }


class XCATSeqGroupedTrain(XCATSeqGroupedBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='train',
            flip_p=0.5,
            num_frames=9,
            **kwargs
        )


class XCATSeqGroupedValidation(XCATSeqGroupedBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='val',
            flip_p=0.0,
            num_frames=9,
            **kwargs
        )


class XCATSeqGroupedTest(XCATSeqGroupedBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data",
            split='test',
            flip_p=0.0,
            num_frames=9,
            **kwargs
        )


def xcat_seq_grouped_collate_fn(batch):
    """
    Collate function for XCATSeqGroupedBase.

    Input: list of dicts from __getitem__, each dict contains:
        - "images": list of num_frames × (H, W) numpy
        - "phases": list of num_frames ints
        - "subject_id": str

    Output: dict with keys:
        - "images": (B, num_frames, 1, H, W) torch tensor
        - "phases": (B, num_frames) torch long tensor
        - "subject_ids": list of str
    """
    B = len(batch)
    num_frames = len(batch[0]["images"])
    H, W = batch[0]["images"][0].shape

    # (B, num_frames, H, W)
    images = np.zeros((B, num_frames, H, W), dtype=np.float32)
    for i, item in enumerate(batch):
        for f in range(num_frames):
            images[i, f] = item["images"][f]

    images_t = torch.from_numpy(images).float().unsqueeze(2)  # (B, num_frames, 1, H, W)

    # phases: list of list → (B, num_frames)
    phases = torch.tensor([item["phases"] for item in batch], dtype=torch.long)

    subject_ids = [item["subject_id"] for item in batch]

    return {
        "images": images_t,
        "phases": phases,
        "subject_ids": subject_ids,
    }


def build_dataloader(split='train', batch_size=1, num_workers=4):
    """
    构建 XCATSeqGrouped DataLoader 的快捷函数。
    batch_size 即为每个 batch 的被试数（每被试 9 张图）。
    """
    if split == 'train':
        dataset = XCATSeqGroupedTrain()
    elif split == 'val':
        dataset = XCATSeqGroupedValidation()
    else:
        dataset = XCATSeqGroupedTest()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=xcat_seq_grouped_collate_fn,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    # 快速测试
    loader = build_dataloader('val', batch_size=2)
    batch = next(iter(loader))
    print("images shape:", batch["images"].shape)   # (B, 9, 1, H, W)
    print("phases shape:", batch["phases"].shape)   # (B, 9)
    print("phases[0]:", batch["phases"][0].tolist())
    print("subject_ids:", batch["subject_ids"])
