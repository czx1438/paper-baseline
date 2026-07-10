import os
import glob
import json
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Raw NPY registration dataset — 9-phase (phase 0 = fixed, phase 1..9 = moving)
# ---------------------------------------------------------------------------


def _build_base_samples(
    num_blocks: int,
    slices_per_phase: int,
    num_phases: int,
    total_files: int,
) -> List[int]:
    """Return the list of raw file indices that correspond to phase 0 (fixed).

    The raw index scheme is:
        raw_idx = block_id * (num_phases * slices_per_phase) + phase_id * slices_per_phase + slice_id

    For phase 0, this simplifies to: raw_idx = block_id * block_size + slice_id
    """
    block_size = num_phases * slices_per_phase  # 171
    indices = []
    for b in range(num_blocks):
        for s in range(slices_per_phase):
            idx = b * block_size + s
            if idx < total_files:
                indices.append(idx)
    return indices


class XCATNPZRegistration(Dataset):
    """
    XCAT 多相位配准数据集（用于配准网络训练）。

    数据结构：
        - fixed 目录 : xcat_data/fixed/fixed/{000..N}.npy   ← phase 0（固定模板）
        - moving 目录: xcat_data/moving/moving/{000..N}.npy ← phase 1..9（演化相位）

    相位组织规则（用户定义）：
        num_phases       = 9          (phase 1..9 = moving)
        slices_per_phase = 19         (每个相位 19 个切片)
        block_size       = 171        (num_phases × slices_per_phase)
        num_blocks       = n_files / block_size ≈ 6
        total base_samples = num_blocks × slices_per_phase = 114

    每个 base_sample = (block_id, slice_id)，对应 9 个 registration 样本：
        moving_idx = block_id * block_size + phase_id * slices_per_phase + slice_id
        fixed_idx  = block_id * block_size + 0 * slices_per_phase + slice_id
        (phase_id = 0..8  →  moving phase 1..9)

    Split 划分（按 base_sample，70/15/15）：
        total  = 114
        train  = 0  .. 79   (80 base_samples  →  720 actual samples)
        val    = 80 .. 96    (17 base_samples  →  153 actual samples)
        test   = 97 .. 113   (17 base_samples  →  153 actual samples)

    同一 base_sample 的 9 个 phase 样本一定属于同一 split。

    返回格式（字典，与 train_mask.py 的 collate_fn 配合）：
        {
            "moving":     moving_t,        # torch.Tensor [1, H, W]
            "fixed":      fixed_t,         # torch.Tensor [1, H, W]
            "phase":      phase,           # int 1..9   (与原始数据 moving 文件名一致)
            "phase_id":   phase_id,        # int 0..8   (内部 0-based, 便于 embedding lookup)
            "moving_idx": moving_idx,      # int 原始文件索引
            "fixed_idx":  fixed_idx,       # int 原始文件索引
            "pairname":   pairname,        # str  "block{block}_slice{slice}_phase{phase}"
        }

    重要约定（与原始数据命名一致）：
        - 相位号 (phase) = 1..9 出现于 moving 文件夹，对应原始索引 0..152
        - phase 0    出现于 fixed 文件夹（fixed/000.npy = "phase 0"），用作 fixed 模板
        - 因此每个 base_sample 的 9 个 moving 样本的 phase 字段就是 1..9：
            phase=1 的 moving_idx = block*171 + 0*19 + slice   (即 moving/000.npy 起那一组)
            phase=9 的 moving_idx = block*171 + 8*19 + slice   (即 moving/152.npy 起那一组)
        - 内部 phase_id (0..8) = phase - 1，仅供模型做 embedding lookup

    为保持与旧代码的兼容性，Dataset 实现了 __iter__ 协议，
    train_mask.py 通过 collate_fn 将 dict 展开为 tuple。
    """

    NUM_PHASES: int = 9          # moving phases = 1..9
    SLICES_PER_PHASE: int = 19   # frames per phase
    BLOCK_SIZE: int = NUM_PHASES * SLICES_PER_PHASE   # 171

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        flip_p: float = 0.5,
        normalize: bool = True,
        split_file: Optional[str] = None,
        # 下面两个参数由 split_file 自动计算，一般不需要显式传入
        num_blocks: int = 6,
        total_files: int = 1026,
    ):
        self.data_root = data_root.rstrip("/")
        self.split = split
        self.flip_p = flip_p if split == "train" else 0.0
        self.normalize = normalize

        # ------------------------------------------------------------------
        # 路径
        # ------------------------------------------------------------------
        self.fixed_dir  = os.path.join(self.data_root, "fixed",  "fixed")
        self.moving_dir = os.path.join(self.data_root, "moving", "moving")

        # ------------------------------------------------------------------
        # Base-sample split（按 block_id × slice_id 划分，不是按 raw file index）
        # ------------------------------------------------------------------
        if split_file is None:
            split_file = os.path.join(self.data_root, "registration_multi_phase_split.json")

        if os.path.exists(split_file):
            with open(split_file) as f:
                cfg = json.load(f)
            info = cfg.get("registration_multi_phase", {})
            split_conf = info.get("split", {})
            train_range = split_conf.get("train", [0, 79])
            val_range   = split_conf.get("val",   [80, 96])
            test_range  = split_conf.get("test",  [97, 113])
            nb = info.get("num_blocks", num_blocks)
            total_f = info.get("total_files", total_files)
        else:
            # 默认 70/15/15 划分：114 base_samples
            train_range = [0, 79]
            val_range   = [80, 96]
            test_range  = [97, 113]
            nb = num_blocks
            total_f = total_files

        if split == "train":
            self.base_range = range(train_range[0], train_range[1] + 1)
        elif split == "val":
            self.base_range = range(val_range[0], val_range[1] + 1)
        else:
            self.base_range = range(test_range[0], test_range[1] + 1)

        # ------------------------------------------------------------------
        # 构建 base_samples: list of (block_id, slice_id)
        # ------------------------------------------------------------------
        self.num_blocks      = nb
        self.total_files     = total_f
        self._block_size     = self.NUM_PHASES * self.SLICES_PER_PHASE   # 171

        # 每个 block 在 base-sample 索引中占 slices_per_phase 个位置
        self.base_samples: List[tuple] = []
        for b in range(nb):
            base_offset = b * self.SLICES_PER_PHASE
            for s in range(self.SLICES_PER_PHASE):
                raw_idx = b * self._block_size + s
                if raw_idx < total_f:
                    self.base_samples.append((b, s))
                else:
                    break

        # 只保留属于当前 split 的 base_samples
        self.base_samples = [
            bs for bs in self.base_samples
            if self._base_idx(bs) in self.base_range
        ]

        self.num_base  = len(self.base_samples)
        self.num_phases = self.NUM_PHASES            # 9
        self._length  = self.num_base * self.num_phases  # 720 / 153 / 153

        print(
            f"[XCATNPZRegistration] split={split}  "
            f"base_samples={self.num_base}  "
            f"total_samples={self._length}  "
            f"(phases_per_base={self.num_phases})"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _base_idx(self, bs: tuple) -> int:
        """(block_id, slice_id) → base-sample index in [0, 114)"""
        b, s = bs
        return b * self.SLICES_PER_PHASE + s

    def _raw_index(self, block_id: int, phase_id: int, slice_id: int) -> int:
        """(block_id, phase_id, slice_id) → raw file index"""
        return block_id * self._block_size + phase_id * self.SLICES_PER_PHASE + slice_id

    def _load_npy(self, idx: int) -> np.ndarray:
        raise NotImplementedError(
            "Override _load_npy in a subclass or set self.fixed_dir / self.moving_dir "
            "to point to the raw .npy directories."
        )

    # ------------------------------------------------------------------
    # Dataset 协议
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        # 解码 idx → (base_idx, phase_id)
        base_idx = idx // self.num_phases
        phase_id = idx % self.num_phases          # 0..8  (内部 0-based)
        phase    = phase_id + 1                    # 1..9  (与 moving 文件名一致)

        block_id, slice_id = self.base_samples[base_idx]

        # moving: 从第 phase_id 个 block 区段里取 (偏移 = phase_id * 19)
        # fixed:  永远是 block_id 区段里的 phase_id=0 (即 fixed/000 起那一组)
        moving_idx = self._raw_index(block_id, phase_id, slice_id)
        fixed_idx  = self._raw_index(block_id, 0,         slice_id)

        # 加载图像
        moving_np = self._load_npy(moving_idx, self.moving_dir)
        fixed_np  = self._load_npy(fixed_idx,  self.fixed_dir)

        # Min-Max 归一化（fixed 和 moving 使用 fixed 的 min/max）
        if self.normalize:
            minv = fixed_np.min()
            maxv = fixed_np.max()
            if maxv - minv > 1e-6:
                fixed_np  = (fixed_np  - minv) / (maxv - minv)
                moving_np = (moving_np - minv) / (maxv - minv)

        # 随机水平翻转（fixed 和 moving 同步）
        if self.flip_p > 0 and np.random.rand() < self.flip_p:
            fixed_np  = np.fliplr(fixed_np).copy()
            moving_np = np.fliplr(moving_np).copy()

        # → torch.Tensor [1, H, W]
        moving_t = torch.from_numpy(moving_np).float().unsqueeze(0)
        fixed_t  = torch.from_numpy(fixed_np).float().unsqueeze(0)

        pairname = f"block{block_id}_slice{slice_id:02d}_phase{phase:02d}"

        return {
            "moving":     moving_t,
            "fixed":      fixed_t,
            "phase":      phase,          # 1..9  (与 moving/ 文件名一致)
            "phase_id":   phase_id,       # 0..8  (内部索引, 用于 embedding)
            "moving_idx": moving_idx,
            "fixed_idx":  fixed_idx,
            "pairname":   pairname,
        }


class XCATNPZRegistrationFromNPY(XCATNPZRegistration):
    """
    直接从原始 .npy 文件加载（无需预处理生成 npz）。
    用于 xcat_data/fixed/fixed/*.npy 和 xcat_data/moving/moving/*.npy。

    文件命名约定：{idx:03d}.npy  （如 000.npy, 019.npy, 038.npy ...）
    """

    def _load_npy(self, idx: int, parent_dir: str) -> np.ndarray:
        path = os.path.join(parent_dir, f"{idx:03d}.npy")
        return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# 兼容性包装：把 dict 输出展开成旧代码期望的 tuple
# (moving_t, fixed_t, segx, segy, pairname, phase, moving_idx, fixed_idx)
# ---------------------------------------------------------------------------

def collate_registration_dicts(batch: List[dict]):
    """
    train_mask.py DataLoader 的 collate_fn。
    把 [{moving, fixed, phase, phase_id, ...}, ...] 展开为:
        (moving_t, fixed_t, segx, segy, pairname,
         phase, phase_id, moving_idx, fixed_idx)
    其中：
        phase    = 1..9  (与原始 moving/ 文件名一致)
        phase_id = 0..8  (内部 0-based, 用于 embedding)
        segx/segy = 零张量（占位），与旧的 npz Dataset 返回格式对齐。
    """
    B = len(batch)
    H = batch[0]["moving"].shape[1]
    W = batch[0]["moving"].shape[2]

    segx = torch.zeros(B, 1, 1, H, W)
    segy = torch.zeros(B, 1, 1, H, W)

    moving_batch  = torch.stack([b["moving"]  for b in batch], dim=0)
    fixed_batch   = torch.stack([b["fixed"]   for b in batch], dim=0)
    phase_batch   = torch.tensor([b["phase"]    for b in batch], dtype=torch.long)   # 1..9
    phase_id_batch = torch.tensor([b["phase_id"] for b in batch], dtype=torch.long)  # 0..8
    moving_idx    = torch.tensor([b["moving_idx"] for b in batch], dtype=torch.long)
    fixed_idx     = torch.tensor([b["fixed_idx"]  for b in batch], dtype=torch.long)
    pairnames     = [b["pairname"] for b in batch]

    return (
        moving_batch, fixed_batch,
        segx, segy,
        pairnames,
        phase_batch, phase_id_batch,
        moving_idx, fixed_idx,
    )


# ---------------------------------------------------------------------------
# 旧版 XCATNPZRegistration（基于预生成的 *_pair.npz 文件）
# 保留以兼容已有工作流，不推荐用于新的多相位训练。
# ---------------------------------------------------------------------------

class XCATNPZRegistrationLegacy(Dataset):
    """
    基于预生成 *_pair.npz 文件的配准数据集（已废弃，推荐使用 XCATNPZRegistrationFromNPY）。

    npz 文件格式：
        img_small : moving image (H, W)
        img_large : fixed  image (H, W)
    或
        moving : ...
        fixed  : ...
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 flip_p: float = 0.5,
                 split_file: Optional[str] = None,
                 normalize: bool = True):
        self.data_root = data_root.rstrip('/')
        self.split = split
        self.flip_p = flip_p if split == 'train' else 0.0
        self.normalize = normalize

        split_dir = os.path.join(self.data_root, split)
        if os.path.isdir(split_dir):
            all_paths = sorted(glob.glob(os.path.join(split_dir, '*_pair.npz')))
            self.npz_paths = all_paths
            self.auto_split = False
        else:
            all_paths = sorted(glob.glob(os.path.join(self.data_root, '*_pair.npz')))
            n = len(all_paths)
            if split_file is None:
                split_file = os.path.join(self.data_root, 'split_indices.json')
            if os.path.exists(split_file):
                with open(split_file) as f:
                    cfg = json.load(f)
                info = cfg.get('registration', cfg.get('vq_ldm', {}))
                split_conf = info.get('split', {})
                idx_range = split_conf.get(split, [0, n - 1])
                self.npz_paths = all_paths[idx_range[0]: idx_range[1] + 1]
                self.auto_split = False
            else:
                if split == 'train':
                    self.npz_paths = all_paths[:int(n * 0.7)]
                elif split == 'val':
                    self.npz_paths = all_paths[int(n * 0.7):int(n * 0.85)]
                else:
                    self.npz_paths = all_paths[int(n * 0.85):]
                self.auto_split = True

        if not self.npz_paths:
            raise FileNotFoundError(
                f"No *_pair.npz files found in {self.data_root}/{split}/."
            )

        print(f"[XCATNPZRegistrationLegacy] split={split}, npz_count={len(self.npz_paths)}, "
              f"auto_split={self.auto_split}")

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        npz_path = self.npz_paths[idx]
        data = np.load(npz_path)

        if 'img_small' in data and 'img_large' in data:
            moving = data['img_small'].astype(np.float32)
            fixed  = data['img_large'].astype(np.float32)
        elif 'moving' in data and 'fixed' in data:
            moving = data['moving'].astype(np.float32)
            fixed  = data['fixed'].astype(np.float32)
        else:
            keys = sorted(data.files)
            raise KeyError(
                f"Unexpected npz keys {keys} in {npz_path}. "
                "Expected ('img_small','img_large') or ('moving','fixed')."
            )

        if self.normalize:
            minv = fixed.min(); maxv = fixed.max()
            if maxv - minv > 1e-6:
                fixed  = (fixed  - minv) / (maxv - minv)
                moving = (moving - minv) / (maxv - minv)

        if np.random.rand() < self.flip_p:
            fixed  = np.fliplr(fixed).copy()
            moving = np.fliplr(moving).copy()

        moving_t = torch.from_numpy(moving).float().unsqueeze(0)
        fixed_t  = torch.from_numpy(fixed).float().unsqueeze(0)
        H, W = fixed_t.shape[1], fixed_t.shape[2]
        segx = torch.zeros(1, 1, H, W)
        segy = torch.zeros(1, 1, H, W)
        basename = os.path.splitext(os.path.basename(npz_path))[0]

        return moving_t, fixed_t, segx, segy, basename


# ---------------------------------------------------------------------------
# VQ 训练数据集：按 block 划分（block0,3,5 = train | block2 = val | block4 = test）
# VQ 训练只需单张图像输入，与 xcat_npz.py 的 XCATNPZRegistration 共享索引映射。
# ---------------------------------------------------------------------------

class XCATNPZVQ(Dataset):
    """
    每个 block：
        fixed：1 phase × 19 slices
        moving：9 phases × 19 slices

    VQ 中每张图像作为独立样本：
        每个 block = 19 + 9×19 = 190 张
    """

    NUM_BLOCKS = 6
    NUM_MOVING_PHASES = 9
    SLICES_PER_PHASE = 19

    MOVING_BLOCK_SIZE = NUM_MOVING_PHASES * SLICES_PER_PHASE  # 171

    SPLIT_BLOCKS = {
        "train": [0, 1, 3, 5],
        "val": [2],
        "test": [4],
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        flip_p: float = 0.5,
        size=None,
    ):
        self.split = split
        self.data_root = data_root.rstrip("/")
        self.flip_p = flip_p if split == "train" else 0.0

        self.fixed_dir = os.path.join(
            self.data_root, "fixed", "fixed"
        )
        self.moving_dir = os.path.join(
            self.data_root, "moving", "moving"
        )

        # 不再只保存一个容易混淆的 raw_idx，
        # 直接保存图像类型、block、phase、slice。
        self.samples = []

        for block_id in self.SPLIT_BLOCKS[split]:
            # 每个 block 的 fixed 只加入一次
            for slice_id in range(self.SLICES_PER_PHASE):
                self.samples.append(
                    ("fixed", block_id, 0, slice_id)
                )

            # 加入9个 moving phase
            for moving_phase_id in range(self.NUM_MOVING_PHASES):
                for slice_id in range(self.SLICES_PER_PHASE):
                    self.samples.append(
                        (
                            "moving",
                            block_id,
                            moving_phase_id,
                            slice_id,
                        )
                    )

        if split == "train":
            import random
            rng = random.Random(42)
            rng.shuffle(self.samples)

        print(
            f"[XCATNPZVQ] split={split}, "
            f"blocks={self.SPLIT_BLOCKS[split]}, "
            f"total_images={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_type, block_id, phase_id, slice_id = self.samples[idx]

        if image_type == "fixed":
            # 如果 fixed 每个 block 只有19张
            fixed_idx = (
                block_id * self.SLICES_PER_PHASE
                + slice_id
            )

            npy_path = os.path.join(
                self.fixed_dir,
                f"{fixed_idx:03d}.npy",
            )

        else:
            moving_idx = (
                block_id * self.MOVING_BLOCK_SIZE
                + phase_id * self.SLICES_PER_PHASE
                + slice_id
            )

            npy_path = os.path.join(
                self.moving_dir,
                f"{moving_idx:03d}.npy",
            )

        image = np.load(npy_path).astype(np.float32)

        minv = image.min()
        maxv = image.max()

        if maxv - minv > 1e-6:
            image = (image - minv) / (maxv - minv)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        if self.flip_p > 0 and np.random.rand() < self.flip_p:
            image = np.fliplr(image).copy()

        # 原版LDM通常使用HWC格式
        image = image[..., None]  # [H, W, 1]

        return {"image": image}


class XCATNPZVQTrain(XCATNPZVQ):
    def __init__(self, data_root: str = "./xcat_data", **kwargs):
        super().__init__(data_root=data_root, split="train", **kwargs)


class XCATNPZVQValidation(XCATNPZVQ):
    def __init__(self, data_root: str = "./xcat_data", **kwargs):
        super().__init__(data_root=data_root, split="val", **kwargs)


class XCATNPZVQTest(XCATNPZVQ):
    def __init__(self, data_root: str = "./xcat_data", **kwargs):
        super().__init__(data_root=data_root, split="test", **kwargs)


# ---------------------------------------------------------------------------
# 别名：旧名仍可用，但打印警告提示迁移
# ---------------------------------------------------------------------------

class XCATNPZRegistration(XCATNPZRegistrationFromNPY):
    """
    别名 → XCATNPZRegistrationFromNPY。
    直接从原始 .npy 读取，支持 9 相位 base-sample split。
    """
    pass
