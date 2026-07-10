"""
Multi-phase dataset for MotionFiLM multi-phase training.

把"同一个 block/slice 的 phase01..phase09" 聚合成单个 sample:
    fixed         : [1, H, W]      (phase 0 的 fixed 模板)
    moving_seq    : [9, 1, H, W]   (phase 1..9 9 个 moving 帧)
    phase_ids     : [9]            (0..8, 内部索引)
    pairname      : str

数据组织约定 (与 xcat_npz.py 完全一致):
    fixed_dir     = <data_root>/fixed/fixed
    moving_dir    = <data_root>/moving/moving
    raw_idx       = block_id * 171 + phase_id * 19 + slice_id     # phase_id=0..8
    文件命名      = {idx:03d}.npy

Split 划分 (与 xcat_npz.py 完全一致):
    按 base_sample (= block_id * 19 + slice_id, 共 114 个) 70/15/15 划分;
    同一个 base_sample 的 9 个 phase 必然属于同一个 split,
    避免信息泄露。
"""
import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_PHASES = 9              # phase 1..9 = moving
SLICES_PER_PHASE = 19
BLOCK_SIZE = NUM_PHASES * SLICES_PER_PHASE   # 171


class MultiPhaseDataset(Dataset):
    """
    Returns:
        fixed_t       : torch.Tensor [1, H, W]      (float32, [0,1] after normalize)
        moving_seq_t  : torch.Tensor [9, 1, H, W]   (float32, [0,1] after normalize)
        phase_ids     : torch.Tensor [9]            (long, 0..8)
        pairname      : str                        (block{block}_slice{slice})
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        flip_p: float = 0.5,
        normalize: bool = True,
        split_file: Optional[str] = None,
        num_blocks: int = 6,
        total_files: int = 1026,
    ):
        self.data_root = data_root.rstrip("/")
        self.split = split
        self.flip_p = flip_p if split == "train" else 0.0
        self.normalize = normalize

        self.fixed_dir = os.path.join(self.data_root, "fixed", "fixed")
        self.moving_dir = os.path.join(self.data_root, "moving", "moving")

        # base-sample split (与 XCATNPZRegistration 保持完全一致)
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

        self.num_blocks = nb
        self.total_files = total_f
        self.num_phases = NUM_PHASES

        # 每个 block 在 base-sample 索引中占 19 个位置
        self.base_samples: List[tuple] = []
        for b in range(nb):
            for s in range(SLICES_PER_PHASE):
                raw_idx = b * BLOCK_SIZE + s
                if raw_idx < total_f:
                    self.base_samples.append((b, s))
                else:
                    break

        # 只保留属于当前 split 的 base_samples
        self.base_samples = [
            bs for bs in self.base_samples
            if self._base_idx(bs) in self.base_range
        ]
        self.num_base = len(self.base_samples)

        print(
            f"[MultiPhaseDataset] split={split}  base_samples={self.num_base}  "
            f"phases_per_base={self.num_phases}  normalize={normalize}  flip_p={self.flip_p}"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _base_idx(self, bs: tuple) -> int:
        b, s = bs
        return b * SLICES_PER_PHASE + s

    def _raw_index(self, block_id: int, phase_id: int, slice_id: int) -> int:
        """(block_id, phase_id, slice_id) → raw file index
           phase_id = 0..8 (内部索引, moving phase 1..9)
        """
        return block_id * BLOCK_SIZE + phase_id * SLICES_PER_PHASE + slice_id

    def _load_npy(self, idx: int, parent_dir: str) -> np.ndarray:
        return np.load(os.path.join(parent_dir, f"{idx:03d}.npy")).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset 协议
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_base

    def __getitem__(self, idx: int):
        block_id, slice_id = self.base_samples[idx]

        # 一次性加载 9 个 moving + 1 个 fixed (相邻帧读盘有系统缓存, 实际开销很小)
        moving_frames = np.zeros((self.num_phases, *self._load_npy(
            self._raw_index(block_id, 0, slice_id), self.moving_dir).shape),
            dtype=np.float32)
        for p in range(self.num_phases):
            moving_frames[p] = self._load_npy(
                self._raw_index(block_id, p, slice_id), self.moving_dir)
        fixed_np = self._load_npy(
            self._raw_index(block_id, 0, slice_id), self.fixed_dir)

        # Min-Max 归一化 (用 fixed 的 min/max 同时缩 fixed + 9 个 moving)
        if self.normalize:
            minv = fixed_np.min()
            maxv = fixed_np.max()
            if maxv - minv > 1e-6:
                fixed_np = (fixed_np - minv) / (maxv - minv)
                moving_frames = (moving_frames - minv) / (maxv - minv)

        # 同步随机水平翻转 (fixed + 9 moving 同步)
        if self.flip_p > 0 and np.random.rand() < self.flip_p:
            fixed_np = np.fliplr(fixed_np).copy()
            moving_frames = moving_frames[:, :, ::-1].copy()

        fixed_t = torch.from_numpy(fixed_np).float().unsqueeze(0)               # [1, H, W]
        moving_seq_t = torch.from_numpy(moving_frames).float().unsqueeze(1)     # [9, 1, H, W]
        phase_ids = torch.arange(self.num_phases, dtype=torch.long)            # [9]  (0..8)

        pairname = f"block{block_id}_slice{slice_id:02d}"

        return {
            "fixed":       fixed_t,
            "moving_seq":  moving_seq_t,
            "phase_ids":   phase_ids,
            "pairname":    pairname,
            "block_id":    block_id,
            "slice_id":    slice_id,
        }


def collate_multiphase(batch: List[dict]):
    """
    collate_fn for MultiPhaseDataset.

    Returns:
        fixed_t       : torch.Tensor [B, 1, H, W]
        moving_seq_t  : torch.Tensor [B, 9, 1, H, W]
        phase_ids     : torch.Tensor [B, 9]
        pairnames     : list[str]
        block_ids     : torch.Tensor [B]
        slice_ids     : torch.Tensor [B]
    """
    B = len(batch)
    fixed_t = torch.stack([b["fixed"] for b in batch], dim=0)
    moving_seq_t = torch.stack([b["moving_seq"] for b in batch], dim=0)
    phase_ids = torch.stack([b["phase_ids"] for b in batch], dim=0)
    pairnames = [b["pairname"] for b in batch]
    block_ids = torch.tensor([b["block_id"] for b in batch], dtype=torch.long)
    slice_ids = torch.tensor([b["slice_id"] for b in batch], dtype=torch.long)
    return fixed_t, moving_seq_t, phase_ids, pairnames, block_ids, slice_ids