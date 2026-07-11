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

Split 划分 (block_id 硬切, 避免信息泄露):
    train_blocks = {0, 1, 3, 5}
    val_blocks   = {2}
    test_blocks  = {4}

    同一个 block 内所有 19 个 slice 同时进入同一个 split;
    注册用 fixed/moving 完全独立于 block 边界，0 信息泄露。
    若想恢复旧的 base-sample 70/15/15 划分：传入 split_file=.../legacy_split.json。
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_PHASES = 9              # phase 1..9 = moving
SLICES_PER_PHASE = 19
BLOCK_SIZE = NUM_PHASES * SLICES_PER_PHASE   # 171

# ---- 按 block_id 硬切的默认 split（你这次训练的新划分） ----
DEFAULT_BLOCK_SPLIT: Dict[str, set] = {
    "train": {0, 1, 3, 5},
    "val":   {2},
    "test":  {4},
}


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
        block_split: Optional[Dict[str, set]] = None,
        num_blocks: int = 6,
        total_files: int = 1026,
    ):
        self.data_root = data_root.rstrip("/")
        self.split = split
        self.flip_p = flip_p if split == "train" else 0.0
        self.normalize = normalize

        self.fixed_dir = os.path.join(self.data_root, "fixed", "fixed")
        self.moving_dir = os.path.join(self.data_root, "moving", "moving")

        self.num_blocks = num_blocks
        self.total_files = total_files
        self.num_phases = NUM_PHASES

        # ------------------------------------------------------------------
        # 1. 解析 split 划分
        #    优先级: 用户显式传入的 block_split > JSON split_file > 默认硬切
        # ------------------------------------------------------------------
        if block_split is not None:
            self.block_split = {k: set(int(x) for x in v)
                                for k, v in block_split.items()}
            self.split_source = "explicit-kwarg"
        else:
            split_file = split_file or os.path.join(
                self.data_root, "registration_multi_phase_split.json"
            )
            if os.path.exists(split_file):
                with open(split_file) as f:
                    cfg = json.load(f)
                info = cfg.get("registration_multi_phase", {})
                split_conf = info.get("split", {})
                self.num_blocks = info.get("num_blocks", num_blocks)
                self.total_files = info.get("total_files", total_files)
                # 仅当 JSON 中存在 block-keyed 字段才用 block 切分，
                # 否则 fallback 到基于 base-range 的旧逻辑并转 block set。
                if split_conf and "blocks" in split_conf:
                    bmap = split_conf["blocks"]
                    self.block_split = {
                        k: set(int(x) for x in v)
                        for k, v in bmap.items() if v
                    }
                    self.split_source = f"json:blocks {split_file}"
                else:
                    self.block_split = self._legacy_range_to_blocks(
                        train_range=split_conf.get("train", [0, 79]),
                        val_range=split_conf.get("val",   [80, 96]),
                        test_range=split_conf.get("test",  [97, 113]),
                    )
                    self.split_source = f"json:legacy-ranges {split_file}"
            else:
                self.block_split = {k: set(v) for k, v in DEFAULT_BLOCK_SPLIT.items()}
                self.split_source = "DEFAULT_BLOCK_SPLIT (hardcoded)"

        if split not in self.block_split:
            raise ValueError(
                f"Unknown split '{split}'. Available: {list(self.block_split)}"
            )
        self.blocks: set = self.block_split[split]

        # ------------------------------------------------------------------
        # 2. 构建 base_samples：枚举所有 (block, slice)，
        #    只保留属于当前 split 的 block 的样本
        # ------------------------------------------------------------------
        all_samples: List[tuple] = []
        for b in range(self.num_blocks):
            for s in range(SLICES_PER_PHASE):
                raw_idx = b * BLOCK_SIZE + s
                if raw_idx < self.total_files:
                    all_samples.append((b, s))
                else:
                    break

        self.base_samples: List[tuple] = [
            (b, s) for (b, s) in all_samples if b in self.blocks
        ]
        self.num_base = len(self.base_samples)

        # blocks 内样本为 0 时立刻报错（避免悄悄训空集）
        if self.num_base == 0:
            raise RuntimeError(
                f"[MultiPhaseDataset] split='{split}' 在 {self.blocks} 内没有样本。\n"
                f"  data_root={self.data_root}\n"
                f"  num_blocks={self.num_blocks}, total_files={self.total_files}\n"
                f"  split_source={self.split_source}"
            )

        blocks_str = "{" + ",".join(str(b) for b in sorted(self.blocks)) + "}"
        print(
            f"[MultiPhaseDataset] split={split:<5}  base_samples={self.num_base}  "
            f"phases_per_base={self.num_phases}  blocks={blocks_str}  "
            f"normalize={normalize}  flip_p={self.flip_p}  "
            f"split_src={self.split_source}"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _legacy_range_to_blocks(
        self,
        train_range=(0, 79),
        val_range=(80, 96),
        test_range=(97, 113),
    ) -> Dict[str, set]:
        """把旧的 base-sample [lo, hi] 区间翻译成 block set，向后兼容旧 JSON。"""
        def to_block_set(rng):
            lo, hi = rng
            return {b for b in range(self.num_blocks)
                    if any(self._base_idx_global((b, s)) in range(lo, hi + 1)
                           for s in range(SLICES_PER_PHASE))}
        return {
            "train": to_block_set(train_range),
            "val":   to_block_set(val_range),
            "test":  to_block_set(test_range),
        }

    @staticmethod
    def _base_idx_global(bs: tuple) -> int:
        b, s = bs
        return b * SLICES_PER_PHASE + s

    def _base_idx(self, bs: tuple) -> int:
        return self._base_idx_global(bs)

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

class PairwisePhaseDataset(MultiPhaseDataset):
    def __len__(self):
        return self.num_base * self.num_phases

    def __getitem__(self, idx):
        base_idx, phase_idx = divmod(idx, self.num_phases)
        block_id, slice_id = self.base_samples[base_idx]

        moving_np = self._load_npy(
            self._raw_index(block_id, phase_idx, slice_id),
            self.moving_dir,
        )
        fixed_np = self._load_npy(
            self._raw_index(block_id, 0, slice_id),
            self.fixed_dir,
        )

        if self.normalize:
            minv, maxv = fixed_np.min(), fixed_np.max()
            if maxv - minv > 1e-6:
                fixed_np = (fixed_np - minv) / (maxv - minv)
                moving_np = (moving_np - minv) / (maxv - minv)

        if self.flip_p > 0 and np.random.rand() < self.flip_p:
            fixed_np = np.fliplr(fixed_np).copy()
            moving_np = np.fliplr(moving_np).copy()

        return {
            "fixed": torch.from_numpy(fixed_np).float().unsqueeze(0),
            "moving": torch.from_numpy(moving_np).float().unsqueeze(0),
            "phase_id": torch.tensor(phase_idx, dtype=torch.long),
            "pairname": f"block{block_id}_slice{slice_id:02d}_phase{phase_idx+1}",
            "block_id": block_id,
            "slice_id": slice_id,
        }
        
def collate_pairwise(batch):
    fixed = torch.stack([x["fixed"] for x in batch])
    moving = torch.stack([x["moving"] for x in batch])
    phase_id = torch.stack([x["phase_id"] for x in batch])
    names = [x["pairname"] for x in batch]
    return fixed, moving, phase_id, names

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