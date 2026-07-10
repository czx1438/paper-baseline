import os
import glob
import json
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class XCATNPZImageBase(Dataset):
    """
    XCAT NPZ 单图数据集（用于 VQ / LDM 训练）。

    数据目录结构：
        data_root/
            *_pair.npz

    npz 文件格式：
        img_small : moving image (H, W)
        img_large : fixed  image (H, W)

    每个 npz 展开为两张单图（fixed + moving），按 split_indices.json 的
    vq_ldm 划分索引切分 train/val/test。若不存在 split 文件，则自动按 70/15/15 划分。

    返回格式与 XCATSeqBase 一致：
        {"image": (1, H, W) tensor in [0,1]}
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 flip_p: float = 0.5,
                 split_file: Optional[str] = None):
        self.data_root = data_root.rstrip('/')
        self.split = split
        self.flip_p = flip_p if split == 'train' else 0.0

        # 收集全部 npz 文件
        all_paths = sorted(glob.glob(os.path.join(self.data_root, '*_pair.npz')))
        if not all_paths:
            raise FileNotFoundError(
                f"No *_pair.npz files found in {self.data_root}."
            )

        # 建立全局索引 -> (npz_path, image_type) 的映射
        # 每个 npz 贡献 2 张单图：fixed 在前，moving 在后
        self.index_map = []
        for npz_path in all_paths:
            self.index_map.append((npz_path, 'fixed'))
            self.index_map.append((npz_path, 'moving'))

        n_total = len(self.index_map)

        # 读取 split 文件（优先 vq_ldm，其次 registration，最后自动划分）
        if split_file is None:
            split_file = os.path.join(self.data_root, 'split_indices.json')

        if os.path.exists(split_file):
            with open(split_file) as f:
                cfg = json.load(f)
            info = cfg.get('vq_ldm', cfg.get('registration', {}))
            split_conf = info.get('split', {})
            idx_range = split_conf.get(split, [0, n_total - 1])
            self.image_indices = list(range(idx_range[0], idx_range[1] + 1))
            self.auto_split = False
        else:
            if split == 'train':
                self.image_indices = list(range(0, int(n_total * 0.7)))
            elif split == 'val':
                self.image_indices = list(range(int(n_total * 0.7), int(n_total * 0.85)))
            else:
                self.image_indices = list(range(int(n_total * 0.85), n_total))
            self.auto_split = True

        print(f"[XCATNPZImageBase] split={split}, "
              f"image_count={len(self.image_indices)}, "
              f"total={n_total}, auto_split={self.auto_split}")

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        global_idx = self.image_indices[idx]
        npz_path, img_type = self.index_map[global_idx]
        data = np.load(npz_path)

        if img_type == 'fixed':
            image = data['img_large'] if 'img_large' in data else data['fixed']
        else:
            image = data['img_small'] if 'img_small' in data else data['moving']

        image = image.astype(np.float32)

        # Min-Max 归一化到 [0, 1]
        minv = image.min()
        maxv = image.max()
        if maxv - minv > 1e-6:
            image = (image - minv) / (maxv - minv)

        # 随机水平翻转（非配对单图，独立翻转）
        if np.random.rand() < self.flip_p:
            image = np.fliplr(image).copy()

        image_t = torch.from_numpy(image).float().unsqueeze(0)
        return {"image": image_t}


class XCATNPZImageTrain(XCATNPZImageBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root=kwargs.get('data_root',
                                  '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep'),
            split='train',
            flip_p=0.5,
            **{k: v for k, v in kwargs.items() if k not in ('data_root', 'split', 'flip_p')}
        )


class XCATNPZImageValidation(XCATNPZImageBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root=kwargs.get('data_root',
                                  '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep'),
            split='val',
            flip_p=0.0,
            **{k: v for k, v in kwargs.items() if k not in ('data_root', 'split', 'flip_p')}
        )


class XCATNPZImageTest(XCATNPZImageBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root=kwargs.get('data_root',
                                  '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep'),
            split='test',
            flip_p=0.0,
            **{k: v for k, v in kwargs.items() if k not in ('data_root', 'split', 'flip_p')}
        )
