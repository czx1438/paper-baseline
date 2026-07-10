import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class XCATNPZSingleBase(Dataset):
    """
    XCAT NPZ 单图数据集（用于 VQ / LDM 训练）。

    与 xcat_Motion_Seq.py 的 XCATSeqBase 对齐：
    - 输入目录下读取所有 *_pair.npz
    - 每个 npz 展开为 2 个单图样本（moving + fixed），统一入池
    - 按 split_indices.json 的 registration 或 vq_ldm 字段做 train/val/test 划分
    - 返回格式为 {"image": tensor}，供 LDM 的 get_input / shared_step 直接使用
    """

    def __init__(self,
                 data_root,
                 split='train',
                 flip_p=0.5,
                 split_file=None):
        self.split = split
        self.data_root = data_root.rstrip('/')
        self.flip_p = flip_p if split == 'train' else 0.0

        all_paths = sorted(glob.glob(os.path.join(self.data_root, '*_pair.npz')))
        if not all_paths:
            raise FileNotFoundError(
                f"No *_pair.npz files found in {self.data_root}."
            )

        if split_file is None:
            split_file = os.path.join(self.data_root, 'split_indices.json')

        if os.path.exists(split_file):
            with open(split_file) as f:
                cfg = json.load(f)
            info = cfg.get('registration', cfg.get('vq_ldm', {}))
            split_conf = info.get('split', {})
            rng = split_conf.get(split)
            if rng is not None:
                start, end = rng
                if start < 0 or end >= len(all_paths) or start > end:
                    raise ValueError(
                        f"[XCATNPZSingleBase] split={split}, range {rng} out of bounds "
                        f"for {len(all_paths)} npz files."
                    )
                all_paths = all_paths[start:end + 1]
            else:
                raise KeyError(
                    f"[XCATNPZSingleBase] split={split} not found in {split_file}."
                )

        self.samples = []
        for npz_path in all_paths:
            data = np.load(npz_path)
            if 'img_small' in data and 'img_large' in data:
                self.samples.append((npz_path, 'moving'))
                self.samples.append((npz_path, 'fixed'))
            elif 'moving' in data and 'fixed' in data:
                self.samples.append((npz_path, 'moving'))
                self.samples.append((npz_path, 'fixed'))
            else:
                keys = sorted(data.files)
                raise KeyError(
                    f"Unexpected npz keys {keys} in {npz_path}. "
                    "Expected ('img_small','img_large') or ('moving','fixed')."
                )

        if split == 'train':
            import random
            random.seed(42)
            random.shuffle(self.samples)

        self._length = len(self.samples)
        self.flip = transforms.RandomHorizontalFlip(p=self.flip_p)

        print(f"[XCATNPZSingleBase] split={split}, npz_files={len(all_paths)}, "
              f"total_samples={self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        npz_path, role = self.samples[i]
        data = np.load(npz_path)

        if role == 'moving':
            image = data['img_small'] if 'img_small' in data else data['moving']
        else:
            image = data['img_large'] if 'img_large' in data else data['fixed']

        image = image.astype(np.float32)
        # minv = image.min()
        # maxv = image.max()
        # if maxv - minv > 1e-6:
        #     image = (image - minv) / (maxv - minv)

        image_t = torch.from_numpy(image)
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        image_t = self.flip(image_t)

        return {"image": image_t}


class XCATNPZSingleTrain(XCATNPZSingleBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='train',
            flip_p=0.5,
            **kwargs
        )


class XCATNPZSingleValidation(XCATNPZSingleBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='val',
            flip_p=0.0,
            **kwargs
        )


class XCATNPZSingleTest(XCATNPZSingleBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='test',
            flip_p=0.0,
            **kwargs
        )
