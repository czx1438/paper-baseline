import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted


class SEYRegistration(Dataset):
    """SEY 肝脏数据集，加载 fixed/moving 图像对."""
    def __init__(self, data_root, split='train', normalize=True):
        super().__init__()
        self.split = split
        self.normalize = normalize
        split_dir = os.path.join(data_root, split)
        self.files = natsorted(glob.glob(os.path.join(split_dir, '*.npz')))
        print(f"  [SEY {split}] loaded {len(self.files)} files from {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        arr = np.load(self.files[index])
        mov = arr["img_small"].astype(np.float32)
        fix = arr["img_large"].astype(np.float32)

        # 归一化：仅在数据未归一化时启用；SEY npz 来自 preprocess_sey.py（已 joint_norm）
        if self.normalize:
            minv, maxv = fix.min(), fix.max()
            if (maxv - minv) > 1e-6:
                fix = (fix - minv) / (maxv - minv)
                mov = (mov - minv) / (maxv - minv)

        # 随机水平翻转
        if self.split == 'train' and np.random.rand() > 0.5:
            fix = np.flip(fix, axis=-1).copy()
            mov = np.flip(mov, axis=-1).copy()

        # 返回格式：[1, H, W]
        mov = torch.from_numpy(mov).unsqueeze(0)
        fix = torch.from_numpy(fix).unsqueeze(0)
        movlab = torch.zeros_like(mov)
        tarlab = torch.zeros_like(fix)
        name = os.path.basename(self.files[index])
        return mov, fix, movlab, tarlab, name
