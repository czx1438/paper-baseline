
import os
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json

class CAMUSBase(Dataset):
    def __init__(self,
                 data_root,
                 isvalid=False,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.isvalid = isvalid
        self.data_root = data_root
        self.allimg_paths = glob.glob(os.path.join(data_root, '*.npy'))
        self.allimg_paths.sort()

        if not isvalid:
            self.image_paths = self.allimg_paths
        else:
            self.image_paths = self.allimg_paths[int(len(self.allimg_paths)*0.99):]

        self._length = len(self.image_paths)

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = np.load(self.image_paths[i])
        #image = self.flip(image)
        iterval = np.max(image) - np.min(image)
        minval = np.min(image)
        image = (image-minval) / iterval
        image = self.flip(torch.from_numpy(image))
        #example["image"] = ((image-np.min(image)) / iterval).astype(np.float32)
        example["image"] = image
        return example


class CAMUSTrain(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/training/image/saving/path", isvalid=False, **kwargs)


class CAMUSValidation(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/val/image/saving/path", isvalid=True, **kwargs)

class ECHOTrain(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/training/image/saving/path", isvalid=False, **kwargs)

class ECHOValidation(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/val/image/saving/path", isvalid=True, **kwargs)



class ACDCTrain(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/training/image/saving/path", isvalid=False, **kwargs)

class ACDCValidation(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/val/image/saving/path", isvalid=True, **kwargs)


class PETProjectionBase(Dataset):
    """
    PET投影数据基类

    适配VQ/Diffusion训练流程

    数据格式要求：
    - 预处理后的.npy文件，每个文件包含 [H, W] 的单通道图像
    - 文件名格式：{basename}_motion.npy, {basename}_clean.npy
    - 或直接是 {basename}.npy

    为什么继承CAMUSBase：
    - 保持与项目的数据加载接口一致
    - 便于替换配置文件即可切换数据源
    """

    def __init__(self,
                 data_root,
                 isvalid=False,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.0,  # PET数据通常不做翻转（角度信息重要）
                 normalize_mode='minmax',  # 预处理后已经是[0,1]，这里只做微调
                 use_paired_data=False,    # 是否使用配对数据训练
                 data_type='motion'):      # 'motion' 或 'clean' 或 'both'
        """
        Args:
            data_root: 预处理后的数据目录
            isvalid: 是否为验证集
            size: 数据尺寸（预处理时已统一，这里主要防错）
            interpolation: 插值方式
            flip_p: 翻转概率（PET通常不建议翻转）
            normalize_mode: 额外归一化模式
            use_paired_data: 是否返回配对数据
            data_type: 'motion' 或 'clean' 或 'both'
        """
        self.isvalid = isvalid
        self.data_root = data_root
        self.data_type = data_type
        self.use_paired_data = use_paired_data
        self.normalize_mode = normalize_mode

        # 加载所有.npy文件
        self.allimg_paths = glob.glob(os.path.join(data_root, '*.npy'))
        self.allimg_paths.sort()

        # 划分训练/验证集（99%训练，1%验证）
        if not isvalid:
            self.image_paths = self.allimg_paths
        else:
            self.image_paths = self.allimg_paths[int(len(self.allimg_paths) * 0.99):]

        self._length = len(self.image_paths)

        # 打印数据集信息
        print(f"PET数据集加载完成: {self._length} 样本")
        print(f"  数据类型: {self.data_type}")
        print(f"  训练/验证: {'验证集' if isvalid else '训练集'}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        """
        返回样本

        关键点：
        - VQ训练需要 'image' 键（见 autoencoder.py get_input方法）
        - 数据格式：[H, W] -> 自动添加channel维度 -> [1, H, W]
        """
        example = {}

        # 加载图像
        image = np.load(self.image_paths[i])

        # 确保是2D数组
        if image.ndim == 3:
            # 如果是[1, H, W]格式，去掉channel
            if image.shape[0] == 1:
                image = image[0]
            # 如果是[H, W, 1]格式
            elif image.shape[2] == 1:
                image = image[:, :, 0]

        # 额外归一化（虽然预处理已完成，这里提供灵活性）
        if self.normalize_mode == 'minmax':
            # 预处理后已经是[0,1]，这里只做安全检查
            image = np.clip(image, 0, 1)
        elif self.normalize_mode == 'standard':
            # 标准化
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-8)

        # 转换为torch tensor并添加channel维度
        image = torch.from_numpy(image.astype(np.float32))

        # PET数据通常不做翻转（角度信息重要）

        # 关键：VQ训练需要 'image' 键
        example["image"] = image

        # 如果使用配对数据，同时返回motion和clean
        if self.use_paired_data:
            # 从文件名推断配对文件
            basename = os.path.basename(self.image_paths[i])
            if '_motion' in basename:
                pair_name = basename.replace('_motion', '_clean')
            elif '_clean' in basename:
                pair_name = basename.replace('_clean', '_motion')
            else:
                pair_name = basename

            pair_path = os.path.join(self.data_root, pair_name)
            if os.path.exists(pair_path):
                pair_image = np.load(pair_path)
                if pair_image.ndim == 3 and pair_image.shape[0] == 1:
                    pair_image = pair_image[0]
                example["image_pair"] = torch.from_numpy(pair_image.astype(np.float32))

        return example


class PETProjectionTrain(PETProjectionBase):
    """PET投影数据训练集"""
    def __init__(self, **kwargs):
        super().__init__(
            data_root="datasets/PET_processed/train",
            isvalid=False,
            **kwargs
        )


class PETProjectionValidation(PETProjectionBase):
    """PET投影数据验证集"""
    def __init__(self, **kwargs):
        super().__init__(
            data_root="datasets/PET_processed/val",
            isvalid=True,
            **kwargs
        )


# =============================================
# XCAT 数据集
# =============================================


class XCATBase(Dataset):
    """
    XCAT 医学图像数据集基类
    数据格式：预处理后的 .npy 单张图像 (512, 512) float32
    用于 VQ-Autoencoder / LDM 训练
    新增三划分支持：train / val / test，比例 70% / 15% / 15%
    """
    def __init__(self,
                 data_root,
                 split='train',     # 'train' | 'val' | 'test'
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 split_file=None    # 可选：指定划分文件路径
                 ):
        self.split = split
        img_dir = os.path.join(data_root, 'images')
        self.all_paths = sorted(glob.glob(os.path.join(img_dir, '*.npy')))

        # 优先从 JSON 文件读取划分
        if split_file is None:
            split_file = os.path.join(data_root, 'split_indices.json')

        if os.path.exists(split_file):
            with open(split_file) as f:
                cfg = json.load(f)
            info = cfg.get('vq_ldm', cfg.get('registration', {}))
            split_conf = info.get('split', {})
            idx_range = split_conf.get(split, [0, len(self.all_paths) - 1])
            self.paths = self.all_paths[idx_range[0]: idx_range[1] + 1]
            print(f"[XCATBase] split={split}, loaded {len(self.paths)} files from split_indices.json")
        else:
            # 兜底：直接按比例划分（兼容旧代码）
            n = len(self.all_paths)
            if split == 'train':
                self.paths = self.all_paths[:int(n * 0.7)]
            elif split == 'val':
                self.paths = self.all_paths[int(n * 0.7):int(n * 0.85)]
            else:  # test
                self.paths = self.all_paths[int(n * 0.85):]

        self._length = len(self.paths)
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = np.load(self.paths[i])
        iterval = np.max(image) - np.min(image)
        minval = np.min(image)
        image = (image - minval) / iterval if iterval > 0 else image
        image = self.flip(torch.from_numpy(image))
        example["image"] = image
        return example


class XCATTrain(XCATBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep",
            split='train',
            **kwargs
        )


class XCATValidation(XCATBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep/",
            split='val',
            **kwargs
        )


class XCATTest(XCATBase):
    """新增：测试集"""
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/XCAT/prep/",
            split='test',
            **kwargs
        )


# =============================================
# SEY 肝脏 CT 数据集
# =============================================


class SEYBase(Dataset):
    """
    SEY 肝脏 CT 医学图像数据集基类
    数据格式：预处理后的 .npz 配对文件，每个文件含
        - img_small (moving, 512x512 float32 [0,1])
        - img_large (fixed,  512x512 float32 [0,1])
    用于 VQ-Autoencoder / LDM 训练（单图训练，无 fixed/moving 区分）

    目录结构（由 preprocess_sey.py 生成）：
        data_root/
            train/*.npz   配对训练集
            val/*.npz     配对验证集
            test/*.npz    配对测试集

    单图样本构造方式：每个 npz 拆成 2 张独立的单图
        索引 i -> 第 i//2 个 npz，取 img_large (i%2==0) 或 img_small (i%2==1)
    """
    def __init__(self,
                 data_root,
                 split='train',     # 'train' | 'val' | 'test'
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.split = split
        self.data_root = data_root
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"[SEYBase] split directory not found: {split_dir}")
        self.npz_paths = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
        if len(self.npz_paths) == 0:
            raise FileNotFoundError(f"[SEYBase] no .npz files under {split_dir}")

        # 每个 npz 拆成 2 张单图（img_large + img_small），下标 i -> npz_idx = i//2, side = i%2
        self._length = 2 * len(self.npz_paths)
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        print(f"[SEYBase] split={split}, loaded {len(self.npz_paths)} npz pairs "
              f"({self._length} single images) from {split_dir}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        npz_idx = i // 2
        side = i % 2  # 0 -> img_large (fixed), 1 -> img_small (moving)
        with np.load(self.npz_paths[npz_idx]) as data:
            image = data['img_large'] if side == 0 else data['img_small']
        image = image.astype(np.float32, copy=False)
        image = self.flip(torch.from_numpy(image))
        example = {"image": image}
        return example


class SEYTrain(SEYBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep",
            split='train',
            **kwargs
        )


class SEYValidation(SEYBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep",
            split='val',
            **kwargs
        )


class SEYTest(SEYBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep",
            split='test',
            **kwargs
        )
