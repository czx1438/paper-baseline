"""
统一数据划分脚本
生成并保存 train / val / test 索引，确保三阶段（VQ、LDM、配准网络）使用完全一致的划分。
划分比例：训练 70% / 验证 15% / 测试 15%
"""
import os
import json
import re
import glob
import numpy as np


def natsorted(lst):
    def natural_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
    return sorted(lst, key=natural_key)

# ==================== 路径配置 ====================
PROJECT_ROOT = "/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main"

VQAUTOENC_CONFIG = {
    "data_root": os.path.join(PROJECT_ROOT, "datasets/XCAT/prep"),
    "images_subdir": "images",          # npy 文件所在子目录
    "split_file": os.path.join(PROJECT_ROOT, "datasets/XCAT/prep/split_indices.json"),
}

REGISTRATION_CONFIG = {
    "data_root": os.path.join(PROJECT_ROOT, "datasets/XCAT/prep"),
    "pattern": "*_pair.npz",            # 配准 npz 文件匹配模式
    "split_file": os.path.join(PROJECT_ROOT, "datasets/XCAT/prep/split_indices.json"),
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


def _make_split(total, train_r, val_r, test_r):
    """根据比例返回 (train_idx, val_idx, test_idx) 的起止索引"""
    n = total
    assert abs(train_r + val_r + test_r - 1.0) < 1e-9, f"比例和必须为1，当前: {train_r}+{val_r}+{test_r}"
    train_end = int(n * train_r)
    val_end   = train_end + int(n * val_r)
    return {
        "train": [0, train_end - 1],          # 闭区间
        "val":   [train_end, val_end - 1],
        "test":  [val_end, n - 1],
    }


def split_vq_ldm(cfg):
    """VQ/LDM 数据划分：从 images_subdir 中读取 .npy 文件"""
    img_dir = os.path.join(cfg["data_root"], cfg["images_subdir"])
    all_paths = natsorted(glob.glob(os.path.join(img_dir, "*.npy")))
    n = len(all_paths)
    print(f"[VQ/LDM] 共找到 {n} 个 .npy 文件")
    split = _make_split(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    return {"vq_ldm": {"n_files": n, "split": split}}


def split_registration(cfg):
    """配准网络数据划分：从 data_root 读取 *_pair.npz 文件"""
    all_paths = natsorted(glob.glob(os.path.join(cfg["data_root"], cfg["pattern"])))
    n = len(all_paths)
    print(f"[Registration] 共找到 {n} 个 _pair.npz 文件")
    split = _make_split(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    return {"registration": {"n_files": n, "split": split}}


def main():
    vq_info  = split_vq_ldm(VQAUTOENC_CONFIG)
    reg_info = split_registration(REGISTRATION_CONFIG)

    combined = {**vq_info, **reg_info}
    combined["meta"] = {
        "train_ratio": TRAIN_RATIO,
        "val_ratio":   VAL_RATIO,
        "test_ratio":  TEST_RATIO,
    }

    out_path = VQAUTOENC_CONFIG["split_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n✅ 划分索引已保存至: {out_path}")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
