#!/usr/bin/env python3
"""
XCAT图像分析：手动指定ROI区域创建mask，计算Dice系数

支持两种ROI定义方式：
1. 交互式选择：使用鼠标框选区域
2. 固定坐标：直接指定心脏和肝脏的边界
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
import glob
import argparse

# ==================== 配置参数 ====================

# 数据路径
DATA_ROOT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data'
FIXED_DIR = os.path.join(DATA_ROOT, 'fixed', 'fixed')
MOVING_DIR = os.path.join(DATA_ROOT, 'moving', 'moving')

# 默认ROI坐标 (x1, y1, x2, y2) - 可根据实际图像调整
# 格式: (左上角x, 左上角y, 右下角x, 右下角y)
# 图像尺寸: 512x512
DEFAULT_ROI = {
    'heart': (280, 230, 340, 370),    # 心脏区域 - 右移、下降、缩小
    'liver': (170, 320, 300, 520),   # 肝脏区域 - 下降、缩小
}

# 亮度阈值选项
# - None: 不使用阈值，使用整个ROI区域
# - 数值(0-100): 使用该百分位阈值提取ROI内的高亮度区域
THRESHOLD_PERCENTILE = 25

# 交互模式
# - True: 运行时弹出窗口让你用鼠标框选ROI
# - False: 使用上面定义的DEFAULT_ROI固定坐标
INTERACTIVE_MODE = False

# ==================== ROI选择器类 ====================

class ROISelector:
    """交互式ROI选择器"""
    def __init__(self, img, title="Select ROI"):
        self.img = img
        self.title = title
        self.roi_coords = None
        self.done = False
        
    def select_roi(self):
        """启动交互式选择"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.img, cmap='gray')
        ax.set_title(f'{self.title}\nClick and drag to select region\nPress "q" to quit without selecting')
        
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.roi_coords = (
                min(x1, x2), min(y1, y2),
                max(x1, x2), max(y1, y2)
            )
            print(f"\nSelected ROI: x=[{self.roi_coords[0]}, {self.roi_coords[2]}], y=[{self.roi_coords[1]}, {self.roi_coords[3]}]")
            self.done = True
            plt.close()
        
        self.rs = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1], minspanx=5, minspany=5)
        
        def on_key(event):
            if event.key == 'q':
                print("\nROI selection cancelled")
                self.done = True
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()
        return self.roi_coords

# ==================== 核心函数 ====================

def normalize(img):
    """归一化图像到[0,1]"""
    minv, maxv = img.min(), img.max()
    if maxv - minv > 1e-6:
        return (img - minv) / (maxv - minv)
    return img

def create_roi_mask(img, roi_coords, threshold_percentile=None):
    """
    基于ROI区域创建mask
    
    Args:
        img: 归一化图像
        roi_coords: (x1, y1, x2, y2) 感兴趣区域坐标
        threshold_percentile: 亮度阈值百分位，None表示不使用阈值
    
    Returns:
        mask: 二值mask
    """
    x1, y1, x2, y2 = roi_coords
    mask = np.zeros_like(img)
    
    if threshold_percentile is not None:
        roi_pixels = img[y1:y2, x1:x2]
        threshold = np.percentile(roi_pixels, threshold_percentile)
        mask[y1:y2, x1:x2] = (roi_pixels > threshold).astype(np.float32)
    else:
        mask[y1:y2, x1:x2] = 1.0
    
    return mask

def dice_score(pred, target):
    """计算Dice系数"""
    intersection = np.sum(pred * target)
    return 2.0 * intersection / (np.sum(pred) + np.sum(target) + 1e-8)

def visualize_roi(img_fixed, img_moving, roi_dict, results):
    """可视化ROI和Dice结果"""
    n_rois = len(roi_dict)
    total_cols = 2 + n_rois * 2  # 原始图 + 差异图 + 各ROI的fixed/moving
    
    fig = plt.figure(figsize=(4 * total_cols, 8))
    
    # 第一行：原始图像
    ax1 = plt.subplot(2, total_cols, 1)
    ax1.imshow(img_fixed, cmap='gray')
    ax1.set_title('Fixed Image')
    ax1.axis('off')
    
    colors = {'heart': 'red', 'liver': 'orange'}
    for name, coords in roi_dict.items():
        x1, y1, x2, y2 = coords
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor=colors.get(name, 'white'), 
                            linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x1 + 5, y1 + 15, name.upper(), color=colors.get(name, 'white'), 
                fontsize=10, fontweight='bold')
    ax1.legend(loc='lower right')
    
    # Moving图像
    ax2 = plt.subplot(2, total_cols, 2)
    ax2.imshow(img_moving, cmap='gray')
    ax2.set_title('Moving Image')
    ax2.axis('off')
    for name, coords in roi_dict.items():
        x1, y1, x2, y2 = coords
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor=colors.get(name, 'white'), 
                            linewidth=2)
        ax2.add_patch(rect)
    
    # 差异图
    ax3 = plt.subplot(2, total_cols, 3)
    diff = np.abs(img_fixed - img_moving)
    im = ax3.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    ax3.set_title('Difference')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # ROI区域的mask - 从第4列开始
    for i, (name, coords) in enumerate(roi_dict.items()):
        x1, y1, x2, y2 = coords
        col_offset = 3 + i * 2  # 第4列开始，每2列一个ROI
        
        # Fixed ROI mask
        ax_fix = plt.subplot(2, total_cols, col_offset + 1)
        mask_fixed = results[name]['mask_fixed']
        ax_fix.imshow(img_fixed, cmap='gray')
        ax_fix.imshow(np.ma.masked_where(mask_fixed == 0, mask_fixed), 
                     cmap='Reds' if name == 'heart' else 'Oranges', alpha=0.5)
        ax_fix.set_title(f'{name.upper()} (Fixed)\nDice={results[name]["dice"]:.4f}')
        ax_fix.axis('off')
        
        # Moving ROI mask
        ax_mov = plt.subplot(2, total_cols, col_offset + 2)
        mask_moving = results[name]['mask_moving']
        ax_mov.imshow(img_moving, cmap='gray')
        ax_mov.imshow(np.ma.masked_where(mask_moving == 0, mask_moving), 
                     cmap='Reds' if name == 'heart' else 'Oranges', alpha=0.5)
        ax_mov.set_title(f'{name.upper()} (Moving)')
        ax_mov.axis('off')
    
    plt.tight_layout()
    return fig

def analyze_with_roi(fixed_path, moving_path, roi_dict, threshold_percentile=None):
    """分析指定ROI区域的配准效果"""
    img_fixed = normalize(np.load(fixed_path))
    img_moving = normalize(np.load(moving_path))
    
    results = {}
    for organ_name, roi_coords in roi_dict.items():
        mask_fixed = create_roi_mask(img_fixed, roi_coords, threshold_percentile)
        mask_moving = create_roi_mask(img_moving, roi_coords, threshold_percentile)
        dice = dice_score(mask_fixed, mask_moving)
        
        x1, y1, x2, y2 = roi_coords
        roi_diff = np.abs(img_fixed[y1:y2, x1:x2] - img_moving[y1:y2, x1:x2])
        
        results[organ_name] = {
            'dice': dice,
            'mask_fixed': mask_fixed,
            'mask_moving': mask_moving,
            'mean_diff': np.mean(roi_diff),
            'max_diff': np.max(roi_diff),
            'roi_area': (x2 - x1) * (y2 - y1)
        }
    
    total_mask_fixed = np.clip(sum(r['mask_fixed'] for r in results.values()), 0, 1)
    total_mask_moving = np.clip(sum(r['mask_moving'] for r in results.values()), 0, 1)
    results['total'] = {'dice': dice_score(total_mask_fixed, total_mask_moving)}
    
    return results, img_fixed, img_moving

# ==================== 主程序 ====================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='XCAT ROI-based Dice Analysis')
    parser.add_argument('--heart', type=int, nargs=4, default=None,
                       help='Heart ROI: x1 y1 x2 y2')
    parser.add_argument('--liver', type=int, nargs=4, default=None,
                       help='Liver ROI: x1 y1 x2 y2')
    parser.add_argument('--threshold', type=float, default=THRESHOLD_PERCENTILE,
                       help='Threshold percentile (0-100), default: no threshold')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive ROI selection')
    parser.add_argument('--test-n', type=int, default=5,
                       help='Number of samples to test')
    args = parser.parse_args()
    
    # 加载图像路径
    fixed_paths = sorted(glob.glob(os.path.join(FIXED_DIR, '*.npy')))
    moving_paths = sorted(glob.glob(os.path.join(MOVING_DIR, '*.npy')))
    print(f"Found {len(fixed_paths)} fixed images and {len(moving_paths)} moving images")
    
    # 加载第一个样本获取图像尺寸
    img_sample = normalize(np.load(fixed_paths[0]))
    h, w = img_sample.shape
    print(f"Image shape: {img_sample.shape}")
    
    # 确定ROI
    if args.interactive:
        print("\n=== Interactive ROI Selection ===")
        print("For each organ, click and drag to select the region.")
        print("Press 'q' to quit without selecting.")
        
        roi_dict = {}
        for organ in ['heart', 'liver']:
            print(f"\n--- Selecting {organ.upper()} ROI ---")
            selector = ROISelector(img_sample, f"Select {organ.upper()} Region")
            coords = selector.select_roi()
            if coords:
                roi_dict[organ] = coords
            else:
                print(f"Using default ROI for {organ}")
                roi_dict[organ] = DEFAULT_ROI[organ]
    else:
        roi_dict = DEFAULT_ROI.copy()
        if args.heart:
            roi_dict['heart'] = tuple(args.heart)
        if args.liver:
            roi_dict['liver'] = tuple(args.liver)
    
    # 显示选定的ROI
    print("\n=== Selected ROI Coordinates ===")
    for name, coords in roi_dict.items():
        x1, y1, x2, y2 = coords
        print(f"{name.upper()}: x=[{x1}, {x2}], y=[{y1}, {y2}], size={x2-x1}x{y2-y1}")
    
    threshold = args.threshold
    print(f"\nThreshold: {'None (full ROI)' if threshold is None else f'{threshold}th percentile'}")
    
    # 分析第一个样本
    print("\n=== First Sample Analysis ===")
    results, img_fixed, img_moving = analyze_with_roi(
        fixed_paths[0], moving_paths[0], roi_dict, threshold
    )
    
    for name in ['heart', 'liver']:
        print(f"\n{name.upper()}:")
        print(f"  Dice Score: {results[name]['dice']:.4f}")
        print(f"  Mean Diff: {results[name]['mean_diff']:.4f}")
        print(f"  Max Diff: {results[name]['max_diff']:.4f}")
    print(f"\nTotal Dice: {results['total']['dice']:.4f}")
    
    # 可视化
    fig = visualize_roi(img_fixed, img_moving, roi_dict, results)
    fig.savefig('xcat_roi_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: xcat_roi_analysis.png")
    
    # 测试多个样本
    print(f"\n=== Testing {args.test_n} Samples ===")
    all_results = {name: [] for name in list(roi_dict.keys()) + ['total']}
    
    n_test = min(args.test_n, len(fixed_paths))
    for i in range(n_test):
        res, _, _ = analyze_with_roi(fixed_paths[i], moving_paths[i], roi_dict, threshold)
        for name in all_results.keys():
            if name in res:
                all_results[name].append(res[name]['dice'])
        
        print(f"Sample {i}: " + ", ".join(f"{n}={all_results[n][-1]:.4f}" 
              for n in all_results.keys()))
    
    print(f"\n=== Average Dice Across {n_test} Samples ===")
    for name, values in all_results.items():
        if values:
            print(f"{name.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
