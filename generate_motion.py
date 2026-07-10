"""
心脏形变真值生成（自动定位心脏中心版 · 平滑位移场 · 收缩-扩张-收缩 · 输出GT场）
=================================================================
改进：
  1. 自动定位心脏中心（ROI亮度阈值筛选 -> 加权质心）
  2. 完整心动周期：收缩(0.95) -> 扩张(1.15) -> 收缩(0.95)
  3. 后向映射，避免前向映射的方向反转
  4. 可视化标注每帧 scale 值
=================================================================
"""
import os, glob, argparse
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

# ---- 运动参数（收缩-扩张-收缩周期）----
MAX_SCALE  = 1.4     # 膨胀峰值  原本1.15
MIN_SCALE  = 0.7     # 收缩状态（起点和终点）  原本0.95
MAX_DY     = 20      # 上下平移峰值(像素)  原本7
NUM_FRAMES = 8
A_FRAC, B_FRAC = 0.10, 0.17      # 椭圆半轴(相对 w,h)
INNER = 0.5                       # 满额半径
N_VIZ = 100                        # 可视化样本数

# ---- 自动定位心脏的搜索区域 ----
SEARCH_REGION = (0.38, 0.70, 0.50, 0.65)


def find_heart_center_with_viz(img):
    h, w = img.shape
    y0, y1 = int(h*SEARCH_REGION[0]), int(h*SEARCH_REGION[1])
    x0, x1 = int(w*SEARCH_REGION[2]), int(w*SEARCH_REGION[3])
    roi_abs = (y0, y1, x0, x1)

    roi = img[y0:y1, x0:x1].astype(np.float32)
    heart_mask = np.zeros((h, w), dtype=np.uint8)

    rmin, rmax = roi.min(), roi.max()
    if rmax - rmin < 1e-6:
        return int(w*0.59), int(h*0.53), roi_abs, heart_mask

    rn = (roi - rmin) / (rmax - rmin)
    thr = np.percentile(rn, 60)
    binmap = (rn > thr).astype(np.uint8)
    heart_mask[y0:y1, x0:x1] = binmap * 255

    yy, xx = np.mgrid[y0:y1, x0:x1]
    if binmap.sum() <= 0:
        return int(w*0.59), int(h*0.53), roi_abs, heart_mask

    total = binmap.sum()
    cx_raw = int((xx * binmap).sum() / total)
    rows = yy[binmap > 0]
    cols = xx[binmap > 0]
    mask_top = int(rows.min())

    # 让椭圆上边界贴合 mask 最高点：cy_top + b = mask_top
    # 临时用 mask_top 作为上边界反推 cy
    a, b = int(w*A_FRAC), int(h*B_FRAC)
    cy = mask_top + b  # 上边界对齐 mask 顶，椭圆整体下移覆盖完整心脏
    cx = cx_raw
    return cx, cy, roi_abs, heart_mask


def make_smooth_field(h, w, dy, scale, ellipse):
    cx, cy, a, b = ellipse
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    yy = yy.astype(np.float32); xx = xx.astype(np.float32)
    r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    weight = np.where(r < INNER, 1.0,
             np.where(r < 1.0, 0.5*(1.0+np.cos(np.pi*(r-INNER)/(1.0-INNER))), 0.0)).astype(np.float32)
    scale_local = 1.0 + (scale - 1.0) * weight
    disp_x = (scale_local - 1.0) * (xx - cx)
    disp_y = (scale_local - 1.0) * (yy - cy) + dy
    return np.stack([disp_y, disp_x], axis=0).astype(np.float32)


def warp(img, disp):
    h, w = img.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # 后向映射：target位置的值 = source的(位置-位移)
    map_y = (yy - disp[0]).astype(np.float32)
    map_x = (xx - disp[1]).astype(np.float32)
    if cv2 is not None:
        return cv2.remap(img.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    from scipy.ndimage import map_coordinates
    return map_coordinates(img.astype(np.float32), [map_y, map_x], order=1, mode='nearest')


def jac_neg_ratio(disp):
    dy, dx = disp[0], disp[1]
    dydy, dydx = np.gradient(dy); dxdy, dxdx = np.gradient(dx)
    jac = (1 + dydy) * (1 + dxdx) - dydx * dxdy
    return float((jac <= 0).mean()), float(jac.min())


def generate_one(fixed, ellipse, num_frames=NUM_FRAMES, max_scale=MAX_SCALE, min_scale=MIN_SCALE, max_dy=MAX_DY):
    """生成序列，明确传入参数避免全局变量依赖"""
    h, w = fixed.shape
    movings, flows = [], []
    wneg, wmin = 0.0, 1e9
    for i in range(num_frames):
        phase = i / num_frames
        angle = phase * np.pi
        s = min_scale + (max_scale - min_scale) * np.sin(angle)
        dy = max_dy * np.sin(angle)
        phi = make_smooth_field(h, w, dy, s, ellipse)
        moving = warp(fixed, phi)
        neg, jmin = jac_neg_ratio(phi)
        wneg = max(wneg, neg); wmin = min(wmin, jmin)
        movings.append(moving); flows.append(phi)
    return np.array(movings, np.float32), np.array(flows, np.float32), wneg, wmin


def viz_sample(fixed, seq, ellipse, save_path, roi_region=None, heart_mask=None, 
               num_frames=NUM_FRAMES, max_scale=MAX_SCALE, min_scale=MIN_SCALE):
    """可视化，明确传入参数"""
    cx, cy, a, b = ellipse
    h, w = fixed.shape
    n = seq.shape[0]
    vmax = fixed.max() if fixed.max() > 0 else 1.0
    
    fig, axes = plt.subplots(1, n + 1, figsize=(2.8*(n+1), 3.2))
    
    def draw(ax, im, title, show_roi=False, show_mask=False):
        ax.imshow(im, cmap='gray', vmin=0, vmax=vmax)
        
        if show_roi and roi_region is not None:
            y0, y1, x0, x1 = roi_region
            rect = Rectangle((x0, y0), x1-x0, y1-y0, 
                           fill=False, edgecolor='yellow', 
                           linestyle='--', linewidth=1.5, alpha=0.8)
            ax.add_patch(rect)
        
        if show_mask and heart_mask is not None:
            mask_rgba = np.zeros((h, w, 4))
            mask_rgba[heart_mask > 0] = [0.0, 1.0, 0.0, 0.35]
            ax.imshow(mask_rgba)
        
        ellipse_patch = Ellipse((cx, cy), 2*a, 2*b, fill=False, 
                               edgecolor='lime', linewidth=1.8)
        ax.add_patch(ellipse_patch)
        ax.plot(cx, cy, 'g+', markersize=12, markeredgewidth=2.5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')
    
    # Fixed帧
    draw(axes[0], fixed, "Fixed\n(scale=1.0)", show_roi=True, show_mask=True)
    
    # 序列帧，标注每帧scale值
    for i in range(n):
        phase = i / num_frames
        angle = phase * np.pi
        s = min_scale + (max_scale - min_scale) * np.sin(angle)
        if s > 1.0:
            status = "↑扩张"
        elif s < 1.0:
            status = "↓收缩"
        else:
            status = "=原始"
        title = f"Frame {i}\nscale={s:.3f} {status}"
        draw(axes[i+1], seq[i], title)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    moving_dir = os.path.join(args.out, "moving_seq")
    flow_dir   = os.path.join(args.out, "gt_flow")
    flown_dir  = os.path.join(args.out, "gt_flow_norm")
    viz_dir    = os.path.join(args.out, "viz")
    for d in (moving_dir,):
        os.makedirs(d, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not paths:
        print(f"[错误] {args.input_dir} 下没有 .npy")
        return
    print(f"找到 {len(paths)} 张 fixed，生成 {NUM_FRAMES} 帧序列（收缩-扩张-收缩周期）")
    print(f"  scale范围: {MIN_SCALE} → {MAX_SCALE} → {MIN_SCALE}")

    gneg, gmin = 0.0, 1e9
    ff_err = []
    centers = []
    
    for k, p in enumerate(paths):
        if args.limit > 0 and k >= args.limit:
            break
        fixed = np.load(p).astype(np.float32)
        if fixed.ndim == 3: fixed = fixed.squeeze()
        h, w = fixed.shape

        cx, cy, roi_abs, heart_mask = find_heart_center_with_viz(fixed)
        a, b = int(w*A_FRAC), int(h*B_FRAC)
        ellipse = (cx, cy, a, b)
        centers.append((cx, cy))

        movings, flows, wneg, wmin = generate_one(fixed, ellipse)
        gneg = max(gneg, wneg); gmin = min(gmin, wmin)
        ff_err.append(float(np.abs(movings[0] - fixed).max()))

        flows_norm = flows.copy()
        flows_norm[:, 0] = flows[:, 0] / ((h - 1) / 2.0)
        flows_norm[:, 1] = flows[:, 1] / ((w - 1) / 2.0)

        base = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(moving_dir, f"{base}.npy"), movings)

        if k < N_VIZ:
            viz_sample(fixed, movings, ellipse, 
                      os.path.join(viz_dir, f"{k:02d}_{base}.png"),
                      roi_region=roi_abs, heart_mask=heart_mask)

        if (k+1) % 50 == 0 or k == len(paths)-1:
            print(f"  {k+1}/{len(paths)} done  |  中心: ({cx}, {cy})")

    centers = np.array(centers)
    print("\n" + "="*60)
    print("完成。生成周期: 收缩(0.95) → 扩张(1.15) → 收缩(0.95)")
    print(f"  雅可比负值占比: {gneg*100:.4f}%")
    print(f"  最小雅可比:     {gmin:.4f}")
    print(f"  第0帧误差:      {max(ff_err):.2e}")
    print(f"  中心cx范围:     [{centers[:,0].min():.0f}, {centers[:,0].max():.0f}]")
    print(f"  中心cy范围:     [{centers[:,1].min():.0f}, {centers[:,1].max():.0f}]")
    print("="*60)


if __name__ == "__main__":
    main()