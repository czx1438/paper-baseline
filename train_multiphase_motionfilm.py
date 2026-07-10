"""
train_multiphase_motionfilm.py
==============================
Multi-phase MotionFiLM 训练脚本 v2 (基于 trajectory consistency)。

核心思想:
    同一个 (block, slice) 的 9 个相位 moving 与同一个 fixed,
    在 LDMMorph(use_motion_film=True) 内分别前向,
    得到 9 组 (disp, warped, motion_code), 然后:

        L = L_reg + lambda_z_acc * L_z_acc + lambda_dvf_acc * L_dvf_acc

其中:
    L_reg        = 9 相位配准 loss 的平均 (NCC + MSE_z)
    L_z_acc     = 二阶 trajectory consistency: mean(w * ||z_{i+1} - 2*z_i + z_{i-1}||^2)
                  w = exp(-alpha * gap_acc_i)  (自适应权重)
    L_dvf_acc   = DVF 加速度平滑: mean(||disp_{i+1} - 2*disp_i + disp_{i-1}||)

默认:
    lambda_z_acc    = 0.005
    lambda_dvf_acc  = 0.001
    alpha_motion_gap = 2.0
    (旧参数 lambda_motion / lambda_periodic 默认 = 0，保留兼容但不参与训练)

不破坏原有 pairwise 训练脚本 (train_mask.py)。
"""
import os
import sys
import csv
import random
import argparse
from argparse import ArgumentParser
from torch.utils.checkpoint import checkpoint

# 兜底 OMP_NUM_THREADS (避免容器继承到非法值导致 libgomp 警告)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

# =============== reproducibility ===============
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ldm.util import instantiate_from_config, default
from omegaconf import OmegaConf

from TransModels.LDMMorph import LDMMorph
from utils.utils import SpatialTransform, jacobian_determinant_vxm, smoothloss, MSE
from ldm.data.xcat_multiphase import MultiPhaseDataset, collate_multiphase


# ===================== argparse =====================
parser = ArgumentParser()
parser.add_argument("--ldm_ckpt", "--resume", dest="ldm_ckpt", type=str,
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-30T21-02-35_xcat-motion-ldm/checkpoints/last.ckpt',
                    help="LDM (LatentDiffusion) pretrained checkpoint (用于注入 LDMMorph 的 first_stage 与 UNet)")
parser.add_argument("--ldm_config", type=str, default=None,
                    help="LDM config yaml; default = xcat_motion-ldm.yaml")
parser.add_argument("--data_root", type=str,
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="xcat_data root (含 fixed/ moving/ registration_multi_phase_split.json)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--iteration", type=int, default=24001)
parser.add_argument("--smth_labda", type=float, default=0.8,
                    help="一阶 smooth 权重 (作用于 disp_stack)")
parser.add_argument("--bending_w", type=float, default=0.0)
parser.add_argument("--jacdet_w", type=float, default=0.0)
parser.add_argument("--beta", type=float, default=0.8,
                    help="图像域 NCC 与 latent MSE 的混合比: beta*NCC + (1-beta)*MSE_z")
# --- v2 新增一致性损失权重 ---
parser.add_argument("--lambda_z_acc", type=float, default=0.005,
                    help="二阶 motion code trajectory consistency 权重")
parser.add_argument("--lambda_dvf_acc", type=float, default=0.001,
                    help="DVF 加速度平滑权重")
parser.add_argument("--alpha_motion_gap", type=float, default=2.0,
                    help="adaptive weight: w = exp(-alpha * gap_acc)")
# --- 旧参数保留但默认禁用 (不参与训练) ---
parser.add_argument("--lambda_motion", type=float, default=0.0,
                    help="[deprecated] 一阶 motion smooth，默认为 0，不参与训练")
parser.add_argument("--lambda_periodic", type=float, default=0.0,
                    help="[deprecated] 周期性 loss，默认为 0，不参与训练，仅记录 debug 值")
# ---
parser.add_argument("--fg_thr", type=float, default=0.05)
parser.add_argument("--checkpoint", type=int, default=5000)
parser.add_argument("--save_dir", type=str,
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/MotionFiLM_MultiPhase_v2')
parser.add_argument("--motion_code_out", type=str, default=None,
                    help="motion_code_seq 落盘路径 (.npz); 默认 <save_dir>/motion_codes.npz")
parser.add_argument("--max_motion_codes", type=int, default=2048,
                    help="最多保存多少个 base-sample 的 motion_code_seq (用于后续 PCA/t-SNE)")
parser.add_argument("--loss_type", type=str, default='ncc', choices=['ncc', 'mse'])
parser.add_argument("--no_ldm", action="store_true",
                    help="使用 CNN-only 编码 (latent loss 仍计算但 LWCA 用 CNN 特征)")
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--vis_interval", type=int, default=1000)


# ===================== utils =====================
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_ldm(opt, configs_list):
    cli = OmegaConf.from_dotlist([])
    cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in configs_list], cli)
    pl_sd = torch.load(opt.ldm_ckpt, map_location='cpu')
    ldm_model = load_model_from_config(cfg.model, pl_sd['state_dict'])
    return ldm_model, cfg


def body_mask(img_tensor, thr=0.05):
    """前景(人体)mask, 与 train_mask.py 完全一致"""
    from scipy.ndimage import binary_fill_holes, label
    arr = img_tensor.detach().cpu().numpy()
    b, c, h, w = arr.shape
    out = np.zeros((b, c, h, w), dtype=np.float32)
    for bi in range(b):
        m = arr[bi, 0] > thr
        m = binary_fill_holes(m)
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = (lab == int(sizes.argmax()))
        out[bi, 0] = m.astype(np.float32)
    return torch.from_numpy(out).to(img_tensor.device)


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


def ncc_loss(fixed, moving, win_size=15, mask=None):
    """local NCC loss - 与 train_mask.py 完全一致"""
    assert fixed.shape == moving.shape
    assert win_size % 2 == 1
    b, c, h, w = fixed.shape
    pad = win_size // 2
    fixed_pad  = F.pad(fixed,  [pad]*4, mode='reflect')
    moving_pad = F.pad(moving, [pad]*4, mode='reflect')
    pf = fixed_pad.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    pm = moving_pad.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    mf = pf.mean(dim=-1); mm = pm.mean(dim=-1)
    cf = pf - mf.unsqueeze(-1); cm = pm - mm.unsqueeze(-1)
    vf = (cf ** 2).mean(dim=-1); vm = (cm ** 2).mean(dim=-1)
    cross = (cf * cm).mean(dim=-1)
    eps = 1e-8
    ncc = cross / (torch.sqrt(vf.clamp(min=eps)) * torch.sqrt(vm.clamp(min=eps)) + eps)
    if mask is not None:
        m = mask.to(ncc.dtype)
        denom = m.sum().clamp(min=1.0)
        ncc_mean = (ncc * m).sum() / denom
    else:
        ncc_mean = ncc.mean()
    return 1.0 - ncc_mean


def ncc_global(x, y, mask=None):
    """
    全图 NCC (batch-wise)，返回每个样本一个值: [B]。

    无需 local window，直接在整图上计算：
        NCC = E[(x - mean(x)) (y - mean(y))] / (std(x) * std(y))

    Args:
        x, y     : [B, 1, H, W]
        mask     : [B, 1, H, W] 可选，优先使用 body_mask
    Returns:
        ncc_vals : [B]  每个 batch 样本一个 NCC 值
    """
    b = x.shape[0]
    eps = 1e-8

    if mask is not None:
        m = mask.to(x.dtype)   # [B,1,H,W]
        # mask 内均值：只累加 mask=1 的像素
        n_pixel = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mean_x  = (x * m).sum(dim=(1, 2, 3), keepdim=True) / n_pixel
        mean_y  = (y * m).sum(dim=(1, 2, 3), keepdim=True) / n_pixel
        # mask 内方差和协方差：用原始 (x - μ) 扣 mask
        var_x   = (((x - mean_x) ** 2) * m).sum(dim=(1, 2, 3), keepdim=True) / n_pixel
        var_y   = (((y - mean_y) ** 2) * m).sum(dim=(1, 2, 3), keepdim=True) / n_pixel
        cov_xy  = (((x - mean_x) * (y - mean_y)) * m).sum(dim=(1, 2, 3), keepdim=True) / n_pixel
    else:
        mean_x = x.mean(dim=(1, 2, 3), keepdim=True)
        mean_y = y.mean(dim=(1, 2, 3), keepdim=True)
        var_x  = x.var (dim=(1, 2, 3), keepdim=True, unbiased=False)
        var_y  = y.var (dim=(1, 2, 3), keepdim=True, unbiased=False)
        cov_xy = ((x - mean_x) * (y - mean_y)).mean(dim=(1, 2, 3), keepdim=True)

    denom = torch.sqrt(var_x.clamp(min=eps)) * torch.sqrt(var_y.clamp(min=eps))
    ncc_vals = (cov_xy.squeeze(-1).squeeze(-1).squeeze(-1) / denom.squeeze(-1).squeeze(-1).squeeze(-1))
    return ncc_vals.clamp(-1.0, 1.0)


def jac_stats(D_f_xy):
    dvf = D_f_xy[0].detach().cpu().numpy()
    _, h, w = dvf.shape
    dvf_px = dvf.copy()
    dvf_px[0] *= h / 2.0
    dvf_px[1] *= w / 2.0
    jd = jacobian_determinant_vxm(dvf_px)
    n_fold = int(np.sum(jd < 0))
    return float(n_fold / jd.size), float(jd.min()), float(jd.max()), n_fold


def encode_score(ldm_model, x, t_enc=1):
    """
    对单张图提取 LDM U-Net 中间 8 个 block 的特征 (2 block / 4 scale)。
    输入: x: [B, 1, H, W], B 一般 = 1
    输出: (s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b)
          每个 [B, Ck//2, Hk, Wk]
    逐 batch 维度循环调用, 避免一次性送大量图片 OOM。
    """
    t_tensor = torch.tensor([t_enc]).cuda()
    out_blocks = [[] for _ in range(8)]
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        zi = ldm_model.get_first_stage_encoding(
            ldm_model.encode_first_stage(xi)
        ).detach()
        noise_i = torch.randn_like(zi)
        x_noisy = ldm_model.q_sample(x_start=zi, t=t_tensor, noise=noise_i)
        outx = ldm_model.apply_model(x_noisy, t=t_tensor, cond=None, return_ids=True)
        ids = outx[1][0]   # 8 个 block
        for k, idx in enumerate([0, 2, 3, 5, 6, 8, 9, 11]):
            out_blocks[k].append(ids[idx])
        del zi, noise_i, x_noisy, outx
    s = [torch.cat(ob, dim=0) for ob in out_blocks]
    return tuple(s)  # (s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b)


def model_forward_one_phase(model, x_one, y, score0, score1, score2, score3, phase_id=None):
    """
    单相位前向。
    Args:
        phase_id: [B] long tensor, 0..8. 默认 None（兼容 pairwise 脚本）。
    Returns: (D_f_xy [B,2,H,W], motion_code [B,16] or None)
    """
    out = model(x_one, y, score0, score1, score2, score3, phase_id=phase_id)
    if len(out) == 3:
        D_f_xy, _, motion_code = out
    else:
        D_f_xy, _ = out
        motion_code = None
    return D_f_xy, motion_code


# ===================== train =====================
def train():
    opt, unknown = parser.parse_known_args()
    print('==[Multi-phase MotionFiLM v2: Trajectory Consistency]==')
    for k, v in vars(opt).items():
        print(f'  {k} = {v}')

    os.makedirs(opt.save_dir, exist_ok=True)
    csv_name = os.path.join(opt.save_dir, 'train_log.csv')
    motion_code_path = opt.motion_code_out or os.path.join(opt.save_dir, 'motion_codes.npz')

    # ----------- LDM -----------
    if opt.ldm_config:
        configs_list = [opt.ldm_config]
    else:
        configs_list = ['/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_motion-ldm.yaml']
    ldm_model, _ = load_ldm(opt, configs_list)
    for p in ldm_model.parameters():
        p.requires_grad = False
    ldm_model.eval()

    # ----------- DataLoaders -----------
    train_loader = Data.DataLoader(
        MultiPhaseDataset(data_root=opt.data_root, split='train', flip_p=0.5, normalize=True),
        batch_size=opt.bs, shuffle=True, num_workers=0,
        collate_fn=collate_multiphase,
    )
    val_loader = Data.DataLoader(
        MultiPhaseDataset(data_root=opt.data_root, split='val', flip_p=0.0, normalize=True),
        batch_size=opt.bs, shuffle=False, num_workers=0,
        collate_fn=collate_multiphase,
    )
    print(f"Multi-phase: train={len(train_loader.dataset)} base-samples, "
          f"val={len(val_loader.dataset)} base-samples  "
          f"×9 phases = {len(train_loader.dataset)*9} / {len(val_loader.dataset)*9} actual pairs")

    # ----------- Model -----------
    model = LDMMorph(128*2, 192*2, 320*2, 448*2,
                     use_ldm=not opt.no_ldm,
                     use_motion_film=True)
    model.cuda()
    total = sum(p.nelement() for p in model.parameters())
    print(f"Number of parameter: {total/1e6:.2f}M")

    transform = SpatialTransform().cuda()
    for p in transform.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_similarity_mse = MSE().loss

    # ----------- CSV 表头 (v2) -----------
    header = [
        'step',
        'loss_total', 'L_reg', 'L_image_ncc', 'L_mse_latent',
        'L_smth', 'L_bend', 'L_jac',
        'L_z_acc', 'L_dvf_acc',
        'gap_mean', 'weight_mean',
        'L_periodic_debug',
        'ncc_p1', 'ncc_p5', 'ncc_p9',
        'jac_neg_p1', 'jac_neg_p5', 'jac_neg_p9',
        'z_jump', 'z_acc_raw', 'mc_std', 'mc_norm',
    ]
    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # motion_code 收集 buffer
    motion_codes_buf = []
    motion_code_pairs = []
    motion_code_buf_max = opt.max_motion_codes

    # 导入二阶损失
    from utils.utils import bending_energy_loss, jacobian_neg_loss

    step = 1
    while step <= opt.iteration:
        for fixed_t, moving_seq_t, phase_ids, pairnames, block_ids, slice_ids in train_loader:
            B = fixed_t.shape[0]
            P = moving_seq_t.shape[1]   # = 9

            fixed_t      = fixed_t.cuda().float()
            moving_seq_t = moving_seq_t.cuda().float()
            phase_ids    = phase_ids.cuda()

            fg = body_mask(fixed_t, thr=opt.fg_thr)

            # --- LDM score 提取 (fixed + 9 moving) ---
            with torch.no_grad():
                # fixed [B,1,H,W] + 9 moving [B,9,1,H,W] → [B, 10, 1, H, W]
                all_imgs = torch.cat([fixed_t.unsqueeze(1), moving_seq_t], dim=1)
                assert all_imgs.shape[1] == P + 1, \
                    f"all_imgs.shape={all_imgs.shape}, expect [B, {P+1}]"
                flat = all_imgs.view(B * (P + 1), 1, *all_imgs.shape[-2:])
                s0a_all, s0b_all, s1a_all, s1b_all, s2a_all, s2b_all, s3a_all, s3b_all = encode_score(ldm_model, flat)

                s0_all = torch.cat([s0a_all, s0b_all], dim=1)  # 128
                s1_all = torch.cat([s1a_all, s1b_all], dim=1)  # 192
                s2_all = torch.cat([s2a_all, s2b_all], dim=1)  # 320
                s3_all = torch.cat([s3a_all, s3b_all], dim=1)  # 448
            s0_fixed,  s0_moving  = s0_all[:B],          s0_all[B:].view(B, P, *s0_all.shape[1:])
            s1_fixed,  s1_moving  = s1_all[:B],          s1_all[B:].view(B, P, *s1_all.shape[1:])
            s2_fixed,  s2_moving  = s2_all[:B],          s2_all[B:].view(B, P, *s2_all.shape[1:])
            s3_fixed,  s3_moving  = s3_all[:B],          s3_all[B:].view(B, P, *s3_all.shape[1:])

            optimizer.zero_grad()

            # =========================================================
            # 第 1 阶段: trajectory loss
            #   只保留 motion_code + 低分 DVF (供 L_dvf_acc)
            #   不保留 warped / latent MSE 计算图
            # =========================================================
            motion_codes = []
            disp_low_list = []

            # =========================================================
            # safety: phase_ids 范围必须是 0..8 (否则 phase_embedding 越界)
            # =========================================================
            assert phase_ids.min() >= 1 and phase_ids.max() <= 9, \
                f"phase_ids out of expected range 1..9, got min={phase_ids.min().item()} max={phase_ids.max().item()}"

            for i in range(P):
                x_i = moving_seq_t[:, i]
                # 4-block concat (与 train_mask.py 一致)
                score0_i = torch.cat([s0_moving[:, i], s0_fixed], dim=1)
                score1_i = torch.cat([s1_moving[:, i], s1_fixed], dim=1)
                score2_i = torch.cat([s2_moving[:, i], s2_fixed], dim=1)
                score3_i = torch.cat([s3_moving[:, i], s3_fixed], dim=1)
                # phase_embedding 需要 0..8；数据是 1..9，所以减 1
                phase_i = phase_ids[:, i] - 1
                assert phase_i.min() >= 0 and phase_i.max() <= 8, \
                    f"phase_i after -1 out of range 0..8, got min={phase_i.min().item()} max={phase_i.max().item()}"

                D_f_xy, m_code = model_forward_one_phase(
                    model, x_i, fixed_t,
                    score0_i, score1_i, score2_i, score3_i,
                    phase_id=phase_i
                )
                motion_codes.append(m_code)

                # 低分 DVF 单独用于 L_dvf_acc (避免保留 full-res 计算图)
                D_low = F.interpolate(
                    D_f_xy, size=(64, 64), mode='bilinear', align_corners=True
                )
                disp_low_list.append(D_low)

                # 立即释放 full-res 计算图引用
                del D_f_xy
                torch.cuda.empty_cache()

            motion_code_seq = torch.stack(motion_codes, dim=1)            # [B, P, 16]
            disp_low_stack  = torch.stack(disp_low_list, dim=1)            # [B, P, 2, 64, 64]

            # 二阶 motion code 差分
            z_acc = (motion_code_seq[:, 2:]
                     - 2.0 * motion_code_seq[:, 1:-1]
                     + motion_code_seq[:, :-2])                              # [B, P-2, 16]
            z_acc_norm = (z_acc ** 2).sum(dim=-1)                            # [B, P-2]

            with torch.no_grad():
                gap = torch.zeros(B, P - 1, device=fixed_t.device)
                for i in range(P - 1):
                    gap[:, i] = 1.0 - ncc_global(
                        moving_seq_t[:, i],
                        moving_seq_t[:, i + 1],
                        mask=fg
                    )
                gap_acc = 0.5 * (gap[:, :-1] + gap[:, 1:])                   # [B, P-2]
                w = torch.exp(-opt.alpha_motion_gap * gap_acc)

            L_z_acc = (w * z_acc_norm).mean()

            disp_acc = (disp_low_stack[:, 2:]
                        - 2.0 * disp_low_stack[:, 1:-1]
                        + disp_low_stack[:, :-2])                            # [B, P-2, 2, 64, 64]
            L_dvf_acc = torch.abs(disp_acc).mean()

            loss_traj = (opt.lambda_z_acc   * L_z_acc
                       + opt.lambda_dvf_acc * L_dvf_acc)
            loss_traj.backward()

            # 释放第 1 阶段残留
            last_motion_code_seq = motion_code_seq.detach().cpu().numpy()
            last_z_acc_norm     = z_acc_norm.detach().cpu().numpy()
            del z_acc, z_acc_norm, disp_acc, disp_low_stack, motion_code_seq
            torch.cuda.empty_cache()

            # =========================================================
            # 第 2 阶段: 逐 phase 配准 loss (NCC + latentMSE + smooth)
            #   每个 phase: forward → loss → backward (loss/P)
            #   任何时候只有 1 个 phase 的计算图在显存
            # =========================================================
            with torch.no_grad():
                z_f = ldm_model.get_first_stage_encoding(
                    ldm_model.encode_first_stage(fixed_t)
                )

            def encode_warped_for_loss(img):
                z = ldm_model.encode_first_stage(img)
                z = ldm_model.get_first_stage_encoding(z)
                return z

            loss_image_log  = []
            loss_z_log      = []
            loss_smth_log   = []

            for i in range(P):
                x_i = moving_seq_t[:, i]
                score0_i = torch.cat([s0_moving[:, i], s0_fixed], dim=1)
                score1_i = torch.cat([s1_moving[:, i], s1_fixed], dim=1)
                score2_i = torch.cat([s2_moving[:, i], s2_fixed], dim=1)
                score3_i = torch.cat([s3_moving[:, i], s3_fixed], dim=1)
                phase_i = phase_ids[:, i] - 1

                D_f_xy, _ = model_forward_one_phase(
                    model, x_i, fixed_t,
                    score0_i, score1_i, score2_i, score3_i,
                    phase_id=phase_i
                )
                _, warped_i = transform(x_i, D_f_xy.permute(0, 2, 3, 1))

                if opt.loss_type == 'ncc':
                    li = ncc_loss(warped_i, fixed_t, mask=fg)
                else:
                    li = masked_mse(warped_i, fixed_t, fg)

                z_w = checkpoint(encode_warped_for_loss, warped_i, use_reentrant=False)
                loss_z_i    = loss_similarity_mse(z_w, z_f)
                loss_smth_i = smoothloss(D_f_xy)

                loss_reg_i = (
                    opt.beta          * li
                    + (1.0 - opt.beta) * loss_z_i
                    + opt.smth_labda  * loss_smth_i
                )
                # 等价于 (sum over P) / P = 9 phase 平均
                (loss_reg_i / P).backward()

                loss_image_log.append(li.detach())
                loss_z_log.append(loss_z_i.detach())
                loss_smth_log.append(loss_smth_i.detach())

                del (D_f_xy, warped_i, z_w,
                     li, loss_z_i, loss_smth_i, loss_reg_i)
                torch.cuda.empty_cache()

            # --- 聚合 loss 做日志 (不参与训练) ---
            loss_image   = torch.stack(loss_image_log).mean()
            loss_mse_lat = torch.stack(loss_z_log).mean()
            loss_smth    = torch.stack(loss_smth_log).mean()
            loss_reg     = (opt.beta * loss_image
                            + (1.0 - opt.beta) * loss_mse_lat
                            + opt.smth_labda * loss_smth)
            loss         = loss_reg + loss_traj.detach()

            optimizer.step()

            # --- 收集 motion_code (第 1 阶段已 stack 过的 motion_code_seq) ---
            # 重新 forward 一遍取 motion_code 仅做记录 (no_grad)
            with torch.no_grad():
                mc_list = []
                for i in range(P):
                    x_i = moving_seq_t[:, i]
                    s0_i = torch.cat([s0_moving[:, i], s0_fixed], dim=1)
                    s1_i = torch.cat([s1_moving[:, i], s1_fixed], dim=1)
                    s2_i = torch.cat([s2_moving[:, i], s2_fixed], dim=1)
                    s3_i = torch.cat([s3_moving[:, i], s3_fixed], dim=1)
                    phase_i = phase_ids[:, i] - 1
                    _, mc = model_forward_one_phase(
                        model, x_i, fixed_t, s0_i, s1_i, s2_i, s3_i,
                        phase_id=phase_i
                    )
                    mc_list.append(mc)
                motion_code_seq = torch.stack(mc_list, dim=1)
            mc_np = motion_code_seq.detach().cpu().numpy()
            for b in range(B):
                if len(motion_codes_buf) >= motion_code_buf_max:
                    break
                motion_codes_buf.append(mc_np[b])
                motion_code_pairs.append(f"{pairnames[b]}")

            # =============================================================
            # 日志
            # =============================================================
            if step == 1 or step % opt.log_interval == 0:
                with torch.no_grad():
                    # 重新 forward 一次只为取 warped (用于相位级 NCC / negR)
                    ncc_p, neg_p = [], []
                    for i in range(P):
                        x_i = moving_seq_t[:, i]
                        s0_i = torch.cat([s0_moving[:, i], s0_fixed], dim=1)
                        s1_i = torch.cat([s1_moving[:, i], s1_fixed], dim=1)
                        s2_i = torch.cat([s2_moving[:, i], s2_fixed], dim=1)
                        s3_i = torch.cat([s3_moving[:, i], s3_fixed], dim=1)
                        phase_i = phase_ids[:, i] - 1
                        Di, _ = model_forward_one_phase(
                            model, x_i, fixed_t, s0_i, s1_i, s2_i, s3_i,
                            phase_id=phase_i
                        )
                        _, wi = transform(x_i, Di.permute(0, 2, 3, 1))
                        ncc_p.append(
                            1.0 - ncc_loss(wi, fixed_t, mask=fg).item()
                        )
                        neg_p.append(jac_stats(Di)[0])
                    ncc_p1, ncc_p5, ncc_p9 = ncc_p[0], ncc_p[4], ncc_p[8]
                    neg_p1, neg_p5, neg_p9 = neg_p[0], neg_p[4], neg_p[8]

                    gap_mean_val    = gap_acc.mean().item()
                    weight_mean_val = w.mean().item()
                    mc_t = torch.as_tensor(last_motion_code_seq, device=motion_code_seq.device)
                    mc_std      = mc_t.std().item()
                    mc_norm_m   = mc_t.norm(dim=-1).mean().item()
                    z_jump = (mc_t[:, 1:] - mc_t[:, :-1]).norm(dim=-1).mean()
                    za_t = torch.as_tensor(last_z_acc_norm, device=motion_code_seq.device)
                    z_acc_raw = za_t.sqrt().mean()

                    print(
                        f"z_jump={z_jump.item():.6f} "
                        f"z_acc_raw={z_acc_raw.item():.6f} "
                        f"mc_std={mc_std:.4f} mc_norm={mc_norm_m:.4f}"
                    )
                    sys.stdout.write(
                        f"\rstep {step} "
                        f"L={loss.item():.4f} "
                        f"L_reg={loss_reg.item():.4f} "
                        f"L_NCC={loss_image.item():.4f} "
                        f"L_zMSE={loss_mse_lat.item():.4f} "
                        f"L_smth={loss_smth.item():.4f} "
                        f"L_z_acc={L_z_acc.item():.8f} "
                        f"L_dvf_acc={L_dvf_acc.item():.8f} "
                        f"gap={gap_mean_val:.3f} "
                        f"w={weight_mean_val:.3f} "
                        f"z_jump={z_jump.item():.4f} "
                        f"z_acc_raw={z_acc_raw.item():.4f} "
                        f"mc_std={mc_std:.4f} "
                        f"mc_norm={mc_norm_m:.4f} "
                        f"NCC_p1/5/9={ncc_p1:.3f}/{ncc_p5:.3f}/{ncc_p9:.3f} "
                        f"negR_p1/5/9={neg_p1*100:.2f}/{neg_p5*100:.2f}/{neg_p9*100:.2f}%")
                    sys.stdout.flush()

                    if step % opt.vis_interval == 0:
                        with open(csv_name, 'a') as f:
                            csv.writer(f).writerow([
                                step,
                                f"{loss.item():.6f}",
                                f"{loss_reg.item():.6f}",
                                f"{loss_image.item():.6f}",
                                f"{loss_mse_lat.item():.6f}",
                                f"{loss_smth.item():.6f}",
                                f"{0.0:.6f}",   # L_bend (deprecated, 默认 0)
                                f"{0.0:.6f}",   # L_jac  (deprecated, 默认 0)
                                f"{L_z_acc.item():.6f}",
                                f"{L_dvf_acc.item():.6f}",
                                f"{gap_mean_val:.6f}",
                                f"{weight_mean_val:.6f}",
                                f"{0.0:.6f}",   # L_periodic (deprecated)
                                f"{ncc_p1:.6f}", f"{ncc_p5:.6f}", f"{ncc_p9:.6f}",
                                f"{neg_p1:.6f}", f"{neg_p5:.6f}", f"{neg_p9:.6f}",
                                f"{z_jump.item():.6f}",
                                f"{z_acc_raw.item():.6f}",
                                f"{mc_std:.6f}",
                                f"{mc_norm_m:.6f}",
                            ])

            # --- checkpoint ---
            if step % opt.checkpoint == 0:
                ck_path = os.path.join(opt.save_dir, f'multiphase_step{step:06d}.pth')
                torch.save(model.state_dict(), ck_path)
                print(f"\n[ckpt] saved {ck_path}")

                if len(motion_codes_buf) > 0:
                    np.savez_compressed(
                        motion_code_path,
                        motion_codes=np.stack(motion_codes_buf, axis=0),
                        pairnames=np.array(motion_code_pairs),
                    )
                    print(f"[motion_code] saved {len(motion_codes_buf)} → {motion_code_path}")

            step += 1
            if step > opt.iteration:
                break

    # --- 训练结束 ---
    if len(motion_codes_buf) > 0:
        np.savez_compressed(
            motion_code_path,
            motion_codes=np.stack(motion_codes_buf, axis=0),
            pairnames=np.array(motion_code_pairs),
        )
        print(f"\n[FINAL] motion_codes: shape={np.stack(motion_codes_buf, axis=0).shape}  → {motion_code_path}")


if __name__ == "__main__":
    train()
