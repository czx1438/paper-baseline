"""
Multi-phase registration 可视化 / 评估脚本
==========================================
跟 train_multiphase_motionfilm.py 完全对齐的数据流:

  - 使用 ldm.data.xcat_multiphase.MultiPhaseDataset
    (block 硬切: train={0,1,3,5}, val={2}, test={4})
  - forward 路径: model(X_i, Y, score_j_i, phase_id=phase_i)
    与 train_multiphase_motionfilm.py:412-416 一致
  - LDM score 完整提取 4 对 (s0a,s0b,s1a,s1b,...) -> 按 moving/fixed 切分
    顺序: cat([s_moving[:, i], s_fixed], dim=1)
  - 输出: 1 个 CSV (每个 (sample, phase) 一行) + 1 张 9 相位轨迹图
"""
import os, sys, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ldm.data.xcat_multiphase import MultiPhaseDataset, collate_multiphase
from utils.utils import SpatialTransform, jacobian_determinant_vxm
from visualize_registration_xcat import (
    DEFAULT_ROI, compute_dice_with_mask,
)
import TransModels.LDMMorph as LDMMorph
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


# ======================== 参数 ========================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, required=True)
parser.add_argument("--data_root", type=str, default='./xcat_data')
parser.add_argument("--ldm_config", type=str,
                    default='./configs/latent-diffusion/xcat_no_motion.yaml')
parser.add_argument("--ldm_ckpt", type=str, required=True)
parser.add_argument("--use_motion_film", action="store_true", default=True)
parser.add_argument("--no_ldm", action="store_true", default=False)
parser.add_argument("--split", type=str, default='test',
                    choices=['train', 'val', 'test'])
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--n_samples", type=int, default=None,
                    help="None = 跑完整个 split")
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--t_enc", type=int, default=1)
parser.add_argument("--no_save_fig", action="store_true")
opt, unknown = parser.parse_known_args()

# ======================== save dir ========================
if opt.save_dir is None:
    suffix = '_motionfilm' if opt.use_motion_film else '_baseline'
    opt.save_dir = f'./logs/visualize_multiphase_{opt.split}{suffix}/'
os.makedirs(opt.save_dir, exist_ok=True)

print('=' * 72)
print(f"Mode: MultiPhase / split={opt.split} / use_motion_film={opt.use_motion_film}")
print(f"Resume:   {opt.resume}")
print(f"LDM Cfg:  {opt.ldm_config}")
print(f"LDM CKPT: {opt.ldm_ckpt}")
print(f"Save Dir: {opt.save_dir}")
print('=' * 72)


# ======================== LDM ========================
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


configs = OmegaConf.merge(OmegaConf.load(opt.ldm_config),
                          OmegaConf.from_dotlist(unknown))
ldm_model = load_model_from_config(configs.model, {"state_dict": None})
pl_sd = torch.load(opt.ldm_ckpt, map_location='cpu')
ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
print(f"LDM loaded: {opt.ldm_ckpt}")
for p in ldm_model.parameters():
    p.requires_grad = False
ldm_model.eval()


# ======================== 配准网络 ========================
model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2,
                          use_ldm=not opt.no_ldm,
                          use_motion_film=opt.use_motion_film).cuda()
state = torch.load(opt.resume, map_location='cuda')
model.load_state_dict(state, strict=False)
print(f"Registration model loaded: {opt.resume}")
model.eval()

transform = SpatialTransform().cuda()
for p in transform.parameters():
    p.requires_grad = False


# ======================== LDM score 提取 (与 train 一致) ========================
def encode_score(ldm_model, x, t_enc=1):
    """逐 batch 调用 LDM U-Net 提取 8 个中间 block 特征 (2 block / 4 scale)"""
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
        for k, idx in enumerate([0, 2, 3, 5, 6, 8, 9, 11]):
            out_blocks[k].append(outx[1][0][idx])
        del zi, noise_i, x_noisy, outx
    return tuple(torch.cat(ob, dim=0) for ob in out_blocks)
    # -> (s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b)


def get_score_per_phase(ldm_model, moving_seq_t, fixed_t, t_enc=1):
    """
    与 train_multiphase_motionfilm.py:368-406 完全一致:
      1) fixed + 9 moving 拼 10 张送 LDM, 提取 s0a..s3b
      2) s_i_all = cat([s_i_a, s_i_b], dim=1)
      3) 切 fixed (前 B) + moving (后 B*P, view 成 [B, P, C_i, ...])
      4) 对每个 phase i:
            score_j_i = cat([s_j_moving[:, i], s_j_fixed], dim=1)
                      # 顺序: moving 在前, fixed 在后
    Returns:
        list of (score0_i, score1_i, score2_i, score3_i) for i in 0..P-1
    """
    B, P = moving_seq_t.shape[0], moving_seq_t.shape[1]
    with torch.no_grad():
        all_imgs = torch.cat([fixed_t.unsqueeze(1), moving_seq_t], dim=1)   # [B, P+1, 1, H, W]
        flat = all_imgs.view(B * (P + 1), 1, *all_imgs.shape[-2:])          # [B*(P+1), 1, H, W]
        s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b = encode_score(ldm_model, flat, t_enc)
        s0_all = torch.cat([s0a, s0b], dim=1)
        s1_all = torch.cat([s1a, s1b], dim=1)
        s2_all = torch.cat([s2a, s2b], dim=1)
        s3_all = torch.cat([s3a, s3b], dim=1)

    s0_fixed, s0_moving = s0_all[:B], s0_all[B:].view(B, P, *s0_all.shape[1:])
    s1_fixed, s1_moving = s1_all[:B], s1_all[B:].view(B, P, *s1_all.shape[1:])
    s2_fixed, s2_moving = s2_all[:B], s2_all[B:].view(B, P, *s2_all.shape[1:])
    s3_fixed, s3_moving = s3_all[:B], s3_all[B:].view(B, P, *s3_all.shape[1:])

    score_list = []
    for i in range(P):
        score0_i = torch.cat([s0_moving[:, i], s0_fixed], dim=1)
        score1_i = torch.cat([s1_moving[:, i], s1_fixed], dim=1)
        score2_i = torch.cat([s2_moving[:, i], s2_fixed], dim=1)
        score3_i = torch.cat([s3_moving[:, i], s3_fixed], dim=1)
        score_list.append((score0_i, score1_i, score2_i, score3_i))
    return score_list


# ======================== 数据集 ========================
dataset = MultiPhaseDataset(
    data_root=opt.data_root,
    split=opt.split,
    flip_p=0.0,
    normalize=True,
)

start = opt.start_idx
end = min(start + (opt.n_samples or len(dataset)), len(dataset))
indices = list(range(start, end))

print(f'\nProcessing split={opt.split}  total in split={len(dataset)}  '
      f'vis=[{start},{end})  count={len(indices)}')


# ======================== 评估循环 ========================
csv_rows = []

for i, idx in enumerate(indices):
    fixed_t, moving_seq_t, phase_ids, pairnames, block_ids, slice_ids = \
        collate_multiphase([dataset[idx]])

    fixed_t      = fixed_t.cuda().float()           # [1, 1, H, W]
    moving_seq_t = moving_seq_t.cuda().float()      # [1, 9, 1, H, W]
    phase_ids    = phase_ids.cuda()

    P = moving_seq_t.shape[1]
    score_list = get_score_per_phase(ldm_model, moving_seq_t, fixed_t, opt.t_enc)

    ncc_b_list, ncc_a_list = [], []
    dice_h_list, dice_l_list = [], []
    jacneg_list = []
    fig_data = []

    for p in range(P):
        x_one = moving_seq_t[:, p]                       # [1, 1, H, W] (沿 phase 维取一张)
        score0_p, score1_p, score2_p, score3_p = score_list[p]

        with torch.no_grad():
            out = model(x_one, fixed_t,
                        score0_p, score1_p, score2_p, score3_p,
                        phase_id=torch.tensor([p], dtype=torch.long).cuda())
            D_f_xy = out[0]
            _, warped_X = transform(x_one, D_f_xy.permute(0, 2, 3, 1))

        # NCC (局部 win=15, 与可视化脚本保持口径)
        img_f = fixed_t[0, 0].cpu().numpy()
        img_x = x_one[0, 0].cpu().numpy()
        img_w = warped_X[0, 0].cpu().numpy()

        ncc_b = _local_ncc(img_f, img_x, win=15)
        ncc_a = _local_ncc(img_f, img_w, win=15)

        # Dice (heart / liver) - 调用可视化脚本里的 ROI + 阈值版本
        dice_dict, _ = compute_dice_with_mask(img_f, img_x, D_f_xy)

        # jacobian (像素域)
        dvf = D_f_xy[0].detach().cpu().numpy()
        dvf_px = dvf.copy()
        dvf_px[0] *= dvf_px.shape[1] / 2.0
        dvf_px[1] *= dvf_px.shape[2] / 2.0
        jd = jacobian_determinant_vxm(dvf_px)
        jac_neg = float(np.sum(jd < 0) / jd.size)

        ncc_b_list.append(ncc_b)
        ncc_a_list.append(ncc_a)
        dice_h_list.append(dice_dict['heart']['after'])
        dice_l_list.append(dice_dict['liver']['after'])
        jacneg_list.append(jac_neg)

        csv_rows.append({
            'idx': idx, 'phase': p,
            'pairname': pairnames[0],
            'block_id': int(block_ids[0]), 'slice_id': int(slice_ids[0]),
            'NCC_Before': ncc_b, 'NCC_After': ncc_a, 'NCC_Delta': ncc_a - ncc_b,
            'Dice_Heart': dice_dict['heart']['after'],
            'Dice_Liver': dice_dict['liver']['after'],
            'Jac_Neg_Ratio': jac_neg,
        })

        fig_data.append({'phase': p, 'mov': img_x, 'fix': img_f, 'warp': img_w,
                         'ncc_b': ncc_b, 'ncc_a': ncc_a,
                         'dice_h': dice_dict['heart']['after'],
                         'dice_l': dice_dict['liver']['after']})

    # 打印每条轨迹平均
    print(f"\n[{i+1}/{len(indices)}] {pairnames[0]} "
          f"(block={int(block_ids[0])}, slice={int(slice_ids[0])})")
    print(f"  NCC:    before={np.mean(ncc_b_list):.4f} -> after={np.mean(ncc_a_list):.4f} "
          f"(delta={np.mean(ncc_a_list) - np.mean(ncc_b_list):+.4f})")
    print(f"  DiceH:  after mean={np.mean(dice_h_list):.4f}")
    print(f"  DiceL:  after mean={np.mean(dice_l_list):.4f}")
    print(f"  JacNeg: mean={np.mean(jacneg_list)*100:.2f}%")

    # 画 9 相位轨迹图 (1 行 9 列 moving+fixed+warped 各一组)
    if not opt.no_save_fig:
        fig, axes = plt.subplots(3, 9, figsize=(27, 9))
        for p in range(9):
            d = fig_data[p]
            axes[0, p].imshow(d['mov'], cmap='gray')
            axes[0, p].set_title(f'p{p} M')
            axes[0, p].axis('off')
            axes[1, p].imshow(d['warp'], cmap='gray')
            axes[1, p].set_title(f'p{p} W\nNCC:{d["ncc_a"]:.3f}')
            axes[1, p].axis('off')
            axes[2, p].imshow(d['fix'] - d['warp'], cmap='hot', vmin=-0.3, vmax=0.3)
            axes[2, p].set_title('Diff')
            axes[2, p].axis('off')
        fig.suptitle(
            f"{pairnames[0]}  NCC mean {np.mean(ncc_b_list):.3f}->{np.mean(ncc_a_list):.3f} "
            f"({np.mean(ncc_a_list)-np.mean(ncc_b_list):+.3f})",
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_png = os.path.join(opt.save_dir, f'sample_{i:03d}_{pairnames[0]}.png')
        plt.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_png}")


# ======================== 写 CSV ========================
csv_path = os.path.join(opt.save_dir, f'stats_{opt.split}.csv')
with open(csv_path, 'w', newline='') as f:
    fieldnames = ['idx', 'phase', 'pairname', 'block_id', 'slice_id',
                  'NCC_Before', 'NCC_After', 'NCC_Delta',
                  'Dice_Heart', 'Dice_Liver', 'Jac_Neg_Ratio']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in csv_rows:
        writer.writerow(r)

print(f"\nCSV saved: {csv_path}")
print(f"Total samples: {len(indices)}  total (idx,phase) rows: {len(csv_rows)}")


# ======================== 局部 NCC helper (与可视化脚本一致) ========================
def _local_ncc(fix_np, mov_np, win=15):
    """滑动窗口 NCC, 返回全图平均 (numpy 输入, 值域 [0,1])"""
    import torch.nn.functional as F
    f = torch.from_numpy(fix_np)[None, None].float()
    m = torch.from_numpy(mov_np)[None, None].float()
    pad = win // 2
    fp = F.pad(f, [pad, pad, pad, pad], mode='reflect')
    mp = F.pad(m, [pad, pad, pad, pad], mode='reflect')
    pf = fp.unfold(2, win, 1).unfold(3, win, 1).contiguous().view(1, 1, *fp.shape[-2:], -1)
    pm = mp.unfold(2, win, 1).unfold(3, win, 1).contiguous().view(1, 1, *mp.shape[-2:], -1)
    mf = pf.mean(dim=-1); mm = pm.mean(dim=-1)
    cf = pf - mf.unsqueeze(-1); cm = pm - mm.unsqueeze(-1)
    vf = (cf ** 2).mean(dim=-1); vm = (cm ** 2).mean(dim=-1)
    cross = (cf * cm).mean(dim=-1)
    eps = 1e-8
    ncc = cross / (torch.sqrt(vf.clamp(min=eps)) * torch.sqrt(vm.clamp(min=eps)) + eps)
    return ncc.mean().item()
