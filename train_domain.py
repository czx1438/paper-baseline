"""
Domain-Adaptive Registration Training (心脏→肝脏 生死线实验)

核心思路：
    - 源域（Source）：XCAT 心脏，配准损失 = NCC + smooth
    - 目标域（Target）：SEY 肝脏，配准损失 = NCC + smooth（NCC 自监督）
    - 域对抗：DANN - GRL 翻转梯度，让网络学到域不变特征

Loss = L_ncc_src + L_ncc_tgt + smooth_src + smooth_tgt + w_domain * L_domain

生死线验证：
    心脏训练直接测肝脏 → 几乎不形变（已确认）
    + 域对抗训练后测肝脏 → 有合理形变 → 方案成立！
"""
import os, glob, sys, json, csv
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from natsort import natsorted

import TransModels.LDMMorph as LDMMorph
import TransModels.DomainDiscriminator as DANN
from utils.utils import (
    SpatialTransform, smoothloss, Dataset_epoch_with_name,
    jacobian_determinant_vxm,
)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default
from ldm.data.xcat_Motion_Seq import XCATSeqRegistration
from omegaconf import OmegaConf


# ============================================================
# Argument Parsing
# ============================================================
parser = ArgumentParser()
parser.add_argument("--resume", type=str, default='',
    dest="resume",
    help="Pretrained registration model (optional, starts from LDM only by default)")
parser.add_argument("--ldm_ckpt", type=str,
    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-06-02T18-07-16_xcat-ldm-vq16-64ch/last.ckpt',
    dest="ldm_ckpt",
    help="LDM checkpoint path (must be provided for feature extraction)")
parser.add_argument("--xcat_path", type=str,
    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
    dest="xcat_path",
    help="XCAT data root")
parser.add_argument("--sey_path", type=str,
    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep',
    dest="sey_path",
    help="SEY liver data root (contains train/, val/, test/)")
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--bs", type=int, dest="bs", default=1)
parser.add_argument("--iteration", type=int, dest="iteration", default=24001)
parser.add_argument("--smth_labda", type=float, dest="smth_labda", default=0.4)
parser.add_argument("--w_domain", type=float, dest="w_domain", default=0.1,
    help="Domain adversarial loss weight (default 0.1)")
parser.add_argument("--grl_warmup_iters", type=int, dest="grl_warmup_iters", default=2000,
    help="GRL warmup iterations (lambda grows from 0 to 1)")
parser.add_argument("--checkpoint_freq", type=int, dest="checkpoint_freq", default=5000)
parser.add_argument("--vis_freq", type=int, dest="vis_freq", default=2000)
parser.add_argument("--no_ldm", action="store_true", dest="no_ldm",
    help="Disable LDM features, use CNN-only")
parser.add_argument("--ldm_config", type=str,
    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_no_motion.yaml',
    dest="ldm_config",
    help="LDM config file")
parser.add_argument("--domain_mode", type=str, default='fixed_motion',
    choices=['fixed_motion', 'xcat'],
    dest="domain_mode",
    help="XCAT data loading mode")
parser.add_argument("--beta", type=float, dest="beta", default=0.8,
    help="NCC vs latent MSE weight: beta*NCC + (1-beta)*MSE")
parser.add_argument("--loss_type", type=str, default='ncc',
    choices=['ncc', 'mse'], dest="loss_type")
parser.add_argument("--save_dir", type=str,
    default=None, dest="save_dir")
opt = parser.parse_args()


# ============================================================
# Defaults & Hyperparameters
# ============================================================
lr          = opt.lr
bs          = opt.bs
iteration   = opt.iteration
smooth_w    = opt.smth_labda
w_domain    = opt.w_domain
grl_warmup  = opt.grl_warmup_iters
t_enc       = 1
beta        = opt.beta
save_root   = opt.save_dir or f'./logs/DA_XCAT2SEY_wd{w_domain}_smooth{smooth_w}_grl{grl_warmup}'
model_dir   = os.path.join(save_root, 'checkpoints')
vis_dir     = os.path.join(save_root, 'visualizations')
csv_path    = os.path.join(save_root, 'training_log.csv')

for d in [model_dir, vis_dir]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# LDM Loading
# ============================================================
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_lgm():
    """Load pretrained LDM. Returns None when --no_ldm is set so callers can
    short-circuit feature extraction entirely."""
    if opt.no_ldm:
        print("[LDM] --no_ldm set, skipping LDM load (saves ~5s and ~1GB)")
        return None
    configs_list = [OmegaConf.load(opt.ldm_config)]
    cli = OmegaConf.from_dotlist([])
    configs = OmegaConf.merge(*configs_list, cli)
    if opt.resume and opt.resume.endswith('.ckpt') and 'ldm' not in opt.resume:
        pl_sd = {"state_dict": None}
    else:
        pl_sd = torch.load(opt.ldm_ckpt, map_location="cpu")
    ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
    print(f"[LDM] loaded from {opt.ldm_ckpt}")
    return ldm_model


# ============================================================
# Data Loaders
# ============================================================
def build_loaders():
    # --- Source: XCAT heart ---
    if opt.domain_mode == 'fixed_motion':
        src_train = XCATSeqRegistration(data_root=opt.xcat_path, split='train', flip_p=0.5)
        src_val   = XCATSeqRegistration(data_root=opt.xcat_path, split='val',   flip_p=0.0)
    else:
        from utils.utils import Dataset_XCAT_Registration
        src_train = Dataset_XCAT_Registration(data_root=opt.xcat_path, split='train', flip_p=0.5)
        src_val   = Dataset_XCAT_Registration(data_root=opt.xcat_path, split='val',   flip_p=0.0)

    src_train_loader = Data.DataLoader(src_train, batch_size=bs, shuffle=True,  num_workers=0, drop_last=True)
    src_val_loader   = Data.DataLoader(src_val,   batch_size=bs, shuffle=False, num_workers=0)

    # --- Target: SEY liver ---
    sey_train_dir = os.path.join(opt.sey_path, 'train')
    sey_val_dir   = os.path.join(opt.sey_path, 'val')
    sey_test_dir  = os.path.join(opt.sey_path, 'test')

    sey_train_paths = natsorted(glob.glob(os.path.join(sey_train_dir, '*.npz')))
    sey_val_paths   = natsorted(glob.glob(os.path.join(sey_val_dir,   '*.npz')))
    sey_test_paths  = natsorted(glob.glob(os.path.join(sey_test_dir,  '*.npz')))

    tgt_train_loader = Data.DataLoader(
        Dataset_epoch_with_name(sey_train_paths),
        batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    tgt_val_loader   = Data.DataLoader(
        Dataset_epoch_with_name(sey_val_paths),
        batch_size=bs, shuffle=False, num_workers=0)
    tgt_test_loader  = Data.DataLoader(
        Dataset_epoch_with_name(sey_test_paths),
        batch_size=bs, shuffle=False, num_workers=0)

    print(f"[Data] Source: {len(src_train)} train, {len(src_val)} val")
    print(f"[Data] Target: {len(tgt_train_loader.dataset)} train, {len(tgt_val_loader.dataset)} val, {len(tgt_test_loader.dataset)} test")
    return src_train_loader, src_val_loader, tgt_train_loader, tgt_val_loader, tgt_test_loader


# ============================================================
# LDM Feature Extraction Helper
# ============================================================
# NOTE: get_ldm_features was removed because it referenced an undefined
# variable `y_noisy`. The correct paired version below is used everywhere.


def get_ldm_scores_pair(img_mov, img_fix):
    """Extract paired LDM features for both moving and fixed.

    Returns None x4 when --no_ldm is set so callers can short-circuit.
    """
    if opt.no_ldm:
        return None, None, None, None
    mov_z  = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(img_mov)).detach()
    fix_z  = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(img_fix)).detach()
    noise  = torch.randn_like(mov_z)
    x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
    y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
    outx = ldm_model.apply_model(x_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
    outy = ldm_model.apply_model(y_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
    score0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
    score1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
    score2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
    score3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]), dim=1)
    return score0, score1, score2, score3


# ============================================================
# Model Initialization
# ============================================================
def build_models():
    reg_model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2, use_ldm=not opt.no_ldm)
    reg_model.cuda()

    domain_disc = DANN.DomainAdversarialModule(in_channels=32)
    domain_disc.cuda()

    if opt.resume and os.path.isfile(opt.resume):
        sd = torch.load(opt.resume, map_location="cuda")
        reg_model.load_state_dict(sd)
        print(f"[RegModel] Loaded from {opt.resume}")

    total_reg = sum(p.nelement() for p in reg_model.parameters()) / 1e6
    total_dom = sum(p.nelement() for p in domain_disc.parameters()) / 1e6
    print(f"[Model] RegModel: {total_reg:.2f}M params | DomainDisc: {total_dom:.2f}M params")
    return reg_model, domain_disc


# ============================================================
# NCC Loss
# ============================================================
def ncc(fixed, moving, win_size=9):
    assert fixed.shape == moving.shape
    b, c, h, w = fixed.shape
    pad = win_size // 2
    fp = F.pad(fixed,  [pad, pad, pad, pad], mode='reflect')
    mp = F.pad(moving, [pad, pad, pad, pad], mode='reflect')
    pf = fp.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    pm = mp.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(b, c, h, w, -1)
    mf = pf.mean(-1); mm = pm.mean(-1)
    cf = pf - mf.unsqueeze(-1); cm = pm - mm.unsqueeze(-1)
    vf = (cf ** 2).mean(-1); vm = (cm ** 2).mean(-1)
    cross = (cf * cm).mean(-1)
    eps = 1e-8
    ncc_val = cross / (torch.sqrt(vf.clamp(min=eps)) * torch.sqrt(vm.clamp(min=eps)) + eps)
    return 1.0 - ncc_val.mean()


# ============================================================
# GRL Lambda Schedule (linear warmup 0 -> 1)
# ============================================================
def get_grl_lambda(step):
    return min(1.0, step / grl_warmup)


# ============================================================
# Compute Registration Loss
# ============================================================
def reg_loss(mov_img, fix_img):
    """Compute NCC + smooth loss for a moving/fixed pair."""
    s0, s1, s2, s3 = get_ldm_scores_pair(mov_img, fix_img)
    disp, feat = reg_model(mov_img, fix_img, s0, s1, s2, s3)  # disp: [B,2,H,W], feat: [B,32,H,W]
    _, warped = transform(mov_img, disp.permute(0, 2, 3, 1))
    loss_ncc = ncc(fix_img, warped)
    loss_smooth = smoothloss(disp)
    return loss_ncc, loss_smooth, disp, feat, warped


# ============================================================
# Domain Loss  (Standard DANN: two separate backward passes)
# ============================================================
# NOTE: domain_adv_loss was rewritten inline as two separate paths inside
# the training loop (see "Backward step 1" / "Backward step 2"). The
# implementation here uses two forward passes of the discriminator to
# avoid PyTorch's inplace-versioning error when stepping the optimizer
# between two backward passes through the same graph.


# ============================================================
# Save Checkpoint
# ============================================================
def save_ckpt(path, reg_sd, dom_sd, step):
    torch.save({**reg_sd, **{f'domain_{k}': v for k, v in dom_sd.items()}, 'step': step}, path)

def load_ckpt(path):
    sd = torch.load(path, map_location="cuda")
    return sd


# ============================================================
# Visualization
# ============================================================
def vis_sample(step, mov, fix, warped, disp, domain_tag,
               ncc_before, ncc_after, save_path):
    mov_np  = mov[0, 0].cpu().numpy()
    fix_np  = fix[0, 0].cpu().numpy()
    warp_np = warped[0, 0].cpu().numpy()
    disp_np = disp[0].cpu().numpy()

    D_disp = disp[0].cpu().numpy()                            # [2, H, W] (channel-first, like train_mask.py)
    disp_hw2 = D_disp                                         # vxm util expects (C, H, W) and transposes internally
    jac = jacobian_determinant_vxm(disp_hw2)
    n_fold = int((jac < 0).sum())
    min_jac = float(jac.min())

    diff_before = np.abs(mov_np - fix_np)
    diff_after  = np.abs(warp_np - fix_np)
    dvf_mag = np.sqrt(D_disp[0]**2 + D_disp[1]**2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    imgs = [mov_np, fix_np, warp_np, diff_before,
            diff_after, jac, dvf_mag,
            np.abs(mov_np - fix_np) - np.abs(warp_np - fix_np)]
    titles = [
        f'Moving\n[domain={domain_tag}]',
        'Fixed (Ref)',
        f'Warped\nNCC: {ncc_before:.4f}->{ncc_after:.4f}',
        'Abs Diff Before',
        'Abs Diff After',
        f'Jac Det [folds={n_fold}, min={min_jac:.3f}]',
        'DVF Magnitude',
        'Diff Reduction'
    ]
    for ax, img, title in zip(axes.flat, imgs, titles):
        if 'Jac' in title or 'DVF' in title or 'Reduction' in title:
            ax.imshow(img, cmap='hot', vmin=0, vmax=0.2 if 'Reduction' not in title else None)
        elif 'Jac' in title:
            ax.imshow(img, cmap='RdBu', vmin=-0.5, vmax=1.5)
        else:
            vmax = max(mov_np.max(), fix_np.max(), warp_np.max())
            ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    fig.suptitle(f'[Step {step}] {domain_tag} | NCC: {ncc_before:.4f}->{ncc_after:.4f} | Disp range: [{disp.min():.3f}, {disp.max():.3f}]', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main Training Loop
# ============================================================
def train():
    global ldm_model, reg_model, domain_disc, transform

    ldm_model, reg_model, domain_disc = None, None, None
    ldm_model = load_lgm()
    reg_model, domain_disc = build_models()
    transform = SpatialTransform().cuda()
    for p in transform.parameters():
        p.requires_grad = False

    src_train_l, src_val_l, tgt_train_l, tgt_val_l, tgt_test_l = build_loaders()

    src_train_iter = iter(src_train_l)
    tgt_train_iter = iter(tgt_train_l)

    optimizer_reg = torch.optim.Adam(reg_model.parameters(), lr=lr)
    optimizer_dom = torch.optim.Adam(domain_disc.parameters(), lr=lr * 0.5)

    lossall = []
    step = 1

    # CSV header
    f_csv = open(csv_path, 'w')
    csv_writer = csv.DictWriter(f_csv, fieldnames=[
        'step', 'loss_total', 'ncc_src', 'ncc_tgt', 'smooth_src', 'smooth_tgt',
        'loss_domain', 'grl_lambda', 'ncc_val_src', 'ncc_val_tgt',
        'ncc_test_tgt', 'disp_mean_src', 'disp_mean_tgt'
    ])
    csv_writer.writeheader()
    f_csv.close()

    while step <= iteration:
        reg_model.train()
        domain_disc.train()

        # ---- Get source batch (XCAT) ----
        try:
            sx, sy, _, _, _ = next(src_train_iter)
        except StopIteration:
            src_train_iter = iter(src_train_l)
            sx, sy, _, _, _ = next(src_train_iter)
        sx = sx.cuda().float()
        sy = sy.cuda().float()

        # ---- Get target batch (SEY) ----
        try:
            tx, ty, _, _, _ = next(tgt_train_iter)
        except StopIteration:
            tgt_train_iter = iter(tgt_train_l)
            tx, ty, _, _, _ = next(tgt_train_iter)
        tx = tx.cuda().float()
        ty = ty.cuda().float()

        # ---- GRL lambda (linear warmup 0 -> 1) ----
        lam = get_grl_lambda(step)
        domain_disc.set_lambda(lam)

        # ---- Forward: Source domain ----
        s0, s1, s2, s3 = get_ldm_scores_pair(sx, sy)
        disp_src, feat_src = reg_model(sx, sy, s0, s1, s2, s3)
        _, warped_src = transform(sx, disp_src.permute(0, 2, 3, 1))
        ncc_src = ncc(sy, warped_src)
        sm_src  = smoothloss(disp_src)

        # ---- Forward: Target domain ----
        t0, t1, t2, t3 = get_ldm_scores_pair(tx, ty)
        disp_tgt, feat_tgt = reg_model(tx, ty, t0, t1, t2, t3)
        _, warped_tgt = transform(tx, disp_tgt.permute(0, 2, 3, 1))
        ncc_tgt = ncc(ty, warped_tgt)
        sm_tgt  = smoothloss(disp_tgt)

        # ---- Domain adversarial loss (computed once, used twice) ----
        # Source (XCAT) -> domain 0, Target (SEY) -> target 1
        # Implementation note: two SEPARATE forward passes through the
        # discriminator (one for D-update without GRL, one for reg-update
        # with GRL). This avoids inplace-versioning errors that arise when
        # stepping optimizer_dom in between two backward passes sharing
        # the same discriminator forward graph.

        # ---- Backward step 1: DISCRIMINATOR update (no GRL) ----
        # D sees real features, gets normal gradient, learns to classify correctly.
        # Re-forward the discriminator so its computation graph is fresh after step().
        optimizer_dom.zero_grad()
        feat_src_d = feat_src.detach(); feat_src_d.requires_grad_(True)
        feat_tgt_d = feat_tgt.detach(); feat_tgt_d.requires_grad_(True)
        logits_src_d = domain_disc.disc_forward(feat_src_d)
        logits_tgt_d = domain_disc.disc_forward(feat_tgt_d)
        d_src = torch.zeros_like(logits_src_d)
        d_tgt = torch.ones_like(logits_tgt_d)
        loss_dom_disc = (F.binary_cross_entropy_with_logits(logits_src_d, d_src) +
                         F.binary_cross_entropy_with_logits(logits_tgt_d, d_tgt)) / 2
        loss_dom_disc.backward()
        optimizer_dom.step()

        # ---- Backward step 2: REGISTRATION-NETWORK update ----
        # Task loss + GRL(domain loss): GRL flips the gradient that flows back to
        # reg params, so reg learns to produce domain-invariant features.
        # Re-forward the discriminator (with GRL) so its graph is fresh after step().
        optimizer_reg.zero_grad()
        logits_src_r = domain_disc.reg_forward(feat_src)
        logits_tgt_r = domain_disc.reg_forward(feat_tgt)
        d_src_r = torch.zeros_like(logits_src_r)
        d_tgt_r = torch.ones_like(logits_tgt_r)
        loss_dom_reg = (F.binary_cross_entropy_with_logits(logits_src_r, d_src_r) +
                        F.binary_cross_entropy_with_logits(logits_tgt_r, d_tgt_r)) / 2
        loss_reg_total = ncc_src + ncc_tgt + sm_src + sm_tgt + w_domain * loss_dom_reg
        loss_reg_total.backward()
        optimizer_reg.step()

        # Display value: use the disc-side loss (real BCE on the discriminator path)
        loss_dom = loss_dom_disc.detach()

        # ---- Logging ----
        lossall.append({
            'step': step, 'ncc_src': ncc_src.item(), 'ncc_tgt': ncc_tgt.item(),
            'sm_src': sm_src.item(), 'sm_tgt': sm_tgt.item(),
            'loss_dom': loss_dom.item(), 'lam': lam,
            'loss_total': loss_reg_total.item()
        })

        sys.stdout.write("\r[Step {0}] L={1:.4f} NCC_src={2:.4f} NCC_tgt={3:.4f} "
                         "Sm_src={4:.4f} Sm_tgt={5:.4f} L_dom={6:.4f} λ={7:.3f} "
                         "Disp_src={8:.4f} Disp_tgt={9:.4f}".format(
            step, loss_reg_total.item(), ncc_src.item(), ncc_tgt.item(),
            sm_src.item(), sm_tgt.item(), loss_dom.item(), lam,
            disp_src.abs().mean().item(), disp_tgt.abs().mean().item()))
        sys.stdout.flush()

        # ---- Visualization ----
        if step % opt.vis_freq == 0:
            with torch.no_grad():
                # Source sample
                vis_sample(step, sx, sy, warped_src, disp_src, 'XCAT',
                           1 - ncc_src.item(), 1 - ncc_src.item(),
                           os.path.join(vis_dir, f'vis_src_{step:06d}.png'))
                # Target sample
                ncc_bef_tgt = 1 - ncc(ty, tx).item()
                ncc_aft_tgt = 1 - ncc(ty, warped_tgt).item()
                vis_sample(step, tx, ty, warped_tgt, disp_tgt, 'SEY',
                           ncc_bef_tgt, ncc_aft_tgt,
                           os.path.join(vis_dir, f'vis_tgt_{step:06d}.png'))

        # ---- Validation ----
        if step % opt.checkpoint_freq == 0:
            reg_model.eval()
            domain_disc.eval()
            with torch.no_grad():
                # Source val
                ncc_src_vals = []
                for xv, yv, _, _, _ in src_val_l:
                    xv, yv = xv.cuda().float(), yv.cuda().float()
                    s0, s1, s2, s3 = get_ldm_scores_pair(xv, yv)
                    d, _ = reg_model(xv, yv, s0, s1, s2, s3)
                    _, w = transform(xv, d.permute(0, 2, 3, 1))
                    ncc_src_vals.append(1 - ncc(yv, w).item())

                # Target val
                ncc_tgt_vals_bef = []
                ncc_tgt_vals_aft = []
                for xv, yv, _, _, _ in tgt_val_l:
                    xv, yv = xv.cuda().float(), yv.cuda().float()
                    ncc_bef = 1 - ncc(yv, xv).item()
                    s0, s1, s2, s3 = get_ldm_scores_pair(xv, yv)
                    d, _ = reg_model(xv, yv, s0, s1, s2, s3)
                    _, w = transform(xv, d.permute(0, 2, 3, 1))
                    ncc_aft = 1 - ncc(yv, w).item()
                    ncc_tgt_vals_bef.append(ncc_bef)
                    ncc_tgt_vals_aft.append(ncc_aft)

                # Target test
                ncc_test_bef = []
                ncc_test_aft = []
                for xv, yv, _, _, _ in tgt_test_l:
                    xv, yv = xv.cuda().float(), yv.cuda().float()
                    ncc_bef = 1 - ncc(yv, xv).item()
                    s0, s1, s2, s3 = get_ldm_scores_pair(xv, yv)
                    d, _ = reg_model(xv, yv, s0, s1, s2, s3)
                    _, w = transform(xv, d.permute(0, 2, 3, 1))
                    ncc_aft = 1 - ncc(yv, w).item()
                    ncc_test_bef.append(ncc_bef)
                    ncc_test_aft.append(ncc_aft)

                mean_src_val = np.mean(ncc_src_vals)
                mean_tgt_bef = np.mean(ncc_tgt_vals_bef)
                mean_tgt_aft = np.mean(ncc_tgt_vals_aft)
                mean_test_bef = np.mean(ncc_test_bef)
                mean_test_aft = np.mean(ncc_test_aft)

            print(f"\n  [Val @ Step {step}]")
            print(f"    XCAT val  NCC: {mean_src_val:.4f}")
            print(f"    SEY  val  NCC: {mean_tgt_bef:.4f} -> {mean_tgt_aft:.4f} (Δ {mean_tgt_aft - mean_tgt_bef:+.4f})")
            print(f"    SEY  test NCC: {mean_test_bef:.4f} -> {mean_test_aft:.4f} (Δ {mean_test_aft - mean_test_bef:+.4f})")
            print(f"    GRL lambda: {lam:.3f}")

            # Save checkpoint
            ckpt_path = os.path.join(model_dir, f'step_{step:06d}.pth')
            save_ckpt(ckpt_path, reg_model.state_dict(), domain_disc.state_dict(), step)
            print(f"    Saved: {ckpt_path}")

            # CSV
            f_csv = open(csv_path, 'a')
            csv_writer = csv.DictWriter(f_csv, fieldnames=[
                'step', 'loss_total', 'ncc_src', 'ncc_tgt', 'smooth_src', 'smooth_tgt',
                'loss_domain', 'grl_lambda', 'ncc_val_src', 'ncc_val_tgt',
                'ncc_test_tgt', 'disp_mean_src', 'disp_mean_tgt'
            ])
            csv_writer.writerow({
                'step': step, 'loss_total': loss_reg_total.item(),
                'ncc_src': ncc_src.item(), 'ncc_tgt': ncc_tgt.item(),
                'smooth_src': sm_src.item(), 'smooth_tgt': sm_tgt.item(),
                'loss_domain': loss_dom.item(), 'grl_lambda': lam,
                'ncc_val_src': mean_src_val,
                'ncc_val_tgt': f'{mean_tgt_bef:.4f}->{mean_tgt_aft:.4f}',
                'ncc_test_tgt': f'{mean_test_bef:.4f}->{mean_test_aft:.4f}',
                'disp_mean_src': disp_src.abs().mean().item(),
                'disp_mean_tgt': disp_tgt.abs().mean().item(),
            })
            f_csv.close()

            reg_model.train()
            domain_disc.train()

        step += 1
        if step > iteration:
            break

    print(f"\nTraining done! Logs: {csv_path}")
    print(f"Models: {model_dir}")


if __name__ == '__main__':
    train()
