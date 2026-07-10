"""
train_temporal_aware.py
========================================================
Training Script for Temporal-Aware LDM-Morph.

Integrates all four contributions:
  1. Temporal Transformer on LDM Features (temporal_transformer.py)
  2. Physics-Informed Temporal Losses (temporal_losses.py)
  3. Spatially-Varying Adaptive Gate (temporal_transformer.py)
  4. Bidirectional Temporal Propagation (temporal_transformer.py)

Usage:
  python train_temporal_aware.py \
      --ldm_ckpt /path/to/ldm/checkpoint.ckpt \
      --xcat_path /path/to/cardiac/phases \
      --context_size 5 \
      --num_phases 5 \
      --temporal_proj_dim 256 \
      --num_temporal_layers 2 \
      --use_bidirectional \
      --lambda_1st 1.0 \
      --lambda_2nd 0.5 \
      --lambda_cycle 0.1 \
      --lr 1e-4 \
      --iteration 24001 \
      --checkpoint 5000

Required data structure:
    xcat_path/
    ├── fixed/fixed/phase0/*.npy   ← Reference phase
    └── moving/moving/
        ├── phase1/*.npy
        ├── phase2/*.npy
        ├── phase3/*.npy
        └── ... (up to phase 9)

If data is in legacy format (fixed/fixed/*.npy + moving/moving/*.npy),
the script will automatically use single-phase mode (ablation baseline).
"""

import os
import sys
import glob
import csv
import json
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from natsort import natsorted
from omegaconf import OmegaConf

# Import LDM infrastructure
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# Import temporal-aware components
from src.temporal import (
    TemporalAwareLDMMorph,
    LDMFeatureExtractor,
    MultiPhaseSequenceDataset,
    PhysicsInformedTemporalLoss,
    temporal_smoothness_loss_1st,
    temporal_smoothness_loss_2nd,
    cyclic_consistency_loss,
    negative_jacobian_loss,
    gate_sparsity_loss,
)
from utils.utils import (
    SpatialTransform,
    smoothloss,
    ncc_loss,
    MSE,
)


# =============================================================================
# Argument Parser
# =============================================================================

parser = ArgumentParser()

# LDM checkpoint
parser.add_argument("--resume", type=str,
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-30T21-02-35_xcat-motion-ldm/checkpoints/last.ckpt',
                    dest="resume",
                    help="Pretrained LDM checkpoint path")
parser.add_argument("--ldm_config", type=str,
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_motion-ldm.yaml',
                    dest="ldm_config",
                    help="LDM config YAML path")

# Training hyperparameters
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int, dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int, dest="iteration", default=24001,
                    help="number of total iterations")
parser.add_argument("--checkpoint", type=int, dest="checkpoint", default=5000,
                    help="frequency of saving models")

# Temporal model hyperparameters — Contributions 1, 3, 4
parser.add_argument("--context_size", type=int, dest="context_size", default=5,
                    help="Number of phases in temporal window (must be odd)")
parser.add_argument("--num_phases", type=int, dest="num_phases", default=5,
                    help="Number of phases to use in the sequence (incl. fixed at index 0)")
parser.add_argument("--temporal_proj_dim", type=int, dest="temporal_proj_dim", default=256,
                    help="Internal dimension of temporal transformer")
parser.add_argument("--num_temporal_layers", type=int, dest="num_temporal_layers", default=2,
                    help="Number of temporal encoder layers per scale")
parser.add_argument("--temporal_num_heads", type=int, dest="temporal_num_heads", default=4,
                    help="Number of attention heads in temporal transformer")
parser.add_argument("--no_bidirectional", action="store_true", dest="no_bidirectional",
                    help="Disable bidirectional temporal propagation (Contribution 4)")
parser.add_argument("--fuse_mode", type=str, dest="fuse_mode", default='concat',
                    choices=['concat', 'add'],
                    help="Fusion mode for bidirectional: concat (recommended) or add")

# Physics-informed loss weights — Contribution 2
parser.add_argument("--lambda_1st", type=float, dest="lambda_1st", default=1.0,
                    help="Weight for 1st-order temporal smoothness loss")
parser.add_argument("--lambda_2nd", type=float, dest="lambda_2nd", default=0.5,
                    help="Weight for 2nd-order temporal smoothness loss")
parser.add_argument("--lambda_cycle", type=float, dest="lambda_cycle", default=0.1,
                    help="Weight for cyclic consistency loss")
parser.add_argument("--lambda_neg_J", type=float, dest="lambda_neg_J", default=0.1,
                    help="Weight for negative Jacobian (anti-folding) loss")
parser.add_argument("--lambda_gate", type=float, dest="lambda_gate", default=0.001,
                    help="Weight for gate sparsity loss")
parser.add_argument("--smooth", type=float, dest="smooth", default=0.001,
                    help="Smoothness loss weight (displacement field regularization)")
parser.add_argument("--beta", type=float, dest="beta", default=0.85,
                    help="Balance between NCC and latent MSE (0-1)")

# Data
parser.add_argument("--xcat_path", type=str,
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    dest="xcat_path",
                    help="Data root for cardiac phases")
parser.add_argument("--flip_p", type=float, dest="flip_p", default=0.5,
                    help="Probability of random horizontal flip")

opt = parser.parse_args()
use_bidirectional = not opt.no_bidirectional

print("=" * 60)
print("Temporal-Aware LDM-Morph Training")
print("=" * 60)
print(f"  Resume LDM from: {opt.resume}")
print(f"  Context size:    {opt.context_size} phases")
print(f"  Total phases:    {opt.num_phases}")
print(f"  Bidirectional:   {use_bidirectional}")
print(f"  Temporal proj:   {opt.temporal_proj_dim} dim")
print(f"  Temporal layers: {opt.num_temporal_layers}")
print(f"  λ_1st={opt.lambda_1st}, λ_2nd={opt.lambda_2nd}, "
      f"λ_cycle={opt.lambda_cycle}, λ_neg_J={opt.lambda_neg_J}")
print(f"  Loss β={opt.beta}, smooth={opt.smooth}")
print("=" * 60)


# =============================================================================
# LDM Model Loading
# =============================================================================

def load_ldm_model(config_path: str, ckpt_path: str):
    """Load pretrained LDM model for feature extraction."""
    configs_list = [OmegaConf.load(config_path)]
    cli = OmegaConf.from_dotlist([])
    configs = OmegaConf.merge(*configs_list, cli)

    print(f"Loading LDM from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(configs.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("LDM loaded successfully")
    return model, configs


# =============================================================================
# LDM Feature Extractor
# =============================================================================

class LDMFeatureExtractor(torch.nn.Module):
    """
    Extract multi-scale LDM features from a batch of images.
    Shared across all phases for consistent temporal feature extraction.
    """
    def __init__(self, ldm_model, t_enc: int = 1):
        super().__init__()
        self.ldm = ldm_model
        self.t_enc = t_enc

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None):
        """
        x: [B, 1, H, W]
        noise: optional shared noise [B, 1, H/8, W/8]
        returns: (score0, score1, score2, score3)
        """
        z = self.ldm.get_first_stage_encoding(self.ldm.encode_first_stage(x))
        if noise is None:
            noise = torch.randn_like(z)
        z_noisy = self.ldm.q_sample(
            x_start=z,
            t=torch.tensor([self.t_enc], device=z.device),
            noise=noise,
        )
        out = self.ldm.apply_model(
            z_noisy,
            t=torch.tensor([self.t_enc], device=z.device),
            cond=None,
            return_ids=True,
        )
        skip_feats = out[1][0]

        score0 = torch.cat([skip_feats[0], skip_feats[2]], dim=1)
        score1 = torch.cat([skip_feats[3], skip_feats[5]], dim=1)
        score2 = torch.cat([skip_feats[6], skip_feats[8]], dim=1)
        score3 = torch.cat([skip_feats[9], skip_feats[11]], dim=1)
        return score0, score1, score2, score3


# =============================================================================
# Multi-Phase Feature Extraction (with Temporal Transformer)
# =============================================================================

def extract_temporal_features(
    extractor: LDMFeatureExtractor,
    phase_sequence: torch.Tensor,
    temporal_fusion_modules: list,
    raw_center_scores: tuple,
) -> tuple:
    """
    Extract temporally-enhanced features for all phases in the sequence.

    Args:
        extractor: LDMFeatureExtractor
        phase_sequence: [B, num_phases, 1, H, W]
        temporal_fusion_modules: list of 4 TemporalAwareFeatureFusion modules
        raw_center_scores: tuple of 4 raw center-phase score tensors

    Returns:
        enhanced_score0-3: temporally-enhanced features for center phase
    """
    B, num_phases, C, H, W = phase_sequence.shape

    # Extract LDM features for all phases
    all_s0, all_s1, all_s2, all_s3 = [], [], [], []
    for p in range(num_phases):
        s0, s1, s2, s3 = extractor(phase_sequence[:, p])
        all_s0.append(s0)
        all_s1.append(s1)
        all_s2.append(s2)
        all_s3.append(s3)

    # Apply temporal fusion on each scale
    enhanced_s0 = temporal_fusion_modules[0](
        torch.stack(all_s0, dim=1), raw_center_scores[0])
    enhanced_s1 = temporal_fusion_modules[1](
        torch.stack(all_s1, dim=1), raw_center_scores[1])
    enhanced_s2 = temporal_fusion_modules[2](
        torch.stack(all_s2, dim=1), raw_center_scores[2])
    enhanced_s3 = temporal_fusion_modules[3](
        torch.stack(all_s3, dim=1), raw_center_scores[3])

    return enhanced_s0, enhanced_s1, enhanced_s2, enhanced_s3


# =============================================================================
# Multi-Phase Forward with Shared Encoder
# =============================================================================

class TemporalAwareLDMMorphWrapper(nn.Module):
    """
    Full Temporal-Aware LDM-Morph with integrated LDM feature extraction.

    This wrapper manages:
      1. Shared LDM feature extraction across all phases
      2. Temporal fusion with bidirectional attention + spatially-varying gate
      3. Full registration forward pass

    Compatible with the original train.py loss interface.
    """
    def __init__(
        self,
        ldm_model,
        channel_1=640,
        channel_2=1024,
        channel_3=1536,
        channel_4=1920,
        num_phases=5,
        context_size=5,
        temporal_proj_dim=256,
        num_temporal_layers=2,
        temporal_num_heads=4,
        use_bidirectional=True,
        fuse_mode='concat',
    ):
        super().__init__()
        self.num_phases = num_phases
        self.context_size = context_size

        # LDM Feature Extractor (shared, frozen)
        self.ldm_extractor = LDMFeatureExtractor(ldm_model)
        for param in self.ldm_extractor.parameters():
            param.requires_grad = False

        # Build Temporal-Aware LDMMorph
        self.reg_model = TemporalAwareLDMMorph(
            channel_1=channel_1,
            channel_2=channel_2,
            channel_3=channel_3,
            channel_4=channel_4,
            num_phases=num_phases,
            temporal_proj_dim=temporal_proj_dim,
            num_temporal_layers=num_temporal_layers,
            use_bidirectional=use_bidirectional,
            temporal_num_heads=temporal_num_heads,
            fuse_mode=fuse_mode,
        )

        # Spatial transform for warping
        self.spatial_transform = SpatialTransform()

    def forward(
        self,
        fixed: torch.Tensor,
        phase_sequence: torch.Tensor,
        extract_ldm_features: bool = True,
    ) -> dict:
        """
        Args:
            fixed: [B, 1, H, W] — reference phase (phase 0)
            phase_sequence: [B, num_phases, 1, H, W] — all phases
                           phase_sequence[:, 0, ...] = fixed
            extract_ldm_features: if True, extract LDM features first

        Returns:
            dict with:
                'disp_field': [B, 2, H, W] — displacement for phase 1 → 0
                'all_displacements': [num_ph-1, B, 2, H, W]
                'warped_image': [B, 1, H, W] — warped moving image
        """
        B, num_ph, C, H, W = phase_sequence.shape

        # Extract LDM features for all phases
        noise = None  # Will be generated once and shared
        z = self.ldm_extractor.ldm.get_first_stage_encoding(
            self.ldm_extractor.ldm.encode_first_stage(phase_sequence[:, 0, :, :, :]))
        noise = torch.randn_like(z)

        all_s0, all_s1, all_s2, all_s3 = [], [], [], []
        for p in range(num_ph):
            s0, s1, s2, s3 = self.ldm_extractor(phase_sequence[:, p, :, :, :], noise=noise)
            all_s0.append(s0)
            all_s1.append(s1)
            all_s2.append(s2)
            all_s3.append(s3)

        # Stack: [num_phases, B, C, H, W] → [B, num_phases, C, H, W]
        stacked_s0 = torch.stack(all_s0, dim=1)
        stacked_s1 = torch.stack(all_s1, dim=1)
        stacked_s2 = torch.stack(all_s2, dim=1)
        stacked_s3 = torch.stack(all_s3, dim=1)

        # Apply temporal fusion (center phase = phase 0 = fixed)
        center_idx = 0
        raw_center_s0 = all_s0[center_idx]
        raw_center_s1 = all_s1[center_idx]
        raw_center_s2 = all_s2[center_idx]
        raw_center_s3 = all_s3[center_idx]

        enhanced_s0 = self.reg_model.temporal_fusion_1(stacked_s0, raw_center_s0)
        enhanced_s1 = self.reg_model.temporal_fusion_2(stacked_s1, raw_center_s1)
        enhanced_s2 = self.reg_model.temporal_fusion_3(stacked_s2, raw_center_s2)
        enhanced_s3 = self.reg_model.temporal_fusion_4(stacked_s3, raw_center_s3)

        # -----------------------------------------------------------------
        # Standard LDMMorph forward using enhanced features
        # -----------------------------------------------------------------
        # CNN adjusters
        cnn_f4 = self.reg_model.c1(enhanced_s0)
        cnn_f8 = self.reg_model.c2(enhanced_s1)
        cnn_f16 = self.reg_model.c3(enhanced_s2)
        cnn_f32 = self.reg_model.c4(enhanced_s3)

        # Use phase 1 as moving (first non-fixed phase)
        moving_img = phase_sequence[:, 1, :, :, :]   # [B, 1, H, W]
        fixed_img = fixed                                  # [B, 1, H, W]

        # LWSA
        img_pair = torch.cat([moving_img, fixed_img], dim=1)
        swin_f4, swin_f8, swin_f16, swin_f32 = self.reg_model.lwsa(img_pair)

        # LWCA
        mov_f4_c = self.reg_model.lwca1(swin_f4, cnn_f4)
        mov_f8_c = self.reg_model.lwca2(swin_f8, cnn_f8)
        mov_f16_c = self.reg_model.lwca3(swin_f16, cnn_f16)
        mov_f32_c = self.reg_model.lwca4(swin_f32, cnn_f32)

        fix_f4_c = self.reg_model.lwca1(cnn_f4, swin_f4)
        fix_f8_c = self.reg_model.lwca2(cnn_f8, swin_f8)
        fix_f16_c = self.reg_model.lwca3(cnn_f16, swin_f16)

        # Decoder
        input_fusion = torch.cat([moving_img, fixed_img], dim=1)
        x_s1 = self.reg_model.avg_pool(input_fusion)
        f4 = self.reg_model.ec1(x_s1)

        x = self.reg_model.up0(mov_f32_c, mov_f16_c, fix_f16_c)
        x = self.reg_model.up1(x, mov_f8_c, fix_f8_c)
        x = self.reg_model.up2(x, mov_f4_c, fix_f4_c)
        x = self.reg_model.up3(x, f4)
        x = self.reg_model.up(x)
        disp_field = self.reg_model.reg_head(x)  # [B, 2, H, W]

        # Warp moving image
        _, warped = self.spatial_transform(moving_img, disp_field.permute(0, 2, 3, 1))

        # Extract all displacement fields (for temporal losses)
        all_displacements = []
        for p in range(1, num_ph):
            mp = phase_sequence[:, p, :, :, :]
            mp_pair = torch.cat([mp, fixed_img], dim=1)
            sf4, sf8, sf16, sf32 = self.reg_model.lwsa(mp_pair)
            cf4 = self.reg_model.c1(all_s0[p])
            cf8 = self.reg_model.c2(all_s1[p])
            cf16 = self.reg_model.c3(all_s2[p])
            cf32 = self.reg_model.c4(all_s3[p])
            mf4 = self.reg_model.lwca1(sf4, cf4)
            mf8 = self.reg_model.lwca2(sf8, cf8)
            mf16 = self.reg_model.lwca3(sf16, cf16)
            mf32 = self.reg_model.lwca4(sf32, cf32)
            ff4 = self.reg_model.lwca1(cf4, sf4)
            ff8 = self.reg_model.lwca2(cf8, sf8)
            ff16 = self.reg_model.lwca3(cf16, sf16)
            mp_x = self.reg_model.avg_pool(torch.cat([mp, fixed_img], dim=1))
            mp_f4 = self.reg_model.ec1(mp_x)
            dx = self.reg_model.up0(mf32, mf16, ff16)
            dx = self.reg_model.up1(dx, mf8, ff8)
            dx = self.reg_model.up2(dx, mf4, ff4)
            dx = self.reg_model.up3(dx, mp_f4)
            dx = self.reg_model.up(dx)
            disp_p = self.reg_model.reg_head(dx)
            all_displacements.append(disp_p)

        all_displacements_tensor = torch.stack(all_displacements, dim=0)  # [num_ph-1, B, 2, H, W]

        return {
            'disp_field': disp_field,
            'all_displacements': all_displacements_tensor,
            'warped_image': warped,
            'moving_img': moving_img,
            'fixed_img': fixed_img,
        }


# =============================================================================
# Training Loop
# =============================================================================

def save_checkpoint(state, save_dir, save_filename, max_model_num=50):
    torch.save(state, os.path.join(save_dir, save_filename))
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))


def train():
    # -------------------------------------------------------------------------
    # Load LDM model
    # -------------------------------------------------------------------------
    ldm_model, configs = load_ldm_model(opt.ldm_config, opt.resume)

    # -------------------------------------------------------------------------
    # Build dataset
    # -------------------------------------------------------------------------
    print(f"\nBuilding dataset from: {opt.xcat_path}")
    train_dataset = MultiPhaseSequenceDataset(
        data_root=opt.xcat_path,
        split='train',
        context_size=opt.context_size,
        flip_p=opt.flip_p,
    )

    val_dataset = MultiPhaseSequenceDataset(
        data_root=opt.xcat_path,
        split='val',
        context_size=opt.context_size,
        flip_p=0.0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.bs,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Dataset mode:  {train_dataset.mode}")

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    print("\nBuilding Temporal-Aware LDMMorph model...")
    model = TemporalAwareLDMMorphWrapper(
        ldm_model=ldm_model,
        channel_1=640,
        channel_2=1024,
        channel_3=1536,
        channel_4=1920,
        num_phases=opt.num_phases,
        context_size=opt.context_size,
        temporal_proj_dim=opt.temporal_proj_dim,
        num_temporal_layers=opt.num_temporal_layers,
        temporal_num_heads=opt.temporal_num_heads,
        use_bidirectional=use_bidirectional,
        fuse_mode=opt.fuse_mode,
    )
    model.cuda()

    total_params = sum(p.nelement() for p in model.reg_model.parameters())
    print(f"  Registration model parameters: {total_params / 1e6:.2f}M")
    print(f"  Temporal fusion params: "
          f"{sum(p.nelement() for p in model.reg_model.temporal_fusion_1.parameters()) / 1e6:.2f}M (per scale)")

    # -------------------------------------------------------------------------
    # Optimizer — differential learning rates (like T-Gated Adapter)
    # -------------------------------------------------------------------------
    # Temporal transformer: higher LR (newly added)
    temporal_params = (
        list(model.reg_model.temporal_fusion_1.parameters()) +
        list(model.reg_model.temporal_fusion_2.parameters()) +
        list(model.reg_model.temporal_fusion_3.parameters()) +
        list(model.reg_model.temporal_fusion_4.parameters())
    )
    # LDM feature extractor: frozen (no grad)
    ldm_params = []  # frozen
    # Decoder + LWCA + LWSA: moderate LR (from pretrained LDMMorph)
    decoder_params = (
        list(model.reg_model.c1.parameters()) +
        list(model.reg_model.c2.parameters()) +
        list(model.reg_model.c3.parameters()) +
        list(model.reg_model.c4.parameters()) +
        list(model.reg_model.lwsa.parameters()) +
        list(model.reg_model.lwca1.parameters()) +
        list(model.reg_model.lwca2.parameters()) +
        list(model.reg_model.lwca3.parameters()) +
        list(model.reg_model.lwca4.parameters()) +
        list(model.reg_model.up0.parameters()) +
        list(model.reg_model.up1.parameters()) +
        list(model.reg_model.up2.parameters()) +
        list(model.reg_model.up3.parameters()) +
        list(model.reg_model.reg_head.parameters()) +
        list(model.reg_model.ec1.parameters())
    )

    optimizer = Adam([
        {'params': ldm_params, 'lr': 0},            # frozen
        {'params': decoder_params, 'lr': opt.lr},   # pretrained backbone: 1e-4
        {'params': temporal_params, 'lr': opt.lr * 5},  # temporal modules: 5e-4
    ], lr=opt.lr)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-7)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    loss_ncc = ncc_loss
    loss_mse = MSE().loss
    loss_smooth = smoothloss

    physics_loss_fn = PhysicsInformedTemporalLoss(
        lambda_1st=opt.lambda_1st,
        lambda_2nd=opt.lambda_2nd,
        lambda_cycle=opt.lambda_cycle,
        lambda_neg_J=opt.lambda_neg_J,
        lambda_gate=opt.lambda_gate,
    )

    spatial_transform = SpatialTransform().cuda()
    for param in spatial_transform.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    model_name_prefix = (
        f"TemporalLDMMorph_ph{opt.num_phases}_ctx{opt.context_size}"
        f"_tpd{opt.temporal_proj_dim}_ntl{opt.num_temporal_layers}"
        f"_bi{int(use_bidirectional)}"
        f"_L1{opt.lambda_1st}_L2{opt.lambda_2nd}_Lcycle{opt.lambda_cycle}"
    )
    model_dir = f"./logs/{model_name_prefix}/"
    csv_name = f"./logs/{model_name_prefix}.csv"

    os.makedirs(model_dir, exist_ok=True)

    f = open(csv_name, 'w')
    with f:
        fnames = ['Index', 'NCC_Val_S', 'OrgNCC_Val_S', 'NCC_Test', 'OrgNCC_Test',
                   'Loss_1st', 'Loss_2nd', 'Loss_Cycle', 'Loss_negJ', 'Loss_Gate']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    print(f"\nOutput directory: {model_dir}")
    print(f"CSV log: {csv_name}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    lossall = np.zeros((8, opt.iteration))
    step = 1
    t_enc = 1

    while step <= opt.iteration:
        model.train()
        for batch in train_loader:
            # -----------------------------------------------------------------
            # Unpack batch
            # -----------------------------------------------------------------
            fixed = batch['fixed'].cuda().float()          # [B, H, W]
            phase_seq = batch['phase_sequence'].cuda().float()  # [B, num_ph, H, W]

            # Add channel dimension
            fixed = fixed.unsqueeze(1)                     # [B, 1, H, W]
            phase_seq = phase_seq.unsqueeze(2)             # [B, num_ph, 1, H, W]

            # Pad if we have fewer phases than num_phases
            num_actual = phase_seq.shape[1]
            if num_actual < opt.num_phases:
                pad = torch.zeros(
                    phase_seq.shape[0], opt.num_phases - num_actual,
                    1, phase_seq.shape[3], phase_seq.shape[4],
                    device=phase_seq.device
                )
                phase_seq = torch.cat([phase_seq, pad], dim=1)

            # -----------------------------------------------------------------
            # Forward pass
            # -----------------------------------------------------------------
            outputs = model(fixed=fixed, phase_sequence=phase_seq)
            disp_field = outputs['disp_field']
            all_displacements = outputs['all_displacements']  # [num_ph-1, B, 2, H, W]
            warped = outputs['warped_image']

            # -----------------------------------------------------------------
            # Similarity losses
            # -----------------------------------------------------------------
            loss_mse_latent = loss_mse(
                model.ldm_extractor.ldm.get_first_stage_encoding(
                    model.ldm_extractor.ldm.encode_first_stage(warped)),
                model.ldm_extractor.ldm.get_first_stage_encoding(
                    model.ldm_extractor.ldm.encode_first_stage(fixed))
            ).detach()  # detach MSE latent to save memory

            loss_ncc_image = loss_ncc(warped, fixed)
            loss_sim = opt.beta * loss_ncc_image + (1 - opt.beta) * loss_mse_latent

            # Standard smoothness
            loss_smooth_val = loss_smooth(disp_field)

            # -----------------------------------------------------------------
            # Physics-informed temporal losses — Contribution 2
            # -----------------------------------------------------------------
            physics_losses = physics_loss_fn(
                displacement_fields=all_displacements,
                gate_values=None,
                num_phases=opt.num_phases,
            )

            # -----------------------------------------------------------------
            # Total loss
            # -----------------------------------------------------------------
            loss = (
                loss_sim
                + opt.smooth * loss_smooth_val
                + physics_losses['total']
            )

            # -----------------------------------------------------------------
            # Backward pass
            # -----------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.reg_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            lossall[:, step] = np.array([
                loss.item(),
                loss_sim.item(),
                loss_smooth_val.item(),
                physics_losses['loss_1st'].item(),
                physics_losses['loss_2nd'].item(),
                physics_losses['loss_cycle'].item(),
                physics_losses['loss_neg_J'].item(),
                physics_losses['loss_gate'].item(),
            ])

            sys.stdout.write(
                f"\r[Step {step}] "
                f"total={loss.item():.4f} "
                f"ncc={loss_ncc_image.item():.4f} "
                f"smth={loss_smooth_val.item():.4f} "
                f"L1st={physics_losses['loss_1st'].item():.4f} "
                f"L2nd={physics_losses['loss_2nd'].item():.4f} "
                f"LnegJ={physics_losses['loss_neg_J'].item():.4f}"
            )
            sys.stdout.flush()

            # -----------------------------------------------------------------
            # Visualization
            # -----------------------------------------------------------------
            if step % 1000 == 0:
                with torch.no_grad():
                    X_cpu = phase_seq[0, 1, 0].cpu().numpy()
                    Y_cpu = fixed[0, 0].cpu().numpy()
                    XY_cpu = warped[0, 0].cpu().numpy()
                    D_cpu = disp_field[0].cpu().numpy()
                    diff_before = np.abs(X_cpu - Y_cpu)
                    diff_after = np.abs(XY_cpu - Y_cpu)

                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    titles = ['Moving (X)', 'Fixed (Y)', 'Warped (X→Y)', 'Diff Before',
                             'Diff After', 'Diff Before (zoom)', 'Disp Field-X', 'Disp Field-Y']
                    imgs = [X_cpu, Y_cpu, XY_cpu, diff_before,
                           diff_after, diff_after[50:150, 50:150], D_cpu[0], D_cpu[1]]
                    img_vmax = max(X_cpu.max(), Y_cpu.max(), XY_cpu.max())
                    for ax, img, title in zip(axes.flat, imgs, titles):
                        ax.imshow(img, cmap='gray', vmin=0, vmax=img_vmax)
                        ax.set_title(f'{title}\n(max={img.max():.4f})', fontsize=10)
                        ax.axis('off')

                    fig.suptitle(
                        f'[Step {step}] loss={loss.item():.4f}\n'
                        f'NCC={1-loss_ncc_image.item():.4f} '
                        f'L1st={physics_losses["loss_1st"].item():.4f} '
                        f'L2nd={physics_losses["loss_2nd"].item():.4f}',
                        fontsize=12)
                    plt.tight_layout()
                    fig.savefig(f'{model_dir}vis_step_{step:06d}.png', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    print(f'\n    [Visualization] Saved to {model_dir}vis_step_{step:06d}.png')

            # -----------------------------------------------------------------
            # Validation checkpoint
            # -----------------------------------------------------------------
            if step % opt.checkpoint == 0:
                model.eval()
                NCCs_Val = []
                NCCs_Val_Org = []

                for xv, in val_loader:
                    xv_fixed = xv['fixed'].cuda().float().unsqueeze(1)
                    xv_seq = xv['phase_sequence'].cuda().float().unsqueeze(2)

                    if xv_seq.shape[1] < opt.num_phases:
                        pad = torch.zeros(xv_seq.shape[0], opt.num_phases - xv_seq.shape[1],
                                        1, xv_seq.shape[3], xv_seq.shape[4], device=xv_seq.device)
                        xv_seq = torch.cat([xv_seq, pad], dim=1)

                    with torch.no_grad():
                        vout = model(xv_fixed, xv_seq)
                        vdisp = vout['disp_field']
                        _, vwarped = spatial_transform(
                            xv_seq[:, 1, :, :, :].squeeze(2), vdisp.permute(0, 2, 3, 1))

                    ncc_s = 1.0 - loss_ncc(vwarped, xv_fixed).item()
                    ncc_org = 1.0 - loss_ncc(xv_seq[:, 1, :, :, :].squeeze(2), xv_fixed).item()
                    NCCs_Val.append(ncc_s)
                    NCCs_Val_Org.append(ncc_org)

                csv_ncc_s = np.mean(NCCs_Val)
                csv_ncc_org = np.mean(NCCs_Val_Org)

                modelname = f'NCCVal_{csv_ncc_s:.4f}_Step_{step:06d}.pth'
                save_checkpoint({
                    'model_state_dict': model.reg_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': step,
                }, model_dir, modelname)

                np.save(os.path.join(model_dir, 'Loss.npy'), lossall)

                print(f'\n    [Validation] NCC_S: {csv_ncc_s:.4f}  '
                      f'OrgNCC_S: {csv_ncc_org:.4f}  '
                      f'Delta: {csv_ncc_s - csv_ncc_org:+.4f}')

                f = open(csv_name, 'a')
                with f:
                    writer = csv.writer(f)
                    writer.writerow([
                        step, csv_ncc_s, csv_ncc_org,
                        -1, -1,
                        physics_losses['loss_1st'].item(),
                        physics_losses['loss_2nd'].item(),
                        physics_losses['loss_cycle'].item(),
                        physics_losses['loss_neg_J'].item(),
                        physics_losses['loss_gate'].item(),
                    ])
                model.train()

            step += 1
            if step > opt.iteration:
                break

        print(f"\n  Epoch complete, step={step}")

    np.save(os.path.join(model_dir, 'Loss.npy'), lossall)
    print(f"\nTraining complete. Results saved to {model_dir}")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train()
