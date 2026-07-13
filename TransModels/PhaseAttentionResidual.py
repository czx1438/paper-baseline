"""Lightweight cross-phase attention and residual DVF refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossPhaseAttentionResidual(nn.Module):
    """Refine frozen pairwise DVFs using context from phases of one slice.

    The final convolution is zero-initialized, so the initial refined DVF is
    exactly the frozen pairwise DVF. Attention is restricted to the phase axis
    of each batch item and never mixes different slices.
    """

    def __init__(
        self,
        code_dim=16,
        num_heads=4,
        hidden_channels=32,
        residual_size=128,
    ):
        super().__init__()
        if code_dim % num_heads != 0:
            raise ValueError("code_dim must be divisible by num_heads")

        self.residual_size = int(residual_size)
        self.code_norm = nn.LayerNorm(code_dim)
        self.phase_attention = nn.MultiheadAttention(
            embed_dim=code_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(code_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(code_dim, code_dim * 2),
            nn.GELU(),
            nn.Linear(code_dim * 2, code_dim),
        )

        # moving, fixed, pairwise-warped, absolute residual, pairwise DVF (2)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(code_dim, hidden_channels),
            nn.PReLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )

        final_layer = self.residual_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def cross_phase_context(self, motion_codes):
        """Return phase-refined codes and attention maps.

        Args:
            motion_codes: [B, P, C], all phases from the same slice per B.
        """
        normalized = self.code_norm(motion_codes)
        #weights表示当前相位作为query对别的相位的关注程度
        #attended是加权融合别的相位信息后的新特征
        attended, weights = self.phase_attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        #最终上下文=当前phase原始运动特征+从其他phase加权融合后的特征
        context = motion_codes + attended
        context = context + self.context_ffn(self.context_norm(context))
        return context, weights

    def refine_phase(
        self,
        moving,
        fixed,
        pairwise_warped,
        pairwise_dvf,
        motion_codes,
        phase_index,
    ):
        """Predict a small residual for one phase.

        Attention is recomputed for each phase during training. This is cheap
        for nine 16-D tokens and lets each phase backpropagate independently,
        avoiding retention of nine image-loss graphs at once.
        """
        context, attention_weights = self.cross_phase_context(motion_codes)
        phase_context = context[:, phase_index]

        size = (self.residual_size, self.residual_size)
        moving_low = F.interpolate(
            moving, size=size, mode="bilinear", align_corners=True
        )
        fixed_low = F.interpolate(
            fixed, size=size, mode="bilinear", align_corners=True
        )
        warped_low = F.interpolate(
            pairwise_warped, size=size, mode="bilinear", align_corners=True
        )
        dvf_low = F.interpolate(
            pairwise_dvf, size=size, mode="bilinear", align_corners=True
        )
        local_input = torch.cat(
            [
                moving_low,
                fixed_low,
                warped_low,
                (warped_low - fixed_low).abs(),
                dvf_low,
            ],
            dim=1,
        )
        local_feature = self.local_encoder(local_input)
        context_feature = self.context_projection(phase_context)
        context_map = context_feature[:, :, None, None].expand(
            -1,
            -1,
            self.residual_size,
            self.residual_size,
        )
        residual_low = self.residual_head(
            torch.cat([local_feature, context_map], dim=1)
        )
        residual = F.interpolate(
            residual_low,
            size=pairwise_dvf.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        refined_dvf = pairwise_dvf + residual
        return refined_dvf, residual, attention_weights

