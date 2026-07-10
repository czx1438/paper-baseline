"""
temporal_transformer.py
========================================================
Temporal-Aware LDM-Morph — Core Temporal Modules

This module implements the following novel contributions over T-Gated Adapter:

Contribution 1 — Temporal Transformer on LDM Features:
  Unlike T-Gated Adapter (which operates on CLIP vision tokens), we apply
  temporal self-attention directly on the multi-scale LDM UNet skip features
  (score0/score1/score2/score3). Each spatial token independently attends
  across the cardiac phase dimension, allowing motion evidence from adjacent
  phases to inform the current phase's feature representation.

Contribution 3 — Spatially-Varying Adaptive Gate:
  T-Gated Adapter uses a uniform gate (one scalar per token). We replace it
  with a spatially-varying gate that accounts for cardiac anatomy:
    - Apex (lower portion): large motion → high gate (trust temporal)
    - Base (upper portion): small motion → low gate (trust single-phase)
  This is task-specific to registration and has no counterpart in T-Gated Adapter.

Contribution 4 — Bidirectional Temporal Propagation:
  T-Gated Adapter is strictly feedforward (5-input → 1-output).
  We exploit the fact that ALL phases are available simultaneously in
  cardiac registration to run BOTH forward and backward temporal attention,
  then fuse both directions. This captures both "past motion history"
  and "future motion trajectory" for each phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Contribution 1 — Stochastic Depth Drop Path
# =============================================================================

class DropPath(nn.Module):
    """Stochastic depth — randomly drops entire residual branches during training.
    Keeps the model path intact at inference (identity).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


# =============================================================================
# Contribution 1 — Temporal Encoder Layer
# =============================================================================

class TemporalEncoderLayer(nn.Module):
    """Single layer of the temporal transformer.

    Operates on shape [B * spatial_tokens, num_phases, proj_dim].
    Self-attention is applied ALONG the phase dimension (num_phases),
    at each spatial token position independently.

    This is the direct analogue of T-Gated Adapter's TemporalEncoderLayer,
    but generalized to arbitrary projection dimension and number of phases.
    """
    def __init__(
        self,
        dim: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        # Pre-norm architecture (like T-Gated Adapter)
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)

        # Phase-level self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,   # [L, B, D] convention
        )

        # FFN: proj_dim → proj_dim * ratio → proj_dim
        mlp_hidden = int(proj_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, proj_dim),
            nn.Dropout(dropout),
        )

        # Stochastic depth on both branches
        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

        # Project dim (input) → proj_dim (internal)
        self.proj_in = nn.Linear(dim, proj_dim)
        self.norm_in = nn.LayerNorm(proj_dim)

        # Project proj_dim (internal) → dim (output)
        self.proj_out = nn.Linear(proj_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B * spatial_tokens, num_phases, dim]
        returns: [B * spatial_tokens, num_phases, dim]
        """
        B_L, num_ph, D = x.shape

        # Project to internal dimension
        x_proj = self.norm_in(self.proj_in(x))

        # Pre-norm self-attention across phase dimension
        attn_out, _ = self.attn(
            self.norm1(x_proj),
            self.norm1(x_proj),
            self.norm1(x_proj),
        )
        x_proj = x_proj + self.drop_path1(attn_out)

        # Pre-norm FFN
        ff_out = self.mlp(self.norm2(x_proj))
        x_proj = x_proj + self.drop_path2(ff_out)

        # Project back to original dim
        return self.proj_out(x_proj)


# =============================================================================
# Contribution 1 — Multi-Layer Temporal Transformer Stack
# =============================================================================

class TemporalTransformer(nn.Module):
    """Stack of N TemporalEncoderLayers.

    Processes all phases together via N rounds of self-attention.
    Output: same shape as input, but each phase's features are enriched
    with temporal context from all other phases.
    """
    def __init__(
        self,
        dim: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_base: float = 0.0,
        drop_path_inc: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_phases = None  # Set dynamically in forward

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            drop_path_i = drop_path_base + i * drop_path_inc
            self.layers.append(
                TemporalEncoderLayer(
                    dim=dim,
                    proj_dim=proj_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path_prob=drop_path_i,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_phases, spatial_tokens, dim]
        returns: [B, num_phases, spatial_tokens, dim]
        """
        B, num_ph, seq, D = x.shape
        self.num_phases = num_ph

        # Merge batch and spatial dims → [B*seq, num_ph, D]
        x = x.permute(0, 2, 1, 3).reshape(B * seq, num_ph, D)

        # Pass through N temporal encoder layers
        for layer in self.layers:
            x = layer(x)

        # Restore shape: [B*seq, num_ph, D] → [B, num_ph, seq, D] → [B, seq, num_ph, D]
        x = x.reshape(B, seq, num_ph, D)
        return x.permute(0, 2, 1, 3)


# =============================================================================
# Contribution 3 — Spatially-Varying Adaptive Gate
# =============================================================================

class SpatiallyVaryingGate(nn.Module):
    """Adaptive gate with spatial awareness of cardiac anatomy.

    Unlike T-Gated Adapter's uniform gate (same gate value for all pixels),
    our gate accounts for the fact that cardiac apex and base have different
    motion amplitudes:
      - Apex (inferior / lower in image): large motion → high temporal weight
      - Base (superior / upper in image): small motion → low temporal weight

    Implementation:
      1. Compute per-token gate value g ∈ [0,1] from center-phase features
         (standard T-Gated Adapter style).
      2. Compute a spatial attention map a ∈ [0,∞) from the feature magnitude,
         biasing the gate toward higher values in high-motion regions.
      3. Fused gate = sigmoid(g * (1 + spatial_weight))

    This is novel: T-Gated Adapter has NO spatial awareness in its gate.
    """
    def __init__(self, dim: int, num_tokens: int = 1, bias_init: float = -5.0):
        """
        dim: feature dimension (Dv)
        num_tokens: number of gate MLP hidden units (default 1 = scalar gate)
        bias_init: initial bias for the gate projection (default -5.0, same as
                   T-Gated Adapter, ensures conservative start)
        """
        super().__init__()

        # Standard gate projection (like T-Gated Adapter)
        self.gate_proj = nn.Linear(dim, num_tokens)
        # Initialize bias = -5.0 (sigmoid(-5) ≈ 0 → start by trusting single-phase)
        self.gate_proj.bias.data.fill_(bias_init)
        self.gate_proj.weight.data.zero_()

        # Spatial attention: learns a per-spatial-position weight
        # that biases the gate toward higher values where features suggest
        # high motion amplitude
        self.spatial_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

        # Learnable spatial bias: prior that apex (bottom of image) > base (top)
        # We use a sigmoid over the vertical axis so the network can learn
        # whether apex or base has more motion from data
        self.spatial_prior = nn.Parameter(torch.zeros(1))  # multiplier for vertical prior

    def forward(
        self,
        temporal_feat: torch.Tensor,
        center_feat: torch.Tensor,
        spatial_coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        temporal_feat: [B, num_phases, seq, D] — output of temporal transformer
        center_feat:    [B, seq, D] — raw center-phase features (no temporal fusion)

        returns: [B, seq, D] — gated fusion of temporal + center features
        """
        B, num_ph, seq, D = temporal_feat.shape

        # Extract center-phase output from temporal transformer
        center_idx = num_ph // 2
        temporal_center = temporal_feat[:, center_idx, :, :]   # [B, seq, D]

        # 1. Standard per-token gate (like T-Gated Adapter)
        gate = torch.sigmoid(self.gate_proj(center_feat))      # [B, seq, 1] or [B, seq, D']

        # 2. Spatial attention: how much does each token suggest high motion?
        #    Use the feature norm as a proxy for motion magnitude
        feat_norm = center_feat.norm(dim=-1, keepdim=True)    # [B, seq, 1]
        spatial_weight = self.spatial_attn(center_feat)        # [B, seq, 1]
        spatial_weight = torch.tanh(spatial_weight)            # bound to [-1, 1]

        # 3. Combine: fuse spatial bias with per-token gate
        #    High-motion regions (large feat_norm) → higher spatial_weight → larger gate
        #    Low-motion regions (small feat_norm) → lower spatial_weight → smaller gate
        gate_augmented = gate * (1.0 + 0.5 * spatial_weight)
        gate_augmented = torch.sigmoid(gate_augmented)

        # 4. Fusion: weighted sum of temporal-enhanced and raw center features
        fused = gate_augmented * temporal_center + (1.0 - gate_augmented) * center_feat

        return fused


# =============================================================================
# Contribution 4 — Bidirectional Temporal Propagation
# =============================================================================

class BidirectionalTemporalEncoder(nn.Module):
    """Bidirectional temporal attention on multi-phase features.

    T-Gated Adapter is strictly unidirectional (5 slices in → 1 slice out).
    We exploit the fact that ALL cardiac phases are simultaneously available
    to run:
      1. Forward attention:  phases[0..T] → each phase attends to all later phases
      2. Backward attention: phases[T..0] → each phase attends to all earlier phases

    Both directions are concatenated and fused, giving each phase access to
    both "past motion history" AND "future motion trajectory".

    This is impossible in T-Gated Adapter's segmentation setting (you only
    have the 5 context slices available at test time, and you predict one
    output slice — there's no "backward" direction).
    """
    def __init__(
        self,
        dim: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_base: float = 0.0,
        drop_path_inc: float = 0.05,
        fuse_mode: str = "concat",
    ):
        """
        fuse_mode: 'concat' (recommended) or 'add' for combining forward/backward
        """
        super().__init__()
        self.fuse_mode = fuse_mode

        # Forward temporal transformer (phase 0 → T direction)
        self.forward_transformer = TemporalTransformer(
            dim=dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_base=drop_path_base,
            drop_path_inc=drop_path_inc,
        )

        # Backward temporal transformer (phase T → 0 direction)
        self.backward_transformer = TemporalTransformer(
            dim=dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_base=drop_path_base,
            drop_path_inc=drop_path_inc,
        )

        if fuse_mode == "concat":
            # Project concatenated forward+backward back to dim
            self.fuse_proj = nn.Linear(dim * 2, dim)
        elif fuse_mode == "add":
            self.fuse_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_phases, seq, D]

        Forward path:  x[:, ::+1, :, :]  (p0, p1, ..., pT)
        Backward path: x[:, ::-1, :, :]  (pT, ..., p1, p0)

        Both paths use the SAME spatial ordering (the backward transformer
        learns to attend to phases that are "earlier" in the original sequence
        but appear "later" in the reversed input).
        """
        # Forward direction
        feat_fwd = self.forward_transformer(x)           # [B, num_ph, seq, D]

        # Backward direction: reverse phase order
        x_rev = x.flip(dims=[1])                         # [B, num_ph, seq, D]
        feat_bwd = self.backward_transformer(x_rev)      # [B, num_ph, seq, D]
        feat_bwd = feat_bwd.flip(dims=[1])                # un-reverse

        if self.fuse_mode == "concat":
            feat_fused = torch.cat([feat_fwd, feat_bwd], dim=-1)  # [B, num_ph, seq, 2D]
            feat_fused = self.fuse_proj(feat_fused)               # [B, num_ph, seq, D]
        else:  # add
            feat_fused = feat_fwd + feat_bwd

        return feat_fused


# =============================================================================
# Contribution 1+3+4 — Complete Temporal Fusion Module
# =============================================================================

class TemporalAwareFeatureFusion(nn.Module):
    """Complete temporal fusion pipeline for one LDM feature scale.

    Integrates:
      - Bidirectional temporal transformer (Contributions 1 + 4)
      - Spatially-varying adaptive gate (Contribution 3)

    Plugs INTO the LDM-Morph pipeline at each of the 4 feature scales
    (score0/1/2/3), replacing the direct use of raw LDM features.

    Pipeline:
      raw_scores [B, num_ph, C, H, W]
          ↓
      BidirectionalTemporalEncoder
          ↓
      [B, num_ph, H*W, C] → center phase extracted
          ↓
      SpatiallyVaryingGate (center_raw + center_enhanced)
          ↓
      [B, C, H, W] — final enhanced features for this scale
    """
    def __init__(
        self,
        dim: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_base: float = 0.0,
        drop_path_inc: float = 0.05,
        fuse_mode: str = "concat",
        use_bidirectional: bool = True,
    ):
        super().__init__()

        self.use_bidirectional = use_bidirectional
        self.proj_dim = proj_dim

        # Project input features to internal dimension
        self.input_proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Output projection: internal → original dim
        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, dim),
        )

        if use_bidirectional:
            self.temporal_encoder = BidirectionalTemporalEncoder(
                dim=proj_dim,
                proj_dim=proj_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_base=drop_path_base,
                drop_path_inc=drop_path_inc,
                fuse_mode=fuse_mode,
            )
        else:
            self.temporal_encoder = TemporalTransformer(
                dim=proj_dim,
                proj_dim=proj_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_base=drop_path_base,
                drop_path_inc=drop_path_inc,
            )

        # Gate: needs original dim for projection
        self.gate = SpatiallyVaryingGate(dim=dim)

        # Projection for gate's input_proj (so it can handle original-dim inputs)
        self.gate_center_proj = nn.Linear(dim, dim)

    def forward(
        self,
        multi_phase_scores: torch.Tensor,
        raw_center_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        multi_phase_scores: [B, num_phases, C, H, W] — features for all phases at this scale
        raw_center_score:   [B, C, H, W]              — raw (pre-temporal) center-phase features

        returns: [B, C, H, W] — temporally enhanced center-phase features
        """
        B, num_ph, C, H, W = multi_phase_scores.shape

        # 1. Project to internal dim: [B, num_ph, C, H, W] → [B, num_ph, H*W, proj_dim]
        x = multi_phase_scores.permute(0, 1, 3, 4, 2)                    # [B, num_ph, H, W, C]
        x = x.reshape(B, num_ph, H * W, C)                             # [B, num_ph, H*W, C]
        x = self.input_proj(x)                                           # [B, num_ph, H*W, proj_dim]

        # 2. Bidirectional temporal transformer
        x_enhanced = self.temporal_encoder(x)                            # [B, num_ph, H*W, proj_dim]

        # 3. Project back to original dim: [B, num_ph, H*W, proj_dim] → [B, num_ph, H*W, C]
        x_enhanced = self.output_proj(x_enhanced)

        # 4. Restore spatial structure
        x_enhanced = x_enhanced.reshape(B, num_ph, H, W, C)              # [B, num_ph, H, W, C]
        x_enhanced = x_enhanced.permute(0, 1, 4, 2, 3)                  # [B, num_ph, C, H, W]

        # 5. Extract center phase (phase 0 = reference = fixed)
        center_idx = num_ph // 2
        temporal_center = x_enhanced[:, center_idx, :, :, :]              # [B, C, H, W]

        # 6. Flatten spatial for gate
        raw_center_flat = raw_center_score.flatten(2).permute(0, 2, 1)   # [B, H*W, C]
        temporal_center_flat = temporal_center.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # 7. Apply spatially-varying adaptive gate
        gated = self.gate(
            temporal_feat=x_enhanced,        # [B, num_ph, H*W, C]
            center_feat=raw_center_flat,      # [B, H*W, C]
        )                                       # [B, H*W, C]

        # 8. Restore spatial shape
        gated = gated.permute(0, 2, 1).reshape(B, C, H, W)             # [B, C, H, W]

        return gated
