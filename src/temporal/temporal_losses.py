"""
temporal_losses.py
========================================================
Physics-Informed Temporal Consistency Losses for Cardiac Motion Registration.

Contribution 2 — Temporal Smoothness on Displacement Fields:
  Unlike T-Gated Adapter (which only constrains input features), we impose
  explicit physical constraints on the OUTPUT displacement fields.

  Cardiac motion is physically smooth:
    - Adjacent phases have similar displacement magnitudes (1st-order continuity)
    - Displacement acceleration is near-zero (2nd-order / jerk continuity)
    - Cardiac cycle is approximately periodic (phase N ≈ phase 0)

  These constraints are fundamentally impossible in segmentation tasks —
  segmentation outputs are discrete labels with no temporal derivative.

Loss terms:
  1. L_temporal_1st:  ||φ_t - φ_{t+1}||²     — 1st-order temporal smoothness
  2. L_temporal_2nd:  ||φ_t - 2φ_{t+1} + φ_{t+2}||² — 2nd-order (jerk) smoothness
  3. L_cycle:          ||φ_N→0||²             — cyclic consistency
  4. L_gate_sparsity:  H(gate)                 — gate entropy (from T-Gated Adapter)
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Contribution 2 — Primary: Temporal Smoothness on Displacement Fields
# =============================================================================

def temporal_smoothness_loss_1st(displacement_fields: torch.Tensor) -> torch.Tensor:
    """
    First-order temporal smoothness on displacement fields.

    For cardiac motion: adjacent phases should have similar displacements
    (the heart doesn't teleport between frames).

    L_1st = Σ_t ||φ_t - φ_{t+1}||² / N

    Args:
        displacement_fields: [B, num_ph-1, 2, H, W] — displacement for each phase
                            displacement_fields[:, 0, :, :, :] = φ_1→0
                            displacement_fields[:, 1, :, :, :] = φ_2→0
                            ...

    Returns:
        scalar tensor — mean squared displacement difference between adjacent phases
    """
    if displacement_fields.shape[1] < 2:
        # Need at least 2 phases for 1st-order smoothness
        return torch.tensor(0.0, device=displacement_fields.device,
                           dtype=displacement_fields.dtype)

    # φ_t - φ_{t+1}: [B, num_ph-2, 2, H, W]
    diff = displacement_fields[:, :-1, :, :, :] - displacement_fields[:, 1:, :, :, :]

    # Mean squared difference
    loss = torch.mean(diff ** 2)
    return loss


def temporal_smoothness_loss_2nd(displacement_fields: torch.Tensor) -> torch.Tensor:
    """
    Second-order temporal smoothness (jerk penalty) on displacement fields.

    For cardiac motion: the ACCELERATION of displacement should be smooth
    (cardiac muscle contracts/expands at roughly constant velocity between phases).

    L_2nd = Σ_t ||φ_t - 2φ_{t+1} + φ_{t+2}||² / N

    This is the finite-difference approximation of:
        L_2nd = ||∂²φ/∂t²||²

    Args:
        displacement_fields: [B, num_ph-1, 2, H, W]

    Returns:
        scalar tensor — mean squared 2nd-order displacement difference
    """
    if displacement_fields.shape[1] < 3:
        # Need at least 3 phases for 2nd-order smoothness
        return torch.tensor(0.0, device=displacement_fields.device,
                           dtype=displacement_fields.dtype)

    # φ_t - 2φ_{t+1} + φ_{t+2}: [B, num_ph-3, 2, H, W]
    phi_t = displacement_fields[:, :-2, :, :, :]      # [B, N-2, 2, H, W]
    phi_t1 = displacement_fields[:, 1:-1, :, :, :]    # [B, N-2, 2, H, W]
    phi_t2 = displacement_fields[:, 2:, :, :, :]      # [B, N-2, 2, H, W]

    jerk = phi_t - 2 * phi_t1 + phi_t2

    loss = torch.mean(jerk ** 2)
    return loss


# =============================================================================
# Contribution 2 — Cyclic Consistency (for full cardiac cycle datasets)
# =============================================================================

def cyclic_consistency_loss(
    displacement_fields: torch.Tensor,
    num_phases: int,
    cycle_target: int = 0,
) -> torch.Tensor:
    """
    Cyclic consistency loss: after N phases, the heart should return close to origin.

    Physically: phase N (last phase in the cycle) should be close to phase 0 (fixed).
    We already predict φ_N→0 (the displacement from phase N to phase 0).
    The displacement from phase N back to phase 0 should be near zero.

    L_cycle = ||φ_cycle_target→0||²

    This is ONLY applicable when we have a complete cardiac cycle in the data
    (phases 0 through N where phase N ≈ phase 0 anatomically).

    Args:
        displacement_fields: [B, num_ph-1, 2, H, W]
        num_phases: total number of phases in the sequence
        cycle_target: which phase to use as the "end of cycle" (default 0 = phase 0)

    Returns:
        scalar tensor — mean squared displacement for the cycle-target phase
    """
    if cycle_target >= displacement_fields.shape[1]:
        return torch.tensor(0.0, device=displacement_fields.device,
                           dtype=displacement_fields.dtype)

    # Use the last predicted phase's displacement (closest to end of cycle)
    cycle_disp = displacement_fields[:, -1, :, :, :]  # [B, 2, H, W]

    loss = torch.mean(cycle_disp ** 2)
    return loss


# =============================================================================
# Contribution 2 — Jacobian Regularization (anti-folding)
# =============================================================================

def jacobian_determinant_2d(disp: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian determinant of a 2D displacement field.

    disp: [B, 2, H, W] — displacement field (x, y)
    Returns: [B, H, W] — Jacobian determinant at each pixel

    det(J) > 0 at all pixels → no folding (diffeomorphic)
    det(J) < 0 → folding (invalid registration)
    """
    B, C, H, W = disp.shape
    assert C == 2, "displacement field must have 2 channels (x, y)"

    # Compute spatial gradients using finite differences
    # ∂disp_x/∂x
    disp_x = disp[:, 0, :, :]  # [B, H, W]
    ddx = torch.zeros_like(disp_x)
    ddx[:, 1:-1, 1:-1] = (disp_x[:, 1:-1, 2:] - disp_x[:, 1:-1, :-2]) / 2.0

    # ∂disp_x/∂y
    dxy = torch.zeros_like(disp_x)
    dxy[:, 1:-1, 1:-1] = (disp_x[:, 2:, 1:-1] - disp_x[:, :-2, 1:-1]) / 2.0

    # ∂disp_y/∂x
    disp_y = disp[:, 1, :, :]  # [B, H, W]
    dyx = torch.zeros_like(disp_y)
    dyx[:, 1:-1, 1:-1] = (disp_y[:, 1:-1, 2:] - disp_y[:, 1:-1, :-2]) / 2.0

    # ∂disp_y/∂y
    dyy = torch.zeros_like(disp_y)
    dyy[:, 1:-1, 1:-1] = (disp_y[:, 2:, 1:-1] - disp_y[:, :-2, 1:-1]) / 2.0

    # Jacobian determinant: det = (1 + ∂u/∂x) * (1 + ∂v/∂y) - ∂u/∂y * ∂v/∂x
    jac = (1.0 + ddx) * (1.0 + dyy) - dxy * dyx

    return jac


def negative_jacobian_loss(disp: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Penalize negative Jacobian determinants (folding).

    L_neg_J = mean(max(0, -det(J))²)

    This ensures the displacement field is diffeomorphic (no folding).
    Applied per-phase and averaged.

    Args:
        disp: [B, num_ph-1, 2, H, W] or [B, 2, H, W]
        eps: small constant for numerical stability

    Returns:
        scalar tensor
    """
    if disp.dim() == 4:
        # Single displacement field [B, 2, H, W]
        disp = disp.unsqueeze(1)  # → [B, 1, 2, H, W]

    B, N, C, H, W = disp.shape
    total_neg = torch.tensor(0.0, device=disp.device, dtype=disp.dtype)

    for i in range(N):
        disp_i = disp[:, i, :, :, :]
        jac = jacobian_determinant_2d(disp_i)  # [B, H, W]
        neg_jac = F.relu(-jac + eps) ** 2       # [B, H, W]
        total_neg += torch.mean(neg_jac)

    return total_neg / max(N, 1)


def jacobian_regularity_loss(disp: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation of Jacobian from 1 (log-barrier for diffeomorphism).

    L_reg = mean((det(J) - 1)²)

    Combined with negative_jacobian_loss, this encourages det(J) ≈ 1 everywhere.

    Args:
        disp: [B, num_ph-1, 2, H, W] or [B, 2, H, W]

    Returns:
        scalar tensor
    """
    if disp.dim() == 4:
        disp = disp.unsqueeze(1)

    B, N, C, H, W = disp.shape
    total_reg = torch.tensor(0.0, device=disp.device, dtype=disp.dtype)

    for i in range(N):
        jac = jacobian_determinant_2d(disp[:, i, :, :, :])
        reg = (jac - 1.0) ** 2
        total_reg += torch.mean(reg)

    return total_reg / max(N, 1)


# =============================================================================
# Gate Sparsity Loss (from T-Gated Adapter, adapted)
# =============================================================================

def gate_sparsity_loss(
    gate_values: torch.Tensor,
    lambda_entropy: float = 0.001,
) -> torch.Tensor:
    """
    Entropy-based gate sparsity regularization (from T-Gated Adapter).

    Encourages gate values to be near 0 or 1 (binary decision:
    "use temporal context" OR "trust single-phase baseline").

    Without this loss, gates tend to settle at ~0.5 (partial blending everywhere),
    which is less interpretable and less robust.

    H(g) = -g·log(g) - (1-g)·log(1-g)  (binary entropy, element-wise)

    Args:
        gate_values: sigmoid-activated gate values [B, seq, D] or [B, ...]
                     Each element should be in (0, 1).
        lambda_entropy: weight for the entropy term (default 0.001)

    Returns:
        scalar tensor
    """
    # Clip to avoid log(0)
    g = gate_values.clamp(min=1e-7, max=1 - 1e-7)

    # Binary entropy: H(g) = -g*log(g) - (1-g)*log(1-g)
    entropy = -(g * torch.log(g) + (1 - g) * torch.log(1 - g))

    return lambda_entropy * torch.mean(entropy)


# =============================================================================
# Combined Physics-Informed Loss
# =============================================================================

class PhysicsInformedTemporalLoss(torch.nn.Module):
    """
    Combined physics-informed loss for temporal-aware cardiac registration.

    Total loss = L_sim + λ_smooth * L_smooth
                       + λ_1st * L_1st
                       + λ_2nd * L_2nd
                       + λ_cycle * L_cycle
                       + λ_neg_J * L_neg_J
                       + λ_gate * L_gate

    where:
      L_sim:    image similarity loss (NCC, from train.py)
      L_smooth: displacement field smoothness (from train.py)
      L_1st:    1st-order temporal smoothness (Contribution 2)
      L_2nd:    2nd-order temporal smoothness (Contribution 2)
      L_cycle:  cyclic consistency (Contribution 2)
      L_neg_J:  negative Jacobian penalty (anti-folding, Contribution 2)
      L_gate:   gate sparsity (from T-Gated Adapter, adapted)
    """
    def __init__(
        self,
        lambda_1st: float = 1.0,
        lambda_2nd: float = 0.5,
        lambda_cycle: float = 0.1,
        lambda_neg_J: float = 0.1,
        lambda_gate: float = 0.001,
    ):
        super().__init__()
        self.lambda_1st = lambda_1st
        self.lambda_2nd = lambda_2nd
        self.lambda_cycle = lambda_cycle
        self.lambda_neg_J = lambda_neg_J
        self.lambda_gate = lambda_gate

    def forward(
        self,
        displacement_fields: torch.Tensor,
        gate_values: torch.Tensor = None,
        num_phases: int = None,
    ) -> dict:
        """
        Compute all physics-informed losses.

        Args:
            displacement_fields: [B, num_ph-1, 2, H, W]
            gate_values: optional gate tensor for sparsity loss
            num_phases: total number of phases

        Returns:
            dict of loss components:
                'loss_1st': first-order temporal smoothness
                'loss_2nd': second-order temporal smoothness
                'loss_cycle': cyclic consistency
                'loss_neg_J': negative Jacobian (anti-folding)
                'loss_gate': gate sparsity
                'total': weighted sum of all
        """
        losses = {}

        losses['loss_1st'] = temporal_smoothness_loss_1st(displacement_fields)
        losses['loss_2nd'] = temporal_smoothness_loss_2nd(displacement_fields)

        if num_phases is not None:
            losses['loss_cycle'] = cyclic_consistency_loss(
                displacement_fields, num_phases)
        else:
            losses['loss_cycle'] = torch.tensor(0.0, device=displacement_fields.device)

        losses['loss_neg_J'] = negative_jacobian_loss(displacement_fields)

        if gate_values is not None:
            losses['loss_gate'] = gate_sparsity_loss(gate_values)
        else:
            losses['loss_gate'] = torch.tensor(0.0, device=displacement_fields.device)

        total = (
            self.lambda_1st * losses['loss_1st'] +
            self.lambda_2nd * losses['loss_2nd'] +
            self.lambda_cycle * losses['loss_cycle'] +
            self.lambda_neg_J * losses['loss_neg_J'] +
            self.lambda_gate * losses['loss_gate']
        )
        losses['total'] = total

        return losses
