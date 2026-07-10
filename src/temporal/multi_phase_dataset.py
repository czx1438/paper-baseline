"""
multi_phase_dataset.py
========================================================
Multi-Phase Cardiac Sequence Dataset for Temporal-Aware LDM-Morph.

Required data structure (after organizing cardiac phases):
    data_root/
    ├── fixed/fixed/phase0/*.npy   ← Reference phase (0)
    └── moving/moving/
        ├── phase1/*.npy           ← Phase 1
        ├── phase2/*.npy           ← Phase 2
        ├── phase3/*.npy           ← Phase 3
        └── ...                    ← Phase 4, 5, ... (up to phase 9)

Each patient/frame has one .npy file per phase (aligned by index).
Dataset loads a WINDOW of N consecutive phases:
    [phase t - ctx//2, ..., phase t, ..., phase t + ctx//2]
as the temporal context, with phase t as the center (moving phase).

NOTE: This is the data preparation step that needs to be done BEFORE training.
The data reorganization script (reorganize_cardiac_phases.py) should be run
first to convert raw phase folders into this structure.
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform, map_coordinates
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# Utilities
# =============================================================================

def apply_motion(img: np.ndarray, motion_type: str) -> np.ndarray:
    """
    Apply synthetic motion deformation to a 2D image.
    Extends the original XCATMotionAugmented._apply_motion() with
    cardiac-specific motion types.

    motion_type options:
      - 'identity': no deformation
      - 'rotate10': 10° rotation (cardiac rotation)
      - 'scale05': anisotropic scale (cardiac through-plane motion proxy)
      - 'warp': elastic B-spline warp (realistic cardiac deformation)
      - 'cardiac_cycle': realistic cardiac cycle simulation
    """
    h, w = img.shape

    if motion_type == 'identity':
        return img.copy()

    elif motion_type == 'rotate10':
        angle = 10 * np.pi / 180
        center = np.array([h / 2, w / 2])
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        offset = center - rot @ center
        return affine_transform(img, rot, offset=offset, order=3,
                               mode='constant', cval=0)

    elif motion_type == 'scale05':
        scale_factors = (1.0, 0.5)
        matrix = np.diag(scale_factors)
        center = np.array([h / 2, w / 2])
        offset = center - matrix @ center
        return affine_transform(img, matrix, offset=offset, order=3,
                               mode='constant', cval=0)

    elif motion_type == 'warp':
        dx = np.random.uniform(-5, 5, size=(4, 4))
        dy = np.random.uniform(-5, 5, size=(4, 4))
        x_grid = np.linspace(0, w - 1, 4)
        y_grid = np.linspace(0, h - 1, 4)
        interp_x = RegularGridInterpolator((y_grid, x_grid), dx,
                                           bounds_error=False, fill_value=0)
        interp_y = RegularGridInterpolator((y_grid, x_grid), dy,
                                           bounds_error=False, fill_value=0)
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=-1)
        x_new = x_coords + interp_x(coords).reshape(h, w)
        y_new = y_coords + interp_y(coords).reshape(h, w)
        return map_coordinates(img, [y_new, x_new], order=3,
                               mode='constant', cval=0)

    elif motion_type == 'cardiac_cycle':
        # Realistic cardiac cycle: radial expansion/contraction from centroid
        # Parameters for a realistic cardiac cycle
        amplitude = np.random.uniform(0.02, 0.08)  # 2-8% radial displacement
        phase_shift = np.random.uniform(0, 2 * np.pi)

        # Create radial displacement field
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        cx, cy = w / 2, h / 2
        r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)

        # Radial displacement (normalized, pointing outward)
        dr = amplitude * r / (r_max + 1e-8)

        # Displacement vectors
        ux = dr * (x_coords - cx) / (r + 1e-8)
        uy = dr * (y_coords - cy) / (r + 1e-8)

        # Apply
        new_x = x_coords + ux
        new_y = y_coords + uy
        return map_coordinates(img, [new_y, new_x], order=3,
                               mode='constant', cval=0)

    else:
        return img.copy()


# =============================================================================
# Multi-Phase Sequence Dataset
# =============================================================================

class MultiPhaseSequenceDataset(Dataset):
    """
    Loads a window of N consecutive cardiac phases for temporal registration.

    Data structure expected:
        data_root/
        ├── fixed/fixed/phase0/*.npy  ← Reference (phase 0 = fixed)
        └── moving/moving/
            ├── phase1/*.npy
            ├── phase2/*.npy
            ├── phase3/*.npy
            └── ...

    Each .npy file is a 2D image (H, W). Files are aligned by patient/frame index.

    For each sample:
      - fixed_image:  phase 0 (reference, NOT augmented)
      - phase_sequence: list of N tensors [phase t-ctx//2, ..., phase t+ctx//2]
                        each is a phase-t image with optional motion augmentation
      - center_phase: index t (the phase being registered)
      - name: patient/frame identifier

    IMPORTANT: The center phase is ALWAYS the "moving" image, registered to phase 0.

    Args:
        data_root: root directory containing fixed/ and moving/ subdirectories
        split: 'train' | 'val' | 'test'
        context_size: number of phases in the temporal window (default 5, odd number)
        flip_p: probability of random horizontal flip
        motion_types: list of synthetic motion augmentation types
        cardiac_motion_p: probability of applying cardiac_cycle augmentation
        normalize_with_fixed: if True, normalize using fixed image min/max (recommended)
        patient_ids: optional list of patient IDs for grouped train/val/test splits
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        context_size: int = 5,
        flip_p: float = 0.5,
        motion_types: list = None,
        cardiac_motion_p: float = 0.3,
        normalize_with_fixed: bool = True,
        patient_ids: list = None,
    ):
        self.data_root = data_root
        self.split = split
        self.context_size = context_size
        self.flip_p = flip_p if split == 'train' else 0.0
        self.motion_types = motion_types or ['identity', 'rotate10', 'scale05', 'warp']
        self.cardiac_motion_p = cardiac_motion_p
        self.normalize_with_fixed = normalize_with_fixed

        assert context_size % 2 == 1, "context_size must be odd"
        self.half_ctx = context_size // 2  # e.g., 2 for context_size=5

        # -----------------------------------------------------------------
        # Discover phase directories
        # -----------------------------------------------------------------
        fixed_base = os.path.join(data_root, 'fixed', 'fixed', 'phase0')
        moving_base = os.path.join(data_root, 'moving', 'moving')

        # Check if new multi-phase structure exists
        if os.path.isdir(fixed_base):
            self.mode = 'multi_phase'
            self._discover_multi_phase(fixed_base, moving_base, patient_ids)
        else:
            # Fallback to legacy single-phase structure (fixed/fixed/*.npy)
            self.mode = 'single_phase'
            self._discover_single_phase(data_root, patient_ids)

        self._build_samples()

        print(f"[MultiPhaseSequenceDataset] mode={self.mode}, split={split}, "
              f"num_patients={len(self.patients)}, total_samples={len(self.samples)}")

    def _discover_multi_phase(self, fixed_base, moving_base, patient_ids):
        """Discover all phases and files in multi-phase structure."""
        # Get list of all .npy files in fixed/phase0
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_base, '*.npy')))
        n = len(self.fixed_paths)

        # Discover available phases
        self.available_phases = []
        for p in range(10):  # Check phases 0-9
            phase_dir = os.path.join(moving_base, f'phase{p}')
            if os.path.isdir(phase_dir):
                n_phase = len(glob.glob(os.path.join(phase_dir, '*.npy')))
                if n_phase > 0:
                    self.available_phases.append(p)
                    setattr(self, f'moving_phase{p}_paths',
                            sorted(glob.glob(os.path.join(phase_dir, '*.npy'))))

        if not self.available_phases:
            raise FileNotFoundError(
                f"No multi-phase directories found in {moving_base}. "
                f"Expected moving/moving/phase{{1..9}}/*.npy"
            )

        print(f"  Discovered phases: {self.available_phases}")
        self.num_phases_total = len(self.available_phases)

        # Determine patient IDs from file indices
        if patient_ids is None:
            self.patients = list(range(n))
        else:
            self.patients = patient_ids[:n]

    def _discover_single_phase(self, data_root, patient_ids):
        """Fallback: single-phase structure (legacy XCAT format)."""
        fixed_dir = os.path.join(data_root, 'fixed', 'fixed')
        moving_dir = os.path.join(data_root, 'moving', 'moving')
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.npy')))
        self.available_phases = [1]  # single "phase 1"
        setattr(self, 'moving_phase1_paths',
                sorted(glob.glob(os.path.join(moving_dir, '*.npy'))))
        n = len(self.fixed_paths)
        if patient_ids is None:
            self.patients = list(range(n))
        else:
            self.patients = patient_ids[:n]
        self.num_phases_total = 1

    def _build_samples(self):
        """Build the sample list: each sample = (patient_idx, center_phase_idx)."""
        self.samples = []
        n = len(self.patients)

        if self.split == 'train':
            n_train = int(n * 0.7)
            patient_range = self.patients[:n_train]
        elif self.split == 'val':
            n_train = int(n * 0.7)
            n_val = int(n * 0.85)
            patient_range = self.patients[n_train:n_val]
        else:  # test
            n_train = int(n * 0.7)
            patient_range = self.patients[n_train:]

        # For each patient, we can center on different phases
        # Only center phases that have enough context (not at boundaries)
        valid_center_phases = []
        for p in self.available_phases:
            phase_idx = self.available_phases.index(p)
            if phase_idx >= self.half_ctx and phase_idx < len(self.available_phases) - self.half_ctx:
                valid_center_phases.append(p)

        if not valid_center_phases:
            # If no valid centers, just use the first available phase
            valid_center_phases = [self.available_phases[0]]

        for patient_idx in patient_range:
            for center_phase in valid_center_phases:
                # Add motion type variation for training
                if self.split == 'train':
                    for mt in self.motion_types:
                        self.samples.append((patient_idx, center_phase, mt))
                else:
                    self.samples.append((patient_idx, center_phase, 'identity'))

        # Shuffle training set
        if self.split == 'train':
            import random
            random.seed(42)
            random.shuffle(self.samples)

    def _load_phase_image(self, patient_idx: int, phase: int) -> np.ndarray:
        """Load a single phase image for a patient."""
        if phase == 0:
            path = self.fixed_paths[patient_idx]
        else:
            paths = getattr(self, f'moving_phase{phase}_paths', None)
            if paths is None:
                raise ValueError(f"Phase {phase} not available")
            path = paths[patient_idx]
        return np.load(path).astype(np.float32)

    def _get_phase_sequence(self, patient_idx: int, center_phase: int,
                            motion_type: str) -> list:
        """Load a window of N phases around center_phase."""
        sequence = []
        phase_indices = self.available_phases

        for offset in range(-self.half_ctx, self.half_ctx + 1):
            target_phase = center_phase + offset
            # Handle boundary: clamp to available phases
            if target_phase not in phase_indices:
                target_phase = center_phase  # repeat center if out of range

            img = self._load_phase_image(patient_idx, target_phase)

            # Apply motion augmentation (only to non-fixed phases)
            if target_phase != 0 and motion_type != 'identity':
                # For cardiac_cycle, use it with probability cardiac_motion_p
                if motion_type == 'cardiac_cycle':
                    if np.random.rand() < self.cardiac_motion_p:
                        img = apply_motion(img, 'cardiac_cycle')
                else:
                    img = apply_motion(img, motion_type)

            sequence.append(img)

        return sequence

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        patient_idx, center_phase, motion_type = self.samples[idx]

        # Load phase sequence
        phase_sequence = self._get_phase_sequence(patient_idx, center_phase, motion_type)

        # Load fixed image (phase 0, no augmentation)
        fixed_img = self._load_phase_image(patient_idx, 0)

        # Min-Max normalization
        if self.normalize_with_fixed:
            minv = fixed_img.min()
            maxv = fixed_img.max()
        else:
            # Use global min/max across all phases in this sample
            all_imgs = [fixed_img] + phase_sequence
            minv = min(img.min() for img in all_imgs)
            maxv = max(img.max() for img in all_imgs)

        eps = 1e-6
        if maxv - minv > eps:
            fixed_img = (fixed_img - minv) / (maxv - minv)
            phase_sequence = [(img - minv) / (maxv - minv) for img in phase_sequence]

        # Random horizontal flip (all images together, preserving alignment)
        if np.random.rand() < self.flip_p:
            fixed_img = np.fliplr(fixed_img).copy()
            phase_sequence = [np.fliplr(img).copy() for img in phase_sequence]

        # Convert to tensors
        fixed_t = torch.from_numpy(fixed_img).float()  # [H, W]

        phase_tensors = [
            torch.from_numpy(img).float() for img in phase_sequence
        ]  # list of [H, W]

        # Stack phase sequence: [num_phases, H, W]
        phase_seq_tensor = torch.stack(phase_tensors, dim=0)

        # Determine center phase index within the sequence
        center_seq_idx = self.half_ctx  # sequence is centered around target

        sample_name = f"patient{patient_idx:04d}_phase{center_phase}"

        return {
            'fixed': fixed_t,                      # [H, W]
            'phase_sequence': phase_seq_tensor,     # [num_phases, H, W]
            'center_seq_idx': center_seq_idx,      # int (which index in sequence is center)
            'center_phase': center_phase,           # int (actual phase number)
            'name': sample_name,
        }


# =============================================================================
# Data Reorganization Utility
# =============================================================================

def reorganize_cardiac_phases(
    raw_data_root: str,
    output_root: str,
    phase_mapping: dict = None,
):
    """
    Reorganize raw cardiac phase data into the multi-phase structure required
    by MultiPhaseSequenceDataset.

    Args:
        raw_data_root: directory containing raw .npy files (all phases mixed)
                       Expected structure:
                           raw_data_root/
                           ├── patient_001/
                           │   ├── phase_0.npy   (H, W)
                           │   ├── phase_1.npy
                           │   └── ...
                           └── ...
        output_root: target directory with reorganized structure
        phase_mapping: optional dict mapping raw phase names to phase indices
                       e.g., {'ED': 0, 'ES': 1, 'mid': 2, ...}

    Example output structure:
        output_root/
        ├── fixed/fixed/phase0/
        │   ├── 000.npy   ← patient 0, phase 0
        │   ├── 001.npy   ← patient 1, phase 0
        │   └── ...
        └── moving/moving/
            ├── phase1/
            ├── phase2/
            └── ...
    """
    import shutil

    if phase_mapping is None:
        phase_mapping = {
            'ED': 0,   # End-Diastole (reference/fixed)
            'ES': 1,   # End-Systole
            'mid': 2,
            'phase3': 3,
            'phase4': 4,
            'phase5': 5,
            'phase6': 6,
            'phase7': 7,
            'phase8': 8,
            'phase9': 9,
        }

    # Find all patient directories
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(raw_data_root, '*'))
        if os.path.isdir(d)
    ])

    # Discover all phases across all patients
    all_phases = set()
    for pdir in patient_dirs:
        for fname in os.listdir(pdir):
            if fname.endswith('.npy'):
                # Try to extract phase name
                stem = fname.replace('.npy', '')
                for key in phase_mapping:
                    if key in stem:
                        all_phases.add(key)
                        break

    print(f"Found patients: {len(patient_dirs)}")
    print(f"Found phases: {all_phases}")

    # Create output directories
    os.makedirs(os.path.join(output_root, 'fixed', 'fixed', 'phase0'), exist_ok=True)
    for phase_name in sorted(all_phases):
        if phase_name != 'ED':  # ED is fixed
            phase_idx = phase_mapping.get(phase_name, -1)
            if phase_idx > 0:
                os.makedirs(
                    os.path.join(output_root, 'moving', 'moving', f'phase{phase_idx}'),
                    exist_ok=True
                )

    # Copy files
    for p_idx, pdir in enumerate(patient_dirs):
        for fname in sorted(os.listdir(pdir)):
            if not fname.endswith('.npy'):
                continue

            # Find phase name
            matched_phase = None
            for key in sorted(phase_mapping.keys(), key=len, reverse=True):
                if key in fname:
                    matched_phase = key
                    break

            if matched_phase is None:
                print(f"Warning: could not match phase for {fname}, skipping")
                continue

            phase_idx = phase_mapping[matched_phase]
            src = os.path.join(pdir, fname)
            dst = None

            if matched_phase == 'ED':
                dst = os.path.join(output_root, 'fixed', 'fixed', 'phase0',
                                   f'{p_idx:06d}.npy')
            else:
                dst = os.path.join(output_root, 'moving', 'moving',
                                   f'phase{phase_idx}', f'{p_idx:06d}.npy')

            if dst:
                shutil.copy2(src, dst)

    print(f"Reorganization complete: {output_root}")
    print(f"  fixed/fixed/phase0/: {len(glob.glob(os.path.join(output_root, 'fixed', 'fixed', 'phase0', '*.npy')))} files")
    for p in range(1, 10):
        n_files = len(glob.glob(os.path.join(output_root, 'moving', 'moving', f'phase{p}', '*.npy')))
        if n_files > 0:
            print(f"  moving/moving/phase{p}/: {n_files} files")
