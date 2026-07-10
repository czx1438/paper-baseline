#!/usr/bin/env python3
"""
Visualize the last 15% of npy files in xcat_data/moving/moving/
Files: 846.npy to 999.npy (154 files)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Paths
data_dir = Path("/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/moving/moving")
output_dir = Path("/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/visualizations")
output_dir.mkdir(exist_ok=True)

# Get all npy files and sort
all_files = sorted([f for f in data_dir.glob("*.npy")], key=lambda x: int(x.stem))
total_files = len(all_files)
print(f"Total npy files: {total_files}")

# Calculate last 15%
num_last_15 = int(total_files * 0.15)
start_idx = total_files - num_last_15
last_15_files = all_files[start_idx:]
print(f"Last 15%: {num_last_15} files (indices {start_idx} to {total_files-1})")
print(f"File range: {last_15_files[0].stem} to {last_15_files[-1].stem}")

# Load a sample to check dimensions
sample = np.load(last_15_files[0])
h, w = sample.shape
print(f"Image shape: {h}x{w}")

# Create grid layout
n_images = len(last_15_files)
cols = 14
rows = (n_images + cols - 1) // cols  # ceiling division
print(f"Grid layout: {rows} rows x {cols} cols = {rows * cols} slots")

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
fig.suptitle(f'XCAT Moving Data - Last 15% (Files {last_15_files[0].stem} to {last_15_files[-1].stem})', 
             fontsize=16, fontweight='bold', y=0.995)

# Flatten axes for easy iteration
axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

# Plot each image
for idx, filepath in enumerate(last_15_files):
    ax = axes_flat[idx]
    data = np.load(filepath)
    
    # Normalize to 0-1 for display
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    ax.imshow(data_norm, cmap='gray', aspect='equal')
    ax.set_title(filepath.stem, fontsize=6, pad=2)
    ax.axis('off')

# Hide unused subplots
for idx in range(n_images, len(axes_flat)):
    axes_flat[idx].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save
output_path = output_dir / "moving_last_15_percent_grid.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved grid visualization to: {output_path}")

# Also create individual samples at regular intervals
sample_indices = [0, len(last_15_files)//4, len(last_15_files)//2, 
                  3*len(last_15_files)//4, len(last_15_files)-1]

fig2, axes2 = plt.subplots(1, len(sample_indices), figsize=(4*len(sample_indices), 4))
fig2.suptitle('Sample Images from Last 15% (Interval Views)', fontsize=14, fontweight='bold')

for i, idx in enumerate(sample_indices):
    ax = axes2[i]
    filepath = last_15_files[idx]
    data = np.load(filepath)
    
    ax.imshow(data, cmap='gray')
    ax.set_title(f'{filepath.stem}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
output_path2 = output_dir / "moving_last_15_percent_samples.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved sample visualization to: {output_path2}")

# Create a 3D-like montage showing the sequence progression
print("\nCreating sequence progression view...")
fig3, ax3 = plt.subplots(figsize=(16, 16))

# Sample every 5th image for a montage
montage_files = last_15_files[::5]
n_cols = 8
n_rows = (len(montage_files) + n_cols - 1) // n_cols

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
fig3.suptitle('Sequence Progression (Every 5th image from last 15%)', 
              fontsize=14, fontweight='bold')

if n_rows > 1:
    axes3_flat = axes3.flatten()
else:
    axes3_flat = axes3

for idx, filepath in enumerate(montage_files):
    if idx >= len(axes3_flat):
        break
    ax = axes3_flat[idx]
    data = np.load(filepath)
    ax.imshow(data, cmap='gray')
    ax.set_title(f'{filepath.stem}', fontsize=7)
    ax.axis('off')

for idx in range(len(montage_files), len(axes3_flat)):
    axes3_flat[idx].axis('off')

plt.tight_layout()
output_path3 = output_dir / "moving_last_15_percent_montage.png"
plt.savefig(output_path3, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved montage to: {output_path3}")

print("\n✓ Visualization complete!")
print(f"  - Grid view: {output_path}")
print(f"  - Sample view: {output_path2}")
print(f"  - Montage view: {output_path3}")
