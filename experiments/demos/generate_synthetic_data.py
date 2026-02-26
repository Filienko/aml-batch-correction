#!/usr/bin/env python3
"""Generate synthetic single cell data for testing.

This script creates a small synthetic dataset that can be used to test
the SCCL pipeline without needing large real datasets.
"""

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl.data import generate_synthetic_data

# Generate synthetic data
print("Generating synthetic single cell data...")

adata = generate_synthetic_data(
    n_cells=2000,
    n_genes=1500,
    n_cell_types=6,
    n_batches=3,
    batch_effect_strength=0.3,
    noise_level=0.5,
    seed=42,
)

# Print info
print(f"\nGenerated data:")
print(f"  Cells: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")
print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
print(f"  Batches: {adata.obs['batch'].nunique()}")

print("\nCell type distribution:")
print(adata.obs['cell_type'].value_counts())

print("\nBatch distribution:")
print(adata.obs['batch'].value_counts())

# Save
output_dir = Path(__file__).parent.parent.parent / "data"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "synthetic_example.h5ad"
adata.write_h5ad(output_file)

print(f"\nSaved synthetic data to: {output_file}")
print("\nYou can now use this data with the SCCL pipeline!")
print("\nExample:")
print(f"  sccl evaluate --data {output_file} --model random_forest --target cell_type")
