#!/usr/bin/env python
"""
Quick diagnostic script to check dataset size and suggest solutions
"""

import scanpy as sc
import numpy as np

DATA_PATH = "data/AML_scAtlas.h5ad"

print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)

print(f"\nDataset info:")
print(f"  Cells: {adata.n_obs:,}")
print(f"  Genes: {adata.n_vars:,}")
print(f"  log2(n_obs) = {np.log2(adata.n_obs):.2f}")

if adata.n_obs > 100000:
    print(f"\n⚠ Large dataset detected!")
    print(f"  The pynndescent library has issues with datasets > 100k cells")
    print(f"  Options:")
    print(f"    1. Use the updated code with automatic subsampling")
    print(f"    2. Manually subsample your data before benchmarking")
    print(f"    3. Use a different neighbor computation method")
else:
    print(f"\n✓ Dataset size should be fine for benchmarking")

# Check embeddings
print(f"\nAvailable embeddings:")
for key in adata.obsm.keys():
    if key.startswith('X_'):
        print(f"  {key}: {adata.obsm[key].shape}")
