#!/usr/bin/env python3
"""Example 3: Batch correction with SCimilarity or scVI.

This example demonstrates:
1. Working with data that has batch effects
2. Using models that handle batch correction
3. Visualizing batch effects before/after
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sccl import Pipeline
from sccl.data import generate_synthetic_data
import scanpy as sc

print("="*60)
print("EXAMPLE 3: Batch Correction")
print("="*60)

# Generate data with strong batch effects
print("\nGenerating data with batch effects...")
adata = generate_synthetic_data(
    n_cells=1500,
    n_genes=1000,
    n_batches=4,
    batch_effect_strength=0.6,  # Strong batch effect
    seed=42,
)

print(f"Generated {adata.n_obs} cells with {adata.obs['batch'].nunique()} batches")

# Preprocess and visualize uncorrected data
print("\nVisualizing uncorrected data...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000)
adata_hvg = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_hvg)
sc.tl.pca(adata_hvg)
sc.pp.neighbors(adata_hvg)
sc.tl.umap(adata_hvg)

# Plot uncorrected
try:
    sc.pl.umap(adata_hvg, color=['batch', 'cell_type'], save='_uncorrected.pdf')
    print("✓ Saved uncorrected UMAP to figures/umap_uncorrected.pdf")
except Exception as e:
    print(f"Could not generate plot: {e}")

# Option 1: Try SCimilarity (if installed)
print("\n" + "="*60)
print("Option 1: SCimilarity (foundation model)")
print("="*60)

try:
    pipeline_scim = Pipeline(model="scimilarity", batch_key="batch")

    # SCimilarity handles batch effects implicitly
    predictions_scim = pipeline_scim.predict(
        adata=adata.copy(),
        target_column="cell_type",
    )

    print("✓ SCimilarity prediction successful!")

except ImportError:
    print("SCimilarity not installed. Install with: pip install scimilarity")
except Exception as e:
    print(f"Error: {e}")

# Option 2: Traditional model (doesn't correct batches)
print("\n" + "="*60)
print("Option 2: Random Forest (no batch correction)")
print("="*60)

pipeline_rf = Pipeline(model="random_forest")

metrics_rf = pipeline_rf.evaluate(
    adata=adata.copy(),
    target_column="cell_type",
    test_size=0.2,
)

print("Random Forest results:")
for metric, value in metrics_rf.items():
    print(f"  {metric:20s}: {value:.4f}")

print("\n✓ Example completed!")
print("\nKey takeaways:")
print("  - Batch effects can hurt model performance")
print("  - SCimilarity and scVI can handle batch effects")
print("  - Traditional models may struggle with strong batch effects")
