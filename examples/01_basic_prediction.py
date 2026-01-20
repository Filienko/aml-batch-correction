#!/usr/bin/env python3
"""Example 1: Basic prediction workflow.

This example demonstrates:
1. Loading data
2. Creating a pipeline
3. Training and predicting with a simple model
4. Evaluating results
"""

import sys
from pathlib import Path
import scanpy as sc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sccl import Pipeline
from sccl.data import generate_synthetic_data

print("="*60)
print("EXAMPLE 1: Basic Prediction Workflow")
print("="*60)

# Step 1: Generate or load data
print("\nStep 1: Loading data...")

# Option A: Generate synthetic data
adata = generate_synthetic_data(n_cells=1000, n_genes=1000, seed=42)

# Option B: Load real data (uncomment to use)
# adata = sc.read_h5ad("data/your_data.h5ad")

print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

# Step 2: Create pipeline with Random Forest model
print("\nStep 2: Creating pipeline with Random Forest...")

pipeline = Pipeline(
    model="random_forest",
    preprocess=True,
)

# Step 3: Evaluate on test set
print("\nStep 3: Evaluating model...")

metrics = pipeline.evaluate(
    adata=adata,
    target_column="cell_type",
    test_size=0.2,
)

# Step 4: Print results
print("\nStep 4: Results")
print("="*60)
for metric, value in metrics.items():
    print(f"{metric:20s}: {value:.4f}")
print("="*60)

print("\n✓ Example completed successfully!")
print("\nKey takeaways:")
print("  - Pipeline is easy to use: just create and call evaluate()")
print("  - Preprocessing happens automatically")
print("  - Multiple metrics computed automatically")
