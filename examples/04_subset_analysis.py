#!/usr/bin/env python3
"""Example 4: Working with data subsets.

This example demonstrates:
1. Subsetting data by studies or cell types
2. Running analysis on specific subsets
3. Practical use case: train on one study, test on another
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sccl import Pipeline
from sccl.data import generate_synthetic_data, subset_data

print("="*60)
print("EXAMPLE 4: Subset Analysis")
print("="*60)

# Generate data
print("\nGenerating data...")
adata = generate_synthetic_data(
    n_cells=2000,
    n_genes=1000,
    n_batches=4,
    seed=42,
)

print(f"Full dataset: {adata.n_obs} cells")
print(f"Batches: {adata.obs['batch'].value_counts().to_dict()}")

# Example 1: Subset to specific batches
print("\n" + "="*60)
print("Example 4.1: Subset to specific batches")
print("="*60)

adata_subset = subset_data(
    adata,
    studies=['Batch_1', 'Batch_2'],  # Keep only first two batches
)

print(f"Subset: {adata_subset.n_obs} cells")
print(f"Batches: {adata_subset.obs['batch'].value_counts().to_dict()}")

# Train on subset
pipeline = Pipeline(model="random_forest")
metrics = pipeline.evaluate(
    adata=adata_subset,
    target_column="cell_type",
    test_size=0.2,
)

print("\nResults on subset:")
for metric, value in metrics.items():
    print(f"  {metric:20s}: {value:.4f}")

# Example 2: Subset to specific cell types
print("\n" + "="*60)
print("Example 4.2: Focus on specific cell types")
print("="*60)

# Get most common cell types
top_cell_types = adata.obs['cell_type'].value_counts().head(3).index.tolist()

adata_celltypes = subset_data(
    adata,
    cell_types=top_cell_types,
)

print(f"Subset: {adata_celltypes.n_obs} cells")
print(f"Cell types: {adata_celltypes.obs['cell_type'].unique()}")

# Example 3: Practical use case - cross-study generalization
print("\n" + "="*60)
print("Example 4.3: Cross-study generalization")
print("="*60)

print("\nScenario: Train on Batch_1, test on Batch_2")

# Split by batch
train_batches = ['Batch_1', 'Batch_3']
test_batches = ['Batch_2']

adata_train = subset_data(adata, studies=train_batches)
adata_test = subset_data(adata, studies=test_batches)

print(f"Training: {adata_train.n_obs} cells from {train_batches}")
print(f"Testing: {adata_test.n_obs} cells from {test_batches}")

# Train on training batches
from sccl.data import preprocess_data
adata_train_prep = preprocess_data(adata_train)
adata_test_prep = preprocess_data(adata_test)

pipeline = Pipeline(model="random_forest", preprocess=False)

# Fit on train
if hasattr(pipeline.model, 'fit'):
    pipeline.model.fit(adata_train_prep, target_column='cell_type')

# Predict on test
predictions = pipeline.model.predict(adata_test_prep, target_column='cell_type')

# Evaluate
from sccl.evaluation import compute_metrics
metrics = compute_metrics(
    y_true=adata_test_prep.obs['cell_type'].values,
    y_pred=predictions,
)

print("\nCross-study generalization results:")
for metric, value in metrics.items():
    print(f"  {metric:20s}: {value:.4f}")

print("\n✓ Example completed!")
print("\nKey takeaways:")
print("  - subset_data() makes it easy to work with specific data slices")
print("  - Can subset by batches, cell types, or custom filters")
print("  - Useful for testing cross-study generalization")
