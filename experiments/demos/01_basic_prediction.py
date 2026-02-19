#!/usr/bin/env python3
"""Example 1: Basic SCimilarity prediction workflow with reference/query datasets.

This example demonstrates:
1. Loading/generating reference and query datasets
2. Computing SCimilarity embeddings
3. Training a classifier on reference embeddings
4. Predicting cell types on query data
5. Evaluating results
"""

import sys
from pathlib import Path
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import generate_synthetic_data, subset_data, preprocess_data
from sccl.evaluation import compute_metrics

print("=" * 60)
print("EXAMPLE 1: SCimilarity Reference/Query Prediction")
print("=" * 60)

# Step 1: Generate reference and query datasets
print("\nStep 1: Generating reference and query datasets...")

# Generate data with multiple batches
adata = generate_synthetic_data(
    n_cells=2000,
    n_genes=2000,
    n_cell_types=5,
    n_batches=4,
    batch_effect_strength=0.4,
    seed=42,
)

# Split into reference (Batch_1, Batch_2) and query (Batch_3, Batch_4)
adata_ref = subset_data(adata, studies=['Batch_1', 'Batch_2'])
adata_query = subset_data(adata, studies=['Batch_3', 'Batch_4'])

print(f"Reference: {adata_ref.n_obs} cells from Batch_1, Batch_2")
print(f"Query:     {adata_query.n_obs} cells from Batch_3, Batch_4")
print(f"Cell types: {adata_ref.obs['cell_type'].unique().tolist()}")

# Step 2: Create SCimilarity pipeline
print("\nStep 2: Creating SCimilarity pipeline...")

try:
    # Using KNN classifier on SCimilarity embeddings
    pipeline = Pipeline(
        model="scimilarity",
        preprocess=True,
        model_params={
            'classifier': 'knn',
            'n_neighbors': 15,
        }
    )

    # Step 3: Train on reference data
    print("\nStep 3: Training on reference data...")

    # Preprocess reference
    adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)

    # Fit the model (computes embeddings + trains classifier)
    pipeline.model.fit(adata_ref_prep, target_column='cell_type')

    # Step 4: Predict on query data
    print("\nStep 4: Predicting on query data...")

    # Preprocess query
    adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

    # Get predictions
    predictions = pipeline.model.predict(adata_query_prep)

    # Step 5: Evaluate results
    print("\nStep 5: Evaluating predictions...")

    y_true = adata_query.obs['cell_type'].values
    metrics = compute_metrics(y_true=y_true, y_pred=predictions)

    print("\n" + "=" * 60)
    print("RESULTS: Cross-batch cell type prediction")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    print("=" * 60)

    # Optional: Get embeddings for visualization
    print("\nStep 6: Getting embeddings for visualization...")
    embeddings = pipeline.model.get_embedding(adata_query_prep)
    print(f"Embedding shape: {embeddings.shape}")

except ImportError as e:
    print(f"\nSCimilarity not available: {e}")
    print("Falling back to Random Forest baseline...")

    # Fallback: use Random Forest
    pipeline = Pipeline(model="random_forest", preprocess=True)

    adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
    adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

    pipeline.model.fit(adata_ref_prep, target_column='cell_type')
    predictions = pipeline.model.predict(adata_query_prep)

    y_true = adata_query.obs['cell_type'].values
    metrics = compute_metrics(y_true=y_true, y_pred=predictions)

    print("\n" + "=" * 60)
    print("RESULTS: Random Forest baseline")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    print("=" * 60)

print("\nKey takeaways:")
print("  - SCimilarity computes embeddings that are batch-invariant")
print("  - Train a classifier (KNN, RF, etc.) on reference embeddings")
print("  - Apply classifier to query embeddings for label transfer")
print("  - No need for explicit batch correction!")
