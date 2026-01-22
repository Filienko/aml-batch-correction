#!/usr/bin/env python3
"""
Quick Start: Cell Type Annotation
==================================

This script demonstrates the complete workflow for cell type annotation:
1. Setup and data loading
2. Initialize pipeline with SCimilarity
3. Get embeddings (optional)
4. Train classifier on reference
5. Predict on query data
6. Evaluate results

Works on headless VM (no plotting).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scanpy as sc
from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column, generate_synthetic_data
from sccl.evaluation import compute_metrics

print("="*80)
print("QUICK START: Cell Type Annotation")
print("="*80)

# ==============================================================================
# STEP 1: Setup and Data Loading
# ==============================================================================

print("\n📦 STEP 1: Setup and Data Loading")
print("-" * 80)

# For demo purposes, generate synthetic data with 2 "studies"
print("Generating synthetic multi-study dataset...")
adata = generate_synthetic_data(n_cells=2000, n_genes=1000, seed=42)

# Add a fake "study" column to simulate multi-study data
import numpy as np
n_cells = adata.n_obs
study_labels = np.random.choice(['reference_study', 'query_study'], size=n_cells, p=[0.6, 0.4])
adata.obs['study'] = study_labels

print(f"✓ Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"  Studies: {adata.obs['study'].unique().tolist()}")
print(f"  Cell types: {adata.obs['cell_type'].nunique()}")

# Auto-detect column names
study_col = get_study_column(adata)
cell_type_col = get_cell_type_column(adata)
print(f"\n✓ Auto-detected columns:")
print(f"  Study column: '{study_col}'")
print(f"  Cell type column: '{cell_type_col}'")

# ==============================================================================
# STEP 2: Split into Reference and Query
# ==============================================================================

print("\n📂 STEP 2: Split into Reference and Query")
print("-" * 80)

# Split by study
adata_ref = subset_data(adata, studies=['reference_study'])
adata_query = subset_data(adata, studies=['query_study'])

print(f"Reference dataset: {adata_ref.n_obs:,} cells")
print(f"  Cell types: {adata_ref.obs[cell_type_col].nunique()}")
print(f"\nQuery dataset: {adata_query.n_obs:,} cells")
print(f"  Cell types (ground truth): {adata_query.obs[cell_type_col].nunique()}")

# ==============================================================================
# STEP 3: Initialize Pipeline - Compare Models
# ==============================================================================

print("\n🔧 STEP 3: Initialize Pipeline")
print("-" * 80)

# We'll compare SCimilarity vs Traditional ML
models_to_test = {
    'Random Forest': ('random_forest', {}),
    'KNN': ('knn', {}),
}

# Note: SCimilarity requires actual model file, skip for synthetic demo
# Uncomment if you have SCimilarity model:
# models_to_test['SCimilarity+KNN'] = ('scimilarity', {
#     'model_path': '/path/to/model_v1.1',
#     'classifier': 'knn'
# })

print(f"Testing {len(models_to_test)} models:")
for name in models_to_test.keys():
    print(f"  • {name}")

# ==============================================================================
# STEP 4: Train and Predict
# ==============================================================================

print("\n🎯 STEP 4: Train Classifier on Reference, Predict on Query")
print("-" * 80)

results = []

for model_name, (model_type, model_params) in models_to_test.items():
    print(f"\n{model_name}:")

    # Create pipeline
    if model_type == 'scimilarity':
        pipeline = Pipeline(model=model_type, model_params=model_params)
    else:
        pipeline = Pipeline(model=model_type)

    # Preprocess reference data
    print("  → Preprocessing reference data...")
    adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)

    # Train on reference
    print("  → Training on reference data...")
    pipeline.model.fit(adata_ref_prep, target_column=cell_type_col)
    print(f"    ✓ Trained on {adata_ref_prep.n_obs:,} cells")

    # Preprocess query data
    print("  → Preprocessing query data...")
    adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

    # Predict on query
    print("  → Predicting on query data...")
    predictions = pipeline.model.predict(adata_query_prep)
    print(f"    ✓ Predicted {len(predictions):,} cells")

    # Evaluate
    metrics = compute_metrics(
        y_true=adata_query.obs[cell_type_col].values,
        y_pred=predictions,
        metrics=['accuracy', 'ari', 'nmi']
    )

    results.append({
        'model': model_name,
        'accuracy': metrics['accuracy'],
        'ari': metrics['ari'],
        'nmi': metrics['nmi'],
    })

    print(f"  → Results:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    ARI:      {metrics['ari']:.4f}")
    print(f"    NMI:      {metrics['nmi']:.4f}")

# ==============================================================================
# STEP 5: Summary
# ==============================================================================

print("\n" + "="*80)
print("📊 SUMMARY: Model Comparison")
print("="*80)

import pandas as pd
results_df = pd.DataFrame(results).set_index('model')
print(results_df.to_string())

best_model = results_df['ari'].idxmax()
best_ari = results_df.loc[best_model, 'ari']

print(f"\n🏆 Best Model: {best_model}")
print(f"   ARI: {best_ari:.4f}")

# ==============================================================================
# STEP 6: Key Takeaways
# ==============================================================================

print("\n" + "="*80)
print("✅ KEY TAKEAWAYS")
print("="*80)
print("""
1. ✓ Easy workflow: load → split → train → predict → evaluate
2. ✓ Automatic column detection (study, cell_type)
3. ✓ Preprocessing handled automatically
4. ✓ Multiple metrics computed (accuracy, ARI, NMI)
5. ✓ Works on headless VM (no plotting required)

Next steps:
  • Try with your own data: modify DATA_PATH above
  • Use SCimilarity with real model file
  • Add more models to comparison
  • Check experiments/paper/ for research-grade benchmarks
""")

print("="*80)
print("✓ Quick Start completed successfully!")
print("="*80)
