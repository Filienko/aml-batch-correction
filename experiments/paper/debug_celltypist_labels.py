"""Debug CellTypist label predictions to see label mismatch."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scanpy as sc
from sccl import Pipeline
from sccl.data import subset_data, get_study_column, get_cell_type_column, preprocess_data
import pandas as pd
import numpy as np

# Paths
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"

print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)

# Get columns
study_col = get_study_column(adata)
cell_type_col = get_cell_type_column(adata)

# Test on same scenario as experiment: beneyto -> jiang
print("\n" + "="*80)
print("Test: beneyto-calabuig-2023 -> jiang_2020 (Same Platform: 10x)")
print("="*80)

# Get reference and query
adata_ref = subset_data(adata, studies=['beneyto-calabuig-2023'])
adata_query = subset_data(adata, studies=['jiang_2020'])

# Subsample for speed
if adata_ref.n_obs > 5000:
    indices = np.random.choice(adata_ref.n_obs, 5000, replace=False)
    adata_ref = adata_ref[indices].copy()

if adata_query.n_obs > 2000:
    indices = np.random.choice(adata_query.n_obs, 2000, replace=False)
    adata_query = adata_query[indices].copy()

print(f"\nReference cell types ({adata_ref.obs[cell_type_col].nunique()} unique):")
ref_types = adata_ref.obs[cell_type_col].value_counts()
print(ref_types)

print(f"\nQuery ground truth cell types ({adata_query.obs[cell_type_col].nunique()} unique):")
query_types = adata_query.obs[cell_type_col].value_counts()
print(query_types)

# Train CellTypist
print("\n" + "="*80)
print("Training CellTypist...")
print("="*80)
pipeline = Pipeline(model='celltypist')

adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
pipeline.model.fit(adata_ref_prep, target_column=cell_type_col)

# Predict
print("\nPredicting on query...")
adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)
predictions = pipeline.model.predict(adata_query_prep, target_column=None)

print(f"\nCellTypist predictions ({len(set(predictions))} unique):")
pred_types = pd.Series(predictions).value_counts()
print(pred_types)

# Compare labels
print("\n" + "="*80)
print("LABEL COMPARISON")
print("="*80)
print("\nGround truth labels (first 20):")
print(adata_query.obs[cell_type_col].values[:20])
print("\nCellTypist predictions (first 20):")
print(predictions[:20])

# Check exact matches
matches = predictions == adata_query.obs[cell_type_col].values
accuracy = matches.sum() / len(matches)
print(f"\nExact match accuracy: {accuracy:.4f}")
print(f"Matches: {matches.sum()} / {len(matches)}")

# Show unique labels in each
print("\n" + "="*80)
print("UNIQUE LABELS")
print("="*80)
print("\nGround truth unique labels:")
print(sorted(set(adata_query.obs[cell_type_col].values)))
print("\nCellTypist unique predictions:")
print(sorted(set(predictions)))

# Find which labels don't overlap
gt_labels = set(adata_query.obs[cell_type_col].values)
pred_labels = set(predictions)
print(f"\nLabels in ground truth but NOT in predictions:")
print(gt_labels - pred_labels)
print(f"\nLabels in predictions but NOT in ground truth:")
print(pred_labels - gt_labels)
print(f"\nLabels in BOTH:")
print(gt_labels & pred_labels)
