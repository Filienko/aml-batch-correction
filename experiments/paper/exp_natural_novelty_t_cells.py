#!/usr/bin/env python3
"""
Experiment: Natural Novelty Detection (T Cells in Zhai vs Zhang)
================================================================

Scenario: 
- Reference: zhai_2022 (Naturally lacks 'T' cells)
- Query:     zhang_2023 (Contains 192 'T' cells)

Fixes:
- Disables CellTypist 'check_expression' to allow training on gene subsets.
- Uses memory-safe loading.
"""

import sys
import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

# --- MEMORY SAFETY ---
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_START_METHOD"] = "forkserver"

# --- MONKEY PATCH CELLTYPIST ---
# Fixes "ValueError: Invalid expression matrix" when using subsets
import celltypist
_original_train = celltypist.train

def _patched_train(*args, **kwargs):
    # Force disable the sum-check because we are training on a gene subset
    kwargs['check_expression'] = False 
    return _original_train(*args, **kwargs)

celltypist.train = _patched_train
# -------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_cell_type_column
from sccl.models.celltypist import CellTypistModel

# --- CONFIGURATION ---
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results" / "natural_novelty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_STUDY = 'zhai_2022'
QUERY_STUDY = 'zhang_2023'
NOVEL_CLASS = 'T' 

def get_scimilarity_embeddings(adata, model_path):
    """Get SCimilarity embeddings."""
    if adata.is_view:
        adata = adata.copy()
    pipeline = Pipeline(model='scimilarity', model_params={'model_path': model_path})
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return pipeline.model.get_embedding(adata)

def main():
    print("="*80)
    print(f"NATURAL NOVELTY EXPERIMENT: {NOVEL_CLASS} cells")
    print("="*80)

    # 1. OPTIMIZED LOAD
    print("\nLoading data (Backed Mode)...")
    adata = sc.read_h5ad(DATA_PATH, backed='r') 
    cell_type_col = get_cell_type_column(adata)
    
    # 2. OPTIMIZED SUBSET
    print("Subsetting studies from disk...")
    adata_ref = subset_data(adata, studies=[REF_STUDY], copy=False)
    adata_query = subset_data(adata, studies=[QUERY_STUDY], copy=False)
    del adata
    gc.collect()

    print(f"Reference: {adata_ref.n_obs} cells")
    print(f"Query:     {adata_query.n_obs} cells")

    # 3. Filter Query for Evaluation
    ref_types = set(adata_ref.obs[cell_type_col].unique())
    query_types = set(adata_query.obs[cell_type_col].unique())
    
    known_types_in_query = list(query_types.intersection(ref_types))
    valid_query_cells = adata_query.obs[cell_type_col].isin(known_types_in_query + [NOVEL_CLASS])
    adata_query = adata_query[valid_query_cells].copy()
    
    y_true_novel = (adata_query.obs[cell_type_col] == NOVEL_CLASS).astype(int).values
    print(f"Eval Query: {adata_query.n_obs} cells ({y_true_novel.sum()} are {NOVEL_CLASS})")

    if y_true_novel.sum() == 0:
        print(f"ERROR: No {NOVEL_CLASS} cells found!")
        return

    # 4. SCimilarity Pipeline
    print("\n[SCimilarity] Computing embeddings...")
    adata_ref_prep = preprocess_data(adata_ref.copy(), n_top_genes=3000)
    adata_query_prep = preprocess_data(adata_query.copy(), n_top_genes=3000)
    
    emb_ref = get_scimilarity_embeddings(adata_ref_prep, MODEL_PATH)
    emb_query = get_scimilarity_embeddings(adata_query_prep, MODEL_PATH)
    
    print("  Calculating distances...")
    nn = NearestNeighbors(n_neighbors=5, n_jobs=-1, metric='euclidean')
    nn.fit(emb_ref)
    dists, _ = nn.kneighbors(emb_query)
    scim_scores = dists.mean(axis=1)
    
    del adata_ref_prep, adata_query_prep, emb_ref, emb_query
    gc.collect()

    # 5. CellTypist Pipeline
    print("\n[CellTypist] Training and scoring...")
    
    print("  Filtering Reference to top 3000 genes...")
    adata_ref_ct = adata_ref.copy()
    sc.pp.normalize_total(adata_ref_ct, target_sum=1e4)
    sc.pp.log1p(adata_ref_ct)
    
    # The crucial step that was crashing:
    sc.pp.highly_variable_genes(adata_ref_ct, n_top_genes=3000, subset=True)
    
    # Train (Now uses patched function to ignore sum check)
    ct_model = CellTypistModel(model=None)
    ct_model.fit(adata_ref_ct, target_column=cell_type_col)
    
    del adata_ref_ct
    gc.collect()
    
    print("  Predicting on Query...")
    import celltypist
    
    adata_query_ct = adata_query.copy()
    sc.pp.normalize_total(adata_query_ct, target_sum=1e4)
    sc.pp.log1p(adata_query_ct)
    
    # Use patched train/annotate just in case, though annotate is usually lenient
    preds = celltypist.annotate(adata_query_ct, model=ct_model._model, majority_voting=False)
    
    ct_scores = 1.0 - preds.probability_matrix.values.max(axis=1)
    
    # Capture confused labels
    pred_labels = preds.predicted_labels['predicted_labels'].values
    t_cell_mask = (y_true_novel == 1)
    confused_labels = pred_labels[t_cell_mask]
    
    from collections import Counter
    print(f"\nCellTypist misclassified the {NOVEL_CLASS} cells as:")
    print(Counter(confused_labels).most_common(5))

    # 6. Evaluation
    print("\n" + "="*80)
    print("RESULTS: Novelty Detection AUROC")
    print("="*80)
    
    scim_auc = roc_auc_score(y_true_novel, scim_scores)
    ct_auc = roc_auc_score(y_true_novel, ct_scores)
    
    print(f"SCimilarity (Distance):   {scim_auc:.4f}")
    print(f"CellTypist (Uncertainty): {ct_auc:.4f}")
    
    if scim_auc > ct_auc:
        print("\n✓ CONCLUSION: SCimilarity is safer. It flagged the T cells as 'Unknown' better.")
    else:
        print("\n✓ CONCLUSION: CellTypist Uncertainty was robust enough.")

    results_file = OUTPUT_DIR / "natural_novelty_scores.csv"
    df_res = pd.DataFrame({
        'Cell_Type': adata_query.obs[cell_type_col].values,
        'Is_Novel': y_true_novel,
        'SCimilarity_Score': scim_scores,
        'CellTypist_Score': ct_scores
    })
    df_res.to_csv(results_file, index=False)
    print(f"\nSaved scores to: {results_file}")

if __name__ == "__main__":
    main()

