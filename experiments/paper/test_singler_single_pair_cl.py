#!/usr/bin/env python3
"""
Standalone SingleR Evaluation Script
======================================
Test SingleR reference-based cell type annotation on multiple reference/query study pairs.

SingleR is a REFERENCE-BASED method - it compares query cells to labeled reference
cells using Spearman correlation to transfer labels. No learnable parameters are
trained; the "training" step is computing reference cell type profiles (medians).

Based on singler Python package (BiocPy):
https://pypi.org/project/singler/
https://github.com/BiocPy/singler

Usage:
    python test_singler_single_pair_cl.py

Requirements:
    pip install singler scanpy numpy pandas scipy scikit-learn
"""

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, classification_report

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl.data import subset_data, get_study_column, get_cell_type_column

# ==============================================================================
# CONFIGURATION  (matches exp_ensemble_embeddings.py)
# ==============================================================================

DATA_PATH = Path("/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad")

MAX_CELLS_PER_STUDY = 15000

SCENARIOS = [
    {
        'name': 'Same-Platform: beneyto (10X Genomics) → Zhang (10X Genomics)',
        'reference': 'beneyto-calabuig-2023',
        'query': 'zhang_2023',
    },
    {
        'name': 'Cross-Platform: Zhai (SORT-seq) → Zhang (10X Genomics)',
        'reference': 'zhai_2022',
        'query': 'zhang_2023',
    },
    {
        'name': 'Cross-Platform: van_galen (Seq-Well) → velten (Muta-Seq)',
        'reference': 'van_galen_2019',
        'query': 'velten_2021',
    },
    {
        'name': 'Cross-Platform: van_galen (Seq-Well) → beneyto (10X Genomics)',
        'reference': 'van_galen_2019',
        'query': 'beneyto-calabuig-2023',
    },
    {
        'name': 'Same-Platform: van_galen (Seq-Well) -> Zhai (SORT-seq)',
        'reference': 'van_galen_2019',
        'query': 'zhai_2022',
    },
]

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# SINGLER UTILITIES
# ==============================================================================

def log_normalize(X, target_sum=1e4):
    """Log-normalize count matrix (scanpy.pp.normalize_total + log1p equivalent)."""
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return np.log1p(X / row_sums * target_sum)


def run_singler(adata_ref, adata_query, cell_type_col):
    """
    Run SingleR annotation.

    SingleR is correlation-based with no learnable parameters. It:
    1. Computes reference cell type profiles (median expression per type)
    2. Computes Spearman correlations between query cells and profiles
    3. Assigns the best-matching cell type to each query cell

    Parameters
    ----------
    adata_ref : AnnData
        Reference data (raw counts in .raw.X or .X)
    adata_query : AnnData
        Query data (raw counts in .raw.X or .X)
    cell_type_col : str
        Column in .obs with cell type labels

    Returns
    -------
    predictions : np.ndarray or None
    """
    try:
        import singler
    except ImportError:
        print("  singler not found. Install with: pip install singler")
        return None

    # Get raw counts and gene names
    if adata_ref.raw is not None:
        X_ref = adata_ref.raw.X
        ref_genes = adata_ref.raw.var_names.values
    else:
        X_ref = adata_ref.X
        ref_genes = adata_ref.var_names.values

    if adata_query.raw is not None:
        X_query = adata_query.raw.X
        query_genes = adata_query.raw.var_names.values
    else:
        X_query = adata_query.X
        query_genes = adata_query.var_names.values

    ref_labels = adata_ref.obs[cell_type_col].values

    # Find common genes
    common_genes = np.intersect1d(ref_genes, query_genes)
    print(f"  Common genes: {len(common_genes)}")

    if len(common_genes) < 100:
        print("  ERROR: Too few common genes between reference and query!")
        return None

    # Subset to common genes
    ref_gene_idx = [np.where(ref_genes == g)[0][0] for g in common_genes]
    query_gene_idx = [np.where(query_genes == g)[0][0] for g in common_genes]

    X_ref_common = X_ref[:, ref_gene_idx]
    X_query_common = X_query[:, query_gene_idx]

    # Log-normalize
    X_ref_norm = log_normalize(X_ref_common)
    X_query_norm = log_normalize(X_query_common)

    # Remove NaN labels from reference
    valid_ref = pd.notna(ref_labels)
    X_ref_norm = X_ref_norm[valid_ref]
    ref_labels_valid = ref_labels[valid_ref]

    print(f"  Reference: {X_ref_norm.shape[0]:,} cells (after NaN removal), {len(np.unique(ref_labels_valid))} types")
    print(f"  Query:     {X_query_norm.shape[0]:,} cells")

    # Run SingleR (expects genes x cells format)
    results = singler.annotate_single(
        test_data=X_query_norm.T,
        test_features=common_genes,
        ref_data=X_ref_norm.T,
        ref_labels=ref_labels_valid,
        ref_features=common_genes,
        num_threads=8,
    )

    return np.asarray(results.column("best"))


# ==============================================================================
# EVALUATION
# ==============================================================================

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics, filtering NaN ground truth labels."""
    valid_mask = pd.notna(y_true)
    y_t = np.asarray(y_true)[valid_mask]
    y_p = np.asarray(y_pred)[valid_mask]

    return {
        'accuracy': accuracy_score(y_t, y_p),
        'ari': adjusted_rand_score(y_t, y_p),
        'f1_macro': f1_score(y_t, y_p, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_t, y_p, average='weighted', zero_division=0),
        'n_cells': len(y_t),
        'n_true_labels': len(np.unique(y_t)),
        'n_pred_labels': len(np.unique(y_p)),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("SingleR Reference-Based Evaluation")
    print("=" * 80)
    print(f"\nData:      {DATA_PATH}")
    print(f"Max cells: {MAX_CELLS_PER_STUDY}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("\nNOTE: SingleR is correlation-based (no learnable parameters).")
    print("      train_time = 0, all time is inference (profile building + correlation).")

    np.random.seed(42)

    # Load atlas (backed mode - memory efficient)
    print("\nLoading atlas...")
    adata = sc.read_h5ad(DATA_PATH, backed='r')
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)
    print(f"  Study column:     {study_col}")
    print(f"  Cell type column: {cell_type_col}")

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {scenario['name']}")
        print('=' * 80)
        print(f"  Reference: {scenario['reference']}")
        print(f"  Query:     {scenario['query']}")

        try:
            # Load reference and query
            adata_ref = subset_data(adata, studies=[scenario['reference']]).to_memory()
            adata_query = subset_data(adata, studies=[scenario['query']]).to_memory()

            # Subsample if needed
            if MAX_CELLS_PER_STUDY and adata_ref.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_ref = adata_ref[indices].copy()
            if MAX_CELLS_PER_STUDY and adata_query.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_query = adata_query[indices].copy()

            print(f"  Ref:   {adata_ref.n_obs:,} cells")
            print(f"  Query: {adata_query.n_obs:,} cells")

            y_true = adata_query.obs[cell_type_col].values

            # SingleR: no explicit training step (correlation-based)
            train_time = 0.0

            # Inference (includes reference profile building + correlation computation)
            infer_start = time.time()
            y_pred = run_singler(adata_ref, adata_query, cell_type_col)
            infer_time = time.time() - infer_start

            if y_pred is None:
                print("  SKIP: SingleR inference failed")
                continue

            metrics = compute_metrics(y_true, y_pred)
            result = {
                'scenario': scenario['name'],
                'reference': scenario['reference'],
                'query': scenario['query'],
                'accuracy': metrics['accuracy'],
                'ari': metrics['ari'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'train_time_sec': train_time,
                'inference_time_sec': infer_time,
                'time_sec': train_time + infer_time,
                'n_cells': metrics['n_cells'],
                'n_true_labels': metrics['n_true_labels'],
                'n_pred_labels': metrics['n_pred_labels'],
            }
            all_results.append(result)

            print(f"\n  Results:")
            print(f"    Accuracy:       {metrics['accuracy']:.4f}")
            print(f"    ARI:            {metrics['ari']:.4f}")
            print(f"    F1 (macro):     {metrics['f1_macro']:.4f}")
            print(f"    F1 (weighted):  {metrics['f1_weighted']:.4f}")
            print(f"    Train time:     {train_time:.1f}s  (no training, correlation-based)")
            print(f"    Inference time: {infer_time:.1f}s")

            print(f"\n  Per-class report:")
            valid_mask = pd.notna(y_true)
            print(classification_report(
                np.asarray(y_true)[valid_mask],
                np.asarray(y_pred)[valid_mask],
                zero_division=0
            ))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['scenario', 'accuracy', 'f1_macro', 'ari', 'inference_time_sec']
        print(df[cols].to_string(index=False))

        print(f"\nAverage performance:")
        print(f"  Accuracy:   {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"  F1 (macro): {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
        print(f"  ARI:        {df['ari'].mean():.4f} ± {df['ari'].std():.4f}")
        print(f"  Infer time: {df['inference_time_sec'].mean():.1f}s ± {df['inference_time_sec'].std():.1f}s")

        output_path = OUTPUT_DIR / "singler_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No scenarios completed successfully.")


if __name__ == "__main__":
    main()
