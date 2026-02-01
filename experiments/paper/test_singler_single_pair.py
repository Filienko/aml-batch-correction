#!/usr/bin/env python3
"""
Standalone SingleR Evaluation Script
=====================================
Test SingleR reference-based cell type annotation on a single reference/query study pair.

SingleR is a REFERENCE-BASED method - it compares query cells to labeled reference
cells to assign cell types. This is different from scTab/ScType which are zero-shot.

Based on singler Python package (BiocPy):
https://pypi.org/project/singler/
https://github.com/BiocPy/singler

Usage:
    python test_singler_single_pair.py

Requirements:
    pip install singler celldex summarizedexperiment
"""

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    classification_report,
    f1_score,
)

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths - adjust these to your setup
DATA_PATH = Path("/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad")

# Study configuration
REFERENCE_STUDY = 'van_galen_2019'
QUERY_STUDY = 'zhang_2023'

# Subsampling for quick testing
MAX_CELLS = 5000
MAX_REF_CELLS = 10000  # Reference can be larger

# Output
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# SINGLER WRAPPER
# ==============================================================================

def run_singler(
    query_matrix,
    query_genes,
    ref_matrix,
    ref_genes,
    ref_labels,
    num_threads=4
):
    """
    Run SingleR annotation.

    Parameters
    ----------
    query_matrix : array-like
        Query expression matrix (cells x genes), log-normalized
    query_genes : array-like
        Gene names for query
    ref_matrix : array-like
        Reference expression matrix (cells x genes), log-normalized
    ref_genes : array-like
        Gene names for reference
    ref_labels : array-like
        Cell type labels for reference cells
    num_threads : int
        Number of threads for parallel processing

    Returns
    -------
    predictions : np.ndarray
        Predicted cell types for query cells
    scores : pd.DataFrame
        Score matrix (optional)
    """
    try:
        import singler
        from summarizedexperiment import SummarizedExperiment
    except ImportError:
        raise ImportError(
            "singler package not found. Install with: pip install singler summarizedexperiment"
        )

    print("  Using singler package")

    # Convert to dense if sparse
    if hasattr(query_matrix, 'toarray'):
        query_matrix = query_matrix.toarray()
    if hasattr(ref_matrix, 'toarray'):
        ref_matrix = ref_matrix.toarray()

    # Ensure numpy arrays
    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    ref_matrix = np.asarray(ref_matrix, dtype=np.float64)
    query_genes = np.asarray(query_genes)
    ref_genes = np.asarray(ref_genes)
    ref_labels = np.asarray(ref_labels)

    # SingleR expects genes x cells (transposed from AnnData format)
    query_matrix_T = query_matrix.T
    ref_matrix_T = ref_matrix.T

    print(f"  Query: {query_matrix_T.shape[1]} cells x {query_matrix_T.shape[0]} genes")
    print(f"  Reference: {ref_matrix_T.shape[1]} cells x {ref_matrix_T.shape[0]} genes")

    # Run SingleR
    print("  Running SingleR annotation...")
    results = singler.annotate_single(
        test_data=query_matrix_T,
        test_features=query_genes,
        ref_data=ref_matrix_T,
        ref_labels=ref_labels,
        ref_features=ref_genes,
        num_threads=num_threads,
    )

    # Extract predictions
    predictions = results.column("best")

    # Convert to numpy array
    predictions = np.asarray(predictions)

    return predictions, results


def run_singler_with_builtin_ref(query_matrix, query_genes, ref_name="BlueprintEncodeData"):
    """
    Run SingleR with built-in celldex reference.

    Parameters
    ----------
    query_matrix : array-like
        Query expression matrix (cells x genes), log-normalized
    query_genes : array-like
        Gene names for query
    ref_name : str
        Name of celldex reference to use

    Returns
    -------
    predictions : np.ndarray
        Predicted cell types
    """
    try:
        import singler
        import celldex
    except ImportError:
        raise ImportError(
            "singler/celldex not found. Install with: pip install singler celldex"
        )

    print(f"  Using built-in reference: {ref_name}")

    # Load reference
    ref_data = getattr(celldex, ref_name)()
    print(f"  Reference loaded: {ref_data.shape}")

    # Convert query to correct format
    if hasattr(query_matrix, 'toarray'):
        query_matrix = query_matrix.toarray()

    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    query_genes = np.asarray(query_genes)

    # Transpose for singler (genes x cells)
    query_matrix_T = query_matrix.T

    # Get reference labels
    ref_labels = ref_data.get_column_data().column("label.main")

    # Run annotation
    results = singler.annotate_single(
        test_data=query_matrix_T,
        test_features=query_genes,
        ref_data=ref_data,
        ref_labels=ref_labels,
    )

    predictions = np.asarray(results.column("best"))

    return predictions, results


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_predictions(y_true, y_pred, method_name="SingleR"):
    """Compute evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {method_name}")
    print('='*60)

    valid_mask = pd.notna(y_true)
    y_true_valid = np.asarray(y_true)[valid_mask]
    y_pred_valid = np.asarray(y_pred)[valid_mask]

    print(f"Cells with valid labels: {len(y_true_valid)}")

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    nmi = normalized_mutual_info_score(y_true_valid, y_pred_valid)
    f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)

    metrics = {
        'method': method_name,
        'accuracy': accuracy,
        'ari': ari,
        'nmi': nmi,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'n_cells': len(y_true_valid),
        'n_true_labels': len(np.unique(y_true_valid)),
        'n_pred_labels': len(np.unique(y_pred_valid)),
    }

    print(f"\nMetrics:")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  ARI:           {ari:.4f}")
    print(f"  NMI:           {nmi:.4f}")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"\nLabel counts:")
    print(f"  Ground truth:  {metrics['n_true_labels']} unique cell types")
    print(f"  Predicted:     {metrics['n_pred_labels']} unique cell types")

    print(f"\nPer-class report:")
    print(classification_report(y_true_valid, y_pred_valid, zero_division=0))

    return metrics


def label_overlap_analysis(y_true, y_pred):
    """Analyze overlap between true and predicted label sets."""
    true_labels = set(np.unique(y_true[pd.notna(y_true)]))
    pred_labels = set(np.unique(y_pred))

    overlap = true_labels & pred_labels
    only_true = true_labels - pred_labels
    only_pred = pred_labels - true_labels

    print(f"\n{'='*60}")
    print("LABEL OVERLAP ANALYSIS")
    print('='*60)
    print(f"Ground truth labels: {len(true_labels)}")
    print(f"Predicted labels:    {len(pred_labels)}")
    print(f"Overlapping labels:  {len(overlap)}")
    print(f"\nLabels only in ground truth ({len(only_true)}):")
    for label in sorted(only_true)[:10]:
        print(f"  - {label}")
    if len(only_true) > 10:
        print(f"  ... and {len(only_true)-10} more")
    print(f"\nLabels only in predictions ({len(only_pred)}):")
    for label in sorted(only_pred)[:10]:
        print(f"  - {label}")
    if len(only_pred) > 10:
        print(f"  ... and {len(only_pred)-10} more")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run SingleR evaluation on a single reference/query pair."""

    print("="*80)
    print("SingleR Single-Pair Evaluation (Reference-Based)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data:           {DATA_PATH}")
    print(f"  Reference:      {REFERENCE_STUDY}")
    print(f"  Query:          {QUERY_STUDY}")
    print(f"  Max ref cells:  {MAX_REF_CELLS}")
    print(f"  Max query cells: {MAX_CELLS}")
    print("\nNOTE: SingleR uses the REFERENCE study to learn cell type signatures,")
    print("      then transfers labels to QUERY cells. This is reference-based annotation.")

    np.random.seed(42)

    # Load data
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print('='*60)

    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded atlas: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Detect columns
    study_col = 'study' if 'study' in adata.obs.columns else 'Study'
    cell_type_candidates = ['cell_type', 'Cell Type', 'celltype', 'cell_label', 'annotation']
    cell_type_col = next((c for c in cell_type_candidates if c in adata.obs.columns), None)

    print(f"Study column:     {study_col}")
    print(f"Cell type column: {cell_type_col}")

    # Extract reference study
    print(f"\n{'='*60}")
    print("EXTRACTING REFERENCE DATA")
    print('='*60)

    ref_mask = adata.obs[study_col] == REFERENCE_STUDY
    adata_ref = adata[ref_mask].copy()

    if MAX_REF_CELLS and adata_ref.n_obs > MAX_REF_CELLS:
        indices = np.random.choice(adata_ref.n_obs, MAX_REF_CELLS, replace=False)
        adata_ref = adata_ref[indices].copy()

    print(f"Reference ({REFERENCE_STUDY}): {adata_ref.n_obs:,} cells")

    ref_labels = adata_ref.obs[cell_type_col].values
    print(f"Reference labels: {len(np.unique(ref_labels[pd.notna(ref_labels)]))} unique types")

    # Extract query study
    print(f"\n{'='*60}")
    print("EXTRACTING QUERY DATA")
    print('='*60)

    query_mask = adata.obs[study_col] == QUERY_STUDY
    adata_query = adata[query_mask].copy()

    if MAX_CELLS and adata_query.n_obs > MAX_CELLS:
        indices = np.random.choice(adata_query.n_obs, MAX_CELLS, replace=False)
        adata_query = adata_query[indices].copy()

    print(f"Query ({QUERY_STUDY}): {adata_query.n_obs:,} cells")

    y_true = adata_query.obs[cell_type_col].values
    print(f"Ground truth labels: {len(np.unique(y_true[pd.notna(y_true)]))} unique types")

    # Preprocess - SingleR needs log-normalized data
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print('='*60)

    # Process reference
    if adata_ref.raw is not None:
        adata_ref_proc = adata_ref.raw.to_adata()
    else:
        adata_ref_proc = adata_ref.copy()

    sc.pp.normalize_total(adata_ref_proc, target_sum=1e4)
    sc.pp.log1p(adata_ref_proc)

    # Process query
    if adata_query.raw is not None:
        adata_query_proc = adata_query.raw.to_adata()
    else:
        adata_query_proc = adata_query.copy()

    sc.pp.normalize_total(adata_query_proc, target_sum=1e4)
    sc.pp.log1p(adata_query_proc)

    # Find common genes
    common_genes = adata_ref_proc.var_names.intersection(adata_query_proc.var_names)
    print(f"Common genes: {len(common_genes)}")

    adata_ref_proc = adata_ref_proc[:, common_genes]
    adata_query_proc = adata_query_proc[:, common_genes]

    # Get gene names
    if 'feature_name' in adata_ref_proc.var.columns:
        ref_genes = adata_ref_proc.var['feature_name'].values
        query_genes = adata_query_proc.var['feature_name'].values
    else:
        ref_genes = adata_ref_proc.var_names.values
        query_genes = adata_query_proc.var_names.values

    print(f"Gene name examples: {ref_genes[:3]}")

    # Run SingleR
    print(f"\n{'='*60}")
    print("RUNNING SINGLER")
    print('='*60)

    start_time = time.time()

    # Get expression matrices
    ref_matrix = adata_ref_proc.X
    query_matrix = adata_query_proc.X

    try:
        y_pred, results = run_singler(
            query_matrix=query_matrix,
            query_genes=query_genes,
            ref_matrix=ref_matrix,
            ref_genes=ref_genes,
            ref_labels=ref_labels,
        )
    except Exception as e:
        print(f"  SingleR failed: {e}")
        print("  Trying alternative: KNN-based label transfer...")

        # Fallback to simple KNN
        from sklearn.neighbors import KNeighborsClassifier

        if hasattr(ref_matrix, 'toarray'):
            ref_matrix = ref_matrix.toarray()
        if hasattr(query_matrix, 'toarray'):
            query_matrix = query_matrix.toarray()

        # Remove cells with missing labels from reference
        valid_ref = pd.notna(ref_labels)
        ref_matrix_valid = ref_matrix[valid_ref]
        ref_labels_valid = ref_labels[valid_ref]

        print(f"  Training KNN on {ref_matrix_valid.shape[0]} reference cells...")
        knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
        knn.fit(ref_matrix_valid, ref_labels_valid)

        print(f"  Predicting on {query_matrix.shape[0]} query cells...")
        y_pred = knn.predict(query_matrix)

    elapsed = time.time() - start_time
    print(f"\nTotal inference time: {elapsed:.1f} seconds")
    print(f"Predicted {len(np.unique(y_pred))} unique cell types")

    # Evaluate
    metrics = evaluate_predictions(y_true, y_pred, method_name="SingleR")
    metrics['time_seconds'] = elapsed
    metrics['reference_study'] = REFERENCE_STUDY
    metrics['query_study'] = QUERY_STUDY

    # Label overlap
    label_overlap_analysis(y_true, y_pred)

    # Save results
    results_df = pd.DataFrame([metrics])
    output_path = OUTPUT_DIR / f"singler_eval_{QUERY_STUDY}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"""
SingleR Evaluation Results:
  Reference:     {REFERENCE_STUDY} ({adata_ref.n_obs:,} cells)
  Query Study:   {QUERY_STUDY}
  Cells:         {metrics['n_cells']:,}

  Accuracy:      {metrics['accuracy']:.4f}
  ARI:           {metrics['ari']:.4f}
  NMI:           {metrics['nmi']:.4f}
  F1 (macro):    {metrics['f1_macro']:.4f}
  F1 (weighted): {metrics['f1_weighted']:.4f}

  Time:          {elapsed:.1f}s

NOTE: SingleR learns from reference labels, so predictions use the same
      label vocabulary as your reference study (no harmonization needed).
""")
    print('='*80)

    return metrics


if __name__ == "__main__":
    main()
