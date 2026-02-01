#!/usr/bin/env python3
"""
Standalone ScType Evaluation Script
====================================
Test ScType marker-based cell type annotation on a single reference/query study pair.

ScType is a marker-based method that uses predefined marker gene databases to
assign cell types. It does NOT require training on reference data.

Following sc-type-py implementation from:
https://github.com/kris-nader/sc-type-py

Usage:
    python test_sctype_single_pair.py

Requirements:
    pip install scanpy pandas numpy openpyxl
    # Or use sctypepy: pip install sctypepy
"""

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
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

# ScType database (Cell Ontology marker genes)
SCTYPE_DB_URL = "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# Tissue type for marker gene filtering
TISSUE_TYPE = "Immune system"  # Options: "Immune system", "Pancreas", "Liver", etc.

# Study configuration
REFERENCE_STUDY = 'van_galen_2019'
QUERY_STUDY = 'zhang_2023'

# Subsampling for quick testing
MAX_CELLS = 5000

# Output
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# SCTYPE FUNCTIONS (from sc-type-py)
# ==============================================================================

def gene_sets_prepare(path_to_db_file, cell_type):
    """
    Prepare gene sets from ScType database.

    Parameters
    ----------
    path_to_db_file : str
        Path or URL to ScTypeDB Excel file
    cell_type : str
        Tissue type to filter (e.g., "Immune system")

    Returns
    -------
    dict with 'gs_positive' and 'gs_negative' gene sets
    """
    # Read database
    cell_markers = pd.read_excel(path_to_db_file)

    # Filter by tissue type
    cell_markers = cell_markers[cell_markers['tissueType'] == cell_type]

    # Prepare gene sets
    gs_positive = {}
    gs_negative = {}

    for _, row in cell_markers.iterrows():
        cell_name = row['cellName']

        # Positive markers
        if pd.notna(row['geneSymbolmore1']):
            genes = [g.strip() for g in str(row['geneSymbolmore1']).split(',')]
            genes = [g for g in genes if g]  # Remove empty strings
            gs_positive[cell_name] = genes

        # Negative markers
        if pd.notna(row['geneSymbolmore2']):
            genes = [g.strip() for g in str(row['geneSymbolmore2']).split(',')]
            genes = [g for g in genes if g]
            gs_negative[cell_name] = genes

    return {'gs_positive': gs_positive, 'gs_negative': gs_negative}


def sctype_score(scRNAseqData, scaled=True, gs=None, gs2=None):
    """
    Calculate ScType scores for each cell.

    Parameters
    ----------
    scRNAseqData : np.ndarray or sparse matrix
        Expression matrix (genes x cells) - NOTE: transposed from AnnData
    scaled : bool
        Whether data is already scaled
    gs : dict
        Positive marker gene sets {cell_type: [genes]}
    gs2 : dict
        Negative marker gene sets {cell_type: [genes]}

    Returns
    -------
    pd.DataFrame
        Score matrix (cells x cell_types)
    """
    if gs is None:
        gs = {}
    if gs2 is None:
        gs2 = {}

    # Get gene names (row names of expression matrix)
    if hasattr(scRNAseqData, 'index'):
        gene_names = scRNAseqData.index.tolist()
        expr_matrix = scRNAseqData.values
    else:
        raise ValueError("scRNAseqData must be a DataFrame with gene names as index")

    # Convert to dense if sparse
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()

    n_genes, n_cells = expr_matrix.shape

    # Get all cell types
    cell_types = list(set(gs.keys()) | set(gs2.keys()))

    # Initialize score matrix
    scores = np.zeros((n_cells, len(cell_types)))

    for ct_idx, cell_type in enumerate(cell_types):
        # Positive markers
        pos_genes = gs.get(cell_type, [])
        pos_idx = [i for i, g in enumerate(gene_names) if g in pos_genes]

        # Negative markers
        neg_genes = gs2.get(cell_type, [])
        neg_idx = [i for i, g in enumerate(gene_names) if g in neg_genes]

        # Calculate score
        if pos_idx:
            pos_expr = expr_matrix[pos_idx, :].mean(axis=0)
        else:
            pos_expr = np.zeros(n_cells)

        if neg_idx:
            neg_expr = expr_matrix[neg_idx, :].mean(axis=0)
        else:
            neg_expr = np.zeros(n_cells)

        # Score = positive - negative
        scores[:, ct_idx] = pos_expr - neg_expr

    return pd.DataFrame(scores, columns=cell_types)


def assign_cell_types(score_matrix, threshold=0):
    """
    Assign cell types based on highest score.

    Parameters
    ----------
    score_matrix : pd.DataFrame
        Score matrix from sctype_score (cells x cell_types)
    threshold : float
        Minimum score threshold (cells below get "Unknown")

    Returns
    -------
    np.ndarray
        Predicted cell type for each cell
    """
    # Get max score and corresponding cell type for each cell
    max_scores = score_matrix.max(axis=1)
    predictions = score_matrix.idxmax(axis=1)

    # Mark low-confidence predictions as Unknown
    predictions[max_scores < threshold] = "Unknown"

    return predictions.values


# ==============================================================================
# ALTERNATIVE: Use sctypepy package if available
# ==============================================================================

def try_sctypepy(adata, tissue_type="Immune system"):
    """
    Try to use the sctypepy package if available.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalized, scaled)
    tissue_type : str
        Tissue type for marker gene database

    Returns
    -------
    predictions : np.ndarray or None
        Predicted cell types, or None if sctypepy not available
    """
    try:
        from sctypepy import sctype

        print("  Using sctypepy package")

        # Run ScType
        results = sctype(
            adata,
            tissue_type=tissue_type,
            db_path=SCTYPE_DB_URL,
        )

        # Get predictions
        if 'sctype_classification' in adata.obs.columns:
            return adata.obs['sctype_classification'].values
        elif 'scType' in adata.obs.columns:
            return adata.obs['scType'].values
        else:
            print("  Warning: Could not find ScType predictions in adata.obs")
            return None

    except ImportError:
        print("  sctypepy not available, using manual implementation")
        return None
    except Exception as e:
        print(f"  sctypepy failed: {e}, using manual implementation")
        return None


# ==============================================================================
# LABEL HARMONIZATION
# ==============================================================================

# Mapping from ScType database labels to common abbreviated labels
SCTYPE_LABEL_MAP = {
    # B cells
    'Naive B cells': 'B',
    'Memory B cells': 'B',
    'B cells': 'B',
    'Plasma cells': 'Plasma',
    'Plasmablasts': 'Plasma',
    'Pro-B cells': 'B',
    'Pre-B cells': 'B',

    # Monocytes
    'Classical monocytes': 'CD14+ Mono',
    'CD14+ Monocytes': 'CD14+ Mono',
    'CD14++ CD16- monocytes': 'CD14+ Mono',
    'Non-classical monocytes': 'CD16+ Mono',
    'CD16+ Monocytes': 'CD16+ Mono',
    'CD14+ CD16+ monocytes': 'CD16+ Mono',
    'Intermediate monocytes': 'CD14+ Mono',

    # NK cells
    'NK cells': 'NK',
    'CD56bright NK cells': 'NK',
    'CD56dim NK cells': 'NK',

    # T cells
    'Naive CD4+ T cells': 'CD4+ T',
    'Memory CD4+ T cells': 'CD4+ T',
    'CD4+ T cells': 'CD4+ T',
    'Naive CD8+ T cells': 'CD8+ T',
    'Memory CD8+ T cells': 'CD8+ T',
    'CD8+ T cells': 'CD8+ T',
    'Tregs': 'Treg',
    'Regulatory T cells': 'Treg',
    'Gamma-delta T cells': 'gdT',
    'NKT cells': 'NKT',

    # Dendritic cells
    'cDC1': 'cDC',
    'cDC2': 'cDC',
    'pDC': 'pDC',
    'Plasmacytoid dendritic cells': 'pDC',
    'Conventional dendritic cells': 'cDC',
    'Dendritic cells': 'DC',

    # Progenitors
    'HSCs': 'HSPC',
    'Hematopoietic stem cells': 'HSPC',
    'HSPCs': 'HSPC',
    'CMPs': 'CMP',
    'Common myeloid progenitors': 'CMP',
    'GMPs': 'GMP',
    'Granulocyte-monocyte progenitors': 'GMP',
    'MEPs': 'MEP',
    'Megakaryocyte-erythroid progenitors': 'MEP',
    'CLPs': 'CLP',

    # Erythroid
    'Erythrocytes': 'Erythroid',
    'Erythroid cells': 'Erythroid',
    'Erythroblasts': 'Erythroid',

    # Granulocytes
    'Neutrophils': 'Neutrophil',
    'Basophils': 'Basophil',
    'Eosinophils': 'Eosinophil',
    'Mast cells': 'Mast',
}


def harmonize_labels(predictions, label_map=None, ground_truth_labels=None):
    """Harmonize ScType predictions to match ground truth vocabulary."""
    if label_map is None:
        label_map = SCTYPE_LABEL_MAP

    predictions = np.asarray(predictions)
    harmonized = predictions.copy()

    # Apply direct mapping
    for sctype_label, target_label in label_map.items():
        mask = predictions == sctype_label
        harmonized[mask] = target_label

    # Try fuzzy matching for unmapped labels
    if ground_truth_labels is not None:
        gt_labels_set = set(np.unique(ground_truth_labels[pd.notna(ground_truth_labels)]))
        unmapped = set(np.unique(harmonized)) - gt_labels_set

        for pred_label in unmapped:
            pred_lower = str(pred_label).lower()
            for gt_label in gt_labels_set:
                gt_lower = gt_label.lower()
                if gt_lower in pred_lower or pred_lower in gt_lower:
                    mask = harmonized == pred_label
                    harmonized[mask] = gt_label
                    print(f"  Fuzzy mapped: '{pred_label}' -> '{gt_label}'")
                    break

    return harmonized


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_predictions(y_true, y_pred, method_name="ScType"):
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
    """Run ScType evaluation on a single reference/query pair."""

    print("="*80)
    print("ScType Single-Pair Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data:        {DATA_PATH}")
    print(f"  Database:    {SCTYPE_DB_URL}")
    print(f"  Tissue:      {TISSUE_TYPE}")
    print(f"  Reference:   {REFERENCE_STUDY}")
    print(f"  Query:       {QUERY_STUDY}")
    print(f"  Max cells:   {MAX_CELLS}")

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

    # Get ground truth
    y_true = adata_query.obs[cell_type_col].values
    print(f"Ground truth labels: {len(np.unique(y_true[pd.notna(y_true)]))} unique types")

    # Preprocess for ScType
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print('='*60)

    # Use raw counts if available
    if adata_query.raw is not None:
        adata_proc = adata_query.raw.to_adata()
    else:
        adata_proc = adata_query.copy()

    # Standard preprocessing
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    sc.pp.scale(adata_proc, max_value=10)

    print(f"Preprocessed: {adata_proc.n_obs} cells x {adata_proc.n_vars} genes")

    # Get gene names
    if 'feature_name' in adata_proc.var.columns:
        gene_names = adata_proc.var['feature_name'].values
    else:
        gene_names = adata_proc.var_names.values

    print(f"Gene name examples: {gene_names[:3]}")

    # Run ScType
    print(f"\n{'='*60}")
    print("RUNNING SCTYPE")
    print('='*60)

    start_time = time.time()

    # Try sctypepy first
    y_pred_raw = try_sctypepy(adata_proc, tissue_type=TISSUE_TYPE)

    if y_pred_raw is None:
        # Manual implementation
        print("  Using manual ScType implementation")

        # Prepare gene sets
        print("  Loading marker database...")
        gs_list = gene_sets_prepare(SCTYPE_DB_URL, TISSUE_TYPE)
        print(f"  Loaded {len(gs_list['gs_positive'])} cell types from database")

        # Prepare expression matrix (genes x cells)
        expr_df = pd.DataFrame(
            adata_proc.X.T if not hasattr(adata_proc.X, 'toarray') else adata_proc.X.T.toarray(),
            index=gene_names
        )

        # Calculate scores
        print("  Calculating ScType scores...")
        scores = sctype_score(
            expr_df,
            scaled=True,
            gs=gs_list['gs_positive'],
            gs2=gs_list['gs_negative']
        )

        # Assign cell types
        y_pred_raw = assign_cell_types(scores, threshold=0)

    elapsed = time.time() - start_time
    print(f"\nTotal inference time: {elapsed:.1f} seconds")
    print(f"Predicted {len(np.unique(y_pred_raw))} unique cell types")

    # Harmonize labels
    print(f"\n{'='*60}")
    print("LABEL HARMONIZATION")
    print('='*60)

    y_pred = harmonize_labels(y_pred_raw, ground_truth_labels=y_true)
    n_mapped = np.sum(y_pred != y_pred_raw)
    print(f"Mapped {n_mapped}/{len(y_pred)} predictions to ground truth vocabulary")

    # Evaluate
    print("\n--- Evaluation with HARMONIZED labels ---")
    metrics = evaluate_predictions(y_true, y_pred, method_name="ScType (harmonized)")
    metrics['time_seconds'] = elapsed
    metrics['reference_study'] = REFERENCE_STUDY
    metrics['query_study'] = QUERY_STUDY

    print("\n--- Evaluation with RAW labels ---")
    metrics_raw = evaluate_predictions(y_true, y_pred_raw, method_name="ScType (raw)")

    # Label overlap
    print("\n--- Label overlap (harmonized) ---")
    label_overlap_analysis(y_true, y_pred)

    # Save results
    results_df = pd.DataFrame([metrics])
    output_path = OUTPUT_DIR / f"sctype_eval_{QUERY_STUDY}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"""
ScType Evaluation Results:
  Query Study:   {QUERY_STUDY}
  Cells:         {metrics['n_cells']:,}

  Accuracy:      {metrics['accuracy']:.4f}
  ARI:           {metrics['ari']:.4f}
  NMI:           {metrics['nmi']:.4f}
  F1 (macro):    {metrics['f1_macro']:.4f}
  F1 (weighted): {metrics['f1_weighted']:.4f}

  Time:          {elapsed:.1f}s
""")
    print('='*80)

    return metrics


if __name__ == "__main__":
    main()
