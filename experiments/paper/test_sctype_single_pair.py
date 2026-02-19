#!/usr/bin/env python3
"""
Standalone ScType Evaluation Script
======================================
Test ScType marker-based cell type annotation on multiple reference/query study pairs.

ScType is a ZERO-SHOT marker-based method - it uses predefined marker gene databases
to score cells without requiring reference data or training. The "reference" study
is listed only to match the evaluation protocol of reference-based methods.

Following sc-type-py implementation from:
https://github.com/kris-nader/sc-type-py

Usage:
    python test_sctype_single_pair.py

Requirements:
    pip install scanpy pandas numpy openpyxl scikit-learn
    # Or use sctypepy: pip install sctypepy
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

# ScType database (Cell Ontology marker genes)
SCTYPE_DB_URL = "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# Tissue type for marker gene filtering
TISSUE_TYPE = "Immune system"

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
# SCTYPE FUNCTIONS
# ==============================================================================

def gene_sets_prepare(path_to_db_file, cell_type):
    """Prepare gene sets from ScType database."""
    cell_markers = pd.read_excel(path_to_db_file)
    cell_markers = cell_markers[cell_markers['tissueType'] == cell_type]

    gs_positive = {}
    gs_negative = {}

    for _, row in cell_markers.iterrows():
        cell_name = row['cellName']
        if pd.notna(row['geneSymbolmore1']):
            genes = [g.strip() for g in str(row['geneSymbolmore1']).split(',')]
            gs_positive[cell_name] = [g for g in genes if g]
        if pd.notna(row['geneSymbolmore2']):
            genes = [g.strip() for g in str(row['geneSymbolmore2']).split(',')]
            gs_negative[cell_name] = [g for g in genes if g]

    return {'gs_positive': gs_positive, 'gs_negative': gs_negative}


def sctype_score(scRNAseqData, gs=None, gs2=None):
    """
    Calculate ScType scores for each cell.

    Parameters
    ----------
    scRNAseqData : pd.DataFrame
        Expression matrix (genes x cells) with gene names as index
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

    gene_names = scRNAseqData.index.tolist()
    expr_matrix = scRNAseqData.values
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()

    n_genes, n_cells = expr_matrix.shape
    cell_types = list(set(gs.keys()) | set(gs2.keys()))
    scores = np.zeros((n_cells, len(cell_types)))

    for ct_idx, cell_type in enumerate(cell_types):
        pos_idx = [i for i, g in enumerate(gene_names) if g in gs.get(cell_type, [])]
        neg_idx = [i for i, g in enumerate(gene_names) if g in gs2.get(cell_type, [])]

        pos_expr = expr_matrix[pos_idx, :].mean(axis=0) if pos_idx else np.zeros(n_cells)
        neg_expr = expr_matrix[neg_idx, :].mean(axis=0) if neg_idx else np.zeros(n_cells)

        scores[:, ct_idx] = pos_expr - neg_expr

    return pd.DataFrame(scores, columns=cell_types)


def assign_cell_types(score_matrix, threshold=0):
    """Assign cell types based on highest score, marking low-confidence as Unknown."""
    max_scores = score_matrix.max(axis=1)
    predictions = score_matrix.idxmax(axis=1).values
    predictions[max_scores < threshold] = "Unknown"
    return predictions


# ==============================================================================
# LABEL HARMONIZATION
# ==============================================================================

SCTYPE_LABEL_MAP = {
    'Naive B cells': 'B', 'Memory B cells': 'B', 'B cells': 'B',
    'Plasma cells': 'Plasma', 'Plasmablasts': 'Plasma',
    'Pro-B cells': 'B', 'Pre-B cells': 'B',
    'Classical monocytes': 'CD14+ Mono', 'CD14+ Monocytes': 'CD14+ Mono',
    'CD14++ CD16- monocytes': 'CD14+ Mono',
    'Non-classical monocytes': 'CD16+ Mono', 'CD16+ Monocytes': 'CD16+ Mono',
    'CD14+ CD16+ monocytes': 'CD16+ Mono', 'Intermediate monocytes': 'CD14+ Mono',
    'NK cells': 'NK', 'CD56bright NK cells': 'NK', 'CD56dim NK cells': 'NK',
    'Naive CD4+ T cells': 'CD4+ T', 'Memory CD4+ T cells': 'CD4+ T', 'CD4+ T cells': 'CD4+ T',
    'Naive CD8+ T cells': 'CD8+ T', 'Memory CD8+ T cells': 'CD8+ T', 'CD8+ T cells': 'CD8+ T',
    'Tregs': 'Treg', 'Regulatory T cells': 'Treg',
    'Gamma-delta T cells': 'gdT', 'NKT cells': 'NKT',
    'cDC1': 'cDC', 'cDC2': 'cDC', 'pDC': 'pDC',
    'Plasmacytoid dendritic cells': 'pDC', 'Conventional dendritic cells': 'cDC',
    'Dendritic cells': 'DC',
    'HSCs': 'HSPC', 'Hematopoietic stem cells': 'HSPC', 'HSPCs': 'HSPC',
    'CMPs': 'CMP', 'Common myeloid progenitors': 'CMP',
    'GMPs': 'GMP', 'Granulocyte-monocyte progenitors': 'GMP',
    'MEPs': 'MEP', 'Megakaryocyte-erythroid progenitors': 'MEP', 'CLPs': 'CLP',
    'Erythrocytes': 'Erythroid', 'Erythroid cells': 'Erythroid', 'Erythroblasts': 'Erythroid',
    'Neutrophils': 'Neutrophil', 'Basophils': 'Basophil',
    'Eosinophils': 'Eosinophil', 'Mast cells': 'Mast',
}


def harmonize_labels(predictions, ground_truth_labels=None):
    """Harmonize ScType predictions to match ground truth vocabulary."""
    predictions = np.asarray(predictions, dtype=str)
    harmonized = predictions.copy()

    for sctype_label, target_label in SCTYPE_LABEL_MAP.items():
        mask = predictions == sctype_label
        harmonized[mask] = target_label

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
                    break

    return harmonized


def run_sctype(adata_query, y_true):
    """
    Run ScType annotation on query data.

    ScType is zero-shot - no reference data used. Returns predictions.
    """
    # Use raw counts if available, then preprocess
    if adata_query.raw is not None:
        adata_proc = adata_query.raw.to_adata()
    else:
        adata_proc = adata_query.copy()

    # Standard preprocessing for ScType
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    sc.pp.scale(adata_proc, max_value=10)

    # Try sctypepy package first
    try:
        from sctypepy import sctype
        print("  Using sctypepy package")
        sctype(adata_proc, tissue_type=TISSUE_TYPE, db_path=SCTYPE_DB_URL)
        for col in ['sctype_classification', 'scType']:
            if col in adata_proc.obs.columns:
                return adata_proc.obs[col].values
    except (ImportError, Exception) as e:
        print(f"  sctypepy unavailable ({e}), using manual implementation")

    # Manual implementation
    gene_names = (adata_proc.var['feature_name'].values
                  if 'feature_name' in adata_proc.var.columns
                  else adata_proc.var_names.values)

    print("  Loading ScType marker database...")
    gs_list = gene_sets_prepare(SCTYPE_DB_URL, TISSUE_TYPE)
    print(f"  Loaded {len(gs_list['gs_positive'])} cell types")

    # Expression matrix: genes x cells
    X = adata_proc.X.T if not hasattr(adata_proc.X, 'toarray') else adata_proc.X.T.toarray()
    expr_df = pd.DataFrame(X, index=gene_names)

    print("  Calculating ScType scores...")
    scores = sctype_score(expr_df, gs=gs_list['gs_positive'], gs2=gs_list['gs_negative'])

    return assign_cell_types(scores, threshold=0)


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
    print("ScType Zero-Shot Evaluation")
    print("=" * 80)
    print(f"\nData:      {DATA_PATH}")
    print(f"Database:  {SCTYPE_DB_URL}")
    print(f"Tissue:    {TISSUE_TYPE}")
    print(f"Max cells: {MAX_CELLS_PER_STUDY}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("\nNOTE: ScType is marker-based and zero-shot (no reference data needed).")
    print("      train_time = 0, all time is inference (scoring + assignment).")

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
        print(f"  Query: {scenario['query']}  (reference not used - zero-shot method)")

        try:
            # Load query data only (ScType doesn't use reference)
            adata_query = subset_data(adata, studies=[scenario['query']]).to_memory()

            if MAX_CELLS_PER_STUDY and adata_query.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_query = adata_query[indices].copy()

            print(f"  Query: {adata_query.n_obs:,} cells")

            y_true = adata_query.obs[cell_type_col].values

            # ScType: zero-shot, no training
            train_time = 0.0

            # Inference
            infer_start = time.time()
            y_pred_raw = run_sctype(adata_query, y_true)
            infer_time = time.time() - infer_start

            # Harmonize labels
            y_pred = harmonize_labels(y_pred_raw, ground_truth_labels=y_true)

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
            print(f"    Train time:     {train_time:.1f}s  (zero-shot, no training)")
            print(f"    Inference time: {infer_time:.1f}s")

            print(f"\n  Per-class report (harmonized):")
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

        output_path = OUTPUT_DIR / "sctype_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No scenarios completed successfully.")


if __name__ == "__main__":
    main()
