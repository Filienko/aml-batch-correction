#!/usr/bin/env python3

"""
Standalone SingleR Evaluation Script - Multiple Scenarios
==========================================================

Test SingleR reference-based cell type annotation on multiple reference/query study pairs.

SingleR is a REFERENCE-BASED method - it compares query cells to labeled reference
cells to assign cell types. This is different from scTab/ScType which are zero-shot.

This version uses h5py to read h5ad files directly, avoiding scanpy/anndata
dependency conflicts.

Based on singler Python package (BiocPy):
https://pypi.org/project/singler/
https://github.com/BiocPy/singler

Usage:
    python test_singler_multiple_scenarios.py

Requirements:
    pip install singler summarizedexperiment h5py numpy pandas scipy scikit-learn
"""

import sys
import warnings
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
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

# Multiple scenarios to test
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
]

SCENARIOS = [
    {
        'name': 'Same-Platform: van_galen (Seq-Well) -> Zhai (SORT-seq)',
        'reference': 'van_galen_2019',
        'query': 'zhai_2022',
    },
]
# Subsampling for quick testing
MAX_CELLS = 10000
MAX_REF_CELLS = 10000  # Reference can be larger

# Output
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# H5AD DATA LOADING (without scanpy/anndata)
# ==============================================================================

class H5ADReader:
    """Minimal h5ad reader using h5py directly."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._file = None

    def __enter__(self):
        self._file = h5py.File(self.filepath, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def get_obs(self):
        """Read obs (cell metadata) as DataFrame."""
        obs = self._file['obs']

        if isinstance(obs, h5py.Dataset):
            return pd.DataFrame(obs[()])
        else:
            data = {}
            index = None

            if '_index' in obs:
                index = obs['_index'][()].astype(str)

            for key in obs.keys():
                if key.startswith('_') or key == '__categories':
                    continue
                try:
                    col_data = obs[key]
                    if isinstance(col_data, h5py.Group):
                        if 'categories' in col_data and 'codes' in col_data:
                            categories = col_data['categories'][()].astype(str)
                            codes = col_data['codes'][()]
                            data[key] = categories[codes]
                    else:
                        arr = col_data[()]
                        if arr.dtype.kind == 'S' or arr.dtype.kind == 'O':
                            arr = arr.astype(str)
                        data[key] = arr
                except Exception:
                    continue

            df = pd.DataFrame(data)
            if index is not None and len(index) == len(df):
                df.index = index
            return df

    def get_var(self, use_raw=True):
        """Read var (gene metadata) as DataFrame."""
        if use_raw and 'raw' in self._file and 'var' in self._file['raw']:
            var = self._file['raw']['var']
        else:
            var = self._file['var']

        if isinstance(var, h5py.Dataset):
            return pd.DataFrame(var[()])
        else:
            data = {}
            index = None

            if '_index' in var:
                index = var['_index'][()].astype(str)

            for key in var.keys():
                if key.startswith('_'):
                    continue
                try:
                    col_data = var[key]
                    if isinstance(col_data, h5py.Group):
                        if 'categories' in col_data and 'codes' in col_data:
                            categories = col_data['categories'][()].astype(str)
                            codes = col_data['codes'][()]
                            data[key] = categories[codes]
                    else:
                        arr = col_data[()]
                        if arr.dtype.kind == 'S' or arr.dtype.kind == 'O':
                            arr = arr.astype(str)
                        data[key] = arr
                except Exception:
                    continue

            df = pd.DataFrame(data)
            if index is not None and len(index) == len(df):
                df.index = index
            return df

    def get_X(self, use_raw=True, cell_indices=None):
        """Read X matrix (optionally from raw)."""
        if use_raw and 'raw' in self._file and 'X' in self._file['raw']:
            X_group = self._file['raw']['X']
        else:
            X_group = self._file['X']

        if isinstance(X_group, h5py.Group):
            data = X_group['data'][()]
            indices = X_group['indices'][()]
            indptr = X_group['indptr'][()]

            if 'shape' in X_group.attrs:
                shape = tuple(X_group.attrs['shape'])
            else:
                n_rows = len(indptr) - 1
                n_cols = indices.max() + 1 if len(indices) > 0 else 0
                shape = (n_rows, n_cols)

            X = csr_matrix((data, indices, indptr), shape=shape)

            if cell_indices is not None:
                X = X[cell_indices]

            return X
        else:
            if cell_indices is not None:
                return X_group[cell_indices]
            return X_group[()]


def load_h5ad_data(filepath, study_col, cell_type_col, study_name, max_cells=None, use_raw=True):
    """Load data for a specific study from h5ad file."""
    with H5ADReader(filepath) as reader:
        print(f"  Reading obs...")
        obs = reader.get_obs()

        if study_col not in obs.columns:
            raise ValueError(f"Study column '{study_col}' not found. Available: {obs.columns.tolist()}")

        mask = obs[study_col] == study_name
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(f"No cells found for study '{study_name}'")

        if max_cells and len(indices) > max_cells:
            np.random.seed(42)
            indices = np.random.choice(indices, max_cells, replace=False)
            indices = np.sort(indices)

        print(f"  Reading var...")
        var = reader.get_var(use_raw=use_raw)

        print(f"  Reading X matrix for {len(indices)} cells...")
        X = reader.get_X(use_raw=use_raw, cell_indices=indices)

        obs = obs.iloc[indices].reset_index(drop=True)

        return X, obs, var


def detect_columns(filepath):
    """Auto-detect study and cell type columns from h5ad file."""
    study_candidates = ['study', 'Study', 'dataset', 'batch', 'sample', 'Batch']
    cell_type_candidates = ['cell_type', 'Cell Type', 'celltype', 'cell_label',
                            'annotation', 'Annotation', 'CellType']

    with H5ADReader(filepath) as reader:
        obs = reader.get_obs()
        columns = obs.columns.tolist()

    study_col = next((c for c in study_candidates if c in columns), None)
    cell_type_col = next((c for c in cell_type_candidates if c in columns), None)

    return study_col, cell_type_col, columns


def list_studies(filepath, study_col):
    """List available studies in the h5ad file."""
    with H5ADReader(filepath) as reader:
        obs = reader.get_obs()
        return obs[study_col].unique().tolist()


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def log_normalize(X, target_sum=1e4):
    """Log-normalize count matrix (like scanpy.pp.normalize_total + log1p)."""
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Normalize to target sum
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    X_norm = X / row_sums * target_sum

    # Log transform
    X_log = np.log1p(X_norm)

    return X_log


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
    """
    try:
        import singler
    except ImportError:
        raise ImportError(
            "singler package not found. Install with: pip install singler"
        )

    print("  Using singler package")

    # Convert to dense if sparse
    if hasattr(query_matrix, 'toarray'):
        query_matrix = query_matrix.toarray()
    if hasattr(ref_matrix, 'toarray'):
        ref_matrix = ref_matrix.toarray()

    # Ensure numpy arrays with correct dtype
    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    ref_matrix = np.asarray(ref_matrix, dtype=np.float64)
    query_genes = np.asarray(query_genes)
    ref_genes = np.asarray(ref_genes)
    ref_labels = np.asarray(ref_labels)

    # SingleR expects genes x cells (transposed from standard format)
    query_matrix_T = query_matrix.T
    ref_matrix_T = ref_matrix.T

    print(f"  Query: {query_matrix_T.shape[1]} cells x {query_matrix_T.shape[0]} genes")
    print(f"  Reference: {ref_matrix_T.shape[1]} cells x {ref_matrix_T.shape[0]} genes")
    print(f"  Reference labels: {len(np.unique(ref_labels))} unique types")

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
# SCENARIO EVALUATION
# ==============================================================================

def run_scenario(scenario, study_col, cell_type_col):
    """Run SingleR evaluation for a single scenario."""
    
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario['name']}")
    print("="*80)
    print(f"  Reference: {scenario['reference']}")
    print(f"  Query:     {scenario['query']}")
    
    np.random.seed(42)
    
    try:
        # Load reference data
        print(f"\n{'='*60}")
        print("LOADING REFERENCE DATA")
        print('='*60)

        X_ref, obs_ref, var_ref = load_h5ad_data(
            DATA_PATH,
            study_col=study_col,
            cell_type_col=cell_type_col,
            study_name=scenario['reference'],
            max_cells=MAX_REF_CELLS,
            use_raw=True
        )

        print(f"Reference ({scenario['reference']}): {X_ref.shape[0]:,} cells x {X_ref.shape[1]:,} genes")

        ref_labels = obs_ref[cell_type_col].values
        print(f"Reference labels: {len(np.unique(ref_labels[pd.notna(ref_labels)]))} unique types")

        # Load query data
        print(f"\n{'='*60}")
        print("LOADING QUERY DATA")
        print('='*60)

        X_query, obs_query, var_query = load_h5ad_data(
            DATA_PATH,
            study_col=study_col,
            cell_type_col=cell_type_col,
            study_name=scenario['query'],
            max_cells=MAX_CELLS,
            use_raw=True
        )

        print(f"Query ({scenario['query']}): {X_query.shape[0]:,} cells x {X_query.shape[1]:,} genes")

        y_true = obs_query[cell_type_col].values
        print(f"Ground truth labels: {len(np.unique(y_true[pd.notna(y_true)]))} unique types")

        # Get gene names
        if 'feature_name' in var_ref.columns:
            ref_genes = var_ref['feature_name'].values
        elif var_ref.index is not None:
            ref_genes = var_ref.index.values
        else:
            ref_genes = np.arange(X_ref.shape[1]).astype(str)

        if 'feature_name' in var_query.columns:
            query_genes = var_query['feature_name'].values
        elif var_query.index is not None:
            query_genes = var_query.index.values
        else:
            query_genes = np.arange(X_query.shape[1]).astype(str)

        # Find common genes
        print(f"\n{'='*60}")
        print("PREPROCESSING")
        print('='*60)

        ref_genes = np.asarray(ref_genes)
        query_genes = np.asarray(query_genes)

        # Find common genes
        common_genes = np.intersect1d(ref_genes, query_genes)
        print(f"Common genes: {len(common_genes)}")

        if len(common_genes) == 0:
            print("ERROR: No common genes found between reference and query!")
            return None

        # Subset to common genes
        ref_gene_mask = np.isin(ref_genes, common_genes)
        query_gene_mask = np.isin(query_genes, common_genes)

        X_ref_common = X_ref[:, ref_gene_mask]
        X_query_common = X_query[:, query_gene_mask]
        ref_genes_common = ref_genes[ref_gene_mask]
        query_genes_common = query_genes[query_gene_mask]

        # Ensure same gene order
        ref_gene_order = {g: i for i, g in enumerate(ref_genes_common)}
        query_reorder = [ref_gene_order[g] for g in query_genes_common if g in ref_gene_order]

        # Log-normalize
        print("  Log-normalizing reference...")
        X_ref_norm = log_normalize(X_ref_common)

        print("  Log-normalizing query...")
        X_query_norm = log_normalize(X_query_common)

        print(f"  Reference: {X_ref_norm.shape}")
        print(f"  Query: {X_query_norm.shape}")

        # Remove cells with missing labels from reference
        valid_ref_mask = pd.notna(ref_labels)
        X_ref_norm = X_ref_norm[valid_ref_mask]
        ref_labels_valid = ref_labels[valid_ref_mask]
        print(f"  Reference after removing NaN labels: {X_ref_norm.shape[0]} cells")

        # Run SingleR
        print(f"\n{'='*60}")
        print("RUNNING SINGLER")
        print('='*60)

        start_time = time.time()

        y_pred, results = run_singler(
            query_matrix=X_query_norm,
            query_genes=ref_genes_common,  # Use same gene names
            ref_matrix=X_ref_norm,
            ref_genes=ref_genes_common,
            ref_labels=ref_labels_valid,
        )

        elapsed = time.time() - start_time
        print(f"\nTotal inference time: {elapsed:.1f} seconds")
        print(f"Predicted {len(np.unique(y_pred))} unique cell types")

        # Evaluate
        metrics = evaluate_predictions(y_true, y_pred, method_name="SingleR")
        metrics['time_seconds'] = elapsed
        metrics['reference_study'] = scenario['reference']
        metrics['query_study'] = scenario['query']
        metrics['scenario_name'] = scenario['name']

        # Label overlap
        label_overlap_analysis(y_true, y_pred)

        # Summary
        print(f"\n{'='*60}")
        print("SCENARIO SUMMARY")
        print('='*60)
        print(f"""
Scenario:      {scenario['name']}
Reference:     {scenario['reference']} ({X_ref_norm.shape[0]:,} cells)
Query Study:   {scenario['query']}
Cells:         {metrics['n_cells']:,}

Accuracy:      {metrics['accuracy']:.4f}
ARI:           {metrics['ari']:.4f}
NMI:           {metrics['nmi']:.4f}
F1 (macro):    {metrics['f1_macro']:.4f}
F1 (weighted): {metrics['f1_weighted']:.4f}

Time:          {elapsed:.1f}s
""")
        print('='*60)

        return metrics
        
    except Exception as e:
        print(f"\nERROR in scenario '{scenario['name']}': {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run SingleR evaluation on multiple scenarios."""

    print("="*80)
    print("SingleR Multi-Scenario Evaluation (Reference-Based)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data:           {DATA_PATH}")
    print(f"  Max ref cells:  {MAX_REF_CELLS}")
    print(f"  Max query cells: {MAX_CELLS}")
    print(f"  Scenarios:      {len(SCENARIOS)}")
    print("\nNOTE: SingleR uses the REFERENCE study to learn cell type signatures,")
    print("      then transfers labels to QUERY cells. This is reference-based annotation.")

    # Detect columns
    print(f"\n{'='*60}")
    print("DETECTING DATA STRUCTURE")
    print('='*60)

    study_col, cell_type_col, all_columns = detect_columns(DATA_PATH)
    print(f"Study column:     {study_col}")
    print(f"Cell type column: {cell_type_col}")

    if study_col is None or cell_type_col is None:
        print("ERROR: Could not detect required columns!")
        return

    # List available studies
    available_studies = list_studies(DATA_PATH, study_col)
    print(f"\nAvailable studies ({len(available_studies)}):")
    for study in sorted(available_studies):
        print(f"  - {study}")

    # Validate scenarios
    print(f"\n{'='*60}")
    print("VALIDATING SCENARIOS")
    print('='*60)
    
    valid_scenarios = []
    for scenario in SCENARIOS:
        ref = scenario['reference']
        query = scenario['query']
        
        if ref not in available_studies:
            print(f"⚠ SKIP: Reference '{ref}' not found")
            continue
        if query not in available_studies:
            print(f"⚠ SKIP: Query '{query}' not found")
            continue
        
        valid_scenarios.append(scenario)
        print(f"✓ Valid: {scenario['name']}")
    
    if not valid_scenarios:
        print("\nERROR: No valid scenarios found!")
        return
    
    print(f"\n{len(valid_scenarios)} valid scenarios to run")

    # Run all scenarios
    all_results = []
    
    for i, scenario in enumerate(valid_scenarios, 1):
        print(f"\n\n{'#'*80}")
        print(f"# SCENARIO {i}/{len(valid_scenarios)}")
        print(f"{'#'*80}")
        
        metrics = run_scenario(scenario, study_col, cell_type_col)
        
        if metrics is not None:
            all_results.append(metrics)
    
    # Save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = OUTPUT_DIR / "singler_eval_all_scenarios.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"All results saved to: {output_path}")
        print('='*80)
        
        # Print summary table
        print(f"\n{'='*80}")
        print("FINAL SUMMARY - ALL SCENARIOS")
        print('='*80)
        print(f"\n{len(all_results)} scenarios completed successfully:\n")
        
        summary_cols = ['scenario_name', 'accuracy', 'ari', 'nmi', 'f1_macro', 'time_seconds']
        summary_df = results_df[summary_cols].copy()
        summary_df.columns = ['Scenario', 'Accuracy', 'ARI', 'NMI', 'F1 (macro)', 'Time (s)']
        
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}")
        
        # Calculate average metrics
        print("\nAverage Performance:")
        print(f"  Accuracy:    {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
        print(f"  ARI:         {results_df['ari'].mean():.4f} ± {results_df['ari'].std():.4f}")
        print(f"  NMI:         {results_df['nmi'].mean():.4f} ± {results_df['nmi'].std():.4f}")
        print(f"  F1 (macro):  {results_df['f1_macro'].mean():.4f} ± {results_df['f1_macro'].std():.4f}")
        print(f"  Time:        {results_df['time_seconds'].mean():.1f}s ± {results_df['time_seconds'].std():.1f}s")
        print('='*80)
    else:
        print("\nNo scenarios completed successfully!")

    return all_results


if __name__ == "__main__":
    main()
