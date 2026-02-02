#!/usr/bin/env python3
"""
SingleR Multi-Scenario Evaluation Script
========================================
Compares Same-Platform vs Cross-Platform performance for SingleR.
"""

import sys
import warnings
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    classification_report,
    f1_score,
)
import singler

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = Path("/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad")

SCENARIOS = [
    {
        'name': 'Same-Platform: Beneyto (10X) -> Zhang (10X)',
        'reference': 'beneyto-calabuig-2023',
        'query': 'zhang_2023',
    },
    {
        'name': 'Cross-Platform: Zhai (SORT-seq) -> Zhang (10X)',
        'reference': 'zhai_2022',
        'query': 'zhang_2023',
    },
    {
        'name': 'Cross-Platform: Van Galen (Seq-Well) -> Velten (Muta-Seq)',
        'reference': 'van_galen_2019',
        'query': 'velten_2021',
    },
    {
        'name': 'Cross-Platform: Van Galen (Seq-Well) -> Beneyto (10X)',
        'reference': 'van_galen_2019',
        'query': 'beneyto-calabuig-2023',
    },
]

# Subsampling for computational efficiency
MAX_CELLS = 10000
MAX_REF_CELLS = 10000

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================================================================
# HELPER CLASSES & FUNCTIONS (Retained from your original)
# ==============================================================================

class H5ADReader:
    """Minimal h5ad reader with robust UTF-8 decoding."""
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._file = None

    def __enter__(self):
        self._file = h5py.File(self.filepath, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file: self._file.close()

    def _decode_array(self, arr):
        """Helper to safely decode byte arrays to UTF-8 strings."""
        if arr.dtype.kind in ['S', 'O']:  # Bytes or Objects
            try:
                # Handle potential mixed types or bytes
                return np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr])
            except AttributeError:
                return arr.astype(str)
        return arr

    def get_obs(self):
        obs = self._file['obs']
        data = {}
        
        # Robustly get index
        if '_index' in obs:
            index = self._decode_array(obs['_index'][()])
        else:
            index = None

        for key in obs.keys():
            if key.startswith('_') or key == '__categories': continue
            col_data = obs[key]
            
            # Handle Categorical (Pandas-style in H5AD)
            if isinstance(col_data, h5py.Group) and 'categories' in col_data:
                categories = self._decode_array(col_data['categories'][()])
                codes = col_data['codes'][()]
                data[key] = categories[codes]
            else:
                data[key] = self._decode_array(col_data[()])
        
        df = pd.DataFrame(data)
        if index is not None: df.index = index
        return df

    def get_var(self, use_raw=True):
        # Check if raw exists and has var
        group = self._file['var']
        if use_raw and 'raw' in self._file and 'var' in self._file['raw']:
            group = self._file['raw']['var']
            
        data = {}
        index = self._decode_array(group['_index'][()]) if '_index' in group else None
        
        for key in group.keys():
            if key.startswith('_'): continue
            col_data = group[key]
            if isinstance(col_data, h5py.Group) and 'categories' in col_data:
                categories = self._decode_array(col_data['categories'][()])
                codes = col_data['codes'][()]
                data[key] = categories[codes]
            else:
                data[key] = self._decode_array(col_data[()])
                
        df = pd.DataFrame(data)
        if index is not None: df.index = index
        return df

    def get_X(self, use_raw=True, cell_indices=None):
        group = self._file['X']
        if use_raw and 'raw' in self._file and 'X' in self._file['raw']:
            group = self._file['raw']['X']
            
        if isinstance(group, h5py.Group):
            data, indices, indptr = group['data'][()], group['indices'][()], group['indptr'][()]
            # Handle cases where shape attribute might be missing
            if 'shape' in group.attrs:
                shape = tuple(group.attrs['shape'])
            else:
                shape = (len(indptr)-1, indices.max()+1)
            X = csr_matrix((data, indices, indptr), shape=shape)
            return X[cell_indices] if cell_indices is not None else X
        
        return group[cell_indices] if cell_indices is not None else group[()]

def log_normalize(X, target_sum=1e4):
    if hasattr(X, 'toarray'): X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return np.log1p(X / row_sums * target_sum)

def load_scenario_data(filepath, study_name, study_col, cell_type_col, max_cells, use_raw=True):
    with H5ADReader(filepath) as reader:
        obs = reader.get_obs()
        mask = obs[study_col] == study_name
        indices = np.where(mask)[0]
        if len(indices) == 0: raise ValueError(f"Study {study_name} not found.")
        
        if max_cells and len(indices) > max_cells:
            np.random.seed(42)
            indices = np.sort(np.random.choice(indices, max_cells, replace=False))
            
        var = reader.get_var(use_raw=use_raw)
        X = reader.get_X(use_raw=use_raw, cell_indices=indices)
        obs = obs.iloc[indices].reset_index(drop=True)
        
        genes = var['feature_name'].values if 'feature_name' in var.columns else var.index.values
        return X, obs, genes

# ==============================================================================
# EVALUATION LOGIC
# ==============================================================================

def run_evaluation_scenario(scenario, study_col, cell_type_col):
    print(f"\n>>> SCENARIO: {scenario['name']}")
    
    # Load Reference
    X_ref, obs_ref, genes_ref = load_scenario_data(
        DATA_PATH, scenario['reference'], study_col, cell_type_col, MAX_REF_CELLS
    )
    ref_labels = obs_ref[cell_type_col].values
    
    # Load Query
    X_query, obs_query, genes_query = load_scenario_data(
        DATA_PATH, scenario['query'], study_col, cell_type_col, MAX_CELLS
    )
    y_true = obs_query[cell_type_col].values

    # Intersect Genes
    common_genes = np.intersect1d(genes_ref, genes_query)
    print(f"  Common genes: {len(common_genes)}")

    ref_idx = [np.where(genes_ref == g)[0][0] for g in common_genes]
    qry_idx = [np.where(genes_query == g)[0][0] for g in common_genes]

    # Process & Normalize
    X_ref_norm = log_normalize(X_ref[:, ref_idx])
    X_query_norm = log_normalize(X_query[:, qry_idx])

    # Filter invalid ref labels
    valid_ref = pd.notna(ref_labels)
    X_ref_norm, ref_labels = X_ref_norm[valid_ref], ref_labels[valid_ref]

    # Run SingleR
    start = time.time()
    results = singler.annotate_single(
        test_data=X_query_norm.T,
        test_features=common_genes,
        ref_data=X_ref_norm.T,
        ref_labels=ref_labels,
        ref_features=common_genes,
        num_threads=8,
    )
    duration = time.time() - start
    y_pred = np.asarray(results.column("best"))

    # Metrics
    valid_qry = pd.notna(y_true)
    y_t, y_p = y_true[valid_qry], y_pred[valid_qry]
    
    metrics = {
        'Scenario': scenario['name'],
        'Accuracy': accuracy_score(y_t, y_p),
        'F1_Macro': f1_score(y_t, y_p, average='macro', zero_division=0),
        'ARI': adjusted_rand_score(y_t, y_p),
        'Time_Sec': duration
    }
    
    print(f"  Done. Accuracy: {metrics['Accuracy']:.4f} | Time: {duration:.1f}s")
    return metrics

def main():
    study_col = 'Study'
    cell_type_col = 'Cell Type'
    
    all_results = []
    
    for scenario in SCENARIOS:
        try:
            res = run_evaluation_scenario(scenario, study_col, cell_type_col)
            all_results.append(res)
        except Exception as e:
            print(f"  Failed scenario {scenario['name']}: {e}")

    # Summary Table
    summary_df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(OUTPUT_DIR / "singler_multi_scenario_results.csv", index=False)
    print(f"\nDetailed results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

