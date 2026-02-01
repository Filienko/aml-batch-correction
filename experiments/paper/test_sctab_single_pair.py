#!/usr/bin/env python3
"""
Standalone scTab Evaluation Script (Minimal Dependencies)
=========================================================
Test scTab foundation model on a single reference/query study pair.

This version uses h5py to read h5ad files directly, avoiding scanpy/anndata
dependency conflicts.

Following official scTab inference steps from:
https://github.com/theislab/scTab

Phase 1: Data Preprocessing
    - Raw count data required (no normalization) - use .raw.X from h5ad
    - Align gene feature space to model's var.parquet order
    - Zero-fill missing genes
    - Note: Data should use same Ensembl release as model (release 104)

Phase 2: Load Trained Model
    - Load checkpoint via torch.load()
    - Load architecture from hparams.yaml
    - Initialize TabNet and load weights
    - Set model to eval mode

Phase 3: Run Model Inference
    - Apply sf-log1p normalization (scale to 10k + log1p)
    - Run batched inference
    - Map integer predictions to labels via cell_type.parquet

Usage:
    python test_sctab_single_pair.py

Requirements:
    - cellnet package: pip install git+https://github.com/theislab/scTab.git
    - h5py, numpy, pandas, torch, pyyaml, scipy, scikit-learn, tqdm
    - scTab checkpoint files
    - merlin model directory with:
        - var.parquet (gene order)
        - categorical_lookup/cell_type.parquet (label mapping)
        - hparams.yaml (in checkpoint parent dir)
"""

import sys
import warnings
import time
from pathlib import Path
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    classification_report,
    f1_score,
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths - adjust these to your setup
DATA_PATH = Path("/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad")

# scTab model paths
SCTAB_CHECKPOINT = Path("scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt")
MERLIN_DIR = Path("merlin_cxg_2023_05_15_sf-log1p_minimal")

# Study configuration
REFERENCE_STUDY = 'van_galen_2019'
QUERY_STUDY = 'zhang_2023'

# Subsampling for quick testing
MAX_CELLS = 5000
BATCH_SIZE = 2048

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

        # Handle different h5ad formats
        if isinstance(obs, h5py.Dataset):
            # Structured array format
            return pd.DataFrame(obs[()])
        else:
            # Group format (newer anndata)
            data = {}
            index = None

            # Get index
            if '_index' in obs:
                index = obs['_index'][()].astype(str)
            elif '__categories' in obs:
                # Handle categorical index
                pass

            # Get columns
            for key in obs.keys():
                if key.startswith('_') or key == '__categories':
                    continue
                try:
                    col_data = obs[key]
                    if isinstance(col_data, h5py.Group):
                        # Categorical data
                        if 'categories' in col_data and 'codes' in col_data:
                            categories = col_data['categories'][()].astype(str)
                            codes = col_data['codes'][()]
                            # Map codes to categories
                            data[key] = categories[codes]
                        else:
                            continue
                    else:
                        arr = col_data[()]
                        if arr.dtype.kind == 'S' or arr.dtype.kind == 'O':
                            arr = arr.astype(str)
                        data[key] = arr
                except Exception as e:
                    print(f"  Warning: Could not read obs column '{key}': {e}")
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

        # Handle different formats
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
                except Exception as e:
                    print(f"  Warning: Could not read var column '{key}': {e}")
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

        # Check if sparse
        if isinstance(X_group, h5py.Group):
            # Sparse matrix
            data = X_group['data'][()]
            indices = X_group['indices'][()]
            indptr = X_group['indptr'][()]

            # Determine shape
            if 'shape' in X_group.attrs:
                shape = tuple(X_group.attrs['shape'])
            else:
                # Infer shape
                n_rows = len(indptr) - 1
                n_cols = indices.max() + 1 if len(indices) > 0 else 0
                shape = (n_rows, n_cols)

            # Create sparse matrix (CSR format is typical for h5ad)
            X = csr_matrix((data, indices, indptr), shape=shape)

            if cell_indices is not None:
                X = X[cell_indices]

            return X
        else:
            # Dense matrix
            if cell_indices is not None:
                return X_group[cell_indices]
            return X_group[()]

    def get_n_obs(self):
        """Get number of cells."""
        if 'obs' in self._file:
            obs = self._file['obs']
            if isinstance(obs, h5py.Dataset):
                return len(obs)
            elif '_index' in obs:
                return len(obs['_index'])
            else:
                # Try to infer from first column
                for key in obs.keys():
                    if not key.startswith('_'):
                        return len(obs[key])
        return 0

    def get_n_vars(self, use_raw=True):
        """Get number of genes."""
        if use_raw and 'raw' in self._file and 'var' in self._file['raw']:
            var = self._file['raw']['var']
        else:
            var = self._file['var']

        if isinstance(var, h5py.Dataset):
            return len(var)
        elif '_index' in var:
            return len(var['_index'])
        return 0


def load_h5ad_data(filepath, study_col, cell_type_col, study_name, max_cells=None, use_raw=True):
    """
    Load data for a specific study from h5ad file.

    Parameters
    ----------
    filepath : str or Path
        Path to h5ad file
    study_col : str
        Column name for study/batch
    cell_type_col : str
        Column name for cell type labels
    study_name : str
        Name of study to extract
    max_cells : int, optional
        Maximum number of cells to return
    use_raw : bool
        Whether to use raw counts (recommended for scTab)

    Returns
    -------
    X : sparse matrix
        Count matrix (cells x genes)
    obs : DataFrame
        Cell metadata
    var : DataFrame
        Gene metadata
    """
    with H5ADReader(filepath) as reader:
        print(f"  Reading obs...")
        obs = reader.get_obs()

        # Filter to study
        if study_col not in obs.columns:
            raise ValueError(f"Study column '{study_col}' not found. Available: {obs.columns.tolist()}")

        mask = obs[study_col] == study_name
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(f"No cells found for study '{study_name}'")

        # Subsample if needed
        if max_cells and len(indices) > max_cells:
            np.random.seed(42)
            indices = np.random.choice(indices, max_cells, replace=False)
            indices = np.sort(indices)

        print(f"  Reading var...")
        var = reader.get_var(use_raw=use_raw)

        print(f"  Reading X matrix for {len(indices)} cells...")
        X = reader.get_X(use_raw=use_raw, cell_indices=indices)

        # Filter obs to selected indices
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
# HELPER FUNCTIONS
# ==============================================================================

def sf_log1p_norm(x):
    """Normalize each cell to have 10000 counts and apply log(x+1) transform."""
    counts = torch.sum(x, dim=1, keepdim=True)
    counts += counts == 0.  # Avoid zero division
    scaling_factor = 10000. / counts
    return torch.log1p(scaling_factor * x)


def streamline_count_matrix(X, gene_names, model_gene_names, use_cellnet=True):
    """
    Align gene space to match model's expected gene order.

    Following scTab official steps:
    - Order of columns in count matrix given by var.parquet
    - Genes must be in exactly same order
    - Zero-fill genes if missing in supplied data

    Parameters
    ----------
    X : sparse matrix (csc format preferred for column slicing)
        Count matrix (cells x genes)
    gene_names : array-like
        Gene names in the input data
    model_gene_names : array-like
        Gene names expected by the model (in correct order from var.parquet)
    use_cellnet : bool
        Try to use cellnet's streamline_count_matrix if available

    Returns
    -------
    aligned_matrix : np.ndarray
        Matrix with genes reordered to match model (zero-filled for missing genes)
    """
    # Try to use cellnet's implementation first
    if use_cellnet:
        try:
            from cellnet.utils.data_loading import streamline_count_matrix as cellnet_streamline
            print("  Using cellnet.utils.data_loading.streamline_count_matrix")
            X_csc = csc_matrix(X) if not isinstance(X, csc_matrix) else X
            return cellnet_streamline(X_csc, gene_names, model_gene_names)
        except ImportError:
            print("  cellnet not available, using manual implementation")

    # Manual implementation (fallback)
    gene_names = np.asarray(gene_names)
    model_gene_names = np.asarray(model_gene_names)

    # Create lookup for input genes
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Find which model genes are in our data
    n_cells = X.shape[0]
    n_model_genes = len(model_gene_names)

    # Create aligned matrix (zeros for missing genes)
    aligned = np.zeros((n_cells, n_model_genes), dtype=np.float32)

    # Convert to array for column access
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    # Fill in genes that exist in both (in model's expected order)
    found_genes = 0
    for i, gene in enumerate(model_gene_names):
        if gene in gene_to_idx:
            aligned[:, i] = X_dense[:, gene_to_idx[gene]]
            found_genes += 1

    print(f"  Gene overlap: {found_genes}/{len(model_gene_names)} model genes found in data")

    return aligned


# ==============================================================================
# SCTAB MODEL
# ==============================================================================

class ScTabInference:
    """scTab model wrapper for inference."""

    def __init__(self, checkpoint_path, merlin_dir, batch_size=2048):
        self.checkpoint_path = Path(checkpoint_path)
        self.merlin_dir = Path(merlin_dir)
        self.batch_size = batch_size

        self.model = None
        self.genes_from_model = None
        self.cell_type_mapping = None
        self.model_params = None
        self.device = None

        self._load_metadata()

    def _load_metadata(self):
        """Load gene ordering and cell type mapping."""
        print("\nLoading scTab model metadata...")

        # Load gene order
        var_path = self.merlin_dir / "var.parquet"
        self.genes_from_model = pd.read_parquet(var_path)
        print(f"  Loaded {len(self.genes_from_model)} genes from model")

        # Load cell type mapping
        cell_type_path = self.merlin_dir / "categorical_lookup" / "cell_type.parquet"
        self.cell_type_mapping = pd.read_parquet(cell_type_path)
        print(f"  Loaded {len(self.cell_type_mapping)} cell type labels")

        # Load model hyperparameters
        hparams_path = self.checkpoint_path.parent / "hparams.yaml"
        with open(hparams_path) as f:
            self.model_params = yaml.full_load(f.read())
        print(f"  Loaded model hyperparameters")

    def _load_model(self):
        """Load TabNet model from checkpoint."""
        if self.model is not None:
            return self.model

        print("\nLoading scTab model from checkpoint...")

        # Load checkpoint (weights_only=False needed for PyTorch 2.6+ as checkpoint contains optimizer state)
        if torch.cuda.is_available():
            ckpt = torch.load(self.checkpoint_path, weights_only=False)
            self.device = 'cuda'
        else:
            ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            self.device = 'cpu'

        print(f"  Using device: {self.device}")

        # Extract TabNet weights
        tabnet_weights = OrderedDict()
        for name, weight in ckpt['state_dict'].items():
            if 'classifier.' in name:
                tabnet_weights[name.replace('classifier.', '')] = weight

        # Import TabNet
        from cellnet.tabnet.tab_network import TabNet

        # Initialize model
        model = TabNet(
            input_dim=self.model_params['gene_dim'],
            output_dim=self.model_params['type_dim'],
            n_d=self.model_params['n_d'],
            n_a=self.model_params['n_a'],
            n_steps=self.model_params['n_steps'],
            gamma=self.model_params['gamma'],
            n_independent=self.model_params['n_independent'],
            n_shared=self.model_params['n_shared'],
            epsilon=self.model_params['epsilon'],
            virtual_batch_size=self.model_params['virtual_batch_size'],
            momentum=self.model_params['momentum'],
            mask_type=self.model_params['mask_type'],
        )

        # Load weights
        model.load_state_dict(tabnet_weights)
        model.eval()

        if self.device == 'cuda':
            model = model.cuda()

        self.model = model
        print("  Model loaded successfully")

        return model

    def predict_from_matrix(self, X, gene_names):
        """
        Predict cell types from count matrix.

        Parameters
        ----------
        X : sparse matrix or array
            Raw count matrix (cells x genes)
        gene_names : array-like
            Gene names matching columns of X

        Returns
        -------
        predictions : np.ndarray
            Predicted cell type labels
        """
        model = self._load_model()

        print(f"  Input: {X.shape[0]} cells x {X.shape[1]} genes")

        gene_names = np.asarray(gene_names)
        model_genes = self.genes_from_model['feature_name'].values

        # Diagnostic: show gene name formats
        print(f"  Input gene examples: {gene_names[:3]}")
        print(f"  Model gene examples: {model_genes[:3]}")

        # Step 1: Subset to genes that exist in model (as per scTab docs)
        # Try different matching strategies if direct match fails
        gene_mask = np.isin(gene_names, model_genes)
        n_direct_match = gene_mask.sum()

        if n_direct_match < 1000:
            print(f"  Direct match found only {n_direct_match} genes, trying alternative matching...")

            # Try case-insensitive matching
            gene_names_lower = np.char.lower(gene_names.astype(str))
            model_genes_lower = np.char.lower(model_genes.astype(str))
            gene_mask_lower = np.isin(gene_names_lower, model_genes_lower)

            if gene_mask_lower.sum() > n_direct_match:
                print(f"  Case-insensitive match: {gene_mask_lower.sum()} genes")
                # Build mapping for case-insensitive
                model_gene_map = {g.lower(): g for g in model_genes}
                gene_names_mapped = np.array([
                    model_gene_map.get(g.lower(), g) for g in gene_names
                ])
                gene_mask = np.isin(gene_names_mapped, model_genes)
                gene_names = gene_names_mapped

            # Try stripping Ensembl version numbers (ENSG00000123456.1 -> ENSG00000123456)
            if gene_mask.sum() < 1000 and any('.' in str(g) for g in gene_names[:100]):
                gene_names_stripped = np.array([str(g).split('.')[0] for g in gene_names])
                model_genes_stripped = np.array([str(g).split('.')[0] for g in model_genes])
                gene_mask_stripped = np.isin(gene_names_stripped, model_genes_stripped)

                if gene_mask_stripped.sum() > gene_mask.sum():
                    print(f"  Version-stripped match: {gene_mask_stripped.sum()} genes")
                    # Rebuild with stripped names
                    model_gene_map = {str(g).split('.')[0]: g for g in model_genes}
                    gene_names = np.array([
                        model_gene_map.get(str(g).split('.')[0], g) for g in gene_names
                    ])
                    gene_mask = np.isin(gene_names, model_genes)

        print(f"  Final gene overlap: {gene_mask.sum()}/{len(model_genes)} model genes found in data")

        if gene_mask.sum() < 1000:
            print("  WARNING: Very low gene overlap! Check gene naming conventions.")
            print(f"  Input uses: {gene_names[:5]}")
            print(f"  Model expects: {model_genes[:5]}")

        # Convert to csc_matrix for efficient column slicing
        if not isinstance(X, csc_matrix):
            X = csc_matrix(X)

        X_subset = X[:, gene_mask]
        gene_names_subset = gene_names[gene_mask]

        print(f"  After subsetting to model genes: {X_subset.shape[1]} genes")

        # Step 2: Align genes to model's expected order
        x_streamlined = streamline_count_matrix(
            X_subset,
            gene_names_subset,
            model_genes
        )

        # Run inference in batches
        n_cells = x_streamlined.shape[0]
        preds = []

        print(f"\nRunning scTab inference on {n_cells:,} cells...")
        print(f"  Batch size: {self.batch_size}")

        with torch.no_grad():
            for i in tqdm(range(0, n_cells, self.batch_size)):
                batch_x = x_streamlined[i:i + self.batch_size]

                # Convert to tensor
                x_tensor = torch.from_numpy(batch_x).float()
                if self.device == 'cuda':
                    x_tensor = x_tensor.cuda()

                # Apply sf-log1p normalization
                x_input = sf_log1p_norm(x_tensor)

                # Predict
                logits, _ = self.model(x_input)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(batch_preds)

        preds = np.hstack(preds)

        # Map integer predictions to cell type labels
        predictions = self.cell_type_mapping.loc[preds]['label'].values

        print(f"  Predicted {len(np.unique(predictions))} unique cell types")

        return predictions


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_predictions(y_true, y_pred, method_name="scTab"):
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    method_name : str
        Name for display

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {method_name}")
    print('='*60)

    # Remove any NaN from ground truth
    valid_mask = pd.notna(y_true)
    y_true_valid = np.asarray(y_true)[valid_mask]
    y_pred_valid = np.asarray(y_pred)[valid_mask]

    print(f"Cells with valid labels: {len(y_true_valid)}")

    # Compute metrics
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

    # Print results
    print(f"\nMetrics:")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  ARI:           {ari:.4f}")
    print(f"  NMI:           {nmi:.4f}")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"\nLabel counts:")
    print(f"  Ground truth:  {metrics['n_true_labels']} unique cell types")
    print(f"  Predicted:     {metrics['n_pred_labels']} unique cell types")

    # Classification report
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
    """Run scTab evaluation on a single reference/query pair."""

    print("="*80)
    print("scTab Single-Pair Evaluation (Minimal Dependencies)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data:       {DATA_PATH}")
    print(f"  Checkpoint: {SCTAB_CHECKPOINT}")
    print(f"  Merlin dir: {MERLIN_DIR}")
    print(f"  Reference:  {REFERENCE_STUDY}")
    print(f"  Query:      {QUERY_STUDY}")
    print(f"  Max cells:  {MAX_CELLS}")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Detect columns
    print(f"\n{'='*60}")
    print("DETECTING DATA STRUCTURE")
    print('='*60)

    study_col, cell_type_col, all_columns = detect_columns(DATA_PATH)
    print(f"Available columns: {all_columns[:10]}..." if len(all_columns) > 10 else f"Available columns: {all_columns}")
    print(f"Study column:     {study_col}")
    print(f"Cell type column: {cell_type_col}")

    if study_col is None:
        print("ERROR: Could not detect study column!")
        return
    if cell_type_col is None:
        print("ERROR: Could not detect cell type column!")
        return

    # List available studies
    available_studies = list_studies(DATA_PATH, study_col)
    print(f"\nAvailable studies: {available_studies}")

    # Check studies exist
    if REFERENCE_STUDY not in available_studies:
        print(f"ERROR: Reference study '{REFERENCE_STUDY}' not found!")
        return
    if QUERY_STUDY not in available_studies:
        print(f"ERROR: Query study '{QUERY_STUDY}' not found!")
        return

    # Load query data
    print(f"\n{'='*60}")
    print("LOADING QUERY DATA")
    print('='*60)

    X_query, obs_query, var_query = load_h5ad_data(
        DATA_PATH,
        study_col=study_col,
        cell_type_col=cell_type_col,
        study_name=QUERY_STUDY,
        max_cells=MAX_CELLS,
        use_raw=True
    )

    print(f"Query ({QUERY_STUDY}): {X_query.shape[0]:,} cells x {X_query.shape[1]:,} genes")

    # Get gene names
    if 'feature_name' in var_query.columns:
        gene_names = var_query['feature_name'].values
    elif var_query.index is not None and len(var_query.index) > 0:
        gene_names = var_query.index.values
    else:
        # Try to find gene name column
        for col in ['gene_name', 'gene_symbol', 'symbol', 'name']:
            if col in var_query.columns:
                gene_names = var_query[col].values
                break
        else:
            gene_names = np.arange(X_query.shape[1]).astype(str)
            print("  Warning: Could not find gene names, using indices")

    print(f"Gene names source: {type(gene_names)}, first 3: {gene_names[:3]}")

    # Get ground truth labels
    y_true = obs_query[cell_type_col].values
    print(f"Ground truth labels: {len(np.unique(y_true[pd.notna(y_true)]))} unique types")

    # Initialize scTab model
    print(f"\n{'='*60}")
    print("INITIALIZING SCTAB")
    print('='*60)

    start_time = time.time()

    sctab = ScTabInference(
        checkpoint_path=SCTAB_CHECKPOINT,
        merlin_dir=MERLIN_DIR,
        batch_size=BATCH_SIZE
    )

    # Run prediction
    print(f"\n{'='*60}")
    print("RUNNING PREDICTION")
    print('='*60)

    y_pred = sctab.predict_from_matrix(X_query, gene_names)

    elapsed = time.time() - start_time
    print(f"\nTotal inference time: {elapsed:.1f} seconds")

    # Evaluate
    metrics = evaluate_predictions(y_true, y_pred, method_name="scTab")
    metrics['time_seconds'] = elapsed
    metrics['reference_study'] = REFERENCE_STUDY
    metrics['query_study'] = QUERY_STUDY

    # Label overlap analysis
    label_overlap_analysis(y_true, y_pred)

    # Save results
    results_df = pd.DataFrame([metrics])
    output_path = OUTPUT_DIR / f"sctab_eval_{QUERY_STUDY}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"""
scTab Evaluation Results:
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
