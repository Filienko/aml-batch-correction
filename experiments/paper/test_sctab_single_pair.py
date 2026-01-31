#!/usr/bin/env python3
"""
Standalone scTab Evaluation Script
===================================
Test scTab foundation model on a single reference/query study pair.

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

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from scipy.sparse import csc_matrix
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
# HELPER FUNCTIONS
# ==============================================================================

def detect_columns(adata):
    """Auto-detect study and cell type columns."""
    study_candidates = ['study', 'Study', 'dataset', 'batch', 'sample']
    cell_type_candidates = ['cell_type', 'Cell Type', 'celltype', 'cell_label', 'annotation']

    study_col = next((c for c in study_candidates if c in adata.obs.columns), None)
    cell_type_col = next((c for c in cell_type_candidates if c in adata.obs.columns), None)

    return study_col, cell_type_col


def subset_study(adata, study_col, study_name, max_cells=None):
    """Extract cells from a specific study, optionally subsampling."""
    mask = adata.obs[study_col] == study_name
    adata_subset = adata[mask].copy()

    if max_cells and adata_subset.n_obs > max_cells:
        indices = np.random.choice(adata_subset.n_obs, max_cells, replace=False)
        adata_subset = adata_subset[indices].copy()

    return adata_subset


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
            return cellnet_streamline(
                csc_matrix(X) if not isinstance(X, csc_matrix) else X,
                gene_names,
                model_gene_names
            )
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

        # Load checkpoint
        if torch.cuda.is_available():
            ckpt = torch.load(self.checkpoint_path)
            device = 'cuda'
        else:
            ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
            device = 'cpu'

        print(f"  Using device: {device}")

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

        if device == 'cuda':
            model = model.cuda()

        self.model = model
        self.device = device
        print("  Model loaded successfully")

        return model

    def predict(self, adata, use_dataloader=True):
        """
        Predict cell types using scTab.

        Following Phase 3 of scTab inference:
        1. Apply sf-log1p normalization (scale to 10k + log1p)
        2. Run batched inference
        3. Map integer predictions to labels via cell_type.parquet

        Parameters
        ----------
        adata : AnnData
            Data with raw counts (use adata.raw if available)
        use_dataloader : bool
            Try to use cellnet's dataloader_factory if available

        Returns
        -------
        predictions : np.ndarray
            Predicted cell type labels
        """
        model = self._load_model()

        # Phase 1: Get raw counts (no normalization yet)
        if adata.raw is not None:
            adata_raw = adata.raw.to_adata()
            print(f"  Using adata.raw (raw counts)")
        else:
            adata_raw = adata.copy()
            print(f"  Using adata.X directly (assuming raw counts)")

        # Get gene names - prefer feature_name to match var.parquet
        if 'feature_name' in adata_raw.var.columns:
            gene_names = adata_raw.var['feature_name'].values
        elif 'gene_ids' in adata_raw.var.columns:
            gene_names = adata_raw.var['gene_ids'].values
        else:
            gene_names = adata_raw.var.index.values

        print(f"  Input: {adata_raw.n_obs} cells x {adata_raw.n_vars} genes")

        # Align genes to model's expected order (from var.parquet)
        # Pass csc_matrix for efficient column slicing
        X = csc_matrix(adata_raw.X) if hasattr(adata_raw.X, 'toarray') else adata_raw.X
        x_streamlined = streamline_count_matrix(
            X,
            gene_names,
            self.genes_from_model['feature_name'].values
        )

        # Try to use cellnet's dataloader_factory
        loader = None
        if use_dataloader:
            try:
                from cellnet.utils.data_loading import dataloader_factory
                loader = dataloader_factory(x_streamlined, batch_size=self.batch_size)
                print(f"  Using cellnet dataloader_factory")
            except ImportError:
                print(f"  cellnet dataloader not available, using manual batching")

        # Phase 3: Run inference
        n_cells = x_streamlined.shape[0]
        preds = []

        print(f"\nRunning scTab inference on {n_cells:,} cells...")
        print(f"  Batch size: {self.batch_size}")

        with torch.no_grad():
            if loader is not None:
                # Use cellnet's dataloader
                for batch in tqdm(loader):
                    x_batch = batch[0]['X']
                    if not isinstance(x_batch, torch.Tensor):
                        x_batch = torch.from_numpy(x_batch).float()
                    if self.device == 'cuda':
                        x_batch = x_batch.cuda()

                    # Apply sf-log1p normalization
                    x_input = sf_log1p_norm(x_batch)
                    logits, _ = self.model(x_input)
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.append(batch_preds)
            else:
                # Manual batching
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

        # Map integer predictions to cell type labels via cell_type.parquet
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
    print("scTab Single-Pair Evaluation")
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

    # Load data
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print('='*60)

    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded atlas: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Detect columns
    study_col, cell_type_col = detect_columns(adata)
    print(f"Study column:     {study_col}")
    print(f"Cell type column: {cell_type_col}")

    # List available studies
    available_studies = adata.obs[study_col].unique()
    print(f"\nAvailable studies: {list(available_studies)}")

    # Check studies exist
    if REFERENCE_STUDY not in available_studies:
        print(f"ERROR: Reference study '{REFERENCE_STUDY}' not found!")
        return
    if QUERY_STUDY not in available_studies:
        print(f"ERROR: Query study '{QUERY_STUDY}' not found!")
        return

    # Extract reference and query
    print(f"\n{'='*60}")
    print("EXTRACTING STUDY DATA")
    print('='*60)

    adata_ref = subset_study(adata, study_col, REFERENCE_STUDY, max_cells=MAX_CELLS)
    print(f"Reference ({REFERENCE_STUDY}): {adata_ref.n_obs:,} cells")

    adata_query = subset_study(adata, study_col, QUERY_STUDY, max_cells=MAX_CELLS)
    print(f"Query ({QUERY_STUDY}): {adata_query.n_obs:,} cells")

    # Get ground truth labels
    y_true = adata_query.obs[cell_type_col].values
    print(f"\nGround truth labels: {len(np.unique(y_true[pd.notna(y_true)]))} unique types")

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

    y_pred = sctab.predict(adata_query)

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
