#!/usr/bin/env python3
"""
Standalone scTab Evaluation Script
====================================
Test scTab zero-shot foundation model on multiple reference/query study pairs.

scTab is a PRE-TRAINED foundation model - it does NOT require reference data.
The "reference" study is included only to match the evaluation protocol of
reference-based methods.

Following official scTab inference steps from:
https://github.com/theislab/scTab

Phase 1: Data Preprocessing
    - Raw count data required (no normalization) - use .raw.X from anndata
    - Align gene feature space to model's var.parquet order
    - Zero-fill missing genes

Phase 2: Load Trained Model
    - Load checkpoint via torch.load()
    - Load architecture from hparams.yaml
    - Initialize TabNet and load weights

Phase 3: Run Model Inference
    - Apply sf-log1p normalization (scale to 10k + log1p)
    - Run batched inference
    - Map integer predictions to labels via cell_type.parquet

Usage:
    python test_sctab_single_pair.py

Requirements:
    - cellnet package: pip install git+https://github.com/theislab/scTab.git
    - scanpy, numpy, pandas, torch, pyyaml, scipy, scikit-learn, tqdm
    - scTab checkpoint files and merlin model directory
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
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, classification_report
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl.data import subset_data, get_study_column, get_cell_type_column

# ==============================================================================
# CONFIGURATION  (matches exp_ensemble_embeddings.py)
# ==============================================================================

DATA_PATH = Path("/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad")

SCTAB_CHECKPOINT = Path("scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt")
MERLIN_DIR = Path("merlin_cxg_2023_05_15_sf-log1p_minimal")

MAX_CELLS_PER_STUDY = 15000
BATCH_SIZE = 2048

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
# SCTAB UTILITIES
# ==============================================================================

# Label mapping from scTab Cell Ontology to abbreviated labels
SCTAB_LABEL_MAP = {
    'B cell': 'B', 'naive B cell': 'B', 'memory B cell': 'B',
    'plasma cell': 'Plasma', 'plasmablast': 'Plasma',
    'CD14-positive monocyte': 'CD14+ Mono',
    'CD14-positive, CD16-negative classical monocyte': 'CD14+ Mono',
    'CD14-low, CD16-positive monocyte': 'CD16+ Mono',
    'CD16-positive, CD14-low monocyte': 'CD16+ Mono',
    'classical monocyte': 'CD14+ Mono',
    'non-classical monocyte': 'CD16+ Mono',
    'intermediate monocyte': 'CD14+ Mono',
    'natural killer cell': 'NK',
    'CD16-negative, CD56-bright natural killer cell, human': 'NK',
    'CD16-positive, CD56-dim natural killer cell, human': 'NK',
    'T cell': 'T',
    'CD4-positive, alpha-beta T cell': 'CD4+ T',
    'CD8-positive, alpha-beta T cell': 'CD8+ T',
    'CD4-positive helper T cell': 'CD4+ T',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ T',
    'CD8-positive, alpha-beta memory T cell': 'CD8+ T',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'CD4+ T',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'CD8+ T',
    'regulatory T cell': 'Treg',
    'gamma-delta T cell': 'gdT',
    'dendritic cell': 'DC',
    'conventional dendritic cell': 'cDC',
    'plasmacytoid dendritic cell': 'pDC',
    'CD1c-positive myeloid dendritic cell': 'cDC',
    'myeloid dendritic cell': 'cDC',
    'hematopoietic stem cell': 'HSPC',
    'hematopoietic multipotent progenitor cell': 'HSPC',
    'common myeloid progenitor': 'CMP',
    'granulocyte monocyte progenitor cell': 'GMP',
    'megakaryocyte-erythroid progenitor cell': 'MEP',
    'common lymphoid progenitor': 'CLP',
    'megakaryocyte': 'MEP',
    'erythroid lineage cell': 'Erythroid',
    'erythrocyte': 'Erythroid',
    'proerythroblast': 'Erythroid',
    'erythroblast': 'Erythroid',
    'neutrophil': 'Neutrophil',
    'basophil': 'Basophil',
    'eosinophil': 'Eosinophil',
    'mast cell': 'Mast',
}


def harmonize_labels(predictions, ground_truth_labels=None):
    """Harmonize scTab Cell Ontology labels to match ground truth vocabulary."""
    predictions = np.asarray(predictions)
    harmonized = predictions.copy()

    for sctab_label, target_label in SCTAB_LABEL_MAP.items():
        mask = predictions == sctab_label
        harmonized[mask] = target_label

    if ground_truth_labels is not None:
        gt_valid = ground_truth_labels[pd.notna(ground_truth_labels)]
        gt_labels_set = set([str(x) for x in np.unique(gt_valid)])
        unmapped = set(np.unique(harmonized)) - gt_labels_set

        for pred_label in unmapped:
            if not isinstance(pred_label, str):
                continue
            pred_lower = pred_label.lower()
            for gt_label in gt_labels_set:
                if not isinstance(gt_label, str):
                    continue
                gt_lower = gt_label.lower()
                if gt_lower in pred_lower or pred_lower in gt_lower:
                    mask = harmonized == pred_label
                    harmonized[mask] = gt_label
                    break

    return harmonized


def sf_log1p_norm(x):
    """Normalize each cell to 10000 counts and apply log1p (scTab preprocessing)."""
    counts = torch.sum(x, dim=1, keepdim=True)
    counts += counts == 0.
    scaling_factor = 10000. / counts
    return torch.log1p(scaling_factor * x)


def run_sctab_inference(adata_query, cell_type_col):
    """
    Run scTab zero-shot cell type prediction on an AnnData query object.

    Parameters
    ----------
    adata_query : AnnData
        Query data (raw counts expected in .raw.X or .X)
    cell_type_col : str
        Column name for ground truth labels in adata_query.obs

    Returns
    -------
    predictions : np.ndarray or None
        Harmonized predicted cell type labels
    """
    if not SCTAB_CHECKPOINT.exists():
        print(f"  scTab checkpoint not found: {SCTAB_CHECKPOINT}")
        return None
    if not MERLIN_DIR.exists():
        print(f"  Merlin directory not found: {MERLIN_DIR}")
        return None

    try:
        # Load model metadata
        var_path = MERLIN_DIR / "var.parquet"
        genes_from_model = pd.read_parquet(var_path)
        model_genes = genes_from_model['feature_name'].values

        cell_type_path = MERLIN_DIR / "categorical_lookup" / "cell_type.parquet"
        cell_type_mapping = pd.read_parquet(cell_type_path)

        hparams_path = SCTAB_CHECKPOINT.parent / "hparams.yaml"
        with open(hparams_path) as f:
            model_params = yaml.full_load(f.read())

        # Get raw counts
        if adata_query.raw is not None:
            X = adata_query.raw.X
            gene_names = adata_query.raw.var_names.values
        else:
            X = adata_query.X
            gene_names = adata_query.var_names.values

        # Match genes to model vocabulary
        gene_mask = np.isin(gene_names, model_genes)
        print(f"  Gene overlap: {gene_mask.sum()}/{len(model_genes)}")

        if gene_mask.sum() < 1000:
            print("  WARNING: Low gene overlap - check gene naming conventions")

        # Subset and align genes
        X_subset = csc_matrix(X)[:, gene_mask]
        gene_names_subset = gene_names[gene_mask]

        gene_to_idx = {g: i for i, g in enumerate(gene_names_subset)}
        n_cells = X_subset.shape[0]
        n_model_genes = len(model_genes)
        aligned = np.zeros((n_cells, n_model_genes), dtype=np.float32)

        X_dense = X_subset.toarray() if hasattr(X_subset, 'toarray') else np.asarray(X_subset)
        for i, gene in enumerate(model_genes):
            if gene in gene_to_idx:
                aligned[:, i] = X_dense[:, gene_to_idx[gene]]

        # Load model
        if torch.cuda.is_available():
            ckpt = torch.load(SCTAB_CHECKPOINT, weights_only=False)
            device = 'cuda'
        else:
            ckpt = torch.load(SCTAB_CHECKPOINT, map_location=torch.device('cpu'), weights_only=False)
            device = 'cpu'

        tabnet_weights = OrderedDict()
        for name, weight in ckpt['state_dict'].items():
            if 'classifier.' in name:
                tabnet_weights[name.replace('classifier.', '')] = weight

        from cellnet.tabnet.tab_network import TabNet

        model = TabNet(
            input_dim=model_params['gene_dim'],
            output_dim=model_params['type_dim'],
            n_d=model_params['n_d'],
            n_a=model_params['n_a'],
            n_steps=model_params['n_steps'],
            gamma=model_params['gamma'],
            n_independent=model_params['n_independent'],
            n_shared=model_params['n_shared'],
            epsilon=model_params['epsilon'],
            virtual_batch_size=model_params['virtual_batch_size'],
            momentum=model_params['momentum'],
            mask_type=model_params['mask_type'],
        )
        model.load_state_dict(tabnet_weights)
        model.eval()
        if device == 'cuda':
            model = model.cuda()

        # Run inference
        preds = []
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, BATCH_SIZE), desc="  scTab inference"):
                batch_x = aligned[i:i + BATCH_SIZE]
                x_tensor = torch.from_numpy(batch_x).float()
                if device == 'cuda':
                    x_tensor = x_tensor.cuda()
                x_input = sf_log1p_norm(x_tensor)
                logits, _ = model(x_input)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(batch_preds)

        preds = np.hstack(preds)
        predictions = cell_type_mapping.loc[preds]['label'].values

        # Harmonize labels
        y_true = adata_query.obs[cell_type_col].values
        predictions = harmonize_labels(predictions, ground_truth_labels=y_true)

        return predictions

    except Exception as e:
        print(f"  scTab error: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    print("scTab Zero-Shot Evaluation")
    print("=" * 80)
    print(f"\nData:       {DATA_PATH}")
    print(f"Checkpoint: {SCTAB_CHECKPOINT}")
    print(f"Merlin dir: {MERLIN_DIR}")
    print(f"Max cells:  {MAX_CELLS_PER_STUDY}")
    print(f"Scenarios:  {len(SCENARIOS)}")

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

        # Load query data (scTab is zero-shot - reference not used for inference)
        adata_query = subset_data(adata, studies=[scenario['query']]).to_memory()

        if MAX_CELLS_PER_STUDY and adata_query.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_query = adata_query[indices].copy()

        print(f"  Query:  {adata_query.n_obs:,} cells  ({scenario['query']})")
        print(f"  NOTE:   scTab is zero-shot, reference data not used for inference")

        y_true = adata_query.obs[cell_type_col].values

        # scTab: train_time = 0 (pre-trained, zero-shot)
        train_time = 0.0

        # Inference
        infer_start = time.time()
        y_pred = run_sctab_inference(adata_query, cell_type_col)
        infer_time = time.time() - infer_start

        if y_pred is None:
            print("  SKIP: scTab inference failed")
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
        print(f"    Train time:     {train_time:.1f}s  (zero-shot, no training)")
        print(f"    Inference time: {infer_time:.1f}s")
        print(f"    True labels:    {metrics['n_true_labels']}")
        print(f"    Pred labels:    {metrics['n_pred_labels']}")

        print(f"\n  Per-class report:")
        valid_mask = pd.notna(y_true)
        print(classification_report(
            np.asarray(y_true)[valid_mask],
            np.asarray(y_pred)[valid_mask],
            zero_division=0
        ))

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

        output_path = OUTPUT_DIR / "sctab_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No scenarios completed successfully.")


if __name__ == "__main__":
    main()
