#!/usr/bin/env python
"""
Intra-Study Label Transfer Evaluation

Complements label_transfer_benchmark.py by evaluating within-study performance.

This tests best-case accuracy without batch effects by:
1. Taking a single study (e.g., van_galen_2019)
2. Splitting into train/test (80/20)
3. Training classifier on train set
4. Evaluating on test set

Use this to:
- Establish upper bound on accuracy (no batch effects)
- Compare inter-study vs intra-study performance
- Validate that low inter-study scores are due to batch effects, not method limitations
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = Path("data/AML_scAtlas.h5ad")
SCIMILARITY_MODEL = "models/model_v1.1"
SCIMILARITY_BATCH_SIZE = 5000

# Study to evaluate (use well-annotated study with many cells)
STUDY = 'van_galen_2019'

# Evaluation parameters
TEST_SIZE = 0.2  # 20% for testing
N_FOLDS = 5      # For cross-validation
RANDOM_STATE = 42

# Output
OUTPUT_DIR = Path("results_intra_study")
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

for d in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_study_data(data_path, study_name):
    """Load and prepare single study data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    adata = sc.read_h5ad(data_path)
    print(f"✓ Loaded atlas: {adata.n_obs:,} cells")

    # Find keys
    study_key = 'Study' if 'Study' in adata.obs.columns else 'study'
    label_key_candidates = ['cell_type_annotation', 'celltype', 'CellType',
                           'cell_type', 'annotation', 'Annotation']
    label_key = next((k for k in label_key_candidates if k in adata.obs.columns), None)

    if label_key is None:
        raise ValueError(f"Cannot find cell type column. Available: {adata.obs.columns.tolist()}")

    # Extract study
    study_mask = adata.obs[study_key] == study_name
    adata_study = adata[study_mask].copy()

    print(f"\n✓ Study: {study_name}")
    print(f"  Cells: {adata_study.n_obs:,}")
    print(f"  Cell types: {adata_study.obs[label_key].nunique()}")

    print(f"\n  Label distribution:")
    for celltype, count in adata_study.obs[label_key].value_counts().head(15).items():
        print(f"    {celltype:30s}: {count:6,} cells")

    return adata_study, label_key


# ==============================================================================
# TRADITIONAL CLASSIFIER
# ==============================================================================

def train_test_traditional(adata, label_key, test_size=0.2, random_state=42):
    """
    Traditional approach with train/test split.
    """
    print("\n" + "=" * 80)
    print("TRADITIONAL CLASSIFIER (Train/Test Split)")
    print("=" * 80)

    # Get raw counts
    if adata.raw is not None:
        adata_full = adata.raw.to_adata()
    else:
        adata_full = adata.copy()

    if 'counts' in adata_full.layers:
        adata_full.X = adata_full.layers['counts'].copy()

    # Normalize
    print("\n  Normalizing...")
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    # Select HVGs
    print("  Selecting HVGs...")
    sc.pp.highly_variable_genes(adata_full, n_top_genes=2000)
    hvgs = adata_full.var['highly_variable']

    X = adata_full[:, hvgs].X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    y = adata_full.obs[label_key].values

    # Remove missing labels
    valid = pd.notna(y)
    X = X[valid]
    y = y[valid]

    # Stratified split
    print(f"\n  Splitting: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train: {len(X_train):,} cells")
    print(f"  Test:  {len(X_test):,} cells")

    # Train
    print("\n  Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predict
    print("  Predicting...")
    y_pred = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)

    print(f"\n  Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ARI:      {ari:.4f}")
    print(f"    NMI:      {nmi:.4f}")

    return {
        'Method': 'Traditional (RF)',
        'Accuracy': accuracy,
        'ARI': ari,
        'NMI': nmi,
        'y_test': y_test,
        'y_pred': y_pred
    }


# ==============================================================================
# SCIMILARITY KNN
# ==============================================================================

def train_test_scimilarity(adata, label_key, test_size=0.2, random_state=42,
                           model_path=None, batch_size=5000):
    """
    SCimilarity approach with train/test split.
    """
    print("\n" + "=" * 80)
    print("SCIMILARITY KNN (Train/Test Split)")
    print("=" * 80)

    from scimilarity import CellAnnotation
    from scimilarity.utils import lognorm_counts, align_dataset

    # Load model
    print("\n  Loading SCimilarity model...")
    ca = CellAnnotation(model_path=model_path)

    # Get full gene set
    if adata.raw is not None:
        adata_full = adata.raw.to_adata()
    else:
        adata_full = adata.copy()

    if 'counts' in adata_full.layers:
        adata_full.X = adata_full.layers['counts'].copy()

    # Gene symbols
    if 'gene_name' in adata_full.var.columns:
        adata_full.var.index = adata_full.var['gene_name']

    # Common genes
    common = adata_full.var.index.intersection(ca.gene_order)
    print(f"  Common genes: {len(common):,}")

    # Subset and reorder
    gene_order_dict = {g: i for i, g in enumerate(ca.gene_order)}
    common_sorted = sorted(common, key=lambda x: gene_order_dict[x])
    adata_subset = adata_full[:, common_sorted].copy()

    # Compute embeddings in batches
    print("  Computing embeddings...")
    n_cells = adata_subset.n_obs
    n_batches = (n_cells + batch_size - 1) // batch_size
    embeddings_list = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_cells)

        batch = adata_subset[start:end].copy()
        batch_aligned = align_dataset(batch, ca.gene_order)

        if 'counts' not in batch_aligned.layers:
            batch_aligned.layers['counts'] = batch_aligned.X.copy()

        batch_norm = lognorm_counts(batch_aligned)
        batch_emb = ca.get_embeddings(batch_norm.X)
        embeddings_list.append(batch_emb)

        del batch, batch_aligned, batch_norm, batch_emb
        gc.collect()

    embeddings = np.vstack(embeddings_list)
    print(f"  ✓ Embeddings: {embeddings.shape}")

    # Get labels
    y = adata_subset.obs[label_key].values
    valid = pd.notna(y)
    embeddings = embeddings[valid]
    y = y[valid]

    # Stratified split
    print(f"\n  Splitting: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train: {len(X_train):,} cells")
    print(f"  Test:  {len(X_test):,} cells")

    # KNN
    print("\n  Training KNN (k=15)...")
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    knn.fit(X_train, y_train)

    # Predict
    print("  Predicting...")
    y_pred = knn.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)

    print(f"\n  Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ARI:      {ari:.4f}")
    print(f"    NMI:      {nmi:.4f}")

    return {
        'Method': 'SCimilarity KNN',
        'Accuracy': accuracy,
        'ARI': ari,
        'NMI': nmi,
        'y_test': y_test,
        'y_pred': y_pred
    }


# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================

def cross_validation_traditional(adata, label_key, n_folds=5):
    """Traditional RF with k-fold cross-validation."""
    print("\n" + "=" * 80)
    print(f"TRADITIONAL CLASSIFIER ({n_folds}-Fold Cross-Validation)")
    print("=" * 80)

    # Prepare data
    if adata.raw is not None:
        adata_full = adata.raw.to_adata()
    else:
        adata_full = adata.copy()

    if 'counts' in adata_full.layers:
        adata_full.X = adata_full.layers['counts'].copy()

    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)
    sc.pp.highly_variable_genes(adata_full, n_top_genes=2000)

    X = adata_full[:, adata_full.var['highly_variable']].X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    y = adata_full.obs[label_key].values
    valid = pd.notna(y)
    X = X[valid]
    y = y[valid]

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    aris = []
    nmis = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/{n_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        ari = adjusted_rand_score(y_test, y_pred)
        nmi = normalized_mutual_info_score(y_test, y_pred)

        accuracies.append(acc)
        aris.append(ari)
        nmis.append(nmi)

        print(f"    Accuracy: {acc:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")

    print(f"\n  Mean ± Std:")
    print(f"    Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"    ARI:      {np.mean(aris):.4f} ± {np.std(aris):.4f}")
    print(f"    NMI:      {np.mean(nmis):.4f} ± {np.std(nmis):.4f}")

    return {
        'Method': 'Traditional (RF) CV',
        'Accuracy_mean': np.mean(accuracies),
        'Accuracy_std': np.std(accuracies),
        'ARI_mean': np.mean(aris),
        'ARI_std': np.std(aris),
        'NMI_mean': np.mean(nmis),
        'NMI_std': np.std(nmis),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("INTRA-STUDY LABEL TRANSFER EVALUATION")
    print("=" * 80)
    print(f"\nStudy: {STUDY}")
    print("\nGoal: Establish best-case accuracy (no batch effects)")
    print("=" * 80)

    # Load data
    adata, label_key = load_study_data(DATA_PATH, STUDY)

    # Train/test split evaluation
    results_traditional = train_test_traditional(
        adata, label_key, TEST_SIZE, RANDOM_STATE
    )

    results_scimilarity = None
    if os.path.exists(SCIMILARITY_MODEL):
        try:
            results_scimilarity = train_test_scimilarity(
                adata, label_key, TEST_SIZE, RANDOM_STATE,
                SCIMILARITY_MODEL, SCIMILARITY_BATCH_SIZE
            )
        except Exception as e:
            print(f"\n✗ SCimilarity failed: {e}")
    else:
        print(f"\n⚠ SCimilarity model not found: {SCIMILARITY_MODEL}")

    # Cross-validation
    results_cv = cross_validation_traditional(adata, label_key, N_FOLDS)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary_data = []

    if results_traditional:
        summary_data.append({
            'Method': 'Traditional (RF)',
            'Evaluation': 'Train/Test Split',
            'Accuracy': f"{results_traditional['Accuracy']:.4f}",
            'ARI': f"{results_traditional['ARI']:.4f}",
            'NMI': f"{results_traditional['NMI']:.4f}",
        })

    if results_scimilarity:
        summary_data.append({
            'Method': 'SCimilarity KNN',
            'Evaluation': 'Train/Test Split',
            'Accuracy': f"{results_scimilarity['Accuracy']:.4f}",
            'ARI': f"{results_scimilarity['ARI']:.4f}",
            'NMI': f"{results_scimilarity['NMI']:.4f}",
        })

    if results_cv:
        summary_data.append({
            'Method': 'Traditional (RF)',
            'Evaluation': f'{N_FOLDS}-Fold CV',
            'Accuracy': f"{results_cv['Accuracy_mean']:.4f} ± {results_cv['Accuracy_std']:.4f}",
            'ARI': f"{results_cv['ARI_mean']:.4f} ± {results_cv['ARI_std']:.4f}",
            'NMI': f"{results_cv['NMI_mean']:.4f} ± {results_cv['NMI_std']:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save
    summary_df.to_csv(METRICS_DIR / "intra_study_results.csv", index=False)
    print(f"\n✓ Results saved: {METRICS_DIR}/intra_study_results.csv")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nIntra-study performance (same study, train/test split):")
    print("  • Represents BEST CASE accuracy (no batch effects)")
    print("  • Use as upper bound for inter-study comparisons")
    print("\nIf inter-study accuracy is much lower:")
    print("  → Indicates batch effects are the main challenge")
    print("  → Not method limitations")
    print("\nNext: Compare to inter-study results from label_transfer_benchmark.py")

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
