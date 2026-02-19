#!/usr/bin/env python3
"""
Experiment: SCimilarity Hyperparameter Optimization
====================================================

Goal: Improve SCimilarity performance by testing different classifiers
and hyperparameters on the embeddings.

Current performance (beneyto → jiang):
- SCimilarity+KNN: ARI=0.200, Acc=45.8%
- CellTypist: ARI=0.446, Acc=71.9%

We'll test:
1. Different classifiers: KNN, RF, LogReg, SVM
2. Different KNN k values
3. Different RF parameters
"""

import sys
import warnings
import gc
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.evaluation import compute_metrics

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CELLS_PER_STUDY = 10000

# Test on challenging scenario: beneyto → jiang (same platform)
# This is where SCimilarity performs poorly (45.8%) vs CellTypist (71.9%)
REFERENCE = 'beneyto-calabuig-2023'
QUERY = 'jiang_2020'

# SCimilarity configurations to test
SCIMILARITY_CONFIGS = {
    # Current baseline
    'KNN-k15': {'classifier': 'knn', 'n_neighbors': 15},

    # Try different k values
    'KNN-k5': {'classifier': 'knn', 'n_neighbors': 5},
    'KNN-k10': {'classifier': 'knn', 'n_neighbors': 10},
    'KNN-k30': {'classifier': 'knn', 'n_neighbors': 30},
    'KNN-k50': {'classifier': 'knn', 'n_neighbors': 50},

    # Random Forest with different parameters
    'RF-default': {'classifier': 'random_forest'},
    'RF-deep': {
        'classifier': 'random_forest',
        'classifier_params': {'n_estimators': 200, 'max_depth': 30}
    },
    'RF-shallow': {
        'classifier': 'random_forest',
        'classifier_params': {'n_estimators': 100, 'max_depth': 10}
    },

    # Logistic Regression (like CellTypist)
    'LogReg': {'classifier': 'logistic_regression'},
    'LogReg-L1': {
        'classifier': 'logistic_regression',
        'classifier_params': {'penalty': 'l1', 'solver': 'saga'}
    },

    # SVM (might be slow)
    # 'SVM-RBF': {'classifier': 'svm'},
    # 'SVM-Linear': {
    #     'classifier': 'svm',
    #     'classifier_params': {'kernel': 'linear'}
    # },
}


def main():
    """Run SCimilarity optimization experiment."""
    print("="*80)
    print("SCimilarity Hyperparameter Optimization")
    print("="*80)
    print(f"\nScenario: {REFERENCE} → {QUERY} (Same Platform: 10x)")
    print(f"Goal: Improve SCimilarity from 45.8% to match CellTypist's 71.9%")
    print(f"\nTesting {len(SCIMILARITY_CONFIGS)} configurations...")
    print("="*80)

    results = []

    # Detect columns
    print("\nLoading data and detecting columns...")
    adata = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)
    print(f"  Study column: '{study_col}'")
    print(f"  Cell type column: '{cell_type_col}'")

    # Get reference and query data
    print(f"\nExtracting reference and query data...")
    adata_ref = subset_data(adata, studies=[REFERENCE])
    adata_query = subset_data(adata, studies=[QUERY])
    del adata
    gc.collect()

    # Subsample if needed
    if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
        print(f"  Subsampling reference: {adata_ref.n_obs:,} → {MAX_CELLS_PER_STUDY:,} cells")
        indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
        adata_ref = adata_ref[indices].copy()

    if adata_query.n_obs > MAX_CELLS_PER_STUDY:
        print(f"  Subsampling query: {adata_query.n_obs:,} → {MAX_CELLS_PER_STUDY:,} cells")
        indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
        adata_query = adata_query[indices].copy()

    print(f"\n  Reference: {adata_ref.n_obs:,} cells, {adata_ref.obs[cell_type_col].nunique()} types")
    print(f"  Query:     {adata_query.n_obs:,} cells, {adata_query.obs[cell_type_col].nunique()} types")

    # Preprocess once (reuse for all configs)
    print("\nPreprocessing data...")
    adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
    adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

    # Test each configuration
    print("\n" + "="*80)
    print("Testing Configurations")
    print("="*80)

    for config_name, config_params in SCIMILARITY_CONFIGS.items():
        print(f"\n[{config_name}]", end=' ')

        try:
            # Create pipeline with configuration
            scim_params = {'model_path': MODEL_PATH}
            scim_params.update(config_params)
            pipeline = Pipeline(model='scimilarity', model_params=scim_params)

            # Train on reference
            pipeline.model.fit(adata_ref_prep, target_column=cell_type_col)

            # Predict on query
            pred = pipeline.model.predict(adata_query_prep, target_column=None)

            # Evaluate
            metrics = compute_metrics(
                y_true=adata_query.obs[cell_type_col].values,
                y_pred=pred,
                metrics=['accuracy', 'ari', 'nmi']
            )

            results.append({
                'config': config_name,
                'accuracy': metrics['accuracy'],
                'ari': metrics['ari'],
                'nmi': metrics['nmi'],
                **config_params
            })

            print(f"✓ Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': config_name,
                'accuracy': 0,
                'ari': 0,
                'nmi': 0,
                **config_params
            })

        finally:
            # Cleanup
            del pipeline
            if 'pred' in locals():
                del pred
            gc.collect()

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    print("\nRanked by Accuracy:")
    print(df_results[['config', 'accuracy', 'ari']].to_string(index=False))

    # Save results
    output_file = OUTPUT_DIR / "scimilarity_optimization.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Show best configuration
    best_config = df_results.iloc[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Config: {best_config['config']}")
    print(f"Accuracy: {best_config['accuracy']:.3f} (baseline: 0.458)")
    print(f"ARI: {best_config['ari']:.3f} (baseline: 0.200)")
    
    improvement = (best_config['accuracy'] - 0.458) / 0.458 * 100
    print(f"\nImprovement: {improvement:+.1f}%")

    # Compare to CellTypist
    celltypist_acc = 0.719
    gap = celltypist_acc - best_config['accuracy']
    print(f"\nCellTypist accuracy: {celltypist_acc:.3f}")
    print(f"Gap to CellTypist: {gap:.3f} ({gap/celltypist_acc*100:.1f}%)")


if __name__ == "__main__":
    main()
