#!/usr/bin/env python3
"""
Experiment: Fair Comparison - Pure vs Semi-Supervised
======================================================

Compares methods fairly by testing both pure supervised and semi-supervised:

PURE SUPERVISED (no query structure):
- CellTypist (majority_voting=False): logistic regression on genes
- SCimilarity (label_propagation=False): classifier on embeddings

SEMI-SUPERVISED (uses query structure):
- CellTypist (majority_voting=True): logistic regression + clustering refinement
- SCimilarity (label_propagation=True): classifier + kNN smoothing

This reveals:
1. Which method has better supervised transfer
2. Which method benefits more from semi-supervised refinement
3. Whether semi-supervised helps same-platform vs cross-platform
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

# Test scenarios
SCENARIOS = [
    {
        'name': 'Same-Platform: beneyto → jiang',
        'reference': 'beneyto-calabuig-2023',
        'query': 'jiang_2020',
        'platform_type': 'SAME',
    },
    {
        'name': 'Cross-Platform: van_galen → jiang',
        'reference': 'van_galen_2019',
        'query': 'jiang_2020',
        'platform_type': 'CROSS',
    },
]

# Methods to test
METHODS = {
    # Pure supervised
    'CellTypist (pure)': {
        'model': 'celltypist',
        'params': {'majority_voting': False},
        'type': 'PURE'
    },
    'SCimilarity+KNN (pure)': {
        'model': 'scimilarity',
        'params': {'classifier': 'knn', 'label_propagation': False, 'model_path': MODEL_PATH},
        'type': 'PURE'
    },
    'SCimilarity+LogReg (pure)': {
        'model': 'scimilarity',
        'params': {'classifier': 'logistic_regression', 'label_propagation': False, 'model_path': MODEL_PATH},
        'type': 'PURE'
    },

    # Semi-supervised
    'CellTypist (semi)': {
        'model': 'celltypist',
        'params': {'majority_voting': True},
        'type': 'SEMI'
    },
    'SCimilarity+KNN (semi)': {
        'model': 'scimilarity',
        'params': {'classifier': 'knn', 'label_propagation': True, 'model_path': MODEL_PATH},
        'type': 'SEMI'
    },
    'SCimilarity+LogReg (semi)': {
        'model': 'scimilarity',
        'params': {'classifier': 'logistic_regression', 'label_propagation': True, 'model_path': MODEL_PATH},
        'type': 'SEMI'
    },
}


def main():
    """Run fair comparison experiment."""
    print("="*80)
    print("Fair Comparison: Pure vs Semi-Supervised")
    print("="*80)
    print("\nPURE SUPERVISED: No query structure used")
    print("  - CellTypist (majority_voting=False)")
    print("  - SCimilarity (label_propagation=False)")
    print("\nSEMI-SUPERVISED: Uses query cell-cell similarity")
    print("  - CellTypist (majority_voting=True)")
    print("  - SCimilarity (label_propagation=True)")
    print("="*80)

    results = []

    # Detect columns
    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"{scenario['name']} ({scenario['platform_type']} platform)")
        print(f"  Reference: {scenario['reference']}")
        print(f"  Query:     {scenario['query']}")
        print('='*80)

        # Get data
        adata_ref = subset_data(adata, studies=[scenario['reference']])
        adata_query = subset_data(adata, studies=[scenario['query']])

        # Subsample
        if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_ref = adata_ref[indices].copy()

        if adata_query.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_query = adata_query[indices].copy()

        print(f"  Reference: {adata_ref.n_obs:,} cells")
        print(f"  Query:     {adata_query.n_obs:,} cells")

        # Preprocess
        adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
        adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

        # Test each method
        for method_name, method_config in METHODS.items():
            print(f"\n  [{method_name}]...", end=' ')

            try:
                # Create pipeline
                pipeline = Pipeline(
                    model=method_config['model'],
                    model_params=method_config['params']
                )

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
                    'scenario': scenario['name'],
                    'platform_type': scenario['platform_type'],
                    'method': method_name,
                    'method_type': method_config['type'],
                    'model': method_config['model'],
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                })

                print(f"✓ Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()

            finally:
                del pipeline
                if 'pred' in locals():
                    del pred
                gc.collect()

        # Cleanup scenario data
        del adata_ref, adata_query, adata_ref_prep, adata_query_prep
        gc.collect()

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    df_results = pd.DataFrame(results)

    # Group by scenario
    for scenario_name in df_results['scenario'].unique():
        print(f"\n{scenario_name}:")
        scenario_df = df_results[df_results['scenario'] == scenario_name].copy()
        scenario_df = scenario_df.sort_values(['method_type', 'accuracy'], ascending=[True, False])

        print("\n" + scenario_df[['method', 'method_type', 'accuracy', 'ari']].to_string(index=False))

        # Compare pure vs semi for each base model
        print("\n  Semi-supervised effect:")
        for base_model in ['CellTypist', 'SCimilarity+KNN', 'SCimilarity+LogReg']:
            pure = scenario_df[scenario_df['method'] == f'{base_model} (pure)']
            semi = scenario_df[scenario_df['method'] == f'{base_model} (semi)']

            if len(pure) > 0 and len(semi) > 0:
                pure_acc = pure.iloc[0]['accuracy']
                semi_acc = semi.iloc[0]['accuracy']
                diff = semi_acc - pure_acc
                pct_change = diff / pure_acc * 100
                symbol = "✓" if diff > 0 else "✗"
                print(f"    {base_model:25s}: {pure_acc:.3f} → {semi_acc:.3f} ({diff:+.3f}, {pct_change:+.1f}%) {symbol}")

    # Save results
    output_file = OUTPUT_DIR / "fair_comparison.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: Method Comparison")
    print("="*80)

    summary = df_results.groupby(['method_type', 'method']).agg({
        'accuracy': 'mean',
        'ari': 'mean'
    }).reset_index()
    summary = summary.sort_values(['method_type', 'accuracy'], ascending=[True, False])

    print("\nAverage Performance:")
    print(summary.to_string(index=False))

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\n1. PURE SUPERVISED comparison shows true transfer learning ability")
    print("   (No query structure used - fair comparison)")
    print("\n2. SEMI-SUPERVISED boost shows benefit of using query manifold")
    print("   (Should help same-platform, might hurt cross-platform)")
    print("\n3. If SCimilarity+semi matches CellTypist+semi:")
    print("   → Embeddings are good, just need semi-supervised refinement")
    print("\n4. If gap remains:")
    print("   → Direct gene expression (CellTypist) > embeddings (SCimilarity)")


if __name__ == "__main__":
    main()
