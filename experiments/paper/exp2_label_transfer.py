#!/usr/bin/env python3
"""
Experiment 2: Label Transfer Benchmark
========================================
Is SCimilarity better than traditional ML for cross-study label transfer?

Setup: Train on van_galen_2019, test on other studies
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data
from sccl.evaluation import compute_metrics

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas_van_galen_subset.h5ad"
# DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas_50k_subset.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_STUDY = 'van_galen_2019'
QUERY_STUDIES = ['zhang_2023', 'beneyto-calabuig-2023', 'jiang_2020', 'velten_2021']

MODELS_TO_TEST = {
    'SCimilarity': 'scimilarity',
    'Random Forest': 'random_forest',
    'SVM': 'svm',
    'KNN': 'knn',
}


def main():
    print("="*80)
    print("EXPERIMENT 2: Label Transfer Benchmark")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(DATA_PATH)

    # Check reference exists
    available_studies = adata.obs['study'].unique() if 'study' in adata.obs else []

    if REFERENCE_STUDY not in available_studies:
        print(f"   ERROR: Reference study '{REFERENCE_STUDY}' not found!")
        print(f"   Available studies: {list(available_studies)[:5]}...")
        return

    # Get valid query studies
    valid_queries = [s for s in QUERY_STUDIES if s in available_studies]

    if not valid_queries:
        print("   ERROR: No query studies found!")
        return

    print(f"   Reference: {REFERENCE_STUDY}")
    print(f"   Query studies: {valid_queries}")

    # Prepare reference data
    print("\n2. Preparing reference data...")
    adata_ref = subset_data(adata, studies=[REFERENCE_STUDY])
    print(f"   Reference: {adata_ref.n_obs:,} cells")

    # Results storage
    results = []

    # Test on each query study
    for query_study in valid_queries:
        print(f"\n{'='*80}")
        print(f"Testing on: {query_study}")
        print('='*80)

        adata_query = subset_data(adata, studies=[query_study])
        print(f"  Query: {adata_query.n_obs:,} cells")

        # Test each model
        for model_name, model_type in MODELS_TO_TEST.items():
            print(f"\n  {model_name}...", end=' ')

            try:
                # Create pipeline
                if model_type == 'scimilarity':
                    pipeline = Pipeline(model=model_type, model_params={'model_path': MODEL_PATH})
                else:
                    pipeline = Pipeline(model=model_type)

                # Train on reference (if needed)
                if hasattr(pipeline.model, 'fit'):
                    adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
                    pipeline.model.fit(adata_ref_prep, target_column='cell_type')

                # Predict on query
                adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)
                pred = pipeline.model.predict(adata_query_prep, target_column=None)

                # Evaluate
                metrics = compute_metrics(
                    y_true=adata_query.obs['cell_type'].values,
                    y_pred=pred,
                    metrics=['accuracy', 'ari', 'nmi', 'f1_macro']
                )

                results.append({
                    'model': model_name,
                    'query_study': query_study,
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                    'f1': metrics['f1_macro'],
                    'n_cells': adata_query.n_obs
                })

                print(f"✓ ARI: {metrics['ari']:.3f}, Acc: {metrics['accuracy']:.3f}")

            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'model': model_name,
                    'query_study': query_study,
                    'accuracy': 0,
                    'ari': 0,
                    'nmi': 0,
                    'f1': 0,
                    'n_cells': adata_query.n_obs
                })

    # Summary
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\nAll Results:")
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("Average Performance by Model:")
    print("="*80)
    avg_by_model = results_df.groupby('model')[['accuracy', 'ari', 'nmi', 'f1']].mean()
    avg_by_model = avg_by_model.sort_values('ari', ascending=False)
    print(avg_by_model.to_string())

    # Save results
    print(f"\n3. Saving results to {OUTPUT_DIR}/")
    results_df.to_csv(OUTPUT_DIR / "exp2_label_transfer.csv", index=False)
    avg_by_model.to_csv(OUTPUT_DIR / "exp2_model_comparison.csv")

    print("   ✓ exp2_label_transfer.csv")
    print("   ✓ exp2_model_comparison.csv")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    best_model = avg_by_model.index[0]
    best_ari = avg_by_model.loc[best_model, 'ari']

    print(f"🏆 Best model: {best_model}")
    print(f"   Average ARI: {best_ari:.4f}")

    scim_ari = avg_by_model.loc['SCimilarity', 'ari'] if 'SCimilarity' in avg_by_model.index else 0

    if best_model == 'SCimilarity':
        print("\n✅ SCimilarity outperforms traditional ML for label transfer")
    elif scim_ari >= best_ari - 0.05:
        print("\n✅ SCimilarity competitive with best traditional ML methods")
    else:
        print("\n⚠️ Traditional ML outperforms SCimilarity")

    print("="*80)


if __name__ == "__main__":
    main()
