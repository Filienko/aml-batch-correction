#!/usr/bin/env python3
"""
Experiment 4: Cross-Study Generalization
=========================================
Leave-one-study-out validation to test robustness
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data
from sccl.evaluation import compute_metrics

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas_van_galen_subset.h5ad"
# DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas_50k_subset.h5ad"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

VAN_GALEN_STUDIES = [
    'van_galen_2019',
    'zhang_2023',
    'beneyto-calabuig-2023',
    'jiang_2020',
    'velten_2021',
    'zhai_2022',
]


def main():
    print("="*80)
    print("EXPERIMENT 4: Cross-Study Generalization")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(DATA_PATH)

    # Get valid studies
    available_studies = adata.obs['study'].unique() if 'study' in adata.obs else []
    valid_studies = [s for s in VAN_GALEN_STUDIES if s in available_studies]

    if len(valid_studies) < 3:
        print(f"   ERROR: Need at least 3 studies, found {len(valid_studies)}")
        return

    print(f"   Using {len(valid_studies)} studies for leave-one-out validation")

    # Subset to valid studies
    adata = subset_data(adata, studies=valid_studies)

    # Leave-one-out validation
    print("\n2. Running leave-one-out validation...")
    results = []

    for held_out_study in valid_studies:
        print(f"\n   Holding out: {held_out_study}...", end=' ')

        # Prepare data
        train_studies = [s for s in valid_studies if s != held_out_study]
        adata_train = subset_data(adata, studies=train_studies)
        adata_test = subset_data(adata, studies=[held_out_study])

        # SCimilarity (doesn't need training on specific studies)
        pipeline = Pipeline(model="scimilarity")
        predictions = pipeline.predict(adata_test.copy())

        # Evaluate
        metrics = compute_metrics(
            y_true=adata_test.obs['cell_type'].values,
            y_pred=predictions,
            metrics=['accuracy', 'ari', 'nmi']
        )

        results.append({
            'held_out_study': held_out_study,
            'n_train_studies': len(train_studies),
            'n_train_cells': adata_train.n_obs,
            'n_test_cells': adata_test.n_obs,
            'accuracy': metrics['accuracy'],
            'ari': metrics['ari'],
            'nmi': metrics['nmi']
        })

        print(f"✓ ARI: {metrics['ari']:.3f}, Acc: {metrics['accuracy']:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\nPer-Study Results:")
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    print(f"  Mean ARI:      {results_df['ari'].mean():.4f} ± {results_df['ari'].std():.4f}")
    print(f"  Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"  Mean NMI:      {results_df['nmi'].mean():.4f} ± {results_df['nmi'].std():.4f}")
    print(f"\n  Min ARI:       {results_df['ari'].min():.4f} ({results_df.loc[results_df['ari'].idxmin(), 'held_out_study']})")
    print(f"  Max ARI:       {results_df['ari'].max():.4f} ({results_df.loc[results_df['ari'].idxmax(), 'held_out_study']})")
    print(f"  Std Dev:       {results_df['ari'].std():.4f}")

    # Save results
    print(f"\n3. Saving results to {OUTPUT_DIR}/")
    results_df.to_csv(OUTPUT_DIR / "exp4_generalization.csv", index=False)
    print("   ✓ exp4_generalization.csv")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    std_dev = results_df['ari'].std()
    mean_ari = results_df['ari'].mean()

    if std_dev < 0.05:
        print("✅ EXCELLENT: Very consistent performance across studies (σ < 0.05)")
    elif std_dev < 0.10:
        print("✅ GOOD: Consistent performance across studies (σ < 0.10)")
    elif std_dev < 0.15:
        print("⚠️ MODERATE: Some variation across studies (σ < 0.15)")
    else:
        print("❌ HIGH: Significant variation across studies")

    print(f"\nMean ARI: {mean_ari:.4f}")
    print(f"Std Dev:  {std_dev:.4f}")

    if mean_ari > 0.70 and std_dev < 0.10:
        print("\n✅ SCimilarity is robust across different AML studies")
    elif mean_ari > 0.70:
        print("\n⚠️ Good average performance but variable across studies")
    else:
        print("\n⚠️ Performance below target threshold")

    print("="*80)


if __name__ == "__main__":
    main()
