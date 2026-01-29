#!/usr/bin/env python3
"""
Test: CellTypist Majority Voting Effect
========================================

Question: How much does majority_voting improve CellTypist performance?

majority_voting=True: SEMI-SUPERVISED
- Clusters the query data (uses query structure!)
- Refines predictions using majority vote within clusters
- NOT pure supervised transfer

majority_voting=False: PURE SUPERVISED
- Only uses logistic regression predictions
- No query data structure used
- True supervised transfer

This will reveal if CellTypist's high performance is due to:
1. Good supervised transfer (majority_voting=False)
2. Semi-supervised refinement (majority_voting=True)
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
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CELLS_PER_STUDY = 10000

# Test scenarios
SCENARIOS = [
    {
        'name': 'Same-Platform: beneyto → jiang',
        'reference': 'beneyto-calabuig-2023',
        'query': 'jiang_2020',
    },
    {
        'name': 'Cross-Platform: van_galen → jiang',
        'reference': 'van_galen_2019',
        'query': 'jiang_2020',
    },
]

def main():
    """Compare CellTypist with and without majority voting."""
    print("="*80)
    print("CellTypist: Majority Voting Effect")
    print("="*80)
    print("\nTesting:")
    print("  1. majority_voting=False (PURE SUPERVISED)")
    print("  2. majority_voting=True  (SEMI-SUPERVISED - uses query structure)")
    print("="*80)

    results = []

    # Detect columns
    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
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

        # Test both configurations
        for use_majority_voting in [False, True]:
            config_name = f"majority_voting={use_majority_voting}"
            method_type = "SEMI-SUPERVISED" if use_majority_voting else "PURE SUPERVISED"

            print(f"\n  [{config_name}] ({method_type})...", end=' ')

            try:
                # Create CellTypist with specific majority_voting setting
                pipeline = Pipeline(
                    model='celltypist',
                    model_params={'majority_voting': use_majority_voting}
                )

                # Train on reference
                pipeline.model.fit(adata_ref_prep, target_column=cell_type_col)

                # Predict on query
                pred = pipeline.model.predict(adata_query_prep, target_column=None)

                # Evaluate
                metrics = compute_metrics(
                    y_true=adata_query.obs[cell_type_col].values,
                    y_pred=pred,
                    metrics=['accuracy', 'ari', 'nmi', 'f1_macro']
                )

                results.append({
                    'scenario': scenario['name'],
                    'majority_voting': use_majority_voting,
                    'method_type': method_type,
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                    'f1_macro': metrics['f1_macro'],
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

    for scenario_name in df_results['scenario'].unique():
        print(f"\n{scenario_name}:")
        scenario_df = df_results[df_results['scenario'] == scenario_name]

        print(scenario_df[['majority_voting', 'method_type', 'accuracy', 'ari', 'f1_macro']].to_string(index=False))

        # Calculate improvement
        pure = scenario_df[scenario_df['majority_voting'] == False].iloc[0]
        semi = scenario_df[scenario_df['majority_voting'] == True].iloc[0]

        acc_improvement = semi['accuracy'] - pure['accuracy']
        ari_improvement = semi['ari'] - pure['ari']

        print(f"\n  Majority Voting Effect:")
        print(f"    Accuracy: {pure['accuracy']:.3f} → {semi['accuracy']:.3f} ({acc_improvement:+.3f}, {acc_improvement/pure['accuracy']*100:+.1f}%)")
        print(f"    ARI:      {pure['ari']:.3f} → {semi['ari']:.3f} ({ari_improvement:+.3f}, {ari_improvement/pure['ari']*100:+.1f}%)")

    # Save results
    output_file = OUTPUT_DIR / "majority_voting_effect.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nIf majority_voting=True gives BIG improvement:")
    print("  → CellTypist's high performance relies on semi-supervised learning")
    print("  → It uses query data structure (clustering) to refine predictions")
    print("  → NOT pure supervised transfer!")
    print("\nIf majority_voting=True gives SMALL improvement:")
    print("  → CellTypist has genuinely good supervised transfer")
    print("  → Logistic regression on genes is truly effective")
    print("  → Performance is realistic for pure label transfer")


if __name__ == "__main__":
    main()
