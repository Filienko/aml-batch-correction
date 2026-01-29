#!/usr/bin/env python3
"""
Experiment: Same-Platform vs Cross-Platform Comparison
=======================================================

Research Question:
How much does platform difference affect label transfer performance?

Comparisons:
1. SAME PLATFORM:  beneyto-calabuig-2023 (10x) → jiang_2020 (10x)
2. CROSS PLATFORM: van_galen_2019 (Seq-Well) → jiang_2020 (10x)
3. CROSS PLATFORM: van_galen_2019 (Seq-Well) → beneyto-calabuig-2023 (10x)

This isolates the effect of sequencing platform on transfer performance.
"""

import sys
import warnings
import gc
warnings.filterwarnings('ignore')

import pandas as pd
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.evaluation import compute_metrics

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Platform information
PLATFORM_INFO = {
    'van_galen_2019': 'Seq-Well',
    'beneyto-calabuig-2023': '10x Genomics',
    'jiang_2020': '10x Genomics',
}

# Transfer scenarios to test
TRANSFER_SCENARIOS = [
    # Same platform transfers (control)
    {
        'name': 'Same-Platform: 10x → 10x',
        'reference': 'beneyto-calabuig-2023',
        'query': 'jiang_2020',
        'ref_platform': '10x Genomics',
        'query_platform': '10x Genomics',
        'platform_match': 'SAME'
    },
    # Cross-platform transfers
    {
        'name': 'Cross-Platform: Seq-Well → 10x (jiang)',
        'reference': 'van_galen_2019',
        'query': 'jiang_2020',
        'ref_platform': 'Seq-Well',
        'query_platform': '10x Genomics',
        'platform_match': 'DIFFERENT'
    },
    {
        'name': 'Cross-Platform: Seq-Well → 10x (beneyto)',
        'reference': 'van_galen_2019',
        'query': 'beneyto-calabuig-2023',
        'ref_platform': 'Seq-Well',
        'query_platform': '10x Genomics',
        'platform_match': 'DIFFERENT'
    },
]

MODELS_TO_TEST = {
    'SCimilarity+KNN': ('scimilarity', {'classifier': 'knn'}),
    'CellTypist': ('celltypist', {}),  # Custom trained on reference data
    # Comment out others temporarily to save memory - add back after first successful run
    # 'SCimilarity+RF': ('scimilarity', {'classifier': 'random_forest'}),
    # 'Random Forest': ('random_forest', {}),
    # 'KNN': ('knn', {}),
}

# Memory optimization - reduced to 10k to prevent OOM
MAX_CELLS_PER_STUDY = 10000  # Further reduced from 20k


def main():
    print("="*80)
    print("EXPERIMENT: Same-Platform vs Cross-Platform Comparison")
    print("="*80)
    print("\nResearch Question:")
    print("  How much does platform difference affect label transfer?")
    print("\nComparisons:")
    for i, scenario in enumerate(TRANSFER_SCENARIOS, 1):
        print(f"  {i}. {scenario['name']}")
        print(f"     {scenario['reference']} ({scenario['ref_platform']}) → {scenario['query']} ({scenario['query_platform']})")

    # Results storage
    results = []

    # Detect columns (load once to get column names, then delete)
    print("\n" + "="*80)
    print("1. Detecting column names...")
    print("="*80)
    adata_temp = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata_temp)
    cell_type_col = get_cell_type_column(adata_temp)
    print(f"   Using study column: '{study_col}'")
    print(f"   Using cell type column: '{cell_type_col}'")
    del adata_temp
    gc.collect()

    # Test each transfer scenario
    print("\n" + "="*80)
    print("2. Testing Transfer Scenarios")
    print("="*80)

    for scenario_idx, scenario in enumerate(TRANSFER_SCENARIOS, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {scenario_idx}/{len(TRANSFER_SCENARIOS)}: {scenario['name']}")
        print(f"  Reference: {scenario['reference']} ({scenario['ref_platform']})")
        print(f"  Query:     {scenario['query']} ({scenario['query_platform']})")
        print(f"  Platform:  {scenario['platform_match']}")
        print('='*80)

        # Load full data fresh for each scenario (memory efficient)
        print("  Loading data...")
        adata = sc.read_h5ad(DATA_PATH)

        # Get reference and query data
        adata_ref = subset_data(adata, studies=[scenario['reference']])
        adata_query = subset_data(adata, studies=[scenario['query']])

        # Delete full dataset immediately
        del adata
        gc.collect()

        # Subsample if needed
        if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
            import numpy as np
            print(f"  Reference: {adata_ref.n_obs:,} cells (subsampling to {MAX_CELLS_PER_STUDY:,})")
            indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_ref = adata_ref[indices].copy()

        if adata_query.n_obs > MAX_CELLS_PER_STUDY:
            import numpy as np
            print(f"  Query: {adata_query.n_obs:,} cells (subsampling to {MAX_CELLS_PER_STUDY:,})")
            indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_query = adata_query[indices].copy()

        print(f"  Reference: {adata_ref.n_obs:,} cells, {adata_ref.obs[cell_type_col].nunique()} types")
        print(f"  Query:     {adata_query.n_obs:,} cells, {adata_query.obs[cell_type_col].nunique()} types")

        # Test each model on this scenario
        for model_name, (model_type, model_params) in MODELS_TO_TEST.items():
            print(f"\n  {model_name}...", end=' ')

            try:
                # Create pipeline
                if model_type == 'scimilarity':
                    scim_params = {'model_path': MODEL_PATH}
                    scim_params.update(model_params)
                    pipeline = Pipeline(model=model_type, model_params=scim_params)
                else:
                    pipeline = Pipeline(model=model_type, model_params=model_params if model_params else None)

                # Train on reference
                if hasattr(pipeline.model, 'fit'):
                    adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
                    pipeline.model.fit(adata_ref_prep, target_column=cell_type_col)

                # Predict on query
                adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)
                pred = pipeline.model.predict(adata_query_prep, target_column=None)

                # Evaluate
                metrics = compute_metrics(
                    y_true=adata_query.obs[cell_type_col].values,
                    y_pred=pred,
                    metrics=['accuracy', 'ari', 'nmi']
                )

                results.append({
                    'scenario': scenario['name'],
                    'reference_study': scenario['reference'],
                    'query_study': scenario['query'],
                    'reference_platform': scenario['ref_platform'],
                    'query_platform': scenario['query_platform'],
                    'platform_match': scenario['platform_match'],
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                    'n_ref_cells': adata_ref.n_obs,
                    'n_query_cells': adata_query.n_obs
                })

                print(f"✓ ARI: {metrics['ari']:.3f}, Acc: {metrics['accuracy']:.3f}")

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'scenario': scenario['name'],
                    'reference_study': scenario['reference'],
                    'query_study': scenario['query'],
                    'reference_platform': scenario['ref_platform'],
                    'query_platform': scenario['query_platform'],
                    'platform_match': scenario['platform_match'],
                    'model': model_name,
                    'accuracy': 0,
                    'ari': 0,
                    'nmi': 0,
                    'n_ref_cells': adata_ref.n_obs,
                    'n_query_cells': adata_query.n_obs
                })

            finally:
                # Memory cleanup
                del pipeline
                if 'adata_ref_prep' in locals():
                    del adata_ref_prep
                if 'adata_query_prep' in locals():
                    del adata_query_prep
                if 'pred' in locals():
                    del pred
                gc.collect()

        # Cleanup after scenario
        del adata_ref, adata_query
        gc.collect()

    # Analysis
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS: All Transfer Scenarios")
    print("="*80)
    print(results_df[['scenario', 'model', 'platform_match', 'ari', 'accuracy']].to_string(index=False))

    # Compare same vs cross platform
    print("\n" + "="*80)
    print("COMPARISON: Same-Platform vs Cross-Platform")
    print("="*80)

    same_platform = results_df[results_df['platform_match'] == 'SAME']
    cross_platform = results_df[results_df['platform_match'] == 'DIFFERENT']

    if len(same_platform) > 0 and len(cross_platform) > 0:
        print("\nSame-Platform Transfer (10x → 10x):")
        same_avg = same_platform.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
        print(same_avg.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nCross-Platform Transfer (Seq-Well → 10x):")
        cross_avg = cross_platform.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
        print(cross_avg.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nPlatform Effect (Cross - Same):")
        platform_effect = cross_avg - same_avg
        print(platform_effect.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Per-scenario breakdown
    print("\n" + "="*80)
    print("Performance by Scenario and Model")
    print("="*80)

    pivot = results_df.pivot_table(
        values='ari',
        index='scenario',
        columns='model',
        aggfunc='mean'
    )
    print("\nARI by Scenario:")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Save results
    print(f"\n" + "="*80)
    print("3. Saving results...")
    print("="*80)
    results_df.to_csv(OUTPUT_DIR / "exp_platform_comparison.csv", index=False)
    print(f"   ✓ {OUTPUT_DIR}/exp_platform_comparison.csv")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Find best performing scenarios
    best_same = same_platform.loc[same_platform['ari'].idxmax()] if len(same_platform) > 0 else None
    best_cross = cross_platform.loc[cross_platform['ari'].idxmax()] if len(cross_platform) > 0 else None

    if best_same is not None:
        print(f"\nBest Same-Platform Transfer:")
        print(f"  Model: {best_same['model']}")
        print(f"  ARI: {best_same['ari']:.4f}")
        print(f"  Accuracy: {best_same['accuracy']:.4f}")

    if best_cross is not None:
        print(f"\nBest Cross-Platform Transfer:")
        print(f"  Model: {best_cross['model']}")
        print(f"  Scenario: {best_cross['scenario']}")
        print(f"  ARI: {best_cross['ari']:.4f}")
        print(f"  Accuracy: {best_cross['accuracy']:.4f}")

    # Platform effect analysis
    if best_same is not None and len(cross_platform) > 0:
        # Get average cross-platform for same model
        cross_model = cross_platform[cross_platform['model'] == best_same['model']]
        if len(cross_model) > 0:
            avg_cross_ari = cross_model['ari'].mean()
            platform_penalty = best_same['ari'] - avg_cross_ari

            print(f"\n" + "="*80)
            print(f"PLATFORM EFFECT ANALYSIS ({best_same['model']})")
            print("="*80)
            print(f"  Same-Platform ARI:  {best_same['ari']:.4f}")
            print(f"  Cross-Platform ARI: {avg_cross_ari:.4f}")
            print(f"  Platform Penalty:   {platform_penalty:+.4f} ARI")
            print(f"  Relative Impact:    {(platform_penalty/best_same['ari'])*100:+.1f}%")

            if platform_penalty > 0.1:
                print("\n❌ LARGE platform effect: Cross-platform transfer significantly worse")
            elif platform_penalty > 0.05:
                print("\n⚠️ MODERATE platform effect: Some degradation in cross-platform")
            else:
                print("\n✅ SMALL platform effect: Model robust to platform differences")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
This experiment isolates the effect of sequencing platform on label transfer:

Same-Platform (10x → 10x):
  - Baseline performance without platform batch effect
  - Only biological/study-specific variation

Cross-Platform (Seq-Well → 10x):
  - Tests robustness to technical platform differences
  - Combines platform + biological variation

Platform Effect = Cross-Platform ARI - Same-Platform ARI
  - Negative = Platform hurts performance
  - ~0 = Model corrects platform differences
  - Positive = Unexpected (check data quality)

Use this to quantify how much platform matters vs other batch effects.
    """)

    print("="*80)


if __name__ == "__main__":
    main()
