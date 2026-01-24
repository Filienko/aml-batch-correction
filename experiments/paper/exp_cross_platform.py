#!/usr/bin/env python3
"""
Experiment: Cross-Platform Robustness
======================================

Research Question:
Can SCimilarity handle batch effects from different experimental platforms?

Setup:
- Reference: van_galen_2019 (Seq-Well)
- Query studies with SAME platform: N/A (van_galen is only Seq-Well)
- Query studies with DIFFERENT platform:
  * beneyto-calabuig-2023 (10x Genomics)
  * jiang_2020 (10x Genomics)
  * zhang_2023 (10x Genomics)

This tests SCimilarity's ability to correct platform-specific batch effects.
"""

import sys
import warnings
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
    'zhang_2023': '10x Genomics',
    'velten_2021': '10x Genomics',
    'zhai_2022': '10x Genomics',
}

REFERENCE_STUDY = 'van_galen_2019'
REFERENCE_PLATFORM = PLATFORM_INFO[REFERENCE_STUDY]

# Query studies (all different platform from reference)
CROSS_PLATFORM_QUERIES = [
    'beneyto-calabuig-2023',
    'jiang_2020',
    'zhang_2023',
]

MODELS_TO_TEST = {
    'SCimilarity+KNN': ('scimilarity', {'classifier': 'knn'}),
    'SCimilarity+RF': ('scimilarity', {'classifier': 'random_forest'}),
    'Random Forest': ('random_forest', {}),
    'KNN': ('knn', {}),
}


def main():
    print("="*80)
    print("EXPERIMENT: Cross-Platform Robustness")
    print("="*80)
    print("\nResearch Question:")
    print("  Can SCimilarity handle batch effects from different sequencing platforms?")
    print("\nSetup:")
    print(f"  Reference: {REFERENCE_STUDY} ({REFERENCE_PLATFORM})")
    print(f"  Query studies (all using different platform):")
    for study in CROSS_PLATFORM_QUERIES:
        print(f"    • {study} ({PLATFORM_INFO.get(study, 'Unknown')})")

    # Load data
    print("\n" + "="*80)
    print("1. Loading data...")
    print("="*80)
    adata = sc.read_h5ad(DATA_PATH)

    # Detect columns
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)
    print(f"   Using study column: '{study_col}'")
    print(f"   Using cell type column: '{cell_type_col}'")

    # Check reference exists
    available_studies = adata.obs[study_col].unique()

    if REFERENCE_STUDY not in available_studies:
        print(f"   ERROR: Reference study '{REFERENCE_STUDY}' not found!")
        return

    # Get valid query studies
    valid_queries = [s for s in CROSS_PLATFORM_QUERIES if s in available_studies]

    if not valid_queries:
        print("   ERROR: No query studies found!")
        return

    print(f"\n   Valid query studies: {len(valid_queries)}")
    for study in valid_queries:
        platform = PLATFORM_INFO.get(study, 'Unknown')
        count = (adata.obs[study_col] == study).sum()
        print(f"     • {study}: {count:,} cells ({platform})")

    # Prepare reference data
    print("\n" + "="*80)
    print("2. Preparing reference data...")
    print("="*80)
    adata_ref = subset_data(adata, studies=[REFERENCE_STUDY])
    print(f"   Reference ({REFERENCE_STUDY}):")
    print(f"     Platform: {REFERENCE_PLATFORM}")
    print(f"     Cells: {adata_ref.n_obs:,}")
    print(f"     Cell types: {adata_ref.obs[cell_type_col].nunique()}")

    # Results storage
    results = []

    # Test on each query study
    print("\n" + "="*80)
    print("3. Cross-Platform Label Transfer")
    print("="*80)

    for query_study in valid_queries:
        query_platform = PLATFORM_INFO.get(query_study, 'Unknown')
        platform_match = "SAME" if query_platform == REFERENCE_PLATFORM else "DIFFERENT"

        print(f"\n{'='*80}")
        print(f"Query: {query_study}")
        print(f"  Reference platform: {REFERENCE_PLATFORM}")
        print(f"  Query platform:     {query_platform}")
        print(f"  Platform match:     {platform_match}")
        print('='*80)

        adata_query = subset_data(adata, studies=[query_study])
        print(f"  Query cells: {adata_query.n_obs:,}")
        print(f"  Cell types:  {adata_query.obs[cell_type_col].nunique()}")

        # Test each model
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
                    'model': model_name,
                    'query_study': query_study,
                    'reference_platform': REFERENCE_PLATFORM,
                    'query_platform': query_platform,
                    'platform_match': platform_match,
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                    'n_cells': adata_query.n_obs
                })

                print(f"✓ ARI: {metrics['ari']:.3f}, Acc: {metrics['accuracy']:.3f}")

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'model': model_name,
                    'query_study': query_study,
                    'reference_platform': REFERENCE_PLATFORM,
                    'query_platform': query_platform,
                    'platform_match': platform_match,
                    'accuracy': 0,
                    'ari': 0,
                    'nmi': 0,
                    'n_cells': adata_query.n_obs
                })

    # Summary
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS: All Cross-Platform Transfers")
    print("="*80)
    print(results_df.to_string(index=False))

    # Group by platform match
    print("\n" + "="*80)
    print("Average Performance by Platform Match")
    print("="*80)

    if 'DIFFERENT' in results_df['platform_match'].values:
        diff_platform = results_df[results_df['platform_match'] == 'DIFFERENT']
        print("\nCross-Platform (Reference: Seq-Well → Query: 10x Genomics):")
        avg_diff = diff_platform.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
        avg_diff = avg_diff.sort_values('ari', ascending=False)
        print(avg_diff.to_string())

    if 'SAME' in results_df['platform_match'].values:
        same_platform = results_df[results_df['platform_match'] == 'SAME']
        print("\nSame Platform:")
        avg_same = same_platform.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
        avg_same = avg_same.sort_values('ari', ascending=False)
        print(avg_same.to_string())

    # Overall average by model
    print("\n" + "="*80)
    print("Overall Average Performance by Model")
    print("="*80)
    avg_by_model = results_df.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
    avg_by_model = avg_by_model.sort_values('ari', ascending=False)
    print(avg_by_model.to_string())

    # Save results
    print(f"\n" + "="*80)
    print("4. Saving results...")
    print("="*80)
    results_df.to_csv(OUTPUT_DIR / "exp_cross_platform.csv", index=False)
    avg_by_model.to_csv(OUTPUT_DIR / "exp_cross_platform_summary.csv")
    print(f"   ✓ {OUTPUT_DIR}/exp_cross_platform.csv")
    print(f"   ✓ {OUTPUT_DIR}/exp_cross_platform_summary.csv")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    best_model = avg_by_model.index[0]
    best_ari = avg_by_model.loc[best_model, 'ari']

    print(f"\n Best Model: {best_model}")
    print(f"   Average ARI (cross-platform): {best_ari:.4f}")

    # Check if SCimilarity models are in results
    scim_models = [m for m in avg_by_model.index if 'SCimilarity' in m]
    if scim_models:
        scim_ari = avg_by_model.loc[scim_models[0], 'ari']

        # Compare to traditional
        trad_models = [m for m in avg_by_model.index if 'SCimilarity' not in m]
        if trad_models:
            best_trad = trad_models[0]
            trad_ari = avg_by_model.loc[best_trad, 'ari']

            improvement = ((scim_ari - trad_ari) / trad_ari) * 100

            print(f"\nSCimilarity vs Traditional ML (cross-platform):")
            print(f"  SCimilarity: ARI = {scim_ari:.4f}")
            print(f"  Best Traditional ({best_trad}): ARI = {trad_ari:.4f}")


if __name__ == "__main__":
    main()
