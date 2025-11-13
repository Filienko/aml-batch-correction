#!/usr/bin/env python
"""
Experiment 1: Cross-Mechanism Batch Correction

Tests batch correction performance across different scRNA-seq technologies:
- Non-droplet technologies: Seq-Well, SORT-Seq, CITEseq, Muta-Seq
- Droplet-based technologies: 10x Genomics Chromium (top 3 largest studies)

This is a harder problem than within-mechanism correction because:
1. Different technologies have fundamentally different biases and characteristics
2. Gene detection rates vary significantly across platforms
3. Library prep differences create systematic variations
4. Technical variation can be as large as biological variation

Note: Study sizes vary considerably (2k to 80k cells per study), which may
affect batch correction performance.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Import evaluation modules
from run_evaluation import (
    detect_batch_key,
    detect_label_key,
    preprocess_adata_exact,
    prepare_uncorrected_embedding_exact,
    load_scvi_embedding,
    run_benchmark_exact,
    compute_scimilarity_corrected,
    force_cleanup,
    optimize_adata_memory,
    print_memory
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# Cross-mechanism studies: 4 non-droplet + 1 droplet baseline
# Manual selection for balanced comparison of diverse technologies
CROSS_MECHANISM_STUDIES = [
    'van_galen_2019',  # Seq-Well (microwell)
    'pei_2020',        # CITEseq (multimodal)
    'velten_2021',     # Muta-Seq (mutation tracking)
    'zhai_2022',       # SORT-Seq (FACS-based)
    'setty_2019'       # 10x Chromium (droplet baseline)
]

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "results_cross_mechanism"
SCIMILARITY_BATCH_SIZE = 1000
SCIMILARITY_ENABLED = True


def load_study_configuration():
    """
    Load study configuration from analysis results.
    If not available, use default configuration.
    """
    global CROSS_MECHANISM_STUDIES

    config_file = "study_analysis_results.csv"

    if os.path.exists(config_file):
        print(f"Loading study configuration from {config_file}")
        df = pd.read_csv(config_file)

        # Select one study from each category
        microwell = df[df['Category'] == 'microwell']
        wellbased = df[df['Category'] == 'well-based']
        droplet = df[df['Category'] == 'droplet'].sort_values('N_cells', ascending=False)

        studies = []
        if not microwell.empty:
            studies.append(microwell.iloc[0]['Actual_Study_Name'])
        if not wellbased.empty:
            studies.append(wellbased.iloc[0]['Actual_Study_Name'])
        if not droplet.empty:
            studies.append(droplet.iloc[0]['Actual_Study_Name'])  # Largest

        CROSS_MECHANISM_STUDIES = studies

        print(f"\n✓ Selected {len(studies)} studies:")
        for study in studies:
            print(f"  - {study}")

    else:
        print(f"⚠ Configuration file not found: {config_file}")
        print(f"  Please run analyze_studies.py first")
        print(f"\n  Or manually set CROSS_MECHANISM_STUDIES in this script")
        sys.exit(1)


def subset_to_cross_mechanism(adata, batch_key='Study'):
    """
    Subset dataset to cross-mechanism studies only.

    Args:
        adata: Full AnnData object
        batch_key: Column name for batch/study

    Returns:
        Subset AnnData with only the selected studies
    """
    print("\n" + "="*80)
    print("SUBSETTING TO CROSS-MECHANISM STUDIES")
    print("="*80)

    print(f"\nOriginal dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Studies in original: {adata.obs[batch_key].nunique()}")

    # Filter to selected studies
    mask = adata.obs[batch_key].isin(CROSS_MECHANISM_STUDIES)
    adata_subset = adata[mask].copy()

    print(f"\nSubset dataset: {adata_subset.n_obs:,} cells × {adata_subset.n_vars:,} genes")
    print(f"Studies in subset: {adata_subset.obs[batch_key].nunique()}")

    # Show breakdown by study with technology classification
    print("\nCells per study:")
    study_counts = adata_subset.obs[batch_key].value_counts()

    # Technology mapping for display
    tech_map = {
        'van_galen_2019': 'Non-droplet (Seq-Well)',
        'zhai_2022': 'Non-droplet (SORT-Seq)',
        'pei_2020': 'Non-droplet (CITEseq)',
        'velten_2021': 'Non-droplet (Muta-Seq)',
    }

    for study, count in study_counts.items():
        tech_category = tech_map.get(study, "Droplet (10x Chromium)")
        print(f"  {study}: {count:,} cells [{tech_category}]")

    # Remove unused categories
    adata_subset.obs[batch_key] = adata_subset.obs[batch_key].cat.remove_unused_categories()

    force_cleanup()

    return adata_subset


def main():
    """
    Main experiment pipeline for cross-mechanism batch correction.
    """
    print("="*80)
    print("EXPERIMENT 1: CROSS-MECHANISM BATCH CORRECTION")
    print("="*80)
    print("\nObjective: Test batch correction across different scRNA-seq technologies")
    print("  - Non-droplet technologies: Seq-Well, SORT-Seq, CITEseq, Muta-Seq")
    print("  - Droplet-based technologies: 10x Genomics Chromium (largest studies)")
    print("\nNote: This tests whether batch correction methods can handle")
    print("      fundamental technology differences, not just experimental batches.")

    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Data file not found: {DATA_PATH}")
        return

    # Studies are hardcoded at the top of this file
    print(f"\nUsing {len(CROSS_MECHANISM_STUDIES)} pre-selected studies:")
    for study in CROSS_MECHANISM_STUDIES:
        print(f"  - {study}")

    # STEP 1: Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    print(f"\nLoading: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print_memory()

    # Detect batch and label keys
    BATCH_KEY = detect_batch_key(adata)
    LABEL_KEY = detect_label_key(adata)
    BATCH_KEY_LOWER = BATCH_KEY.lower()

    if BATCH_KEY_LOWER not in adata.obs.columns:
        adata.obs[BATCH_KEY_LOWER] = adata.obs[BATCH_KEY].copy()

    # STEP 2: Subset to cross-mechanism studies
    adata = subset_to_cross_mechanism(adata, BATCH_KEY)

    # Optimize memory
    adata = optimize_adata_memory(adata)
    force_cleanup()

    # STEP 3: Preprocess
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING")
    print("="*80)

    adata = preprocess_adata_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # STEP 4: Uncorrected PCA
    print("\n" + "="*80)
    print("STEP 3: UNCORRECTED PCA")
    print("="*80)

    try:
        adata = prepare_uncorrected_embedding_exact(adata, BATCH_KEY_LOWER)
        force_cleanup()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # STEP 5: Load scVI embeddings (if available)
    print("\n" + "="*80)
    print("STEP 4: SCVI EMBEDDINGS (OPTIONAL)")
    print("="*80)

    if os.path.exists(SCVI_PATH):
        print(f"⚠ Note: scVI embeddings were computed on full dataset")
        print(f"  Attempting to subset to cross-mechanism studies...")

        try:
            adata_scvi_full = sc.read_h5ad(SCVI_PATH)

            # Find cells from our subset in the scVI file
            common_cells = adata.obs_names.intersection(adata_scvi_full.obs_names)

            if len(common_cells) > 0:
                print(f"  ✓ Found {len(common_cells):,} / {adata.n_obs:,} cells in scVI file")

                # Reorder to match our subset
                adata_scvi_subset = adata_scvi_full[adata.obs_names]
                adata.obsm['X_scVI'] = adata_scvi_subset.X.copy()

                print(f"  ✓ Added scVI embeddings: {adata.obsm['X_scVI'].shape}")

                del adata_scvi_full, adata_scvi_subset
            else:
                print(f"  ✗ No common cells found in scVI file")

            force_cleanup()

        except Exception as e:
            print(f"  ✗ Could not load scVI embeddings: {e}")
    else:
        print(f"  ℹ scVI file not found: {SCVI_PATH}")

    # STEP 6: SCimilarity
    if SCIMILARITY_ENABLED and os.path.exists(SCIMILARITY_MODEL):
        print("\n" + "="*80)
        print("STEP 5: SCIMILARITY")
        print("="*80)

        try:
            adata = compute_scimilarity_corrected(
                adata,
                model_path=SCIMILARITY_MODEL,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            force_cleanup()
        except Exception as e:
            print(f"✗ SCimilarity failed: {e}")
            import traceback
            traceback.print_exc()

    # STEP 7: Benchmarking
    print("\n" + "="*80)
    print("STEP 6: BENCHMARKING")
    print("="*80)

    results = {}

    # Uncorrected
    print("\n--- Uncorrected ---")
    df_unc = run_benchmark_exact(adata, 'X_uncorrected', OUTPUT_DIR, 'Uncorrected', N_JOBS)
    if df_unc is not None:
        results['Uncorrected'] = df_unc
    force_cleanup()

    # scVI
    if 'X_scVI' in adata.obsm:
        print("\n--- scVI ---")
        df_scvi = run_benchmark_exact(adata, 'X_scVI', OUTPUT_DIR, 'scVI', N_JOBS)
        if df_scvi is not None:
            results['scVI'] = df_scvi
        force_cleanup()

    # SCimilarity
    if 'X_scimilarity' in adata.obsm:
        print("\n--- SCimilarity ---")
        df_scim = run_benchmark_exact(adata, 'X_scimilarity', OUTPUT_DIR, 'SCimilarity', N_JOBS)
        if df_scim is not None:
            results['SCimilarity'] = df_scim
        force_cleanup()

    # STEP 8: Save results
    if len(results) > 0:
        print("\n" + "="*80)
        print("FINAL RESULTS - CROSS-MECHANISM")
        print("="*80)

        combined = pd.concat(results.values(), keys=results.keys())
        combined.index = combined.index.droplevel(1)

        # Add metadata
        combined['Experiment'] = 'Cross-Mechanism'
        combined['N_studies'] = len(CROSS_MECHANISM_STUDIES)
        combined['N_cells'] = adata.n_obs
        combined['Studies'] = ', '.join(CROSS_MECHANISM_STUDIES)

        output_file = os.path.join(OUTPUT_DIR, "cross_mechanism_results.csv")
        combined.to_csv(output_file)
        print(f"\n✓ Results saved: {output_file}")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        print(f"\n{'Method':<20s} {'Total':>10s} {'Batch':>10s} {'Bio':>10s}")
        print("-" * 52)

        for method in combined.index[:len(results)]:  # Only show methods, not duplicates
            total = combined.loc[method, 'Total']
            batch = combined.loc[method, 'Batch correction']
            bio = combined.loc[method, 'Bio conservation']
            print(f"{method:<20s} {total:>10.4f} {batch:>10.4f} {bio:>10.4f}")

        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nCross-mechanism batch correction is HARDER than within-mechanism because:")
        print("  • Different technologies have different gene detection rates")
        print("  • Library prep differences create systematic biases")
        print("  • Technical variation can be as large as biological variation")
        print("\nExpected results:")
        print("  • Lower batch correction scores than within-mechanism")
        print("  • Greater challenge for all methods")
        print("  • Some cell types may cluster by technology rather than biology")

    print("\n" + "="*80)
    print("✓ EXPERIMENT 1 COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nNext: Run experiment_within_mechanism.py to compare with within-mechanism correction")


if __name__ == "__main__":
    main()
