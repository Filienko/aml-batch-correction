#!/usr/bin/env python
"""
Experiment 2: Within-Mechanism Batch Correction

Tests batch correction performance within droplet-based scRNA-seq technologies.
All selected studies use 10x Genomics Chromium platform (or similar droplet-based).

Studies included:
- oetjen_2018: 10x Genomics Single Cell 3′
- beneyto-calabuig-2023: 10x Genomics Chromium Single Cell 3′
- jiang_2020: 10x Genomics Chromium Single Cell 3′
- zheng_2017: 10x Genomics GemCode Single-Cell 3′
- setty_2019: 10x Chromium
- petti_2019: 10x Genomics Chromium Single Cell 5′
- mumme_2023: 10x Genomics Chromium (3′ v3 and 5′ v1)
- zhang_2023: 10x Genomics Chromium

This is an easier problem than cross-mechanism because:
1. Same core technology (droplet-based encapsulation)
2. Similar gene detection characteristics
3. Batch effects are primarily due to biological/experimental factors, not technology
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

# These will be populated after analyzing the dataset
# Expected: all droplet-based (10x) studies
WITHIN_MECHANISM_STUDIES = None

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "results_within_mechanism"
SCIMILARITY_BATCH_SIZE = 1000
SCIMILARITY_ENABLED = True


def load_study_configuration():
    """
    Load study configuration from analysis results.
    If not available, use default configuration.
    """
    global WITHIN_MECHANISM_STUDIES

    config_file = "study_analysis_results.csv"

    if os.path.exists(config_file):
        print(f"Loading study configuration from {config_file}")
        df = pd.read_csv(config_file)

        # Select all droplet-based studies
        droplet = df[df['Category'] == 'droplet']

        if not droplet.empty:
            WITHIN_MECHANISM_STUDIES = droplet['Actual_Study_Name'].tolist()

            print(f"\n✓ Selected {len(WITHIN_MECHANISM_STUDIES)} droplet-based studies:")
            for study in WITHIN_MECHANISM_STUDIES:
                n_cells = droplet[droplet['Actual_Study_Name'] == study]['N_cells'].values[0]
                print(f"  - {study}: {n_cells:,} cells")

            total_cells = droplet['N_cells'].sum()
            print(f"\nTotal cells: {total_cells:,}")
        else:
            print("✗ No droplet-based studies found in configuration")
            sys.exit(1)

    else:
        print(f"⚠ Configuration file not found: {config_file}")
        print(f"  Please run analyze_studies.py first")
        print(f"\n  Or manually set WITHIN_MECHANISM_STUDIES in this script")
        sys.exit(1)


def subset_to_within_mechanism(adata, batch_key='Study'):
    """
    Subset dataset to within-mechanism (droplet-based) studies only.

    Args:
        adata: Full AnnData object
        batch_key: Column name for batch/study

    Returns:
        Subset AnnData with only droplet-based studies
    """
    print("\n" + "="*80)
    print("SUBSETTING TO WITHIN-MECHANISM STUDIES (DROPLET-BASED)")
    print("="*80)

    print(f"\nOriginal dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Studies in original: {adata.obs[batch_key].nunique()}")

    # Filter to selected studies
    mask = adata.obs[batch_key].isin(WITHIN_MECHANISM_STUDIES)
    adata_subset = adata[mask].copy()

    print(f"\nSubset dataset: {adata_subset.n_obs:,} cells × {adata_subset.n_vars:,} genes")
    print(f"Studies in subset: {adata_subset.obs[batch_key].nunique()}")

    # Show breakdown by study
    print("\nCells per study (all droplet-based):")
    study_counts = adata_subset.obs[batch_key].value_counts().sort_values(ascending=False)
    for study, count in study_counts.items():
        pct = 100 * count / adata_subset.n_obs
        print(f"  {study}: {count:,} cells ({pct:.1f}%)")

    # Remove unused categories
    adata_subset.obs[batch_key] = adata_subset.obs[batch_key].cat.remove_unused_categories()

    # Check balance
    min_cells = study_counts.min()
    max_cells = study_counts.max()
    ratio = max_cells / min_cells if min_cells > 0 else float('inf')

    print(f"\nBalance:")
    print(f"  Smallest study: {min_cells:,} cells")
    print(f"  Largest study: {max_cells:,} cells")
    print(f"  Ratio: {ratio:.1f}x")

    if ratio > 10:
        print(f"  ⚠ Studies are unbalanced (>10x difference)")
        print(f"    This may affect batch correction performance")

    force_cleanup()

    return adata_subset


def main():
    """
    Main experiment pipeline for within-mechanism batch correction.
    """
    print("="*80)
    print("EXPERIMENT 2: WITHIN-MECHANISM BATCH CORRECTION")
    print("="*80)
    print("\nObjective: Test batch correction within same technology (droplet-based)")
    print("  - All studies use 10x Genomics Chromium or similar droplet platform")
    print("  - Batch effects are from experimental/biological factors, not technology")

    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Data file not found: {DATA_PATH}")
        return

    # Load study configuration
    load_study_configuration()

    if WITHIN_MECHANISM_STUDIES is None or len(WITHIN_MECHANISM_STUDIES) == 0:
        print("\n✗ No studies configured for within-mechanism experiment")
        return

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

    # STEP 2: Subset to within-mechanism studies
    adata = subset_to_within_mechanism(adata, BATCH_KEY)

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
        print(f"  Attempting to subset to within-mechanism studies...")

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
        print("FINAL RESULTS - WITHIN-MECHANISM")
        print("="*80)

        combined = pd.concat(results.values(), keys=results.keys())
        combined.index = combined.index.droplevel(1)

        # Add metadata
        combined['Experiment'] = 'Within-Mechanism'
        combined['N_studies'] = len(WITHIN_MECHANISM_STUDIES)
        combined['N_cells'] = adata.n_obs
        combined['Studies'] = ', '.join(WITHIN_MECHANISM_STUDIES)

        output_file = os.path.join(OUTPUT_DIR, "within_mechanism_results.csv")
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
        print("\nWithin-mechanism batch correction is EASIER than cross-mechanism because:")
        print("  • Same core technology (droplet encapsulation)")
        print("  • Similar gene detection rates across studies")
        print("  • Batch effects are primarily experimental, not technological")
        print("\nExpected results:")
        print("  • Higher batch correction scores than cross-mechanism")
        print("  • Better biological signal preservation")
        print("  • More consistent cell type clustering across batches")

    print("\n" + "="*80)
    print("✓ EXPERIMENT 2 COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
