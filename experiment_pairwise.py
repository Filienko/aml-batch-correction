#!/usr/bin/env python
"""
Pairwise Batch Correction Comparison

Tests batch correction on pairs of studies to match SCimilarity paper methodology.
The original SCimilarity paper compared "two kidney datasets, two PBMC datasets,
two lung datasets" - i.e., pairwise comparisons.

This is a fairer test because:
1. Batch correction metrics work better with 2 batches
2. Clearer interpretation (biology vs batch)
3. SCimilarity designed for pairwise integration
4. Less advantage for multi-batch optimizers like Harmony
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
import run_evaluation
from run_evaluation import (
    detect_batch_key,
    detect_label_key,
    preprocess_adata_exact,
    prepare_uncorrected_embedding_exact,
    load_scvi_embedding,
    run_benchmark_exact,
    compute_scimilarity_corrected,
    compute_harmony_corrected,
    force_cleanup,
    optimize_adata_memory,
    print_memory
)

# ============================================================================
# CONFIGURATION - EDIT THESE TWO STUDIES
# ============================================================================

DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# ============== CONFIGURE YOUR TWO STUDIES HERE ==============
STUDY_A = 'van_galen_2019'  # Seq-Well
STUDY_B = 'setty_2019'       # 10x Chromium

# Or try other pairs:
# STUDY_A = 'oetjen_2018'           # 10x
# STUDY_B = 'beneyto-calabuig-2023' # 10x
# ============================================================

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = f"results_pairwise_{STUDY_A}_vs_{STUDY_B}"
SCIMILARITY_BATCH_SIZE = 1000
SCIMILARITY_ENABLED = True


def subset_to_two_studies(adata, study_a, study_b, batch_key='Study'):
    """
    Subset dataset to exactly two studies.

    Args:
        adata: Full AnnData object
        study_a: First study name
        study_b: Second study name
        batch_key: Column name for batch/study

    Returns:
        Subset AnnData with only two studies
    """
    print("\n" + "="*80)
    print("SUBSETTING TO TWO STUDIES (PAIRWISE COMPARISON)")
    print("="*80)

    print(f"\nOriginal dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Studies in original: {adata.obs[batch_key].nunique()}")

    # Filter to two studies
    mask = adata.obs[batch_key].isin([study_a, study_b])
    adata_subset = adata[mask].copy()

    print(f"\nSubset dataset: {adata_subset.n_obs:,} cells × {adata_subset.n_vars:,} genes")
    print(f"Studies in subset: {adata_subset.obs[batch_key].nunique()}")

    # Show breakdown by study
    print("\nCells per study:")
    study_counts = adata_subset.obs[batch_key].value_counts()
    for study, count in study_counts.items():
        pct = 100 * count / adata_subset.n_obs
        print(f"  {study}: {count:,} cells ({pct:.1f}%)")

    # Check balance
    counts = study_counts.values
    ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

    print(f"\nBalance ratio: {ratio:.1f}x")
    if ratio > 5:
        print(f"  ⚠ Studies are unbalanced (>5x difference)")
        print(f"    This may affect batch correction performance")

    # Remove unused categories
    adata_subset.obs[batch_key] = adata_subset.obs[batch_key].cat.remove_unused_categories()

    force_cleanup()

    return adata_subset


def main():
    """
    Main experiment pipeline for pairwise batch correction.
    """
    print("="*80)
    print("PAIRWISE BATCH CORRECTION COMPARISON")
    print("="*80)
    print(f"\nComparing: {STUDY_A} vs {STUDY_B}")
    print("\nObjective: Test batch correction on two studies")
    print("  - Simpler than multi-batch (8 studies)")
    print("  - Matches SCimilarity paper methodology")
    print("  - Clearer interpretation of batch vs biology")

    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Data file not found: {DATA_PATH}")
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
    run_evaluation.BATCH_KEY = detect_batch_key(adata)
    run_evaluation.LABEL_KEY = detect_label_key(adata)
    run_evaluation.BATCH_KEY_LOWER = run_evaluation.BATCH_KEY.lower()

    BATCH_KEY = run_evaluation.BATCH_KEY
    LABEL_KEY = run_evaluation.LABEL_KEY
    BATCH_KEY_LOWER = run_evaluation.BATCH_KEY_LOWER

    if BATCH_KEY_LOWER not in adata.obs.columns:
        adata.obs[BATCH_KEY_LOWER] = adata.obs[BATCH_KEY].copy()

    # STEP 2: Subset to two studies
    adata = subset_to_two_studies(adata, STUDY_A, STUDY_B, BATCH_KEY)

    # STEP 2.5: Load scVI (before preprocessing)
    print("\n" + "="*80)
    print("STEP 2.5: SCVI EMBEDDINGS (BEFORE PREPROCESSING)")
    print("="*80)

    if os.path.exists(SCVI_PATH):
        try:
            print(f"  Loading scVI file: {SCVI_PATH}")
            adata_scvi = sc.read_h5ad(SCVI_PATH)
            print(f"  scVI: {adata_scvi.n_obs:,} cells × {adata_scvi.n_vars} features")
            print(f"  Main: {adata.n_obs:,} cells (after study subsetting)")

            scvi_has_numeric = all(str(idx).isdigit() for idx in adata_scvi.obs_names[:100])

            if scvi_has_numeric and adata_scvi.n_obs >= adata.n_obs:
                print(f"  ✓ scVI uses numeric indices - using row positions")

                # Load full data to get original indices
                print(f"  Loading original data to find cell positions...")
                adata_full_temp = sc.read_h5ad(DATA_PATH)

                mask = adata_full_temp.obs[BATCH_KEY].isin([STUDY_A, STUDY_B])
                original_indices = np.where(mask)[0]

                print(f"  ✓ Found {len(original_indices):,} matching cells")

                del adata_full_temp
                force_cleanup()

                if len(original_indices) == adata.n_obs:
                    adata.obsm['X_scVI'] = adata_scvi.X[original_indices].copy()
                    print(f"  ✓ Added scVI embeddings: {adata.obsm['X_scVI'].shape}")
            else:
                print(f"  ⚠ Using standard loading...")
                adata = load_scvi_embedding(adata, SCVI_PATH)

            del adata_scvi
            force_cleanup()

        except Exception as e:
            print(f"  ✗ Error loading scVI: {e}")
    else:
        print(f"  ℹ scVI file not found: {SCVI_PATH}")

    # Optimize memory
    adata = optimize_adata_memory(adata)
    force_cleanup()

    # STEP 3: Preprocess
    print("\n" + "="*80)
    print("STEP 3: PREPROCESSING")
    print("="*80)

    adata = preprocess_adata_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # STEP 4: Uncorrected PCA
    print("\n" + "="*80)
    print("STEP 4: UNCORRECTED PCA")
    print("="*80)

    try:
        adata = prepare_uncorrected_embedding_exact(adata, BATCH_KEY_LOWER)
        force_cleanup()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # STEP 5: SCimilarity
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

    # STEP 6: Harmony
    print("\n" + "="*80)
    print("STEP 6: HARMONY")
    print("="*80)

    try:
        adata = compute_harmony_corrected(adata, BATCH_KEY, N_JOBS)
        force_cleanup()
    except Exception as e:
        print(f"✗ Harmony failed: {e}")
        import traceback
        traceback.print_exc()

    # STEP 7: Benchmarking
    print("\n" + "="*80)
    print("STEP 7: BENCHMARKING")
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

    # Harmony
    if 'X_harmony' in adata.obsm:
        print("\n--- Harmony ---")
        df_harmony = run_benchmark_exact(adata, 'X_harmony', OUTPUT_DIR, 'Harmony', N_JOBS)
        if df_harmony is not None:
            results['Harmony'] = df_harmony
        force_cleanup()

    # STEP 8: Save results
    if len(results) > 0:
        print("\n" + "="*80)
        print("FINAL RESULTS - PAIRWISE COMPARISON")
        print("="*80)

        combined = pd.concat(results.values(), keys=results.keys())
        combined.index = combined.index.droplevel(1)

        # Add metadata
        combined['Experiment'] = 'Pairwise'
        combined['Study_A'] = STUDY_A
        combined['Study_B'] = STUDY_B
        combined['N_cells'] = adata.n_obs

        output_file = os.path.join(OUTPUT_DIR, f"pairwise_{STUDY_A}_vs_{STUDY_B}.csv")
        combined.to_csv(output_file)
        print(f"\n✓ Results saved: {output_file}")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        print(f"\nComparison: {STUDY_A} vs {STUDY_B}")
        print(f"Cells: {adata.n_obs:,}")

        print(f"\n{'Method':<20s} {'Total':>10s} {'Batch':>10s} {'Bio':>10s}")
        print("-" * 52)

        for method in combined.index[:len(results)]:
            total = combined.loc[method, 'Total']
            batch = combined.loc[method, 'Batch correction']
            bio = combined.loc[method, 'Bio conservation']
            print(f"{method:<20s} {total:>10.4f} {batch:>10.4f} {bio:>10.4f}")

        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nPairwise comparison advantages:")
        print("  • Simpler task - only 2 batches to mix")
        print("  • Clearer interpretation - biology vs batch")
        print("  • Matches SCimilarity paper methodology")
        print("  • Less advantage for multi-batch optimizers")
        print("\nExpected results:")
        print("  • SCimilarity should be more competitive")
        print("  • Batch correction scores should be higher overall")
        print("  • Easier to verify if cells cluster by biology")

    print("\n" + "="*80)
    print("✓ PAIRWISE EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"\nTo test another pair, edit STUDY_A and STUDY_B at top of script")


if __name__ == "__main__":
    main()
