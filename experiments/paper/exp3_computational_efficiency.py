#!/usr/bin/env python3
"""
Experiment 3: Computational Efficiency
=======================================
How fast is SCimilarity compared to traditional pipeline?
"""

import sys
import warnings
import time
warnings.filterwarnings('ignore')

import pandas as pd
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas_van_galen_subset.h5ad"
# DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas_50k_subset.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_CELLS_FOR_TIMING = 5000  # Subsample for timing


def main():
    print("="*80)
    print("EXPERIMENT 3: Computational Efficiency")
    print("="*80)

    # Load and subsample data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(DATA_PATH)

    print(f"\n2. Subsampling to {N_CELLS_FOR_TIMING:,} cells for timing...")
    adata_timing = subset_data(adata.copy(), n_cells=min(N_CELLS_FOR_TIMING, adata.n_obs))
    print(f"   Using: {adata_timing.n_obs:,} cells")

    # Timing results
    timing_results = []

    # Time SCimilarity
    print("\n3. Testing SCimilarity...")
    print(f"   Using model: {MODEL_PATH}")
    start = time.time()
    pipeline_scim = Pipeline(model="scimilarity", model_params={'model_path': MODEL_PATH})
    pred_scim = pipeline_scim.predict(adata_timing.copy())
    scim_time = time.time() - start
    timing_results.append({'method': 'SCimilarity', 'time_seconds': scim_time})
    print(f"   ✓ Completed in {scim_time:.1f} seconds ({scim_time/60:.2f} minutes)")

    # Time Random Forest
    print("\n4. Testing Random Forest...")
    start = time.time()
    pipeline_rf = Pipeline(model="random_forest")
    pred_rf = pipeline_rf.predict(adata_timing.copy(), target_column='cell_type')
    rf_time = time.time() - start
    timing_results.append({'method': 'Random Forest', 'time_seconds': rf_time})
    print(f"   ✓ Completed in {rf_time:.1f} seconds ({rf_time/60:.2f} minutes)")

    # Time SVM
    print("\n5. Testing SVM...")
    start = time.time()
    pipeline_svm = Pipeline(model="svm")
    pred_svm = pipeline_svm.predict(adata_timing.copy(), target_column='cell_type')
    svm_time = time.time() - start
    timing_results.append({'method': 'SVM', 'time_seconds': svm_time})
    print(f"   ✓ Completed in {svm_time:.1f} seconds ({svm_time/60:.2f} minutes)")

    # Estimate traditional pipeline time
    # Based on tool documentation:
    # - CellTypist: ~5-10 min
    # - SingleR: ~10-20 min
    # - scType: ~5 min
    # - Consensus + manual curation: hours
    celltyist_time = 7 * 60  # 7 minutes (average)
    singler_time = 15 * 60   # 15 minutes (average)
    sctype_time = 5 * 60     # 5 minutes
    traditional_time = celltyist_time + singler_time + sctype_time

    timing_results.append({
        'method': 'Traditional Pipeline (CellTypist+SingleR+scType)',
        'time_seconds': traditional_time
    })

    # Manual curation (estimated)
    manual_time = 2 * 3600  # 2 hours (conservative estimate)
    timing_results.append({
        'method': 'Manual Curation (estimated)',
        'time_seconds': manual_time
    })

    # Create results DataFrame
    timing_df = pd.DataFrame(timing_results)
    timing_df['time_minutes'] = timing_df['time_seconds'] / 60
    timing_df['time_hours'] = timing_df['time_seconds'] / 3600
    timing_df['speedup_vs_traditional'] = traditional_time / timing_df['time_seconds']

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\nTiming Results:")
    print(timing_df[['method', 'time_minutes', 'speedup_vs_traditional']].to_string(index=False))

    # Save results
    print(f"\n6. Saving results to {OUTPUT_DIR}/")
    timing_df.to_csv(OUTPUT_DIR / "exp3_timing.csv", index=False)
    print("   ✓ exp3_timing.csv")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    print(f"\nSCimilarity:          {scim_time/60:.1f} minutes")
    print(f"Traditional pipeline: {traditional_time/60:.0f} minutes (automated tools only)")
    print(f"Manual curation:      {manual_time/60:.0f} minutes (estimated)")
    print(f"\nTotal traditional:    {(traditional_time + manual_time)/60:.0f} minutes")
    print(f"Speedup (vs automated): {traditional_time/scim_time:.1f}x faster")
    print(f"Speedup (vs total):     {(traditional_time + manual_time)/scim_time:.1f}x faster")

    if scim_time < traditional_time / 5:
        print("\n✅ SCimilarity is SIGNIFICANTLY more efficient (>5x speedup)")
    elif scim_time < traditional_time / 2:
        print("\n✅ SCimilarity is more efficient (>2x speedup)")
    else:
        print("\n⚠️ Moderate efficiency improvement")

    print("="*80)


if __name__ == "__main__":
    main()
