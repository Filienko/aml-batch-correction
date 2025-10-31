
#!/usr/bin/env python
"""
Extended Batch Correction Evaluation - MEMORY OPTIMIZED
For large datasets (>700k cells)

Key optimizations:
- Smaller batch size for SCimilarity (5000 -> 2000 cells)
- Aggressive garbage collection
- Process gene alignment in batches
- Reduce memory footprint throughout
"""

import os
import sys
from batch_correction_evaluation import (
    prepare_uncorrected_embedding,
    load_scvi_embedding,
    compute_scimilarity_embedding,
    run_scib_benchmark,
    compare_methods
)
import scanpy as sc
import gc
import pandas as pd

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR LARGE DATASETS
# ============================================================================

# Data paths
DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# Metadata keys
BATCH_KEY = "Sample"  # Use "sample" or "study" depending on your data
LABEL_KEY = "Cell Type"  # Cell type annotation column

# Analysis parameters
N_HVGS = 2000
N_JOBS = 16
OUTPUT_DIR = "batch_correction_extended"

SCIMILARITY_BATCH_SIZE = 1000



# Save options
SAVE_COMBINED_ADATA = False  # Set to False to save memory

# ============================================================================
# MEMORY OPTIMIZATION HELPERS
# ============================================================================

def print_memory_usage():
    """Print current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"  Memory: {mem_info.rss / 1024**3:.2f} GB")
    except ImportError:
        pass

def aggressive_cleanup():
    """Force garbage collection and clear caches"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("EXTENDED BATCH CORRECTION EVALUATION")
    print("="*80)

    print(f"\nMemory settings:")
    print(f"  SCimilarity batch size: {SCIMILARITY_BATCH_SIZE:,} cells")
    print(f"  Parallel jobs: {N_JOBS}")
    print_memory_usage()

    # Check files
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Error: Data file not found: {DATA_PATH}")
        return

    # Load data with memory optimization
    print(f"\nLoading data from: {DATA_PATH}")
    print("  (This may take a few minutes for large files...)")
    
    try:
        adata = sc.read_h5ad(DATA_PATH)
        print(f"✓ Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
        print_memory_usage()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Verify metadata
    batch_key = BATCH_KEY
    label_key = LABEL_KEY
    
    # Check for batch key variations
    if batch_key not in adata.obs.columns:
        if batch_key.lower() in adata.obs.columns:
            batch_key = batch_key.lower()
            print(f"  Using batch key: '{batch_key}' (lowercase)")
        else:
            print(f"\n❌ Error: Batch key '{BATCH_KEY}' not found!")
            print(f"Available columns: {adata.obs.columns.tolist()}")
            return
    
    # Check for label key variations
    if label_key not in adata.obs.columns:
        if label_key.lower() in adata.obs.columns:
            label_key = label_key.lower()
            print(f"  Using label key: '{label_key}' (lowercase)")
        elif "celltype" in adata.obs.columns:
            label_key = "celltype"
            print(f"  Using label key: '{label_key}'")
        elif "cell_type" in adata.obs.columns:
            label_key = "cell_type"
            print(f"  Using label key: '{label_key}'")
        else:
            print(f"\n❌ Error: Label key '{LABEL_KEY}' not found!")
            print(f"Available columns: {adata.obs.columns.tolist()}")
            return

    print(f"\n✓ Batch key: {batch_key} ({adata.obs[batch_key].nunique()} batches)")
    print(f"✓ Label key: {label_key} ({adata.obs[label_key].nunique()} types)")

    # Check for required data layers
    if 'counts' not in adata.layers:
        print("\n⚠ Warning: No 'counts' layer found.")
        print("  Attempting to use .X as counts...")
        adata.layers['counts'] = adata.X.copy()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine what to run
    EVAL_UNCORRECTED = True
    EVAL_SCVI = True
    EVAL_SCIMILARITY = True
    
    # Store results
    results = {}

    # ========================================================================
    # PREPARE EMBEDDINGS
    # ========================================================================
    print("\n" + "="*80)
    print("PREPARING EMBEDDINGS")
    print("="*80)

    # 1. Uncorrected (PCA)
    if EVAL_UNCORRECTED:
        print("\n1. Computing Uncorrected PCA...")
        print_memory_usage()
        try:
            adata = prepare_uncorrected_embedding(adata, n_hvgs=N_HVGS)
            aggressive_cleanup()
            print_memory_usage()
        except Exception as e:
            print(f"⚠ Warning: Uncorrected PCA failed: {e}")
            EVAL_UNCORRECTED = False

    # 2. scVI
    if EVAL_SCVI:
        print("\n2. Loading pre-computed scVI embeddings...")
        print_memory_usage()
        if os.path.exists(SCVI_PATH):
            try:
                adata = load_scvi_embedding(adata, scvi_path=SCVI_PATH)
                aggressive_cleanup()
                print_memory_usage()
            except Exception as e:
                print(f"⚠ Warning: Failed to load scVI embeddings: {e}")
                EVAL_SCVI = False
        else:
            print(f"⚠ Warning: scVI file not found: {SCVI_PATH}")
            EVAL_SCVI = False

    # 3. SCimilarity
    if EVAL_SCIMILARITY:
        print("\n3. Computing SCimilarity embeddings...")
        print("  This is the memory-intensive step.")
        print(f"  Processing in batches of {SCIMILARITY_BATCH_SIZE:,} cells")
        print_memory_usage()
        
        try:
            adata = compute_scimilarity_embedding(
                adata,
                model_path=SCIMILARITY_MODEL,
                use_full_gene_set=False,  # Use HVGs to reduce memory
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            aggressive_cleanup()
            print_memory_usage()
            print("✓ SCimilarity completed successfully!")
        except MemoryError as e:
            print(f"\n❌ Out of memory during SCimilarity computation!")
            print(f"\nSuggested fixes:")
            print(f"  1. Reduce SCIMILARITY_BATCH_SIZE to {SCIMILARITY_BATCH_SIZE // 2}")
            print(f"  2. Close other applications to free RAM")
            print(f"  3. Consider running on a machine with more memory")
            print(f"  4. Use a subset of your data for testing")
            return
        except Exception as e:
            print(f"⚠ Warning: SCimilarity failed: {e}")
            import traceback
            traceback.print_exc()
            EVAL_SCIMILARITY = False

    # ========================================================================
    # RUN BENCHMARKING
    # ========================================================================
    print("\n" + "="*80)
    print("RUNNING SCIB BENCHMARKING")
    print("="*80)
    # 1. Uncorrected
    if EVAL_UNCORRECTED:
        print("\n" + "-"*80)
        print("1. Evaluating: Uncorrected (PCA)")
        print("-"*80)
        print_memory_usage()
        try:
            results['Uncorrected'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_uncorrected',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
            aggressive_cleanup()
        except Exception as e:
            print(f"⚠ Warning: Uncorrected benchmarking failed: {e}")

    # 2. scVI
    if EVAL_SCVI:
        print("\n" + "-"*80)
        print("2. Evaluating: scVI")
        print("-"*80)
        print_memory_usage()
        try:
            results['scVI'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_scVI',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
            aggressive_cleanup()
        except Exception as e:
            print(f"⚠ Warning: scVI benchmarking failed: {e}")

    # 3. SCimilarity
    if EVAL_SCIMILARITY:
        print("\n" + "-"*80)
        print("3. Evaluating: SCimilarity")
        print("-"*80)
        print_memory_usage()
        try:
            results['SCimilarity'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_scimilarity',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
            aggressive_cleanup()
        except Exception as e:
            print(f"⚠ Warning: SCimilarity benchmarking failed: {e}")

    # ========================================================================
    # COMPARE METHODS
    # ========================================================================
    if len(results) > 1:
        print("\n" + "="*80)
        print("COMPARING ALL METHODS")
        print("="*80)

        try:
            combined = compare_methods(results, output_dir=OUTPUT_DIR)

            # Print final summary
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)

            print("\nMethod Performance (sorted by Total score):")
            print("-" * 80)
            for method in combined.sort_values('Total', ascending=False).index:
                total = combined.loc[method, 'Total']
                batch = combined.loc[method, 'Batch correction']
                bio = combined.loc[method, 'Bio conservation']
                print(f"  {method:20s}: Total={total:.3f}, Batch={batch:.3f}, Bio={bio:.3f}")

            # Winner
            winner = combined['Total'].idxmax()
            winner_score = combined.loc[winner, 'Total']

            print(f"\n🏆 Best method: {winner}")
            print(f"   Total score: {winner_score:.3f}")
            print(f"   Batch correction: {combined.loc[winner, 'Batch correction']:.3f}")
            print(f"   Bio conservation: {combined.loc[winner, 'Bio conservation']:.3f}")
        except Exception as e:
            print(f"⚠ Warning: Method comparison failed: {e}")
    elif len(results) == 0:
        print("\n⚠ Warning: No methods were successfully evaluated!")
    else:
        print(f"\n✓ Only one method evaluated: {list(results.keys())[0]}")

    # Save processed data (optional, disabled by default to save memory)
    if SAVE_COMBINED_ADATA and len(results) > 0:
        output_file = os.path.join(OUTPUT_DIR, "adata_with_all_embeddings.h5ad")
        try:
            print(f"\nSaving data with all embeddings...")
            adata.write_h5ad(output_file)
            print(f"✓ Saved data to: {output_file}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save combined adata: {e}")

    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")
    print("="*80)
    print_memory_usage()

    if len(results) > 1:
        print("\nKey files:")
        print(f"  - {OUTPUT_DIR}/combined_metrics.csv")


if __name__ == "__main__":
    main()

