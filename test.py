#!/usr/bin/env python
"""
Test Script for Memory-Optimized SCimilarity
Tests the approach on a small subset before running on full 748k dataset
"""

import os
import sys
import numpy as np
import scanpy as sc
from batch_correction_evaluation import compute_scimilarity_embedding
import gc

def test_scimilarity_memory_optimization(
    data_path="data/AML_scAtlas.h5ad",
    subset_size=10000,
    batch_sizes=[5000, 2000, 1000]
):
    """
    Test SCimilarity embedding computation with different batch sizes
    
    Args:
        data_path: Path to full dataset
        subset_size: Number of cells to test with
        batch_sizes: List of batch sizes to try
    """
    
    print("="*80)
    print("TESTING MEMORY-OPTIMIZED SCIMILARITY")
    print("="*80)
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Load full dataset
    print(f"\nLoading data from: {data_path}")
    adata_full = sc.read_h5ad(data_path)
    print(f"✓ Full dataset: {adata_full.n_obs:,} cells × {adata_full.n_vars:,} genes")
    
    # Create subset for testing
    print(f"\nCreating test subset of {subset_size:,} cells...")
    np.random.seed(42)
    subset_indices = np.random.choice(adata_full.n_obs, subset_size, replace=False)
    adata_subset = adata_full[subset_indices].copy()
    print(f"✓ Test subset: {adata_subset.n_obs:,} cells × {adata_subset.n_vars:,} genes")
    
    # Clean up full dataset from memory
    del adata_full
    gc.collect()
    
    # Ensure counts layer exists
    if 'counts' not in adata_subset.layers:
        print("  Adding counts layer from .X")
        adata_subset.layers['counts'] = adata_subset.X.copy()
    
    # Try different batch sizes
    results = {}
    
    for batch_size in batch_sizes:
        print("\n" + "-"*80)
        print(f"Testing with batch_size={batch_size}")
        print("-"*80)
        
        try:
            # Create a fresh copy for each test
            adata_test = adata_subset.copy()
            
            # Compute embeddings
            import time
            start = time.time()
            
            adata_test = compute_scimilarity_embedding(
                adata_test,
                model_path="models/model_v1.1",
                use_full_gene_set=False,  # Use HVGs to save memory
                batch_size=batch_size
            )
            
            elapsed = time.time() - start
            
            # Check result
            if 'X_scimilarity' in adata_test.obsm:
                embedding_shape = adata_test.obsm['X_scimilarity'].shape
                print(f"\n✓ SUCCESS with batch_size={batch_size}")
                print(f"  Embedding shape: {embedding_shape}")
                print(f"  Time: {elapsed:.1f} seconds")
                print(f"  Speed: {subset_size/elapsed:.0f} cells/sec")
                
                results[batch_size] = {
                    'success': True,
                    'time': elapsed,
                    'shape': embedding_shape,
                    'cells_per_sec': subset_size/elapsed
                }
            else:
                print(f"\n✗ FAILED: No embeddings created")
                results[batch_size] = {'success': False, 'error': 'No embeddings'}
            
            # Clean up
            del adata_test
            gc.collect()
            
        except MemoryError as e:
            print(f"\n✗ MEMORY ERROR with batch_size={batch_size}")
            print(f"  Error: {e}")
            results[batch_size] = {'success': False, 'error': 'OOM'}
            gc.collect()
            
        except Exception as e:
            print(f"\n✗ ERROR with batch_size={batch_size}")
            print(f"  Error: {e}")
            results[batch_size] = {'success': False, 'error': str(e)}
            gc.collect()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\nTest dataset: {subset_size:,} cells")
    print(f"Batch sizes tested: {batch_sizes}")
    print("\nResults:")
    print("-"*80)
    
    for batch_size, result in results.items():
        if result['success']:
            print(f"  {batch_size:5d} cells/batch: ✓ SUCCESS")
            print(f"         Time: {result['time']:.1f}s")
            print(f"         Speed: {result['cells_per_sec']:.0f} cells/sec")
        else:
            print(f"  {batch_size:5d} cells/batch: ✗ FAILED ({result['error']})")
    
    # Provide recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION FOR FULL DATASET")
    print("="*80)
    
    successful_sizes = [bs for bs, r in results.items() if r['success']]
    
    if successful_sizes:
        recommended = min(successful_sizes)  # Use smallest successful size for safety
        full_size = 748679  # Your full dataset size
        
        estimated_time = (full_size / subset_size) * results[recommended]['time']
        estimated_hours = estimated_time / 3600
        
        print(f"\n✓ Recommended batch size: {recommended}")
        print(f"  Based on successful test with {subset_size:,} cells")
        print(f"\nEstimated time for full dataset ({full_size:,} cells):")
        print(f"  ~{estimated_hours:.1f} hours")
        print(f"\nTo use this setting, edit run_evaluation_optimized.py:")
        print(f"  SCIMILARITY_BATCH_SIZE = {recommended}")
    else:
        print("\n❌ No batch sizes worked!")
        print("   Try:")
        print("   1. Test with smaller batch sizes (e.g., 500, 250)")
        print("   2. Close other applications to free memory")
        print("   3. Use a machine with more RAM")
    
    print()


def quick_memory_check():
    """Quick check of available system memory"""
    print("\n" + "="*80)
    print("SYSTEM MEMORY CHECK")
    print("="*80)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        
        print(f"\nTotal RAM: {mem.total / 1024**3:.1f} GB")
        print(f"Available: {mem.available / 1024**3:.1f} GB")
        print(f"Used: {mem.used / 1024**3:.1f} GB ({mem.percent}%)")
        
        if mem.available < 8 * 1024**3:  # Less than 8GB
            print("\n⚠ WARNING: Low available memory!")
            print("  Close other applications before running")
        else:
            print("\n✓ Sufficient memory available")
            
    except ImportError:
        print("\npsutil not installed. Install with: pip install psutil")
        print("(This is optional but helps monitor memory)")


if __name__ == "__main__":
    print("\nSCimilarity Memory Optimization Test")
    print("This will test different batch sizes on a subset of your data\n")
    
    # Check memory first
    quick_memory_check()
    
    # Run test
    print("\n" + "="*80)
    input("Press Enter to start test (Ctrl+C to cancel)...")
    
    test_scimilarity_memory_optimization(
        data_path="data/AML_scAtlas.h5ad",
        subset_size=10000,  # Test with 10k cells
        batch_sizes=[5000, 2000, 1000]  # Try these batch sizes
    )

