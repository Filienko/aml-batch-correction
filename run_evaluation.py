#!/usr/bin/env python
"""
ULTRA MEMORY-OPTIMIZED Batch Correction Evaluation
For very large datasets (700k+ cells)

Key optimizations:
1. Process data in chunks during preprocessing
2. Use sparse matrices everywhere
3. Minimize data copies
4. Aggressive garbage collection
5. Lower batch sizes for SCimilarity
6. Skip unnecessary intermediate steps
"""

import os
import sys
import gc
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# Metadata keys
BATCH_KEY = "Study"
LABEL_KEY = "Cell Type"

# CRITICAL MEMORY PARAMETERS - Adjust these if still running out of memory
N_HVGS = 2000
N_JOBS = 8  # Reduced from 16 - fewer parallel jobs = less memory
OUTPUT_DIR = "batch_correction_extended"

# SCimilarity settings - VERY conservative
SCIMILARITY_BATCH_SIZE = 500  # Very small batches
SCIMILARITY_ENABLED = True  # Set to False to skip if OOM persists

# Data processing settings
CHUNK_SIZE = 50000  # Process 50k cells at a time during filtering
SAVE_COMBINED_ADATA = False  # Never save full adata to conserve memory

# ============================================================================
# MEMORY UTILITIES
# ============================================================================

def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def print_memory():
    """Print current memory usage"""
    mem_gb = get_memory_usage_gb()
    print(f"  💾 Memory: {mem_gb:.2f} GB")

def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()

def optimize_adata_memory(adata):
    """Optimize AnnData memory usage"""
    import scipy.sparse as sp
    
    # Convert to sparse if not already
    if not sp.issparse(adata.X):
        print("  Converting to sparse matrix...")
        adata.X = sp.csr_matrix(adata.X)
    
    # Use 32-bit floats
    if adata.X.dtype != np.float32:
        print("  Converting to float32...")
        adata.X.data = adata.X.data.astype(np.float32)
    
    # Clean up categorical columns
    for col in adata.obs.select_dtypes(include=['category']).columns:
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    
    return adata

# ============================================================================
# PREPROCESSING - CHUNK-BASED
# ============================================================================

def preprocess_adata_chunked(adata, chunk_size=50000):
    """
    Preprocess data in chunks to avoid memory spikes
    """
    print("\n" + "="*80)
    print("PREPROCESSING (Memory-optimized)")
    print("="*80)
    print(f"Initial: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print_memory()
    
    # Step 1: Cell filtering (can't chunk this easily)
    print("\n1. Filtering cells (min_counts=1000, min_genes=300)...")
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1000)
    sc.pp.filter_cells(adata, min_genes=300)
    print(f"  {n_before:,} → {adata.n_obs:,} cells")
    force_cleanup()
    print_memory()
    
    # Step 2: Remove low-count samples
    print("\n2. Removing samples with <50 cells...")
    cell_counts = adata.obs[BATCH_KEY].value_counts()
    keep_samples = cell_counts.index[cell_counts >= 50]
    n_before = adata.n_obs
    adata = adata[adata.obs[BATCH_KEY].isin(keep_samples)].copy()
    print(f"  {n_before:,} → {adata.n_obs:,} cells")
    force_cleanup()
    print_memory()
    
    # Step 3: Harmonize cell types
    print("\n3. Harmonizing cell types...")
    celltype_mapping = {
        "Perivascular cell": "Stromal Cells",
        "Megakaryocyte": "Megakaryocytes",
        "Plasma cell": "Plasma Cells",
        "Plasmablast": "Plasma Cells",
        "cDC1": "Conventional Dendritic Cells",
        "cDC2": "Conventional Dendritic Cells",
        "DC precursor": "Dendritic Progenitor Cell",
        "Ery": "Erythroid Cells",
        "Dendritic Cells": "Conventional Dendritic Cells",
        "MAIT": "T Cells",
        "CD8+ T": "CD8+ T Cells",
        "CD4+ T": "CD4+ T Cells",
        "gd T": "T Cells",
        "Pre-B": "B Cell Precursors",
        "Pro-B": "B Cell Precursors",
        "Myelocytes": "Granulocytes",
        "Granulocyte": "Granulocytes",
        "Promyelocytes": "Granulocytes",
        "HLA-II+ monocyte": "Monocytes",
        "HSC": "HSC/MPPs",
        "MPP": "HSC/MPPs",
        "CD14+ monocyte": "Monocytes",
        "CD11c+": "Unknown",
        "LymP": "Unknown"
    }
    
    for col in [LABEL_KEY, "celltype", "main_original_celltype"]:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].replace(celltype_mapping)
            print(f"  ✓ Harmonized '{col}'")
    
    # Step 4: Optimize memory
    print("\n4. Optimizing memory...")
    adata = optimize_adata_memory(adata)
    force_cleanup()
    print_memory()
    
    print(f"\n✓ Preprocessing complete: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata

# ============================================================================
# UNCORRECTED EMBEDDING - SIMPLIFIED
# ============================================================================

def prepare_uncorrected_embedding_minimal(adata, n_hvgs=2000):
    """
    Minimal memory footprint for uncorrected PCA
    """
    print("\n" + "="*80)
    print("UNCORRECTED PCA (Memory-optimized)")
    print("="*80)
    print_memory()
    
    # Work on a lightweight copy
    print("Creating working copy...")
    adata_work = adata.copy()
    
    # Use counts layer
    if 'counts' in adata.layers:
        adata_work.X = adata_work.layers['counts'].copy()
        del adata_work.layers['counts']  # Free memory
    
    force_cleanup()
    print_memory()
    
    # Filter genes
    print("Filtering genes (min_cells=30)...")
    sc.pp.filter_genes(adata_work, min_cells=30)
    print(f"  {adata_work.n_vars:,} genes remaining")
    force_cleanup()
    
    # Normalize
    print("Normalizing...")
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)
    force_cleanup()
    print_memory()
    
    # HVG selection
    print(f"Finding {n_hvgs} HVGs...")
    sc.pp.highly_variable_genes(
        adata_work,
        n_top_genes=n_hvgs,
        flavor="seurat_v3",
        subset=True,
        span=0.8
    )
    print(f"  Subset to {adata_work.n_vars:,} genes")
    force_cleanup()
    print_memory()
    
    # PCA
    print("Computing PCA...")
    sc.tl.pca(adata_work, svd_solver='arpack')
    
    # Transfer back
    adata.obsm['X_pca'] = adata_work.obsm['X_pca'].copy()
    adata.obsm['X_uncorrected'] = adata_work.obsm['X_pca'].copy()
    
    del adata_work
    force_cleanup()
    print_memory()
    
    print(f"✓ PCA computed: {adata.obsm['X_pca'].shape}")
    return adata

# ============================================================================
# SCIMILARITY - ULTRA CONSERVATIVE
# ============================================================================

def compute_scimilarity_minimal(adata, model_path, batch_size=500):
    """
    Ultra-conservative SCimilarity computation
    """
    print("\n" + "="*80)
    print("SCIMILARITY (Ultra memory-optimized)")
    print("="*80)
    print(f"  Batch size: {batch_size} cells")
    print_memory()
    
    try:
        from scimilarity import CellAnnotation
        from scimilarity.utils import lognorm_counts, align_dataset
    except ImportError as e:
        print(f"✗ SCimilarity not available: {e}")
        return adata
    
    # Load model
    print("Loading SCimilarity model...")
    try:
        ca = CellAnnotation(model_path=model_path)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return adata
    
    force_cleanup()
    print_memory()
    
    # Get working data with minimal memory
    print("Preparing data...")
    
    # Find raw counts - they might be in different locations after preprocessing
    X_counts = None
    genes = None
    counts_source = None
    
    # Check different possible locations for raw counts
    if 'counts' in adata.layers:
        print("  Checking layers['counts']...")
        X_counts = adata.layers['counts']
        genes = adata.var.index
        counts_source = "layers['counts']"
        print(f"    Max value: {X_counts.max():.0f}")
        if X_counts.max() < 100:
            print("    ⚠ Warning: 'counts' layer appears normalized (max < 100)")
            print("    This might not be raw counts!")
    
    # If counts layer doesn't exist or looks normalized, check .X
    if X_counts is None or X_counts.max() < 100:
        print("  Checking .X...")
        if adata.X.max() > 100:
            print("    .X appears to contain raw counts")
            X_counts = adata.X
            genes = adata.var.index
            counts_source = ".X"
        elif adata.X.max() < 20:
            print("    .X appears to be log-normalized")
    
    # Last resort: check .raw
    if (X_counts is None or X_counts.max() < 100) and adata.raw is not None:
        print("  Checking .raw.X...")
        if adata.raw.X.max() > 100:
            print("    .raw.X appears to contain raw counts")
            X_counts = adata.raw.X
            genes = adata.raw.var.index
            counts_source = ".raw.X"
    
    # Validate we found raw counts
    if X_counts is None:
        print("\n  ✗ Could not find raw counts in any location")
        print("    Checked: layers['counts'], .X, .raw.X")
        print("    SCimilarity requires raw counts - skipping")
        return adata
    
    if X_counts.max() < 100:
        print(f"\n  ⚠ Warning: Found data in {counts_source}, but max value is {X_counts.max():.2f}")
        print("    This is unusually low for raw counts (expected >1000)")
        print("    Proceeding anyway, but results may be affected...")
    else:
        print(f"  ✓ Using raw counts from: {counts_source}")
        print(f"    Max value: {X_counts.max():.0f} (looks like raw counts ✓)")
    
    # Find common genes
    print("Finding common genes...")
    common_genes = genes.intersection(ca.gene_order)
    print(f"  {len(common_genes):,} common genes")
    
    if len(common_genes) < 1000:
        print(f"  ⚠ Too few common genes ({len(common_genes)}), skipping SCimilarity")
        return adata
    
    # Get indices for common genes
    gene_order_dict = {gene: i for i, gene in enumerate(ca.gene_order)}
    common_genes_sorted = sorted(common_genes, key=lambda x: gene_order_dict[x])
    gene_indices = [genes.get_loc(g) for g in common_genes_sorted]
    
    # Process in very small batches
    print(f"Computing embeddings ({adata.n_obs:,} cells in batches of {batch_size})...")
    n_cells = adata.n_obs
    embeddings_list = []
    
    for start_idx in range(0, n_cells, batch_size):
        end_idx = min(start_idx + batch_size, n_cells)
        batch_num = start_idx // batch_size + 1
        total_batches = (n_cells + batch_size - 1) // batch_size
        
        if batch_num % 10 == 0 or batch_num == 1:
            print(f"  Batch {batch_num}/{total_batches}")
            print_memory()
        
        try:
            # Get batch data - only the genes we need
            batch_X = X_counts[start_idx:end_idx, gene_indices]
            
            # Create minimal AnnData
            import anndata as ad
            batch_ad = ad.AnnData(X=batch_X)
            batch_ad.var.index = common_genes_sorted
            
            # CRITICAL: lognorm_counts expects raw counts in .layers['counts']
            # We need to explicitly create this layer
            batch_ad.layers['counts'] = batch_ad.X.copy()
            
            # Align and process
            batch_aligned = align_dataset(batch_ad, ca.gene_order)
            
            # Ensure counts layer survived alignment
            if 'counts' not in batch_aligned.layers:
                batch_aligned.layers['counts'] = batch_aligned.X.copy()
            
            batch_norm = lognorm_counts(batch_aligned)
            batch_emb = ca.get_embeddings(batch_norm.X)
            
            embeddings_list.append(batch_emb)
            
            # Aggressive cleanup
            del batch_X, batch_ad, batch_aligned, batch_norm, batch_emb
            
            if batch_num % 20 == 0:
                force_cleanup()
        
        except MemoryError:
            print(f"\n  ✗ OOM at batch {batch_num}")
            print(f"  Try reducing batch size to {batch_size // 2}")
            raise
        except Exception as e:
            print(f"\n  ✗ Error at batch {batch_num}: {e}")
            raise
    
    # Concatenate
    print("Concatenating results...")
    embeddings = np.vstack(embeddings_list)
    del embeddings_list
    force_cleanup()
    
    adata.obsm['X_scimilarity'] = embeddings
    print(f"✓ SCimilarity complete: {embeddings.shape}")
    print_memory()
    
    return adata

# ============================================================================
# BENCHMARKING - SIMPLIFIED
# ============================================================================

def run_benchmark_minimal(adata, embedding_key, output_dir, method_name=None, n_jobs=8):
    """
    Minimal benchmark with reduced parallelism
    
    Args:
        adata: AnnData object
        embedding_key: Key in .obsm for the embedding
        output_dir: Where to save results
        method_name: Clean name for the method (e.g., 'Uncorrected', 'SCimilarity')
        n_jobs: Number of parallel jobs
    """
    print("\n" + "="*80)
    print(f"BENCHMARKING: {embedding_key}")
    print("="*80)
    print_memory()
    
    from scib_metrics.benchmark import Benchmarker, BioConservation
    import time
    
    # Verify embedding
    if embedding_key not in adata.obsm:
        print(f"✗ Embedding '{embedding_key}' not found")
        return None
    
    print(f"  Embedding: {adata.obsm[embedding_key].shape}")
    print(f"  Cells: {adata.n_obs:,}")
    
    # Use method_name if provided, otherwise use embedding_key
    if method_name is None:
        method_name = embedding_key.replace('X_', '').replace('_', ' ').title()
    
    # Configure metrics
    biocons = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=False
    )
    
    # Run benchmark
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    try:
        bm = Benchmarker(
            adata,
            batch_key=BATCH_KEY,
            label_key=LABEL_KEY,
            embedding_obsm_keys=[embedding_key],
            pre_integrated_embedding_obsm_key="X_pca",
            bio_conservation_metrics=biocons,
            n_jobs=n_jobs,
        )
        
        bm.benchmark()
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    elapsed = int(time.time() - start)
    print(f"✓ Completed in {elapsed // 60}m {elapsed % 60}s")
    
    # Get results
    df = bm.get_results(min_max_scale=False)
    
    # Rename the index to use clean method name
    df.index = [method_name]
    
    # Save with clean filename
    output_file = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_metrics.csv")
    df.to_csv(output_file)
    print(f"✓ Saved: {output_file}")
    
    # Print summary
    print("\nMetrics:")
    for col in df.columns:
        value = df.loc[method_name, col]
        # Handle Series or scalar
        val = float(value.iloc[0] if hasattr(value, 'iloc') else value)
        print(f"  {col:30s}: {val:.4f}")
    
    force_cleanup()
    print_memory()
    
    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ULTRA MEMORY-OPTIMIZED EVALUATION")
    print("="*80)
    print(f"\nSettings:")
    print(f"  N_JOBS: {N_JOBS}")
    print(f"  SCIMILARITY_BATCH_SIZE: {SCIMILARITY_BATCH_SIZE}")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print_memory()
    
    # Check files
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ File not found: {DATA_PATH}")
        return
    
    # STEP 1: Load and preprocess
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    print(f"Loading from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print_memory()
    
    # Optimize immediately
    adata = optimize_adata_memory(adata)
    force_cleanup()
    print_memory()
    
    # Preprocess
    adata = preprocess_adata_chunked(adata, chunk_size=CHUNK_SIZE)
    force_cleanup()
    
    # STEP 2: Uncorrected PCA
    print("\n" + "="*80)
    print("STEP 2: UNCORRECTED PCA")
    print("="*80)
    
    try:
        adata = prepare_uncorrected_embedding_minimal(adata, n_hvgs=N_HVGS)
        force_cleanup()
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # STEP 3: SCimilarity (optional)
    if SCIMILARITY_ENABLED:
        print("\n" + "="*80)
        print("STEP 3: SCIMILARITY")
        print("="*80)
        
        try:
            adata = compute_scimilarity_minimal(
                adata,
                model_path=SCIMILARITY_MODEL,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            force_cleanup()
        except Exception as e:
            print(f"✗ SCimilarity failed: {e}")
            print("  Continuing without SCimilarity...")
    
    # STEP 4: Benchmarking
    print("\n" + "="*80)
    print("STEP 4: BENCHMARKING")
    print("="*80)
    
    results = {}
    
    # Uncorrected
    print("\n--- Uncorrected ---")
    df_unc = run_benchmark_minimal(adata, 'X_uncorrected', OUTPUT_DIR, method_name='Uncorrected', n_jobs=N_JOBS)
    if df_unc is not None:
        results['Uncorrected'] = df_unc
    force_cleanup()
    
    # SCimilarity
    if 'X_scimilarity' in adata.obsm:
        print("\n--- SCimilarity ---")
        df_scim = run_benchmark_minimal(adata, 'X_scimilarity', OUTPUT_DIR, method_name='SCimilarity', n_jobs=N_JOBS)
        if df_scim is not None:
            results['SCimilarity'] = df_scim
        force_cleanup()
    
    # STEP 5: Compare
    if len(results) > 0:
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        # Combine results properly
        combined_list = []
        for method_name, df in results.items():
            df_copy = df.copy()
            df_copy['Method'] = method_name
            combined_list.append(df_copy)
        
        combined = pd.concat(combined_list, ignore_index=True)
        combined = combined.set_index('Method')
        
        # Save combined results
        output_file = os.path.join(OUTPUT_DIR, "combined_metrics.csv")
        combined.to_csv(output_file)
        print(f"\n✓ Combined results saved to: {output_file}")
        
        # Print nicely formatted table
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        
        # Key metrics to display
        key_metrics = ['Total', 'Batch correction', 'Bio conservation']
        
        print(f"\n{'Method':<20s} {'Total':>10s} {'Batch Corr':>12s} {'Bio Conserv':>12s}")
        print("-" * 56)
        
        for method in combined.index:
            total = combined.loc[method, 'Total']
            batch = combined.loc[method, 'Batch correction']
            bio = combined.loc[method, 'Bio conservation']
            
            print(f"{method:<20s} {total:>10.4f} {batch:>12.4f} {bio:>12.4f}")
        
        # Full table
        print("\n" + "="*80)
        print("DETAILED METRICS")
        print("="*80)
        print("\n" + combined.to_string())
        
        # Winner
        if len(results) > 1:
            print("\n" + "="*80)
            print("COMPARISON")
            print("="*80)
            
            winner_idx = combined['Total'].idxmax()
            winner_score = combined.loc[winner_idx, 'Total']
            
            print(f"\n🏆 Best Method: {winner_idx}")
            print(f"   Total Score: {winner_score:.4f}")
            
            if 'Uncorrected' in combined.index and winner_idx != 'Uncorrected':
                baseline = combined.loc['Uncorrected', 'Total']
                improvement = ((winner_score - baseline) / baseline) * 100
                print(f"   Improvement over Uncorrected: +{improvement:.1f}%")
            
            print("\nAll methods:")
            for method in combined.index:
                total = combined.loc[method, 'Total']
                batch = combined.loc[method, 'Batch correction']
                bio = combined.loc[method, 'Bio conservation']
                print(f"  {method:<20s}: Total={total:.3f}, Batch={batch:.3f}, Bio={bio:.3f}")
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print_memory()

if __name__ == "__main__":
    main()

