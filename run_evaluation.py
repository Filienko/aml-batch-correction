#!/usr/bin/env python
"""
FIXED: Batch Correction Evaluation - Auto-detect Column Names
Automatically detects whether columns are capitalized or lowercase
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
# AUTO-DETECT COLUMN NAMES
# ============================================================================

def detect_batch_key(adata):
    """Auto-detect batch key (sample or study)"""
    for key in ['sample', 'Sample', 'study', 'Study', 'batch', 'Batch']:
        if key in adata.obs.columns:
            print(f"  ✓ Detected batch key: '{key}' ({adata.obs[key].nunique()} unique values)")
            return key
    raise ValueError("No batch column found! Expected 'sample', 'Sample', 'study', or 'Study'")

def detect_label_key(adata):
    """Auto-detect cell type label key"""
    for key in ['celltype', 'Cell Type', 'cell_type', 'main_original_celltype', 'annotation']:
        if key in adata.obs.columns:
            print(f"  ✓ Detected label key: '{key}' ({adata.obs[key].nunique()} unique types)")
            return key
    raise ValueError("No cell type column found! Expected 'celltype', 'Cell Type', etc.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# These will be auto-detected
BATCH_KEY = None  # Will be set to "Sample" or "sample"
LABEL_KEY = None  # Will be set to "Cell Type" or "celltype"
BATCH_KEY_LOWER = None  # Lowercase version for preprocessing

# CRITICAL MEMORY PARAMETERS
N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "batch_correction_results"

# SCimilarity settings
SCIMILARITY_BATCH_SIZE = 500
SCIMILARITY_ENABLED = True

# Data processing settings
CHUNK_SIZE = 50000

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
    
    if not sp.issparse(adata.X):
        print("  Converting to sparse matrix...")
        adata.X = sp.csr_matrix(adata.X)
    
    if adata.X.dtype != np.float32:
        print("  Converting to float32...")
        adata.X.data = adata.X.data.astype(np.float32)
    
    for col in adata.obs.select_dtypes(include=['category']).columns:
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    
    return adata

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_adata_exact(adata, batch_key_lower):
    """
    EXACT preprocessing from AML-scAtlas
    batch_key_lower: lowercase version of batch key for HVG selection
    """
    print("\n" + "="*80)
    print("PREPROCESSING (Exact AML-scAtlas replication)")
    print("="*80)
    print(f"Initial: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print_memory()
    
    # Step 1: Cell filtering
    print("\n1. Filtering cells (min_counts=1000, min_genes=300)...")
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1000)
    sc.pp.filter_cells(adata, min_genes=300)
    print(f"  {n_before:,} → {adata.n_obs:,} cells")
    force_cleanup()
    
    # Step 2: Remove low-count samples (using the actual batch column)
    print(f"\n2. Removing {BATCH_KEY}s with <50 cells...")
    cell_counts = adata.obs[BATCH_KEY].value_counts()
    keep_samples = cell_counts.index[cell_counts >= 50]
    n_before = adata.n_obs
    adata = adata[adata.obs[BATCH_KEY].isin(keep_samples)].copy()
    print(f"  {n_before:,} → {adata.n_obs:,} cells")
    force_cleanup()
    
    # Step 3: Filter genes
    print("\n3. Filtering genes (min_cells=30)...")
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=30)
    print(f"  {n_before:,} → {adata.n_vars:,} genes")
    force_cleanup()
    
    # Step 4: Harmonize cell types (if applicable)
    print("\n4. Harmonizing cell types...")
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
    
    # Apply to any cell type columns present
    for col in [LABEL_KEY, 'celltype', 'Cell Type', 'main_original_celltype']:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].replace(celltype_mapping)
            print(f"  ✓ Harmonized '{col}'")
    
    # Step 5: Optimize memory
    print("\n5. Optimizing memory...")
    adata = optimize_adata_memory(adata)
    force_cleanup()
    print_memory()
    
    print(f"\n✓ Preprocessing complete: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata

# ============================================================================
# UNCORRECTED EMBEDDING
# ============================================================================

def prepare_uncorrected_embedding_exact(adata, batch_key_lower):
    """
    EXACT uncorrected PCA from AML-scAtlas
    batch_key_lower: lowercase version for HVG batch_key parameter
    """
    print("\n" + "="*80)
    print("UNCORRECTED PCA (Exact AML-scAtlas replication)")
    print("="*80)
    print_memory()
    
    # Create working copy
    print("Creating working copy...")
    adata_work = adata.copy()
    
    # Use raw counts from layers
    if 'counts' in adata.layers:
        print("  Using counts from .layers['counts']")
        adata_work.X = adata_work.layers['counts'].copy()
    else:
        print("  ⚠ Warning: No 'counts' layer, using .X as-is")
    
    force_cleanup()
    
    # Normalize
    print("Log Normalization...")
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)
    
    # Save normalized counts
    adata_work.raw = adata_work
    adata_work.layers["normalised_counts"] = adata_work.X.copy()
    
    force_cleanup()
    
    # HVG selection - use lowercase batch key
    print(f"Finding {N_HVGS} highly variable genes...")
    print(f"  Using batch_key='{batch_key_lower}' for HVG selection")
    
    sc.pp.highly_variable_genes(
        adata_work,
        n_top_genes=N_HVGS,
        flavor="seurat_v3",
        layer="counts",
        batch_key=batch_key_lower,  # Use lowercase version
        subset=True,
        span=0.8
    )
    print(f"  Subset to {adata_work.n_vars:,} genes")
    
    force_cleanup()
    
    # PCA
    print("Computing PCA...")
    sc.tl.pca(adata_work, svd_solver='arpack', use_highly_variable=True)
    
    # Transfer to original
    adata.obsm['X_pca'] = adata_work.obsm['X_pca'].copy()
    adata.obsm['X_uncorrected'] = adata_work.obsm['X_pca'].copy()
    
    del adata_work
    force_cleanup()
    
    print(f"✓ PCA computed: {adata.obsm['X_pca'].shape}")
    return adata

# ============================================================================
# SCIMILARITY (Optional)
# ============================================================================

def compute_scimilarity_minimal(adata, model_path, batch_size=500):
    """Ultra-conservative SCimilarity computation"""
    print("\n" + "="*80)
    print("SCIMILARITY")
    print("="*80)
    
    try:
        from scimilarity import CellAnnotation
        from scimilarity.utils import lognorm_counts, align_dataset
    except ImportError as e:
        print(f"✗ SCimilarity not available: {e}")
        return adata
    
    print(f"  Batch size: {batch_size} cells")
    print_memory()
    
    # Load model
    print("Loading SCimilarity model...")
    try:
        ca = CellAnnotation(model_path=model_path)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return adata
    
    force_cleanup()
    
    # Find raw counts
    X_counts = None
    genes = None
    
    if 'counts' in adata.layers:
        X_counts = adata.layers['counts']
        genes = adata.var.index
        print(f"  ✓ Using raw counts from layers['counts']")
    elif adata.raw is not None:
        X_counts = adata.raw.X
        genes = adata.raw.var.index
        print(f"  ✓ Using raw counts from .raw.X")
    else:
        print(f"  ✗ No raw counts found")
        return adata
    
    # Find common genes
    common_genes = genes.intersection(ca.gene_order)
    print(f"  {len(common_genes):,} common genes")
    
    if len(common_genes) < 1000:
        print(f"  ⚠ Too few common genes, skipping")
        return adata
    
    # Process in batches
    gene_order_dict = {gene: i for i, gene in enumerate(ca.gene_order)}
    common_genes_sorted = sorted(common_genes, key=lambda x: gene_order_dict[x])
    gene_indices = [genes.get_loc(g) for g in common_genes_sorted]
    
    print(f"Computing embeddings in batches of {batch_size}...")
    n_cells = adata.n_obs
    embeddings_list = []
    
    for start_idx in range(0, n_cells, batch_size):
        end_idx = min(start_idx + batch_size, n_cells)
        batch_num = start_idx // batch_size + 1
        total_batches = (n_cells + batch_size - 1) // batch_size
        
        if batch_num % 10 == 0 or batch_num == 1:
            print(f"  Batch {batch_num}/{total_batches}")
        
        try:
            batch_X = X_counts[start_idx:end_idx, gene_indices]
            
            import anndata as ad
            batch_ad = ad.AnnData(X=batch_X)
            batch_ad.var.index = common_genes_sorted
            batch_ad.layers['counts'] = batch_ad.X.copy()
            
            batch_aligned = align_dataset(batch_ad, ca.gene_order)
            if 'counts' not in batch_aligned.layers:
                batch_aligned.layers['counts'] = batch_aligned.X.copy()
            
            batch_norm = lognorm_counts(batch_aligned)
            batch_emb = ca.get_embeddings(batch_norm.X)
            
            embeddings_list.append(batch_emb)
            
            del batch_X, batch_ad, batch_aligned, batch_norm, batch_emb
            
            if batch_num % 20 == 0:
                force_cleanup()
        
        except Exception as e:
            print(f"\n  ✗ Error at batch {batch_num}: {e}")
            raise
    
    # Concatenate
    embeddings = np.vstack(embeddings_list)
    del embeddings_list
    force_cleanup()
    
    adata.obsm['X_scimilarity'] = embeddings
    print(f"✓ SCimilarity complete: {embeddings.shape}")
    
    return adata

# ============================================================================
# BENCHMARKING
# ============================================================================

def run_benchmark_exact(adata, embedding_key, output_dir, method_name=None, n_jobs=8):
    """EXACT benchmark configuration from AML-scAtlas"""
    print("\n" + "="*80)
    print(f"BENCHMARKING: {embedding_key}")
    print("="*80)
    print_memory()
    
    from scib_metrics.benchmark import Benchmarker, BioConservation
    import time
    
    if embedding_key not in adata.obsm:
        print(f"✗ Embedding '{embedding_key}' not found")
        return None
    
    print(f"  Embedding: {adata.obsm[embedding_key].shape}")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Batch key: {BATCH_KEY}")
    print(f"  Label key: {LABEL_KEY}")
    
    if method_name is None:
        method_name = embedding_key.replace('X_', '').replace('_', ' ').title()
    
    # EXACT bioconservation config from paper
    biocons = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    try:
        bm = Benchmarker(
            adata,
            batch_key=BATCH_KEY,  # Use detected key
            label_key=LABEL_KEY,  # Use detected key
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
    
    # Rename index properly
    if embedding_key in df.index:
        df = df.rename(index={embedding_key: method_name})
    else:
        df.index = [method_name]
    
    # Save
    output_file = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_metrics.csv")
    df.to_csv(output_file)
    print(f"✓ Saved: {output_file}")
    
    # Print summary
    print("\nMetrics:")
    for col in df.columns:
        value = df.loc[method_name, col]
        print(f"  {col:30s}: {value:.4f}")
    
    force_cleanup()
    
    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    global BATCH_KEY, LABEL_KEY, BATCH_KEY_LOWER
    
    print("="*80)
    print("BATCH CORRECTION EVALUATION - Auto-detect Column Names")
    print("="*80)
    
    # Check files
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ File not found: {DATA_PATH}")
        return
    
    # STEP 1: Load and detect columns
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA & AUTO-DETECTING COLUMNS")
    print("="*80)
    
    print(f"Loading from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Auto-detect column names
    print("\nAuto-detecting column names...")
    try:
        BATCH_KEY = detect_batch_key(adata)
        LABEL_KEY = detect_label_key(adata)
        
        # Create lowercase version for HVG selection
        # Paper uses lowercase 'sample' for HVG batch_key
        BATCH_KEY_LOWER = BATCH_KEY.lower()
        
        print(f"\n  Will use for benchmarking:")
        print(f"    BATCH_KEY = '{BATCH_KEY}'")
        print(f"    LABEL_KEY = '{LABEL_KEY}'")
        print(f"  Will use for HVG selection:")
        print(f"    batch_key = '{BATCH_KEY_LOWER}' (lowercase)")
        
        # Create lowercase column if it doesn't exist
        if BATCH_KEY_LOWER not in adata.obs.columns:
            print(f"\n  Creating lowercase column '{BATCH_KEY_LOWER}' for HVG selection...")
            adata.obs[BATCH_KEY_LOWER] = adata.obs[BATCH_KEY].copy()
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nYour data columns:")
        print(adata.obs.columns.tolist())
        return
    
    print_memory()
    
    # Optimize memory
    adata = optimize_adata_memory(adata)
    force_cleanup()
    
    # STEP 2: Preprocess
    adata = preprocess_adata_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()
    
    # STEP 3: Uncorrected PCA
    print("\n" + "="*80)
    print("STEP 2: UNCORRECTED PCA")
    print("="*80)
    
    try:
        adata = prepare_uncorrected_embedding_exact(adata, BATCH_KEY_LOWER)
        force_cleanup()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 4: SCimilarity (optional)
    if SCIMILARITY_ENABLED and os.path.exists(SCIMILARITY_MODEL):
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
    
    # STEP 5: Benchmarking
    print("\n" + "="*80)
    print("STEP 4: BENCHMARKING")
    print("="*80)
    
    results = {}
    
    # Uncorrected
    print("\n--- Uncorrected ---")
    df_unc = run_benchmark_exact(adata, 'X_uncorrected', OUTPUT_DIR, method_name='Uncorrected', n_jobs=N_JOBS)
    if df_unc is not None:
        results['Uncorrected'] = df_unc
    force_cleanup()
    
    # SCimilarity
    if 'X_scimilarity' in adata.obsm:
        print("\n--- SCimilarity ---")
        df_scim = run_benchmark_exact(adata, 'X_scimilarity', OUTPUT_DIR, method_name='SCimilarity', n_jobs=N_JOBS)
        if df_scim is not None:
            results['SCimilarity'] = df_scim
        force_cleanup()
    
    # STEP 6: Results
    if len(results) > 0:
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        # Combine results
        combined = pd.concat(results.values(), keys=results.keys())
        combined.index = combined.index.droplevel(1)
        
        # Save
        output_file = os.path.join(OUTPUT_DIR, "combined_metrics.csv")
        combined.to_csv(output_file)
        print(f"\n✓ Combined results saved to: {output_file}")
        
        # Print table
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        
        print(f"\n{'Method':<20s} {'Total':>10s} {'Batch Corr':>12s} {'Bio Conserv':>12s}")
        print("-" * 56)
        
        for method in combined.index:
            total = combined.loc[method, 'Total']
            batch = combined.loc[method, 'Batch correction']
            bio = combined.loc[method, 'Bio conservation']
            
            print(f"method:{method} total:{total} batch:{batch} bio:{bio}")
        
        print("\n" + "="*80)
        print("DATA INFO")
        print("="*80)
        print(f"Batch key used: '{BATCH_KEY}' ({adata.obs[BATCH_KEY].nunique()} unique)")
        print(f"Label key used: '{LABEL_KEY}' ({adata.obs[LABEL_KEY].nunique()} types)")
        print(f"Cells: {adata.n_obs:,}")
        print(f"Genes after preprocessing: {adata.n_vars:,}")
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print_memory()

if __name__ == "__main__":
    main()

