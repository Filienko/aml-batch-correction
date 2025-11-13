#!/usr/bin/env python
"""
CORRECTED: Batch Correction Evaluation with Proper SCimilarity Integration

Key fixes for SCimilarity:
1. Use raw counts (not normalized)
2. Align genes BEFORE normalization using align_dataset()
3. Normalize using lognorm_counts() from scimilarity.utils
4. Use full gene set (not HVGs)
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

# Import the corrected SCimilarity function
from scimilarity import CellAnnotation
from scimilarity.utils import lognorm_counts, align_dataset

# ============================================================================
# AUTO-DETECT COLUMN NAMES
# ============================================================================

def detect_batch_key(adata):
    """Auto-detect batch key"""
    for key in ['Study', 'Sample', 'Batch']:
        if key in adata.obs.columns:
            print(f"  ✓ Detected batch key: '{key}' ({adata.obs[key].nunique()} unique values)")
            return key
    raise ValueError("No batch column found!")

def detect_label_key(adata):
    """Auto-detect cell type label key"""
    for key in ['celltype', 'Cell Type', 'cell_type', 'main_original_celltype', 'annotation']:
        if key in adata.obs.columns:
            print(f"  ✓ Detected label key: '{key}' ({adata.obs[key].nunique()} unique types)")
            return key
    raise ValueError("No cell type column found!")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

BATCH_KEY = None
LABEL_KEY = None
BATCH_KEY_LOWER = None

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "batch_correction_results_fixed"

# CRITICAL: Reduced for 733k cells!
SCIMILARITY_BATCH_SIZE = 1000  # Reduced from 5000 to avoid OOM
SCIMILARITY_ENABLED = True

# ============================================================================
# MEMORY UTILITIES
# ============================================================================

def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def print_memory():
    mem_gb = get_memory_usage_gb()
    print(f"  💾 Memory: {mem_gb:.2f} GB")

def force_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()

def optimize_adata_memory(adata):
    import scipy.sparse as sp
    
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    
    if adata.X.dtype != np.float32:
        adata.X.data = adata.X.data.astype(np.float32)
    
    for col in adata.obs.select_dtypes(include=['category']).columns:
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    
    return adata

# ============================================================================
# PREPROCESSING (FOR UNCORRECTED PCA ONLY)
# ============================================================================

def preprocess_adata_exact(adata, batch_key_lower):
    """
    Preprocessing for uncorrected PCA baseline
    NOTE: SCimilarity will use the ORIGINAL data, not this preprocessed version
    """
    global BATCH_KEY, LABEL_KEY

    print("\n" + "="*80)
    print("PREPROCESSING FOR UNCORRECTED PCA")
    print("="*80)
    print(f"Initial: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print_memory()
    
    # Cell filtering
    print("\n1. Filtering cells (min_counts=1000, min_genes=300)...")
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1000)
    sc.pp.filter_cells(adata, min_genes=300)
    print(f"  {n_before:,} → {adata.n_obs:,} cells")
    force_cleanup()
    
    # Remove low-count samples
    print(f"\n2. Removing {BATCH_KEY}s with <50 cells...")
    cell_counts = adata.obs[BATCH_KEY].value_counts()
    keep_samples = cell_counts.index[cell_counts >= 50]
    n_before = adata.n_obs
    adata = adata[adata.obs[BATCH_KEY].isin(keep_samples)].copy()
    print(f"  {n_before:,} → {adata.n_obs:,} cells")
    force_cleanup()
    
    # Filter genes
    print("\n3. Filtering genes (min_cells=30)...")
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=30)
    print(f"  {n_before:,} → {adata.n_vars:,} genes")
    force_cleanup()
    
    # Harmonize cell types
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
    
    for col in [LABEL_KEY, 'celltype', 'Cell Type', 'main_original_celltype']:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].replace(celltype_mapping)
    
    # Optimize memory
    adata = optimize_adata_memory(adata)
    force_cleanup()
    
    print(f"\n✓ Preprocessing complete: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata

# ============================================================================
# UNCORRECTED PCA EMBEDDING
# ============================================================================

def prepare_uncorrected_embedding_exact(adata, batch_key_lower):
    """Compute uncorrected PCA baseline"""
    print("\n" + "="*80)
    print("UNCORRECTED PCA")
    print("="*80)
    print_memory()
    
    adata_work = adata.copy()
    
    if 'counts' in adata.layers:
        adata_work.X = adata_work.layers['counts'].copy()
    
    force_cleanup()
    
    # Normalize
    print("Normalizing...")
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)
    
    adata_work.raw = adata_work
    adata_work.layers["normalised_counts"] = adata_work.X.copy()
    
    force_cleanup()
    
    # HVGs
    print(f"Finding {N_HVGS} highly variable genes...")
    sc.pp.highly_variable_genes(
        adata_work,
        n_top_genes=N_HVGS,
        flavor="seurat_v3",
        layer="counts",
        batch_key=batch_key_lower,
        subset=True,
        span=0.8
    )
    print(f"  Subset to {adata_work.n_vars:,} genes")
    
    force_cleanup()
    
    # PCA
    print("Computing PCA...")
    sc.tl.pca(adata_work, svd_solver='arpack', use_highly_variable=True)
    
    adata.obsm['X_pca'] = adata_work.obsm['X_pca'].copy()
    adata.obsm['X_uncorrected'] = adata_work.obsm['X_pca'].copy()
    
    del adata_work
    force_cleanup()
    
    print(f"✓ PCA computed: {adata.obsm['X_pca'].shape}")
    return adata

# ============================================================================
# SCVI LOADING
# ============================================================================

def load_scvi_embedding(adata, scvi_path="data/AML_scAtlas_X_scVI.h5ad"):
    """
    Load pre-computed scVI embeddings with graceful error handling.
    
    This function:
    - Handles file not found (skips gracefully)
    - Handles cell count mismatches (subsets to common cells)
    - Handles cell order differences (reorders automatically)
    - Never crashes the pipeline
    
    Args:
        adata: Main AnnData object
        scvi_path: Path to scVI embedding file
    
    Returns:
        adata: With scVI embeddings added if successful,
               or subset if cells don't match,
               or unchanged if loading fails
    """
    print("\n" + "="*80)
    print("SCVI EMBEDDINGS")
    print("="*80)
    
    # Check if file exists
    if not os.path.exists(scvi_path):
        print(f"  ℹ File not found: {scvi_path}")
        print(f"  Skipping scVI evaluation")
        return adata
    
    try:
        print(f"  Loading scVI embeddings...")
        print(f"  File: {scvi_path}")
        
        adata_scvi = sc.read_h5ad(scvi_path)
        
        print(f"  scVI file: {adata_scvi.n_obs:,} cells × {adata_scvi.n_vars:,} features")
        print(f"  Main data: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        
        # Case 1: Perfect match
        if adata_scvi.n_obs == adata.n_obs and (adata.obs_names == adata_scvi.obs_names).all():
            print(f"  ✓ Perfect match (same cells, same order)")
            
            adata.obsm['X_scVI'] = adata_scvi.X.copy()
            print(f"  ✓ Added scVI embeddings: {adata.obsm['X_scVI'].shape}")
            
            del adata_scvi
            force_cleanup()
            return adata
        
        # Case 2: Same count, different order
        elif adata_scvi.n_obs == adata.n_obs:
            print(f"  ⚠ Same cell count but different order")
            print(f"  Reordering to match main data...")
            
            try:
                adata_scvi_reordered = adata_scvi[adata.obs_names]
                adata.obsm['X_scVI'] = adata_scvi_reordered.X.copy()
                
                print(f"  ✓ Successfully reordered")
                print(f"  ✓ Added scVI embeddings: {adata.obsm['X_scVI'].shape}")
                
                del adata_scvi, adata_scvi_reordered
                force_cleanup()
                return adata
            
            except KeyError:
                print(f"  ✗ Cannot reorder - cell IDs don't match")
                print(f"  ℹ Checking if IDs are just numeric indices...")

                # Case 2b: Same count, IDs don't match, but might be in same order
                # This happens when scVI file has numeric indices ('0', '1', '2')
                # but main data has cell barcodes
                try:
                    # Check if scVI IDs are just numeric indices
                    scvi_ids_numeric = all(str(idx).isdigit() for idx in adata_scvi.obs_names[:100])

                    if scvi_ids_numeric:
                        print(f"  ✓ scVI file uses numeric indices")
                        print(f"  ℹ Assuming embeddings are in same order as main data")
                        print(f"  ⚠ WARNING: This assumes scVI embeddings were computed on same data in same order!")

                        # Trust the order and copy directly
                        adata.obsm['X_scVI'] = adata_scvi.X.copy()

                        print(f"  ✓ Added scVI embeddings: {adata.obsm['X_scVI'].shape}")

                        del adata_scvi
                        force_cleanup()
                        return adata

                except Exception as e:
                    print(f"  ℹ Could not verify numeric indices: {e}")
                    pass
                # Fall through to mismatch handling

        # Case 3: Different counts - find common cells OR use numeric indices
        print(f"  ⚠ Cell count mismatch")
        print(f"  Searching for common cells...")

        common_cells = adata.obs_names.intersection(adata_scvi.obs_names)

        if len(common_cells) == 0:
            print(f"  ✗ No common cells found by name matching")

            # Check if scVI uses numeric indices (common when computed on full data)
            print(f"  ℹ Checking if scVI file uses numeric indices...")
            scvi_ids_numeric = all(str(idx).isdigit() for idx in adata_scvi.obs_names[:100])

            if scvi_ids_numeric:
                print(f"  ✓ scVI file uses numeric indices")
                print(f"  ⚠ scVI: {adata_scvi.n_obs:,} cells, Current: {adata.n_obs:,} cells")
                print(f"")
                print(f"  ✗ CANNOT LOAD: scVI was computed on full dataset,")
                print(f"    but current data is a SUBSET (e.g., specific studies only).")
                print(f"")
                print(f"  SOLUTIONS:")
                print(f"    1. Compute scVI on this specific subset, OR")
                print(f"    2. Run scVI on full data first, then subset BOTH,")
                print(f"    3. Use scVI only for full-dataset experiments")
                print(f"")
                print(f"  Skipping scVI for this experiment.")
                del adata_scvi
                force_cleanup()
                return adata
            else:
                print(f"  ✗ Cannot match cells - different naming and counts")
                print(f"  Skipping scVI evaluation")
                return adata
        
        print(f"  Found {len(common_cells):,} common cells ({100*len(common_cells)/adata.n_obs:.1f}%)")
        
        # Create subset with common cells
        print(f"  Creating subset with common cells...")
        
        adata_subset = adata[common_cells].copy()
        adata_scvi_subset = adata_scvi[common_cells]
        
        adata_subset.obsm['X_scVI'] = adata_scvi_subset.X.copy()
        
        print(f"  ✓ Subset created: {adata_subset.n_obs:,} cells")
        print(f"  ✓ Added scVI embeddings: {adata_subset.obsm['X_scVI'].shape}")
        print(f"  ⚠ Note: Returning subset of original data!")
        
        del adata_scvi, adata_scvi_subset
        force_cleanup()
        
        return adata_subset
    
    except Exception as e:
        print(f"  ✗ Error loading scVI embeddings:")
        print(f"    {type(e).__name__}: {e}")
        print(f"  Skipping scVI evaluation")
        
        # Print abbreviated traceback
        import traceback
        tb_lines = traceback.format_exc().split('\n')
        print(f"  Traceback (last 3 lines):")
        for line in tb_lines[-4:-1]:
            print(f"    {line}")
        
        return adata


# ============================================================================
# SCIMILARITY - CORRECTED VERSION
# ============================================================================

def compute_scimilarity_corrected(adata, model_path, batch_size=1000):
    """
    MEMORY-OPTIMIZED SCimilarity for large datasets (733k cells).
    
    Key changes for your dataset:
    1. Smaller batch size (1000 instead of 5000)
    2. Process normalization in chunks to avoid OOM
    3. More aggressive garbage collection
    4. Monitor memory usage
    """
    print("\n" + "="*80)
    print("SCIMILARITY - MEMORY-OPTIMIZED FOR LARGE DATASET")
    print("="*80)
    
    # Memory monitoring
    import psutil
    process = psutil.Process(os.getpid())
    
    def print_mem():
        mem_gb = process.memory_info().rss / 1e9
        print(f"    💾 Memory: {mem_gb:.1f} GB")
    
    # Load model
    print("\n1. Loading SCimilarity model...")
    try:
        ca = CellAnnotation(model_path=model_path)
        print(f"   ✓ Model loaded")
        print(f"   Model expects {len(ca.gene_order):,} genes")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return adata
    
    print_mem()
    
    # Get raw counts
    print("\n2. Extracting raw counts...")
    
    raw_counts = None
    gene_names = None
    
    if 'counts' in adata.layers:
        raw_counts = adata.layers['counts']
        gene_names = adata.var_names
        print(f"   ✓ Using .layers['counts']")
    elif adata.raw is not None:
        raw_counts = adata.raw.X
        gene_names = adata.raw.var_names
        print(f"   ✓ Using .raw.X")
    else:
        x_max = adata.X.max()
        if x_max > 100:
            raw_counts = adata.X
            gene_names = adata.var_names
            print(f"   ✓ Using .X (max={x_max:.0f})")
        else:
            print(f"   ✗ No raw counts found")
            return adata
    
    n_cells = adata.n_obs
    n_genes = len(gene_names)
    
    print(f"   Dataset: {n_cells:,} cells × {n_genes:,} genes")
    print(f"   Count max: {raw_counts.max():.0f}")
    
    # Estimate memory
    dense_mem_gb = (n_cells * n_genes * 4) / 1e9
    print(f"   ⚠ Full dense array would need ~{dense_mem_gb:.1f} GB")
    print(f"   → Using batched processing to avoid this")
    
    force_cleanup()
    print_mem()
    
    # Find common genes
    print(f"\n3. Finding common genes...")
    common_genes = set(gene_names).intersection(set(ca.gene_order))
    print(f"   Common: {len(common_genes):,} ({100*len(common_genes)/len(ca.gene_order):.1f}%)")
    
    if len(common_genes) < 5000:
        print(f"   ⚠ Only {len(common_genes):,} common genes")
    
    # Create gene mapping
    gene_order_dict = {gene: i for i, gene in enumerate(ca.gene_order)}
    common_genes_sorted = sorted(common_genes, key=lambda x: gene_order_dict[x])
    gene_indices = [gene_names.get_loc(g) for g in common_genes_sorted]
    
    # Compute embeddings in batches
    print(f"\n4. Computing embeddings (batch_size={batch_size:,})...")
    n_batches = (n_cells + batch_size - 1) // batch_size
    print(f"   Total batches: {n_batches}")
    print(f"   ⚠ Large dataset - this will take 15-30 minutes")
    
    embeddings_list = []
    
    import scipy.sparse as sp
    import anndata as ad
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_cells)
        
        # Progress updates every 10 batches or at start
        if batch_idx % 10 == 0 or batch_idx == 0:
            print(f"\n   Batch {batch_idx+1}/{n_batches}: {start_idx:,}-{end_idx:,}")
            print_mem()
        
        try:
            # Extract batch (only common genes to save memory)
            if sp.issparse(raw_counts):
                batch_counts = raw_counts[start_idx:end_idx, gene_indices].toarray()
            else:
                batch_counts = raw_counts[start_idx:end_idx, gene_indices]
            
            # Create mini AnnData
            batch_ad = ad.AnnData(X=batch_counts)
            batch_ad.var_names = common_genes_sorted
            
            # Align to model
            batch_aligned = align_dataset(batch_ad, ca.gene_order)
            
            # Prepare for normalization
            if 'counts' not in batch_aligned.layers:
                batch_aligned.layers['counts'] = batch_aligned.X.copy()
            
            # Normalize (THIS was where it got killed before)
            batch_norm = lognorm_counts(batch_aligned)
            
            # Get embeddings
            batch_emb = ca.get_embeddings(batch_norm.X)
            embeddings_list.append(batch_emb)
            
            # Aggressive cleanup
            del batch_counts, batch_ad, batch_aligned, batch_norm, batch_emb
            
            # More frequent GC for large datasets
            if batch_idx % 5 == 0:
                force_cleanup()
        
        except MemoryError as e:
            print(f"\n   ✗ OUT OF MEMORY at batch {batch_idx+1}!")
            print(f"   Current batch_size: {batch_size}")
            print(f"   SOLUTION: Reduce batch_size to {batch_size // 2} and retry")
            print(f"   In run_evaluation_fixed.py, change:")
            print(f"     SCIMILARITY_BATCH_SIZE = {batch_size // 2}")
            raise MemoryError(f"Reduce SCIMILARITY_BATCH_SIZE to {batch_size // 2}")
        
        except Exception as e:
            print(f"\n   ✗ Error at batch {batch_idx+1}: {e}")
            raise
    
    # Concatenate
    print(f"\n5. Concatenating results...")
    embeddings = np.vstack(embeddings_list)
    
    del embeddings_list
    force_cleanup()
    
    print(f"   ✓ Final shape: {embeddings.shape}")
    
    # Add to adata
    adata.obsm['X_scimilarity'] = embeddings
    
    print(f"\n✓ SCimilarity complete!")
    print_mem()
    print("="*80)
    
    return adata

# ============================================================================
# HARMONY BATCH CORRECTION
# ============================================================================

def compute_harmony_corrected(adata, batch_key, n_jobs=8):
    """
    Compute Harmony batch correction.

    Harmony is a fast integration method that works directly on PCA embeddings.
    It iteratively adjusts principal components to remove batch effects while
    preserving biological variation.

    Reference: Korsunsky et al. (2019) Nature Methods
    """
    print("\n" + "="*80)
    print("HARMONY BATCH CORRECTION")
    print("="*80)
    print_memory()

    try:
        # Harmony requires PCA as input
        if 'X_pca' not in adata.obsm:
            print("  ✗ X_pca not found! Run uncorrected PCA first.")
            return adata

        print(f"  Input: X_pca with shape {adata.obsm['X_pca'].shape}")
        print(f"  Batch key: {batch_key}")
        print(f"  Batches: {adata.obs[batch_key].nunique()}")

        # Create working copy to avoid modifying the original
        adata_work = adata.copy()

        print("\n  Running Harmony integration...")
        import time
        start = time.time()

        # Run Harmony
        sc.external.pp.harmony_integrate(
            adata_work,
            batch_key,
            basis='X_pca',
            adjusted_basis='X_harmony',
            max_iter_harmony=10,
            verbose=False
        )

        elapsed = int(time.time() - start)
        print(f"  ✓ Completed in {elapsed // 60}m {elapsed % 60}s")

        # Copy result back to original adata
        adata.obsm['X_harmony'] = adata_work.obsm['X_harmony'].copy()

        print(f"  ✓ Harmony embedding: {adata.obsm['X_harmony'].shape}")

        del adata_work
        force_cleanup()

        print("✓ Harmony complete!")
        print_memory()

        return adata

    except Exception as e:
        print(f"  ✗ Harmony failed: {e}")
        import traceback
        traceback.print_exc()
        return adata


# ============================================================================
# BENCHMARKING
# ============================================================================

def run_benchmark_exact(adata, embedding_key, output_dir, method_name=None, n_jobs=8):
    """Run scIB benchmark"""
    global BATCH_KEY, LABEL_KEY

    print("\n" + "="*80)
    print(f"BENCHMARKING: {embedding_key}")
    print("="*80)

    from scib_metrics.benchmark import Benchmarker, BioConservation
    import time
    
    if embedding_key not in adata.obsm:
        print(f"✗ Embedding '{embedding_key}' not found")
        return None
    
    print(f"  Embedding: {adata.obsm[embedding_key].shape}")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Batch: {BATCH_KEY}")
    print(f"  Label: {LABEL_KEY}")
    
    if method_name is None:
        method_name = embedding_key.replace('X_', '').title()
    
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
            batch_key=BATCH_KEY,
            label_key=LABEL_KEY,
            embedding_obsm_keys=[embedding_key],
            pre_integrated_embedding_obsm_key="X_pca",
            bio_conservation_metrics=biocons,
            n_jobs=n_jobs,
        )
        
        bm.benchmark()
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    elapsed = int(time.time() - start)
    print(f"✓ Completed in {elapsed // 60}m {elapsed % 60}s")
    
    df = bm.get_results(min_max_scale=False)
    
    # Debug: show what was returned
    print(f"\n  Debug: Results shape: {df.shape}")
    print(f"  Debug: Index values: {df.index.tolist()}")
    
    # Handle case where results include multiple embeddings
    # We only want the row for our specific embedding
    if len(df.index) > 1:
        print(f"  ℹ Multiple results returned ({len(df.index)} rows)")
        
        # Find the row that matches our embedding_key
        if embedding_key in df.index:
            print(f"  → Selecting row for '{embedding_key}'")
            df = df.loc[[embedding_key]]
        else:
            # Take the first row if exact match not found
            print(f"  ⚠ Embedding key '{embedding_key}' not in index")
            print(f"  → Using first row: '{df.index[0]}'")
            df = df.iloc[[0]]
    
    # Rename the index to our method name
    df.index = [method_name]
    
    output_file = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_metrics.csv")
    df.to_csv(output_file)
    print(f"✓ Saved: {output_file}")
    
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
    print("BATCH CORRECTION EVALUATION - CORRECTED SCIMILARITY")
    print("="*80)
    
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ File not found: {DATA_PATH}")
        return
    
    # STEP 1: Load and detect
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    print(f"Loading: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Detect columns
    try:
        BATCH_KEY = detect_batch_key(adata)
        LABEL_KEY = detect_label_key(adata)
        BATCH_KEY_LOWER = BATCH_KEY.lower()
        
        if BATCH_KEY_LOWER not in adata.obs.columns:
            adata.obs[BATCH_KEY_LOWER] = adata.obs[BATCH_KEY].copy()
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Verify raw counts for SCimilarity
    print("\nVerifying data for SCimilarity...")
    if 'counts' in adata.layers:
        print(f"  ✓ .layers['counts'] found (max={adata.layers['counts'].max():.0f})")
    elif adata.raw is not None:
        print(f"  ✓ .raw.X found (max={adata.raw.X.max():.0f})")
    else:
        print(f"  ⚠ No raw counts in layers or .raw")
        print(f"    .X max={adata.X.max():.2f}")
    
    adata = optimize_adata_memory(adata)
    force_cleanup()
    
    # STEP 2: Preprocess for uncorrected PCA
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
        return
    
    # STEP 4: Load scVI embeddings (optional)
    print("\n" + "="*80)
    print("STEP 3: SCVI EMBEDDINGS (OPTIONAL)")
    print("="*80)
    
    adata = load_scvi_embedding(adata, SCVI_PATH)
    force_cleanup()
    
    # STEP 5: SCimilarity (CORRECTED)
    if SCIMILARITY_ENABLED and os.path.exists(SCIMILARITY_MODEL):
        print("\n" + "="*80)
        print("STEP 4: SCIMILARITY (CORRECTED)")
        print("="*80)
        
        try:
            adata = compute_scimilarity_corrected(
                adata,
                model_path=SCIMILARITY_MODEL,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            force_cleanup()
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 6: Benchmarking
    print("\n" + "="*80)
    print("STEP 5: BENCHMARKING")
    print("="*80)
    
    results = {}
    
    # Uncorrected
    print("\n--- Uncorrected ---")
    df_unc = run_benchmark_exact(adata, 'X_uncorrected', OUTPUT_DIR, 'Uncorrected', N_JOBS)
    if df_unc is not None:
        results['Uncorrected'] = df_unc
    force_cleanup()
    
    # scVI (if available)
    if 'X_scVI' in adata.obsm:
        print("\n--- scVI ---")
        df_scvi = run_benchmark_exact(adata, 'X_scVI', OUTPUT_DIR, 'scVI', N_JOBS)
        if df_scvi is not None:
            results['scVI'] = df_scvi
        force_cleanup()
    else:
        print("\n--- scVI ---")
        print("  ℹ scVI embeddings not available, skipping")
    
    # SCimilarity
    if 'X_scimilarity' in adata.obsm:
        print("\n--- SCimilarity ---")
        df_scim = run_benchmark_exact(adata, 'X_scimilarity', OUTPUT_DIR, 'SCimilarity', N_JOBS)
        if df_scim is not None:
            results['SCimilarity'] = df_scim
        force_cleanup()
    else:
        print("\n--- SCimilarity ---")
        print("  ℹ SCimilarity embeddings not available, skipping")
    
    # STEP 7: Results
    if len(results) > 0:
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        combined = pd.concat(results.values(), keys=results.keys())
        combined.index = combined.index.droplevel(1)
        
        output_file = os.path.join(OUTPUT_DIR, "combined_metrics.csv")
        combined.to_csv(output_file)
        print(f"\n✓ Results saved: {output_file}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        print(f"\n{'Method':<20s} {'Total':>10s} {'Batch':>10s} {'Bio':>10s}")
        print("-" * 52)
        
        for method in combined.index:
            total = combined.loc[method, 'Total']
            batch = combined.loc[method, 'Batch correction']
            bio = combined.loc[method, 'Bio conservation']
            print(f"{method:<20s} {total:>10.4f} {batch:>10.4f} {bio:>10.4f}")
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

