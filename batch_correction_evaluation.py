#!/usr/bin/env python
# coding: utf-8

"""
Batch Correction Evaluation Pipeline - MEMORY OPTIMIZED VERSION
Compares Uncorrected, Harmony, scVI, scANVI, and SCimilarity using scIB metrics

This script replicates the evaluation setup from the AML Atlas paper with
optimizations for large datasets (>700k cells).
"""

import os
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker, BioConservation
import time
from typing import Optional
import anndata as ad

# SCimilarity specific imports
from scimilarity import CellAnnotation
from scimilarity.utils import lognorm_counts, align_dataset

# Optional imports
try:
    import scib
    SCIB_AVAILABLE = True
except ImportError:
    SCIB_AVAILABLE = False
    print("Warning: scib not available for Harmony integration")

try:
    import scvi
    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False
    print("Warning: scvi-tools not available")

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Configure scanpy
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))

def preprocess_adata(adata: ad.AnnData) -> ad.AnnData:
    """Apply the same preprocessing as the original AML Atlas analysis."""
    
    print("Preprocessing dataset...")
    
    # 1. Cell filtering (matching original)
    print(f"  Starting: {adata.n_obs:,} cells")
    sc.pp.filter_cells(adata, min_counts=1000)
    sc.pp.filter_cells(adata, min_genes=300)
    print(f"  After cell filtering: {adata.n_obs:,} cells")
    
    # 2. Remove samples with low cell counts
    cell_counts = adata.obs['Sample'].value_counts()
    keep = cell_counts.index[cell_counts >= 50]
    adata = adata[adata.obs['Sample'].isin(keep)].copy()
    print(f"  After sample filtering: {adata.n_obs:,} cells")
    
    # 3. Harmonize cell type names (EXACT mapping from original)
    print("  Harmonizing cell types...")
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
    
    # Apply to your cell type column (adjust column name as needed)
    if "main_original_celltype" in adata.obs.columns:
        adata.obs["main_original_celltype"] = adata.obs["main_original_celltype"].replace(celltype_mapping)
    if "celltype" in adata.obs.columns:
        adata.obs["celltype"] = adata.obs["celltype"].replace(celltype_mapping)
    
    print("✓ Preprocessing complete")
    return adata

def prepare_uncorrected_embedding(adata: ad.AnnData, n_hvgs: int = 2000) -> ad.AnnData:
    """
    Prepare uncorrected PCA embedding - handling pre-normalized data correctly.
    
    Args:
        adata: AnnData with raw counts in .layers['counts'] and normalized data in .X
        n_hvgs: Number of highly variable genes
    
    Returns:
        AnnData with X_pca and X_uncorrected embeddings
    """
    print(f"\nPreparing uncorrected embedding...")
    print(f"Input: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Check data state
    x_max = adata.X.max()
    counts_max = adata.layers['counts'].max()
    print(f"  adata.X max: {x_max:.2f} (already normalized)")
    print(f"  adata.layers['counts'] max: {counts_max:.0f} (raw counts)")
    
    # Create working copy with RAW COUNTS as .X
    print("  Creating working copy with raw counts...")
    adata_work = adata.copy()
    adata_work.X = adata_work.layers['counts'].copy()  # ← KEY: Use raw counts!
    
    print(f"  Working data X max: {adata_work.X.max():.0f}")
    
    # Now follow the original workflow exactly
    
    # 1. Filter genes
    print("  Filtering genes (min_cells=30)...")
    n_genes_before = adata_work.n_vars
    sc.pp.filter_genes(adata_work, min_cells=30)
    print(f"    {n_genes_before:,} → {adata_work.n_vars:,} genes")
    
    # 2. Normalize to depth 10,000
    print("  Normalizing to target_sum=1e4...")
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    
    # 3. Log transform
    print("  Log1p transform...")
    sc.pp.log1p(adata_work)
    print(f"    After log1p - X max: {adata_work.X.max():.2f}")
    
    # 4. Store normalized data
    adata_work.raw = adata_work
    adata_work.layers["normalised_counts"] = adata_work.X.copy()
    
    # 5. Identify HVGs using raw counts, then SUBSET
    print(f"  Computing {n_hvgs} highly variable genes...")
    sc.pp.highly_variable_genes(
        adata_work,
        n_top_genes=n_hvgs,
        flavor="seurat_v3",
        layer="counts",  # Uses raw counts for HVG calculation
        batch_key="Sample",
        subset=True,  # CRITICAL: Subset to HVGs
        span=0.8
    )
    print(f"    Subset to {adata_work.n_vars:,} HVGs")
    
    # 6. Optional: Scale (helps with PCA stability)
    # sc.pp.scale(adata_work, max_value=10)
    
    # 7. Compute PCA on normalized, log-transformed, HVG-subset data
    print("  Computing PCA...")
    sc.tl.pca(adata_work, svd_solver='arpack', use_highly_variable=True)
    print(f"    PCA shape: {adata_work.obsm['X_pca'].shape}")
    
    # 8. Transfer PCA back to original object
    adata.obsm['X_pca'] = adata_work.obsm['X_pca'].copy()
    adata.obsm['X_uncorrected'] = adata_work.obsm['X_pca'].copy()
    
    print(f"✓ Uncorrected PCA computed: {adata.obsm['X_pca'].shape}")
    
    return adata

def load_scvi_embedding(
    adata: ad.AnnData,
    scvi_path: str = "data/AML_scAtlas_X_scVI.h5ad"
) -> ad.AnnData:
    """
    Load pre-computed scVI embeddings and add to adata object.

    Args:
        adata: Original AnnData object
        scvi_path: Path to scVI-corrected embeddings (must be .h5ad)

    Returns:
        AnnData object with scVI embeddings in .obsm['X_scVI']
    """
    print(f"\nLoading scVI embeddings from {scvi_path}...")

    if not os.path.exists(scvi_path):
        raise FileNotFoundError(f"scVI embedding file not found: {scvi_path}")

    # Load scVI embeddings
    adata_scvi = sc.read_h5ad(scvi_path)

    if adata.n_obs != adata_scvi.n_obs:
        raise ValueError(
            f"Cell number mismatch! Main adata has {adata.n_obs} cells, "
            f"but scVI file has {adata_scvi.n_obs} cells."
        )
    
    print(f"  ✓ Cell counts match ({adata.n_obs}). Assuming same cell order.")
    adata.obsm['X_scVI'] = adata_scvi.X.copy()

    print(f"✓ scVI embeddings loaded: {adata.obsm['X_scVI'].shape}")

    return adata

def compute_scimilarity_embedding(
    adata: ad.AnnData,
    model_path: str = "model_v1.1",
    use_full_gene_set: bool = True,
    batch_size: int = 5000,
    subsample_for_alignment: bool = True,
    max_genes_for_full_alignment: int = 20000
) -> ad.AnnData:
    """
    Compute SCimilarity embeddings for batch correction evaluation.
    MEMORY-OPTIMIZED VERSION for large datasets.

    Args:
        adata: AnnData object with raw counts in .layers['counts']
        model_path: Path to SCimilarity model
        use_full_gene_set: Whether to use full gene set (recommended)
        batch_size: Number of cells to process at once (reduce if OOM)
        subsample_for_alignment: If True, subsample genes before alignment for large datasets
        max_genes_for_full_alignment: Max genes to keep before alignment if subsampling

    Returns:
        AnnData object with SCimilarity embeddings in .obsm['X_scimilarity']
    """
    print(f"\nComputing SCimilarity embeddings...")
    print(f"  Dataset size: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # For large datasets, use batched processing
    if adata.n_obs > 10000:
        print(f"  Large dataset detected - using batched processing")
        print(f"  Batch size: {batch_size:,} cells")

    # Initialize SCimilarity model
    try:
        ca = CellAnnotation(model_path=model_path)
        print(f"✓ SCimilarity model loaded from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load SCimilarity model: {e}")

    # Prepare data for SCimilarity
    print("  Preparing data for SCimilarity...")
    if use_full_gene_set:
        # Use full gene set (recommended to avoid gene overlap issues)
        if adata.raw is None:
            print("  No .raw attribute found, using main object")
            adata_full.X = adata_full.layers['counts'].copy()
            adata_full = adata.copy()
        else:
            print("  Using .raw attribute for full gene set")
            # Don't convert to full AnnData yet - work with view to save memory
            adata_full = adata.raw.to_adata()
            if adata_full.X.max() < 100:
                if 'counts' in adata.layers:
                    adata_full.X = adata.layers['counts'].copy()
    else:
        if 'highly_variable' in adata.var.columns:
            print(f"  Using 'highly_variable' genes (use_full_gene_set=False)")
            adata_full = adata[:, adata.var['highly_variable']].copy()
            print(f"  ✓ Subsetted to {adata_full.shape[1]} HVGs")
        else:
            adata_full = adata.copy()
    
    print(f"  Working with: {adata_full.n_obs:,} cells × {adata_full.n_vars:,} genes")
    
    # Ensure gene symbols are in index
    if 'gene_name' in adata_full.var.columns:
        print("  Setting gene symbols from 'gene_name' column")
        adata_full.var.index = adata_full.var['gene_name']
    elif 'gene_symbols' in adata_full.var.columns:
        print("  Setting gene symbols from 'gene_symbols' column")
        adata_full.var.index = adata_full.var['gene_symbols']

    # Get the intersection of genes between dataset and model
    print("  Finding gene intersection with SCimilarity model...")
    common_genes = adata_full.var.index.intersection(ca.gene_order)
    print(f"  ✓ Found {len(common_genes):,} common genes")
    
    if len(common_genes) < 5000:
        print(f"  ⚠ Warning: Only {len(common_genes)} common genes found.")
        print(f"    This may affect embedding quality.")
    
    # Reorder genes to match SCimilarity model order
    gene_order_dict = {gene: i for i, gene in enumerate(ca.gene_order)}
    common_genes_sorted = sorted(common_genes, key=lambda x: gene_order_dict[x])
    
    # Subset to common genes BEFORE copying to save memory
    print(f"  Subsetting to common genes...")
    adata_subset = adata_full[:, common_genes_sorted].copy()
    print(f"  ✓ Subsetted dataset: {adata_subset.shape}")
    
    # Free memory immediately
    del adata_full
    import gc
    gc.collect()

    # Compute embeddings in batches for memory efficiency
    print("  Computing embeddings in batches...")
    
    n_cells = adata_subset.n_obs
    embeddings_list = []
    
    for start_idx in range(0, n_cells, batch_size):
        end_idx = min(start_idx + batch_size, n_cells)
        batch_num = start_idx // batch_size + 1
        total_batches = (n_cells + batch_size - 1) // batch_size
        
        print(f"    Batch {batch_num}/{total_batches}: cells {start_idx:,} to {end_idx:,}")
        
        try:
            # Get batch - work with a copy to avoid memory issues
            batch_adata = adata_subset[start_idx:end_idx].copy()
            
            # Align with model gene order (adds zeros for missing genes)
            print(f"      Aligning genes...", end='', flush=True)
            batch_aligned = align_dataset(batch_adata, ca.gene_order)
            print(f" {batch_aligned.shape}", flush=True)
            
            # Normalize
            print(f"      Normalizing...", end='', flush=True)
            batch_norm = lognorm_counts(batch_aligned)
            print(" done", flush=True)
            
            # Compute embeddings for batch
            print(f"      Computing embeddings...", end='', flush=True)
            batch_embeddings = ca.get_embeddings(batch_norm.X)
            print(f" {batch_embeddings.shape}", flush=True)
            
            embeddings_list.append(batch_embeddings)
            
            # Free memory aggressively
            del batch_adata, batch_aligned, batch_norm, batch_embeddings
            gc.collect()
            
        except MemoryError as e:
            print(f"\n    ⚠ Memory error in batch {batch_num}")
            print(f"    Try reducing SCIMILARITY_BATCH_SIZE in run_evaluation.py")
            print(f"    Current: {batch_size}, suggested: {batch_size // 2}")
            raise MemoryError(f"Out of memory processing batch {batch_num}") from e
        except Exception as e:
            print(f"\n    ✗ Error in batch {batch_num}: {e}")
            raise RuntimeError(f"Failed to compute embeddings for batch {batch_num}: {e}")
    
    # Concatenate all batches
    print("  Concatenating batch results...")
    embeddings = np.vstack(embeddings_list)
    print(f"  ✓ Embeddings computed: {embeddings.shape}")
    
    # Clean up
    del embeddings_list, adata_subset
    gc.collect()

    # Add to original object
    adata.obsm['X_scimilarity'] = embeddings

    print(f"✓ SCimilarity embeddings added: {adata.obsm['X_scimilarity'].shape}")

    return adata


def run_scib_benchmark(
    adata: ad.AnnData,
    batch_key: str,
    label_key: str,
    embedding_key: str,
    output_dir: str = "results",
    n_jobs: int = 8
) -> pd.DataFrame:
    """
    Run scIB benchmarking on a single embedding.
    Matches the original AML Atlas benchmarking setup.

    Args:
        adata: AnnData object with embeddings
        batch_key: Key in .obs for batch information
        label_key: Key in .obs for cell type labels
        embedding_key: Key in .obsm for embedding to evaluate
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame with scIB metrics
    """
    print(f"\n{'='*80}")
    print(f"Running scIB benchmark for: {embedding_key}")
    print(f"{'='*80}")

    # Verify embedding exists and is valid
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")
    
    embedding = adata.obsm[embedding_key]
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Dataset size: {adata.n_obs:,} cells")
    
    if adata.n_obs == 0:
        raise ValueError("Dataset is empty!")
    
    if embedding.shape[0] == 0:
        raise ValueError(f"Embedding '{embedding_key}' is empty!")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure bio conservation metrics (matching original)
    biocons = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=False
    )

    # Run benchmarker
    start = time.time()
    try:
        # Suppress warnings about log2(large numbers)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                  message='.*divide by zero encountered in log2.*')
            
            bm = Benchmarker(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_obsm_keys=[embedding_key],
                pre_integrated_embedding_obsm_key="X_pca",
                bio_conservation_metrics=biocons,
                n_jobs=n_jobs,
            )

            bm.benchmark()

    except (ValueError, OverflowError) as e:
        if "infinity" in str(e).lower() or "overflow" in str(e).lower():
            print(f"\n⚠ Error with pynndescent neighbor computation: {e}")
            print("  This is likely due to a library version issue.")
            print("  The original AML Atlas used the same dataset size successfully.")
            print("\n  Suggested fixes:")
            print("  1. Update scib-metrics: pip install --upgrade scib-metrics")
            print("  2. Update pynndescent: pip install --upgrade pynndescent")
            print("  3. Try: pip install pynndescent==0.5.10 scib-metrics==0.4.1")
            raise RuntimeError(f"Neighbor computation failed. Try updating libraries.") from e
        else:
            raise
    except Exception as e:
        print(f"✗ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    end = time.time()
    elapsed = int(end - start)
    print(f"\n✓ Benchmark completed in {elapsed // 60} min {elapsed % 60} sec")

    # Get results
    df = bm.get_results(min_max_scale=False)

    # Rename index to be more descriptive
    benchmark_id = f"{batch_key}_{label_key}_{embedding_key}"
    df = df.rename(index={embedding_key: benchmark_id})

    # Save results
    output_file = os.path.join(output_dir, f"{benchmark_id}.csv")
    df.to_csv(output_file)
    print(f"✓ Results saved to: {output_file}")

    # Print summary
    print("\nMetrics Summary:")
    print("-" * 80)
    for col in df.columns:
        value = df.loc[benchmark_id, col]
        print(f"  {col:30s}: {value:.4f}")

    return df


def compare_methods(
    results_dict: dict,
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Compare multiple batch correction methods.

    Args:
        results_dict: Dictionary mapping method names to result DataFrames
        output_dir: Directory to save comparison results

    Returns:
        Combined DataFrame with all results
    """
    print(f"\n{'='*80}")
    print("Comparing batch correction methods")
    print(f"{'='*80}\n")

    # Combine all results
    all_results = []
    for method_name, df in results_dict.items():
        df_copy = df.copy()
        df_copy.index = [method_name]
        all_results.append(df_copy)

    combined = pd.concat(all_results)

    # Save combined results
    os.makedirs(output_dir, exist_ok=True)
    combined_file = os.path.join(output_dir, "combined_metrics.csv")
    combined.to_csv(combined_file)
    print(f"✓ Combined results saved to: {combined_file}")

    # Print comparison table
    print("\nComparison Table:")
    print("=" * 80)
    print(combined.to_string())

    # Calculate and display overall scores
    if 'Batch correction' in combined.columns and 'Bio conservation' in combined.columns:
        print("\n" + "=" * 80)
        print("Overall Scores:")
        print("=" * 80)
        for method in combined.index:
            batch_score = combined.loc[method, 'Batch correction']
            bio_score = combined.loc[method, 'Bio conservation']
            total_score = combined.loc[method, 'Total']
            print(f"{method:30s}: Batch={batch_score:.4f}, Bio={bio_score:.4f}, Total={total_score:.4f}")

    return combined

