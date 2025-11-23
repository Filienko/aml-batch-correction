#!/usr/bin/env python
"""
Phase 2: SCimilarity Experiment (The Foundation Model Solution)

This script:
1. Loads the raw, unintegrated dataset from Phase 1
2. Projects every cell into the SCimilarity latent space (NO training, just inference)
3. Generates UMAP from SCimilarity embeddings
4. Visualizes the FM solution (Figure 1C)
5. Compares with ground truth atlas

Goal: Show that a foundation model can automatically solve batch effects
      without the complex pipeline used in the atlas
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

# Suppress warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(8, 8))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
DATA_DIR = Path("results_atlas_replication/data")
RAW_PROBLEM_PATH = DATA_DIR / "merged_raw_problem.h5ad"

# Output
OUTPUT_DIR = Path("results_atlas_replication")
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_OUT_DIR = OUTPUT_DIR / "data"

# SCimilarity
SCIMILARITY_MODEL_PATH = "models/model_v1.1"
SCIMILARITY_BATCH_SIZE = 5000  # Process cells in batches to avoid OOM

# Create directories
for dir_path in [OUTPUT_DIR, FIGURES_DIR, DATA_OUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SCIMILARITY PROJECTION
# ==============================================================================

def project_to_scimilarity(adata, model_path, batch_size=5000):
    """
    Project raw data into SCimilarity latent space.

    Key points:
    - NO training or fine-tuning
    - Just inference using pre-trained model
    - This is the "foundation model" approach

    Args:
        adata: Raw AnnData object
        model_path: Path to SCimilarity model
        batch_size: Number of cells to process at once

    Returns:
        adata with SCimilarity embeddings in .obsm['X_scimilarity']
    """
    print("=" * 80)
    print("PROJECTING TO SCIMILARITY LATENT SPACE")
    print("=" * 80)

    print(f"\nInput dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Import SCimilarity
    try:
        from scimilarity import CellAnnotation
        from scimilarity.utils import lognorm_counts, align_dataset
    except ImportError:
        raise ImportError(
            "SCimilarity not installed. Install with: pip install scimilarity"
        )

    # Load model
    print(f"\nLoading SCimilarity model from: {model_path}")
    ca = CellAnnotation(model_path=model_path)
    print(f"✓ Model loaded")
    print(f"  Supports {len(ca.gene_order):,} genes")
    print(f"  Embedding dimension: {ca.embedding_size}")

    # Prepare data
    print("\nPreparing data for SCimilarity...")

    # Ensure we have raw counts
    if 'counts' in adata.layers:
        print("  Using .layers['counts']")
        adata_work = adata.copy()
        adata_work.X = adata_work.layers['counts'].copy()
    elif adata.X.max() > 100:
        print("  Assuming .X contains raw counts (max value > 100)")
        adata_work = adata.copy()
    else:
        print("  ⚠ Warning: .X appears normalized (max < 100)")
        print("  Proceeding anyway...")
        adata_work = adata.copy()

    # Ensure gene symbols in index
    if 'gene_name' in adata_work.var.columns:
        print("  Setting gene symbols from 'gene_name'")
        adata_work.var.index = adata_work.var['gene_name']
    elif 'gene_symbols' in adata_work.var.columns:
        print("  Setting gene symbols from 'gene_symbols'")
        adata_work.var.index = adata_work.var['gene_symbols']
    else:
        print("  Using current var.index as gene symbols")

    # Find gene intersection
    print(f"\nFinding gene intersection with SCimilarity...")
    common_genes = adata_work.var.index.intersection(ca.gene_order)
    print(f"✓ Found {len(common_genes):,} / {len(ca.gene_order):,} genes")

    if len(common_genes) < 5000:
        print(f"  ⚠ Warning: Only {len(common_genes):,} common genes")
        print(f"  Embedding quality may be reduced")

    # Subset to common genes (preserving model order)
    gene_order_dict = {gene: i for i, gene in enumerate(ca.gene_order)}
    common_genes_sorted = sorted(common_genes, key=lambda x: gene_order_dict[x])

    adata_subset = adata_work[:, common_genes_sorted].copy()
    print(f"✓ Subset dataset: {adata_subset.shape}")

    # Free memory
    del adata_work
    gc.collect()

    # Compute embeddings in batches
    print(f"\nComputing embeddings (batch size: {batch_size:,} cells)...")

    n_cells = adata_subset.n_obs
    n_batches = (n_cells + batch_size - 1) // batch_size
    embeddings_list = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_cells)

        print(f"  Batch {batch_idx + 1}/{n_batches}: cells {start_idx:,} to {end_idx:,}")

        # Get batch
        batch_adata = adata_subset[start_idx:end_idx].copy()

        # Align with SCimilarity gene order (adds zeros for missing genes)
        batch_aligned = align_dataset(batch_adata, ca.gene_order)
        print(f"    Aligned: {batch_aligned.shape}")

        # Ensure counts layer
        if 'counts' not in batch_aligned.layers:
            batch_aligned.layers['counts'] = batch_aligned.X.copy()

        # Normalize
        batch_norm = lognorm_counts(batch_aligned)

        # Compute embeddings
        batch_embeddings = ca.get_embeddings(batch_norm.X)
        print(f"    Embeddings: {batch_embeddings.shape}")

        embeddings_list.append(batch_embeddings)

        # Free memory
        del batch_adata, batch_aligned, batch_norm, batch_embeddings
        gc.collect()

    # Concatenate
    print("\nConcatenating batch results...")
    embeddings = np.vstack(embeddings_list)
    print(f"✓ Final embeddings: {embeddings.shape}")

    # Add to original object
    adata.obsm['X_scimilarity'] = embeddings

    # Clean up
    del embeddings_list, adata_subset
    gc.collect()

    print(f"\n✓ SCimilarity projection complete!")
    print(f"  Embeddings stored in .obsm['X_scimilarity']")

    return adata


def compute_scimilarity_umap(adata):
    """
    Compute UMAP from SCimilarity embeddings.
    """
    print("\n" + "=" * 80)
    print("COMPUTING UMAP FROM SCIMILARITY EMBEDDINGS")
    print("=" * 80)

    if 'X_scimilarity' not in adata.obsm:
        raise ValueError("No SCimilarity embeddings found! Run projection first.")

    print(f"\nComputing neighbors in SCimilarity space...")
    sc.pp.neighbors(adata, use_rep='X_scimilarity', n_neighbors=15)

    print("Computing UMAP...")
    sc.tl.umap(adata)

    print(f"✓ UMAP computed: {adata.obsm['X_umap'].shape}")

    return adata


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_scimilarity_solution(adata, batch_key, label_key, output_dir):
    """
    Visualize the SCimilarity solution (Figure 1C).

    This should show:
    - Cells clustered by BIOLOGY (like the ground truth)
    - NOT by batch (batch correction achieved)
    - WITHOUT any of the manual work!
    """
    print("\n" + "=" * 80)
    print("VISUALIZING SCIMILARITY SOLUTION (Figure 1C)")
    print("=" * 80)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Color by cell type (should be clustered)
    if label_key in adata.obs.columns and adata.obs[label_key].notna().any():
        sc.pl.umap(
            adata,
            color=label_key,
            ax=axes[0],
            show=False,
            title="SCimilarity Solution: Cell Types (Biology Preserved!)",
            legend_loc='right margin',
            legend_fontsize=8,
            frameon=False
        )
    else:
        # Compute Leiden clusters as proxy
        print("  No cell type labels, computing Leiden clusters...")
        sc.tl.leiden(adata, resolution=0.5)
        sc.pl.umap(
            adata,
            color='leiden',
            ax=axes[0],
            show=False,
            title="SCimilarity Solution: Leiden Clusters",
            legend_loc='right margin',
            legend_fontsize=8,
            frameon=False
        )

    # Panel 2: Color by batch (should be MIXED)
    sc.pl.umap(
        adata,
        color=batch_key,
        ax=axes[1],
        show=False,
        title="SCimilarity Solution: Batches (Automatically Mixed!)",
        legend_loc='right margin',
        legend_fontsize=8,
        frameon=False
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / "fig1c_scimilarity_solution.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_file}")
    plt.close()


def create_comparison_figure(adata_raw, adata_scim, batch_key, label_key, output_dir):
    """
    Create a side-by-side comparison:
    - Before SCimilarity (raw problem)
    - After SCimilarity (FM solution)
    """
    print("\n" + "=" * 80)
    print("CREATING BEFORE/AFTER COMPARISON")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Row 1: Raw Problem
    # Compute UMAP for raw if needed
    if 'X_umap' not in adata_raw.obsm:
        print("\nComputing UMAP for raw data...")
        adata_raw_viz = adata_raw.copy()
        sc.pp.normalize_total(adata_raw_viz, target_sum=1e4)
        sc.pp.log1p(adata_raw_viz)
        sc.pp.highly_variable_genes(adata_raw_viz, n_top_genes=2000,
                                     batch_key=batch_key, subset=True)
        sc.tl.pca(adata_raw_viz)
        sc.pp.neighbors(adata_raw_viz, use_rep='X_pca')
        sc.tl.umap(adata_raw_viz)
        adata_raw.obsm['X_umap'] = adata_raw_viz.obsm['X_umap'].copy()
        del adata_raw_viz
        gc.collect()

    # Raw: Batches
    sc.pl.umap(
        adata_raw,
        color=batch_key,
        ax=axes[0, 0],
        show=False,
        title="BEFORE: Raw Data - Batches (Separated)",
        legend_loc='right margin',
        legend_fontsize=8
    )

    # Raw: Cell types (if available)
    if label_key in adata_raw.obs.columns and adata_raw.obs[label_key].notna().any():
        sc.pl.umap(
            adata_raw,
            color=label_key,
            ax=axes[0, 1],
            show=False,
            title="BEFORE: Raw Data - Cell Types (Obscured)",
            legend_loc='right margin',
            legend_fontsize=8
        )
    else:
        axes[0, 1].text(0.5, 0.5, 'No cell type labels\navailable',
                       ha='center', va='center', fontsize=14)
        axes[0, 1].set_title("BEFORE: No Labels")

    # Row 2: SCimilarity Solution
    # SCimilarity: Batches
    sc.pl.umap(
        adata_scim,
        color=batch_key,
        ax=axes[1, 0],
        show=False,
        title="AFTER: SCimilarity - Batches (Mixed!)",
        legend_loc='right margin',
        legend_fontsize=8
    )

    # SCimilarity: Cell types (if available)
    if label_key in adata_scim.obs.columns and adata_scim.obs[label_key].notna().any():
        sc.pl.umap(
            adata_scim,
            color=label_key,
            ax=axes[1, 1],
            show=False,
            title="AFTER: SCimilarity - Cell Types (Preserved!)",
            legend_loc='right margin',
            legend_fontsize=8
        )
    else:
        # Use Leiden
        if 'leiden' not in adata_scim.obs.columns:
            sc.tl.leiden(adata_scim, resolution=0.5)
        sc.pl.umap(
            adata_scim,
            color='leiden',
            ax=axes[1, 1],
            show=False,
            title="AFTER: SCimilarity - Leiden Clusters",
            legend_loc='right margin',
            legend_fontsize=8
        )

    plt.suptitle(
        "Foundation Model Batch Correction: Before vs After",
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout()

    # Save
    output_file = output_dir / "fig_comparison_before_after.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison figure saved: {output_file}")
    plt.close()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """
    Main execution pipeline for Phase 2.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: SCIMILARITY EXPERIMENT (FM SOLUTION)")
    print("=" * 80)
    print("\nGoal: Show foundation model can automatically solve batch effects\n")

    # Load raw problem from Phase 1
    print("=" * 80)
    print("LOADING RAW PROBLEM DATASET")
    print("=" * 80)

    if not RAW_PROBLEM_PATH.exists():
        raise FileNotFoundError(
            f"\n✗ Raw problem file not found: {RAW_PROBLEM_PATH}\n"
            f"  Run Phase 1 first: python phase1_ground_truth.py"
        )

    print(f"\nLoading from: {RAW_PROBLEM_PATH}")
    adata_raw = sc.read_h5ad(RAW_PROBLEM_PATH)
    print(f"✓ Loaded: {adata_raw.n_obs:,} cells × {adata_raw.n_vars:,} genes")

    # Detect keys
    batch_key_candidates = ['Study', 'study', 'batch', 'Batch', 'dataset_of_origin']
    batch_key = next((k for k in batch_key_candidates if k in adata_raw.obs.columns), None)

    label_key_candidates = ['celltype', 'CellType', 'cell_type', 'cell_type_annotation']
    label_key = next((k for k in label_key_candidates if k in adata_raw.obs.columns), None)

    if batch_key is None:
        print("\n⚠ Could not detect batch key")
        print(f"Available: {adata_raw.obs.columns.tolist()}")
        batch_key = input("Enter batch column name: ")

    print(f"\nBatch key: '{batch_key}'")
    print(f"Label key: '{label_key}' {'(found)' if label_key else '(not found)'}")

    # Project to SCimilarity
    adata_scim = project_to_scimilarity(
        adata_raw,
        SCIMILARITY_MODEL_PATH,
        batch_size=SCIMILARITY_BATCH_SIZE
    )

    # Compute UMAP
    adata_scim = compute_scimilarity_umap(adata_scim)

    # Visualize
    visualize_scimilarity_solution(adata_scim, batch_key, label_key, FIGURES_DIR)

    # Create before/after comparison
    create_comparison_figure(adata_raw, adata_scim, batch_key, label_key, FIGURES_DIR)

    # Save result
    output_file = DATA_OUT_DIR / "scimilarity_solution.h5ad"
    adata_scim.write(output_file)
    print(f"\n✓ SCimilarity solution saved: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"\n✓ Processed: {adata_scim.n_obs:,} cells")
    print(f"✓ Embedding dimension: {adata_scim.obsm['X_scimilarity'].shape[1]}")

    print(f"\nOutputs:")
    print(f"  Figures:")
    print(f"    - {FIGURES_DIR / 'fig1c_scimilarity_solution.pdf'}")
    print(f"    - {FIGURES_DIR / 'fig_comparison_before_after.pdf'}")
    print(f"  Data:")
    print(f"    - {output_file}")

    print(f"\nNext steps:")
    print(f"  Run Phase 3: python phase3_quantitative_benchmark.py")

    return adata_scim, batch_key, label_key


if __name__ == "__main__":
    adata_scim, batch_key, label_key = main()
