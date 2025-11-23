#!/usr/bin/env python
"""
Phase 1: Data and Model Setup

This script:
1. Loads the fully processed AML scAtlas (159 patients) - the "Ground Truth"
2. Extracts key metadata (batch and biology labels)
3. Visualizes the atlas UMAP (Figure 1A - Gold Standard)
4. Creates the "raw problem" by merging unintegrated data
5. Visualizes the raw problem UMAP (Figure 1B - Massive batch effects)
6. Initializes SCimilarity foundation model

Goal: Establish the ground truth and demonstrate the problem that needs solving
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

# Suppress warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(8, 8))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
DATA_DIR = Path("data")
ATLAS_PATH = DATA_DIR / "AML_scAtlas.h5ad"
SCVI_PATH = DATA_DIR / "AML_scAtlas_X_scVI.h5ad"

# Raw data paths (if available separately)
RAW_DIR = DATA_DIR / "raw"
VAN_GALEN_RAW = RAW_DIR / "van_galen_2019_raw.h5ad"
ABBAS_RAW = RAW_DIR / "abbas_2021_raw.h5ad"
WANG_RAW = RAW_DIR / "wang_2024_raw.h5ad"

# Output
OUTPUT_DIR = Path("results_atlas_replication")
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_OUT_DIR = OUTPUT_DIR / "data"

# SCimilarity model
SCIMILARITY_MODEL_PATH = "models/model_v1.1"

# Create directories
for dir_path in [OUTPUT_DIR, FIGURES_DIR, DATA_OUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# STEP 1: LOAD GROUND TRUTH ATLAS
# ==============================================================================

def load_ground_truth_atlas(atlas_path):
    """
    Load the fully processed AML scAtlas (159 patients).

    This is our "Gold Standard" - the result of:
    - scVI integration
    - CellTypist + SingleR + scType consensus
    - Manual curation with marker genes
    - Custom LSC annotation

    Returns:
        adata: AnnData object with ground truth annotations
        batch_key: Name of the batch/study column
        label_key: Name of the cell type annotation column
    """
    print("=" * 80)
    print("STEP 1: LOADING GROUND TRUTH ATLAS")
    print("=" * 80)

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"\n✗ Atlas file not found: {atlas_path}\n"
            f"  Please download the AML scAtlas data.\n"
            f"  See DATA_SOURCES.md for instructions."
        )

    print(f"\nLoading atlas from: {atlas_path}")
    adata = sc.read_h5ad(atlas_path)
    print(f"✓ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Identify batch and label keys
    print("\nIdentifying metadata columns...")
    print(f"Available columns: {list(adata.obs.columns)}")

    # Common batch key names
    batch_key_candidates = [
        'dataset_of_origin', 'Study', 'study', 'batch', 'Batch',
        'dataset', 'Dataset', 'sample_origin'
    ]
    batch_key = next((k for k in batch_key_candidates if k in adata.obs.columns), None)

    if batch_key is None:
        print("\n⚠ Warning: Could not auto-detect batch key.")
        print("Available columns:", adata.obs.columns.tolist())
        batch_key = input("Enter the batch/study column name: ")

    print(f"✓ Batch key: '{batch_key}'")
    print(f"  Unique batches: {adata.obs[batch_key].nunique()}")
    print(f"  Batch names: {sorted(adata.obs[batch_key].unique())[:10]}...")

    # Common label key names
    label_key_candidates = [
        'cell_type_annotation', 'celltype', 'CellType', 'cell_type',
        'cell_annotation', 'annotation', 'Annotation', 'final_annotation'
    ]
    label_key = next((k for k in label_key_candidates if k in adata.obs.columns), None)

    if label_key is None:
        print("\n⚠ Warning: Could not auto-detect cell type label key.")
        print("Available columns:", adata.obs.columns.tolist())
        label_key = input("Enter the cell type annotation column name: ")

    print(f"✓ Label key: '{label_key}'")
    print(f"  Unique cell types: {adata.obs[label_key].nunique()}")
    print(f"  Cell type names: {sorted(adata.obs[label_key].unique())[:10]}...")

    # Check for UMAP
    if 'X_umap' in adata.obsm:
        print(f"✓ UMAP coordinates found: {adata.obsm['X_umap'].shape}")
    else:
        print("⚠ No pre-computed UMAP found, will compute later")

    # Check for scVI embedding
    if 'X_scVI' in adata.obsm:
        print(f"✓ scVI embedding found: {adata.obsm['X_scVI'].shape}")
    elif SCVI_PATH.exists():
        print(f"  Loading scVI from separate file: {SCVI_PATH}")
        adata_scvi = sc.read_h5ad(SCVI_PATH)
        if adata_scvi.n_obs == adata.n_obs:
            adata.obsm['X_scVI'] = adata_scvi.X.copy()
            print(f"  ✓ scVI embedding loaded: {adata.obsm['X_scVI'].shape}")
        else:
            print(f"  ✗ Cell count mismatch: {adata_scvi.n_obs} vs {adata.n_obs}")
    else:
        print("⚠ No scVI embedding found")

    return adata, batch_key, label_key


def visualize_ground_truth(adata, batch_key, label_key, output_dir):
    """
    Visualize the ground truth atlas UMAP (Figure 1A).

    This should show:
    - Cells clustered by BIOLOGY (cell_type_annotation)
    - NOT clustered by BATCH (dataset_of_origin)
    - This is what good batch correction looks like
    """
    print("\n" + "=" * 80)
    print("VISUALIZING GROUND TRUTH ATLAS (Figure 1A)")
    print("=" * 80)

    # Compute UMAP if not available
    if 'X_umap' not in adata.obsm:
        print("\nComputing UMAP from scVI embedding...")
        if 'X_scVI' in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=15)
        else:
            print("⚠ No scVI embedding, using PCA...")
            if 'X_pca' not in adata.obsm:
                sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15)

        sc.tl.umap(adata)
        print(f"✓ UMAP computed: {adata.obsm['X_umap'].shape}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Color by cell type (BIOLOGY)
    sc.pl.umap(
        adata,
        color=label_key,
        ax=axes[0],
        show=False,
        title="Ground Truth Atlas: Cell Types (Biology)",
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
        title="Ground Truth Atlas: Batches (Well-Mixed)",
        legend_loc='right margin',
        legend_fontsize=8,
        frameon=False
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / "fig1a_ground_truth_atlas.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_file}")
    plt.close()

    # Print statistics
    print("\nGround Truth Statistics:")
    print(f"  Total cells: {adata.n_obs:,}")
    print(f"  Number of batches: {adata.obs[batch_key].nunique()}")
    print(f"  Number of cell types: {adata.obs[label_key].nunique()}")
    print(f"\n  Cell type distribution:")
    celltype_counts = adata.obs[label_key].value_counts()
    for ct, count in celltype_counts.head(10).items():
        print(f"    {ct}: {count:,} ({count/adata.n_obs*100:.1f}%)")

    print(f"\n  Batch distribution:")
    batch_counts = adata.obs[batch_key].value_counts()
    for batch, count in batch_counts.head(10).items():
        print(f"    {batch}: {count:,} ({count/adata.n_obs*100:.1f}%)")


# ==============================================================================
# STEP 2: CREATE THE "RAW PROBLEM"
# ==============================================================================

def create_raw_problem(adata_atlas, batch_key, label_key, output_dir):
    """
    Create the "raw problem" dataset.

    Two approaches:
    1. If raw datasets available separately: Merge them without integration
    2. Otherwise: Use the atlas but "undo" the integration by using raw counts

    Goal: Show what the data looks like WITHOUT batch correction
    """
    print("\n" + "=" * 80)
    print("STEP 2: CREATING RAW PROBLEM DATASET")
    print("=" * 80)

    # Check if separate raw files exist
    raw_files = [VAN_GALEN_RAW, ABBAS_RAW, WANG_RAW]
    raw_exists = [f.exists() for f in raw_files]

    if any(raw_exists):
        print("\nFound raw dataset files:")
        for f, exists in zip(raw_files, raw_exists):
            print(f"  {f.name}: {'✓' if exists else '✗'}")

        # Load and merge raw datasets
        adata_raw = merge_raw_datasets(raw_files, batch_key, label_key)
    else:
        print("\nNo separate raw files found.")
        print("Extracting raw counts from atlas...")
        adata_raw = extract_raw_from_atlas(adata_atlas, batch_key, label_key)

    # Save raw problem dataset
    output_file = output_dir / "merged_raw_problem.h5ad"
    adata_raw.write(output_file)
    print(f"\n✓ Raw problem dataset saved: {output_file}")

    return adata_raw


def merge_raw_datasets(raw_files, batch_key, label_key):
    """
    Load and merge raw datasets without any integration.
    """
    print("\nMerging raw datasets...")

    datasets = []
    for file_path in raw_files:
        if not file_path.exists():
            continue

        print(f"  Loading {file_path.name}...")
        adata = sc.read_h5ad(file_path)

        # Add study name to obs
        study_name = file_path.stem.replace('_raw', '')
        adata.obs[batch_key] = study_name

        print(f"    {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        datasets.append(adata)

    if len(datasets) == 0:
        raise ValueError("No raw datasets found!")

    # Concatenate (this will handle gene alignment automatically)
    print(f"\nConcatenating {len(datasets)} datasets...")
    adata_merged = datasets[0].concatenate(
        datasets[1:],
        batch_key='_temp_batch',
        join='outer',  # Keep all genes
        fill_value=0
    )

    # Clean up batch column
    adata_merged.obs[batch_key] = adata_merged.obs['_temp_batch']
    adata_merged.obs.drop(columns=['_temp_batch'], inplace=True)

    print(f"✓ Merged dataset: {adata_merged.n_obs:,} cells × {adata_merged.n_vars:,} genes")

    return adata_merged


def extract_raw_from_atlas(adata_atlas, batch_key, label_key):
    """
    Extract raw counts from the atlas (undo normalization/integration).
    """
    print("\nExtracting raw counts from atlas...")

    # Try to get raw counts
    if adata_atlas.raw is not None:
        print("  Using .raw attribute")
        adata_raw = adata_atlas.raw.to_adata()
        # Transfer obs
        adata_raw.obs = adata_atlas.obs.copy()
    elif 'counts' in adata_atlas.layers:
        print("  Using .layers['counts']")
        adata_raw = adata_atlas.copy()
        adata_raw.X = adata_raw.layers['counts'].copy()
    else:
        print("  ⚠ Warning: No raw counts found, using current .X")
        print("  This may already be normalized!")
        adata_raw = adata_atlas.copy()

    print(f"✓ Raw dataset: {adata_raw.n_obs:,} cells × {adata_raw.n_vars:,} genes")

    return adata_raw


def visualize_raw_problem(adata_raw, batch_key, label_key, output_dir):
    """
    Visualize the raw problem UMAP (Figure 1B).

    This should show:
    - Cells clustered by BATCH (dataset_of_origin) - MASSIVE batch effects!
    - NOT by biology
    - This is the problem we need to solve
    """
    print("\n" + "=" * 80)
    print("VISUALIZING RAW PROBLEM (Figure 1B)")
    print("=" * 80)

    # Basic preprocessing for PCA
    print("\nPreprocessing for visualization...")
    adata_viz = adata_raw.copy()

    # Normalize
    sc.pp.normalize_total(adata_viz, target_sum=1e4)
    sc.pp.log1p(adata_viz)

    # HVGs
    sc.pp.highly_variable_genes(
        adata_viz,
        n_top_genes=2000,
        batch_key=batch_key,
        subset=True
    )

    # PCA
    sc.tl.pca(adata_viz, svd_solver='arpack')

    # Neighbors + UMAP
    sc.pp.neighbors(adata_viz, use_rep='X_pca')
    sc.tl.umap(adata_viz)

    print(f"✓ UMAP computed: {adata_viz.obsm['X_umap'].shape}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Color by batch (should be SEPARATED)
    sc.pl.umap(
        adata_viz,
        color=batch_key,
        ax=axes[0],
        show=False,
        title="Raw Problem: Batches (Massive Batch Effects!)",
        legend_loc='right margin',
        legend_fontsize=8,
        frameon=False
    )

    # Panel 2: Color by cell type (should be mixed/unclear)
    if label_key in adata_viz.obs.columns:
        sc.pl.umap(
            adata_viz,
            color=label_key,
            ax=axes[1],
            show=False,
            title="Raw Problem: Cell Types (Obscured by Batch)",
            legend_loc='right margin',
            legend_fontsize=8,
            frameon=False
        )
    else:
        axes[1].text(0.5, 0.5, 'No cell type labels\nin raw data',
                    ha='center', va='center', fontsize=14)
        axes[1].set_title("Raw Problem: No Labels Available")

    plt.tight_layout()

    # Save
    output_file = output_dir / "fig1b_raw_problem.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_file}")
    plt.close()


# ==============================================================================
# STEP 3: INITIALIZE SCIMILARITY
# ==============================================================================

def initialize_scimilarity(model_path):
    """
    Initialize the SCimilarity foundation model.

    This is a pre-trained model - we will NOT fine-tune it.
    We will just use it for inference (projection into latent space).
    """
    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZING SCIMILARITY FOUNDATION MODEL")
    print("=" * 80)

    try:
        from scimilarity import CellAnnotation

        print(f"\nLoading SCimilarity from: {model_path}")
        ca = CellAnnotation(model_path=model_path)
        print(f"✓ SCimilarity model loaded successfully")
        print(f"  Model supports {len(ca.gene_order):,} genes")
        print(f"  Embedding dimension: {ca.embedding_size}")

        return ca

    except ImportError:
        print("\n✗ Error: scimilarity package not installed")
        print("  Install with: pip install scimilarity")
        return None
    except Exception as e:
        print(f"\n✗ Error loading SCimilarity: {e}")
        print(f"  Model path: {model_path}")
        print(f"  Make sure the model files are downloaded")
        return None


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """
    Main execution pipeline for Phase 1.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: DATA AND MODEL SETUP")
    print("=" * 80)
    print("\nGoal: Establish ground truth and demonstrate the problem\n")

    # Step 1: Load ground truth
    adata_atlas, batch_key, label_key = load_ground_truth_atlas(ATLAS_PATH)

    # Visualize ground truth (Figure 1A)
    visualize_ground_truth(adata_atlas, batch_key, label_key, FIGURES_DIR)

    # Step 2: Create raw problem
    adata_raw = create_raw_problem(
        adata_atlas, batch_key, label_key, DATA_OUT_DIR
    )

    # Visualize raw problem (Figure 1B)
    visualize_raw_problem(adata_raw, batch_key, label_key, FIGURES_DIR)

    # Step 3: Initialize SCimilarity
    ca = initialize_scimilarity(SCIMILARITY_MODEL_PATH)

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print(f"\n✓ Ground truth atlas: {adata_atlas.n_obs:,} cells")
    print(f"✓ Raw problem dataset: {adata_raw.n_obs:,} cells")
    print(f"✓ SCimilarity model: {'Loaded' if ca else 'Failed'}")

    print(f"\nOutputs:")
    print(f"  Figures:")
    print(f"    - {FIGURES_DIR / 'fig1a_ground_truth_atlas.pdf'}")
    print(f"    - {FIGURES_DIR / 'fig1b_raw_problem.pdf'}")
    print(f"  Data:")
    print(f"    - {DATA_OUT_DIR / 'merged_raw_problem.h5ad'}")

    print(f"\nNext steps:")
    print(f"  Run Phase 2: python phase2_scimilarity_projection.py")

    return adata_atlas, adata_raw, batch_key, label_key, ca


if __name__ == "__main__":
    adata_atlas, adata_raw, batch_key, label_key, ca = main()
