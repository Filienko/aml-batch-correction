#!/usr/bin/env python
# coding: utf-8

"""
Batch Correction Evaluation Pipeline
Compares Uncorrected, Harmony, scVI, scANVI, and SCimilarity using scIB metrics

This script replicates the evaluation setup from the AML Atlas paper.
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


def prepare_uncorrected_embedding(adata: ad.AnnData, n_hvgs: int = 2000) -> ad.AnnData:
    """
    Prepare uncorrected PCA embedding for comparison.
    Matches the original AML Atlas preprocessing.

    Args:
        adata: AnnData object with raw counts in .layers['counts']
        n_hvgs: Number of highly variable genes to use

    Returns:
        AnnData object with uncorrected PCA in .obsm['X_pca']
    """
    print(f"\nPreparing uncorrected embedding with {n_hvgs} HVGs...")

    # Work on a copy
    adata_work = adata.copy()

    # Normalize and log-transform (matching original: target_sum=1e4)
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)

    # Identify highly variable genes (matching original parameters)
    sc.pp.highly_variable_genes(
        adata_work,
        n_top_genes=n_hvgs,
        flavor='seurat_v3',
        layer='counts',
        batch_key='Sample',
        subset=False,
        span=0.8
    )

    # Subset to HVGs and compute PCA
    adata_hvg = adata_work[:, adata_work.var['highly_variable']].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver='arpack')

    # Transfer PCA back to original object
    adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
    adata.obsm['X_uncorrected'] = adata_hvg.obsm['X_pca']  # For benchmarking

    print(f"✓ Uncorrected PCA computed: {adata.obsm['X_pca'].shape}")

    return adata


def compute_harmony_embedding(
    adata: ad.AnnData,
    batch_key: str = "Sample"
) -> ad.AnnData:
    """
    Compute Harmony batch correction embeddings using scib.ig.harmony.
    Matches the original AML Atlas implementation.

    Args:
        adata: AnnData object with PCA in .obsm['X_pca']
        batch_key: Key in .obs for batch information

    Returns:
        AnnData object with Harmony embeddings in .obsm['X_harmony']
    """
    print(f"\nComputing Harmony embeddings...")
    print(f"  Batch key: {batch_key}")

    if not SCIB_AVAILABLE:
        raise ImportError("Harmony requires scib. Install with: pip install scib")

    # Check if PCA exists
    if 'X_pca' not in adata.obsm:
        print("  No PCA found, computing PCA first...")
        adata = prepare_uncorrected_embedding(adata, n_hvgs=2000)

    # Run Harmony using scib implementation (matches original)
    print("  Running Harmony integration...")
    try:
        scib.ig.harmony(adata, batch=batch_key)
        print(f"  ✓ Harmony completed")
    except Exception as e:
        raise RuntimeError(f"Failed to run Harmony: {e}")

    # scib.ig.harmony stores result in .obsm['X_emb']
    # Transfer to .obsm['X_harmony'] for consistency
    adata.obsm['X_harmony'] = adata.obsm['X_emb'].copy()

    print(f"✓ Harmony embeddings added: {adata.obsm['X_harmony'].shape}")

    return adata


def train_scvi_model(
    adata: ad.AnnData,
    batch_key: str = "Sample",
    n_layers: int = 2,
    n_latent: int = 30,
    save_dir: Optional[str] = None
) -> ad.AnnData:
    """
    Train scVI model and compute embeddings.
    Matches the original AML Atlas implementation.

    Args:
        adata: AnnData object with raw counts in .layers['counts']
        batch_key: Key in .obs for batch information
        n_layers: Number of layers in VAE
        n_latent: Latent dimensionality
        save_dir: Directory to save model (optional)

    Returns:
        AnnData object with scVI embeddings in .obsm['X_scVI']
    """
    print(f"\nTraining scVI model...")
    print(f"  Batch key: {batch_key}")
    print(f"  Data shape: {adata.n_obs} cells × {adata.n_vars} genes")

    if not SCVI_AVAILABLE:
        raise ImportError("scVI requires scvi-tools. Install with: pip install scvi-tools")

    # Check for raw counts
    if 'counts' not in adata.layers:
        raise ValueError("scVI requires raw counts in .layers['counts']")

    # Setup anndata for scVI (matching original)
    print("  Setting up AnnData for scVI...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)

    # Train scVI model (matching original parameters)
    print("  Training scVI...")
    vae = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood="nb")
    vae.train()
    print("  ✓ scVI training complete")

    # Save model if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"scvi_{batch_key}_model")
        vae.save(model_path, overwrite=True)
        print(f"  ✓ Model saved to: {model_path}")

    # Get latent representation
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    print(f"✓ scVI embeddings added: {adata.obsm['X_scVI'].shape}")

    return adata


def load_scvi_embedding(
    adata: ad.AnnData,
    scvi_path: str = "data/AML_scAtlas_X_scVI.h5ad"
) -> ad.AnnData:
    """
    Load pre-computed scVI embeddings and add to adata object.

    Args:
        adata: Original AnnData object
        scvi_path: Path to scVI-corrected embeddings

    Returns:
        AnnData object with scVI embeddings in .obsm['X_scVI']
    """
    print(f"\nLoading scVI embeddings from {scvi_path}...")

    if not os.path.exists(scvi_path):
        raise FileNotFoundError(f"scVI embedding file not found: {scvi_path}")

    # Load scVI embeddings
    adata_scvi = sc.read_h5ad(scvi_path)

    # Verify cell alignment
    if not all(adata.obs_names == adata_scvi.obs_names):
        print("Warning: Cell names don't match. Attempting to align...")
        common_cells = adata.obs_names.intersection(adata_scvi.obs_names)
        adata = adata[common_cells].copy()
        adata_scvi = adata_scvi[common_cells].copy()
        print(f"  Aligned to {len(common_cells)} common cells")

    # Transfer scVI embeddings
    adata.obsm['X_scVI'] = adata_scvi.X.copy()

    print(f"✓ scVI embeddings loaded: {adata.obsm['X_scVI'].shape}")

    return adata


def train_scanvi_model(
    adata: ad.AnnData,
    batch_key: str = "Sample",
    label_key: str = "Cell Type",
    scvi_model_path: Optional[str] = None,
    n_layers: int = 2,
    n_latent: int = 30,
    max_epochs: int = 20,
    save_dir: Optional[str] = None
) -> ad.AnnData:
    """
    Train scANVI (semi-supervised scVI) model.
    Matches the original AML Atlas implementation.

    Args:
        adata: AnnData object with raw counts in .layers['counts']
        batch_key: Key in .obs for batch information
        label_key: Key in .obs for cell type labels
        scvi_model_path: Path to pre-trained scVI model (optional)
        n_layers: Number of layers in VAE
        n_latent: Latent dimensionality
        max_epochs: Maximum training epochs
        save_dir: Directory to save model (optional)

    Returns:
        AnnData object with scANVI embeddings in .obsm['X_scANVI']
    """
    print(f"\nTraining scANVI model...")
    print(f"  Batch key: {batch_key}")
    print(f"  Label key: {label_key}")

    if not SCVI_AVAILABLE:
        raise ImportError("scANVI requires scvi-tools. Install with: pip install scvi-tools")

    # Check for raw counts
    if 'counts' not in adata.layers:
        raise ValueError("scANVI requires raw counts in .layers['counts']")

    # Setup anndata for scVI
    print("  Setting up AnnData for scVI...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)

    # Train or load scVI model
    if scvi_model_path and os.path.exists(scvi_model_path):
        print(f"  Loading pre-trained scVI model from {scvi_model_path}...")
        vae = scvi.model.SCVI.load(scvi_model_path, adata)
    else:
        print("  Training scVI model first...")
        vae = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood="nb")
        vae.train()
        print("  ✓ scVI training complete")

    # Train scANVI from scVI model (matching original implementation)
    print("  Training scANVI model...")
    try:
        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata,
            labels_key=label_key,
            unlabeled_category="Unknown",
        )

        lvae.train(max_epochs=max_epochs, n_samples_per_label=100)
        print("  ✓ scANVI training complete")
    except Exception as e:
        raise RuntimeError(f"Failed to train scANVI: {e}")

    # Save model if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"scanvi_{batch_key}_model")
        lvae.save(model_path, overwrite=True)
        print(f"  ✓ Model saved to: {model_path}")

    # Get latent representation
    adata.obsm['X_scANVI'] = lvae.get_latent_representation(adata)

    # Store predicted annotations
    adata.obs['scANVI_annotations'] = lvae.predict(adata)

    print(f"✓ scANVI embeddings added: {adata.obsm['X_scANVI'].shape}")

    return adata


def compute_scimilarity_embedding(
    adata: ad.AnnData,
    model_path: str = "model_v1.1",
    use_full_gene_set: bool = True,
    batch_size: int = 10000
) -> ad.AnnData:
    """
    Compute SCimilarity embeddings for batch correction evaluation.
    Uses batched processing for memory efficiency with large datasets.

    Args:
        adata: AnnData object with raw counts in .layers['counts']
        model_path: Path to SCimilarity model
        use_full_gene_set: Whether to use full gene set (recommended)
        batch_size: Number of cells to process at once (reduce if OOM)

    Returns:
        AnnData object with SCimilarity embeddings in .obsm['X_scimilarity']
    """
    print(f"\nComputing SCimilarity embeddings...")
    print(f"  Dataset size: {adata.n_obs:,} cells")
    
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
    if use_full_gene_set:
        # Use full gene set (recommended to avoid gene overlap issues)
        if adata.raw is None:
            print("  No .raw attribute found, using main object")
            adata_full = adata.copy()
        else:
            print("  Using .raw attribute for full gene set")
            adata_full = adata.raw.to_adata()
    else:
        if 'highly_variable' in adata.var.columns:
            print(f"  Using 'highly_variable' genes (use_full_gene_set=False)")
            adata_full = adata[:, adata.var['highly_variable']].copy()
            print(f"  ✓ Subsetted to {adata_full.shape[1]} HVGs")
        else:
            adata_full = adata.copy()
    # Ensure gene symbols are in index
    if 'gene_name' in adata_full.var.columns:
        print("  Setting gene symbols from 'gene_name' column")
        adata_full.var.index = adata_full.var['gene_name']
    elif 'gene_symbols' in adata_full.var.columns:
        print("  Setting gene symbols from 'gene_symbols' column")
        adata_full.var.index = adata_full.var['gene_symbols']

    # Store counts for SCimilarity
    #if 'counts' in adata_full.layers:
    #    print("  Using counts from .layers['counts']")
    #    adata_full.X = adata_full.layers['counts'].copy()

    # Align with SCimilarity gene order
    print("  Aligning genes with SCimilarity model...")
    try:
        adata_scim = align_dataset(adata_full, ca.gene_order)
        adata_scim = lognorm_counts(adata_scim)
        print(f"  ✓ Aligned to {adata_scim.shape[1]} genes")
    except Exception as e:
        raise RuntimeError(f"Failed to align dataset: {e}")

    # Compute embeddings in batches for memory efficiency
    print("  Computing embeddings in batches...")
    
    n_cells = adata_scim.n_obs
    embeddings_list = []
    
    import gc
    
    for start_idx in range(0, n_cells, batch_size):
        end_idx = min(start_idx + batch_size, n_cells)
        batch_num = start_idx // batch_size + 1
        total_batches = (n_cells + batch_size - 1) // batch_size
        
        print(f"    Batch {batch_num}/{total_batches}: cells {start_idx:,} to {end_idx:,}")
        
        try:
            # Get batch data
            batch_data = adata_scim.X[start_idx:end_idx]
            
            # Compute embeddings for batch
            batch_embeddings = ca.get_embeddings(batch_data)
            embeddings_list.append(batch_embeddings)
            
            # Free memory
            del batch_data
            gc.collect()
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute embeddings for batch {batch_num}: {e}")
    
    # Concatenate all batches
    print("  Concatenating batch results...")
    embeddings = np.vstack(embeddings_list)
    print(f"  ✓ Embeddings computed: {embeddings.shape}")
    
    # Clean up
    del embeddings_list, adata_scim, adata_full
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

    # Create visualizations
    create_comparison_plots(combined, output_dir)

    return combined


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """
    Create comparison plots for batch correction metrics.

    Args:
        df: DataFrame with metrics for all methods
        output_dir: Directory to save plots
    """
    print(f"\nCreating comparison plots...")

    # 1. Batch correction vs Bio conservation scatter plot
    if 'Batch correction' in df.columns and 'Bio conservation' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        scatter = ax.scatter(
            df['Batch correction'],
            df['Bio conservation'],
            c=range(len(df)),
            s=200,
            alpha=0.6,
            cmap='viridis'
        )

        # Add method labels
        for method, row in df.iterrows():
            ax.annotate(
                method,
                (row['Batch correction'], row['Bio conservation']),
                xytext=(10, -5),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

        ax.set_xlabel('Batch Correction Score', fontsize=12)
        ax.set_ylabel('Bio Conservation Score', fontsize=12)
        ax.set_title('Batch Correction vs Biological Conservation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_file = os.path.join(output_dir, "batch_vs_bio_scatter.png")
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {scatter_file}")
        plt.close()

    # 2. Radar plot for all metrics
    create_radar_plot(df, output_dir)

    # 3. Heatmap of all metrics
    create_heatmap(df, output_dir)


def create_radar_plot(df: pd.DataFrame, output_dir: str):
    """Create radar plot comparing all metrics across methods."""
    import numpy as np

    # Select key metrics for radar plot
    metrics = ['Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI',
               'KBET', 'Graph connectivity', 'PCR comparison']

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 3:
        print("  ⚠ Not enough metrics for radar plot")
        return

    # Number of variables
    num_vars = len(available_metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each method
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for idx, (method, row) in enumerate(df.iterrows()):
        values = row[available_metrics].tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, size=10)

    # Set y-limits
    ax.set_ylim(0, 1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.title('Batch Correction Metrics Comparison', size=14, fontweight='bold', pad=20)

    radar_file = os.path.join(output_dir, "metrics_radar_plot.png")
    plt.savefig(radar_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {radar_file}")
    plt.close()


def create_heatmap(df: pd.DataFrame, output_dir: str):
    """Create heatmap of all metrics."""
    import seaborn as sns

    # Normalize metrics to 0-1 scale for visualization
    df_norm = (df - df.min()) / (df.max() - df.min())

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))

    sns.heatmap(
        df_norm,
        annot=df,  # Show actual values
        fmt='.3f',
        cmap='Blues',
        cbar_kws={'label': 'Normalized Score'},
        linewidths=0.5,
        ax=ax
    )

    plt.title('Batch Correction Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Methods', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    heatmap_file = os.path.join(output_dir, "metrics_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {heatmap_file}")
    plt.close()
