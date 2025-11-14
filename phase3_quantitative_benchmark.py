#!/usr/bin/env python
"""
Phase 3: Quantitative Batch-Effect Benchmarking

This script quantitatively compares three conditions:
1. **Problem**: Raw, unintegrated data (massive batch effects)
2. **Gold Standard**: Expert-curated AML scAtlas (scVI + manual curation)
3. **FM Solution**: SCimilarity latent space (automated, no manual work)

Metrics:
- Batch Mixing: LISI, kBET (higher = better batch integration)
- Biology Conservation: ARI, NMI (higher = better preservation of cell types)

Key Hypothesis:
- FM Solution batch mixing ≥ Gold Standard
- FM Solution biology conservation > Gold Standard (!)
  → If true, suggests expert curation may have "over-corrected"
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
from typing import Dict, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results_atlas_replication/data")
ATLAS_PATH = DATA_DIR / "AML_scAtlas.h5ad"
RAW_PROBLEM_PATH = RESULTS_DIR / "merged_raw_problem.h5ad"
SCIM_SOLUTION_PATH = RESULTS_DIR / "scimilarity_solution.h5ad"

# Output
OUTPUT_DIR = Path("results_atlas_replication")
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create directories
for dir_path in [METRICS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# BATCH MIXING METRICS
# ==============================================================================

def compute_lisi_score(adata, batch_key, use_rep='X_pca', perplexity=30):
    """
    Compute Local Inverse Simpson's Index (LISI) for batch mixing.

    LISI measures local diversity of batches around each cell.
    - Perfect mixing = number of batches
    - No mixing = 1
    - Higher is better

    Args:
        adata: AnnData object with embeddings
        batch_key: Column in .obs for batch labels
        use_rep: Which embedding to use
        perplexity: Neighborhood size (default 30)

    Returns:
        median_lisi: Median LISI score across all cells
        lisi_scores: Per-cell LISI scores
    """
    print(f"\n  Computing LISI (batch mixing)...")

    try:
        import harmonypy as hm
        from scipy.spatial.distance import pdist, squareform

        # Get embedding
        if use_rep not in adata.obsm:
            raise ValueError(f"Embedding '{use_rep}' not found")

        X = adata.obsm[use_rep]
        batches = adata.obs[batch_key].values

        # Simple LISI implementation
        # For each cell, count how many different batches are in its neighborhood
        from sklearn.neighbors import NearestNeighbors

        n_neighbors = min(perplexity, len(adata) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        # Compute Simpson index for each cell's neighborhood
        lisi_scores = []
        for i in range(len(adata)):
            neighbor_batches = batches[indices[i]]
            # Count frequency of each batch
            unique, counts = np.unique(neighbor_batches, return_counts=True)
            # Simpson's index
            proportions = counts / counts.sum()
            simpson = np.sum(proportions ** 2)
            # Inverse Simpson (LISI)
            lisi = 1.0 / simpson if simpson > 0 else 1.0
            lisi_scores.append(lisi)

        lisi_scores = np.array(lisi_scores)
        median_lisi = np.median(lisi_scores)

        print(f"    Median LISI: {median_lisi:.3f} (max possible: {adata.obs[batch_key].nunique()})")

        return median_lisi, lisi_scores

    except ImportError:
        print("    ✗ harmonypy not available, using approximate LISI")
        # Fallback: use simple neighborhood diversity
        from sklearn.neighbors import NearestNeighbors

        X = adata.obsm[use_rep]
        batches = adata.obs[batch_key].values
        n_neighbors = min(30, len(adata) - 1)

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        # Count unique batches in neighborhood
        lisi_scores = []
        for i in range(len(adata)):
            neighbor_batches = batches[indices[i]]
            n_unique = len(np.unique(neighbor_batches))
            lisi_scores.append(n_unique)

        lisi_scores = np.array(lisi_scores)
        median_lisi = np.median(lisi_scores)

        print(f"    Approx LISI: {median_lisi:.3f}")
        return median_lisi, lisi_scores


def compute_kbet_acceptance(adata, batch_key, use_rep='X_pca', k=25):
    """
    Compute k-nearest neighbor Batch Effect Test (kBET) acceptance rate.

    kBET tests if batches are uniformly distributed in local neighborhoods.
    - Rejection rate close to 0 = good mixing
    - Rejection rate close to 1 = poor mixing
    We report 1 - rejection_rate (higher = better)

    Args:
        adata: AnnData object
        batch_key: Batch column
        use_rep: Embedding key
        k: Number of neighbors

    Returns:
        acceptance_rate: Fraction of neighborhoods with good batch mixing
    """
    print(f"\n  Computing kBET (batch mixing)...")

    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import chisquare

    X = adata.obsm[use_rep]
    batches = adata.obs[batch_key].values
    batch_categories = np.unique(batches)
    n_batches = len(batch_categories)

    # Expected batch frequencies (uniform)
    batch_freq = pd.Series(batches).value_counts(normalize=True)
    expected_freq = batch_freq.values

    # Compute k-NN
    k_actual = min(k, len(adata) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual)
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # Test each neighborhood
    accepted = 0
    total = 0

    # Sample neighborhoods for efficiency
    sample_size = min(1000, len(adata))
    sample_indices = np.random.choice(len(adata), sample_size, replace=False)

    for i in sample_indices:
        neighbor_batches = batches[indices[i]]

        # Observed frequencies
        observed = []
        for batch in batch_categories:
            count = np.sum(neighbor_batches == batch)
            observed.append(count)

        # Expected frequencies
        expected = expected_freq * k_actual

        # Chi-square test
        try:
            _, p_value = chisquare(observed, expected)
            if p_value > 0.05:  # Accept null hypothesis (uniform distribution)
                accepted += 1
        except:
            pass

        total += 1

    acceptance_rate = accepted / total if total > 0 else 0

    print(f"    kBET acceptance rate: {acceptance_rate:.3f} (higher = better)")

    return acceptance_rate


# ==============================================================================
# BIOLOGICAL CONSERVATION METRICS
# ==============================================================================

def compute_ari(adata, label_key, use_rep='X_pca', resolution=0.5):
    """
    Compute Adjusted Rand Index (ARI) between clusters and ground truth labels.

    ARI measures agreement between two partitions:
    - 1.0 = perfect agreement
    - 0.0 = random agreement
    - Higher = better preservation of biology

    Args:
        adata: AnnData object
        label_key: Ground truth cell type labels
        use_rep: Embedding for clustering
        resolution: Leiden resolution

    Returns:
        ari_score: ARI between Leiden clusters and true labels
    """
    print(f"\n  Computing ARI (biology conservation)...")

    if label_key not in adata.obs.columns:
        print(f"    ✗ Label key '{label_key}' not found")
        return np.nan

    from sklearn.metrics import adjusted_rand_score

    # Compute clusters if not available
    if 'leiden' not in adata.obs.columns:
        # Compute neighbors if needed
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, use_rep=use_rep)

        # Leiden clustering
        sc.tl.leiden(adata, resolution=resolution)

    # Compute ARI
    true_labels = adata.obs[label_key].values
    cluster_labels = adata.obs['leiden'].values

    # Remove NaN labels
    valid_mask = pd.notna(true_labels)
    true_labels = true_labels[valid_mask]
    cluster_labels = cluster_labels[valid_mask]

    if len(true_labels) == 0:
        print(f"    ✗ No valid labels found")
        return np.nan

    ari = adjusted_rand_score(true_labels, cluster_labels)

    print(f"    ARI: {ari:.3f} (1.0 = perfect agreement)")

    return ari


def compute_nmi(adata, label_key, use_rep='X_pca', resolution=0.5):
    """
    Compute Normalized Mutual Information (NMI).

    NMI measures information shared between clusters and true labels.
    - 1.0 = perfect information preservation
    - 0.0 = no shared information
    - Higher = better

    Args:
        adata: AnnData object
        label_key: True cell type labels
        use_rep: Embedding
        resolution: Clustering resolution

    Returns:
        nmi_score: NMI between clusters and true labels
    """
    print(f"\n  Computing NMI (biology conservation)...")

    if label_key not in adata.obs.columns:
        print(f"    ✗ Label key '{label_key}' not found")
        return np.nan

    from sklearn.metrics import normalized_mutual_info_score

    # Compute clusters if needed
    if 'leiden' not in adata.obs.columns:
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, use_rep=use_rep)
        sc.tl.leiden(adata, resolution=resolution)

    true_labels = adata.obs[label_key].values
    cluster_labels = adata.obs['leiden'].values

    # Remove NaN
    valid_mask = pd.notna(true_labels)
    true_labels = true_labels[valid_mask]
    cluster_labels = cluster_labels[valid_mask]

    if len(true_labels) == 0:
        return np.nan

    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    print(f"    NMI: {nmi:.3f} (1.0 = perfect information)")

    return nmi


def compute_cell_type_asw(adata, label_key, use_rep='X_pca'):
    """
    Compute Average Silhouette Width (ASW) for cell types.

    ASW measures how well cell types are separated in the embedding.
    - 1.0 = perfect separation
    - 0.0 = no separation
    - -1.0 = wrong separation

    Returns:
        asw: Average silhouette width
    """
    print(f"\n  Computing Cell Type ASW (biology conservation)...")

    if label_key not in adata.obs.columns:
        print(f"    ✗ Label key '{label_key}' not found")
        return np.nan

    from sklearn.metrics import silhouette_score

    X = adata.obsm[use_rep]
    labels = adata.obs[label_key].values

    # Remove NaN
    valid_mask = pd.notna(labels)
    X = X[valid_mask]
    labels = labels[valid_mask]

    if len(np.unique(labels)) < 2:
        print(f"    ✗ Need at least 2 cell types")
        return np.nan

    # Sample if too large
    if len(X) > 10000:
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        X = X[sample_idx]
        labels = labels[sample_idx]

    asw = silhouette_score(X, labels, sample_size=min(5000, len(X)))

    print(f"    ASW: {asw:.3f} (1.0 = perfect separation)")

    return asw


# ==============================================================================
# COMPREHENSIVE BENCHMARKING
# ==============================================================================

def benchmark_method(adata, batch_key, label_key, use_rep, method_name):
    """
    Run all metrics for one method.

    Returns:
        results: Dictionary of metric scores
    """
    print("=" * 80)
    print(f"BENCHMARKING: {method_name}")
    print("=" * 80)
    print(f"  Dataset: {adata.n_obs:,} cells")
    print(f"  Embedding: {use_rep}")
    print(f"  Embedding shape: {adata.obsm[use_rep].shape}")

    results = {
        'Method': method_name,
        'N_Cells': adata.n_obs,
        'N_Batches': adata.obs[batch_key].nunique(),
        'N_CellTypes': adata.obs[label_key].nunique() if label_key in adata.obs.columns else np.nan,
    }

    # Batch mixing metrics
    print("\nBATCH MIXING METRICS:")
    lisi, _ = compute_lisi_score(adata, batch_key, use_rep)
    kbet = compute_kbet_acceptance(adata, batch_key, use_rep)

    results['LISI'] = lisi
    results['kBET'] = kbet
    results['Batch_Mixing_Score'] = (lisi / adata.obs[batch_key].nunique() + kbet) / 2

    # Biology conservation metrics
    if label_key in adata.obs.columns:
        print("\nBIOLOGY CONSERVATION METRICS:")
        ari = compute_ari(adata, label_key, use_rep)
        nmi = compute_nmi(adata, label_key, use_rep)
        asw = compute_cell_type_asw(adata, label_key, use_rep)

        results['ARI'] = ari
        results['NMI'] = nmi
        results['ASW'] = asw
        results['Bio_Conservation_Score'] = np.nanmean([ari, nmi, asw])

        # Overall score
        results['Overall_Score'] = np.nanmean([
            results['Batch_Mixing_Score'],
            results['Bio_Conservation_Score']
        ])
    else:
        print("\n⚠ No cell type labels available, skipping bio conservation")
        results['ARI'] = np.nan
        results['NMI'] = np.nan
        results['ASW'] = np.nan
        results['Bio_Conservation_Score'] = np.nan
        results['Overall_Score'] = results['Batch_Mixing_Score']

    print("\n" + "=" * 80)
    print(f"RESULTS: {method_name}")
    print("=" * 80)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")

    return results


def visualize_comparison(results_df, output_dir):
    """
    Create visualization comparing all methods.
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    # Figure 1: Bar chart of main metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('Batch_Mixing_Score', 'Batch Mixing\n(Higher = Better)', axes[0, 0]),
        ('Bio_Conservation_Score', 'Biology Conservation\n(Higher = Better)', axes[0, 1]),
        ('Overall_Score', 'Overall Score\n(Higher = Better)', axes[1, 0]),
    ]

    for metric, title, ax in metrics_to_plot:
        if metric in results_df.columns:
            data = results_df[['Method', metric]].dropna()
            ax.barh(data['Method'], data[metric], color=['red', 'blue', 'green'][:len(data)])
            ax.set_xlabel('Score')
            ax.set_title(title, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)

    # Detailed metrics table in last subplot
    axes[1, 1].axis('off')
    table_data = results_df[['Method', 'LISI', 'kBET', 'ARI', 'NMI', 'ASW']].round(3)
    table = axes[1, 1].table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1, 1].set_title('Detailed Metrics', fontweight='bold', pad=20)

    plt.suptitle(
        'Quantitative Batch Correction Comparison\n'
        'Problem vs Gold Standard vs Foundation Model',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    # Save
    output_file = output_dir / "fig2_quantitative_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison figure saved: {output_file}")
    plt.close()

    # Figure 2: Heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    heatmap_data = results_df.set_index('Method')[
        ['LISI', 'kBET', 'ARI', 'NMI', 'ASW']
    ].T

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Score'},
        ax=ax
    )
    ax.set_title('Metrics Heatmap (All Methods)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Metric')

    plt.tight_layout()
    output_file = output_dir / "fig2_metrics_heatmap.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved: {output_file}")
    plt.close()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """
    Main benchmarking pipeline.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: QUANTITATIVE BATCH-EFFECT BENCHMARKING")
    print("=" * 80)
    print("\nGoal: Quantitatively prove FM ≥ Expert curation\n")

    all_results = []

    # Detect keys from SCimilarity solution
    print("Loading SCimilarity solution to detect keys...")
    adata_temp = sc.read_h5ad(SCIM_SOLUTION_PATH)
    batch_key_candidates = ['Study', 'study', 'batch', 'Batch', 'dataset_of_origin']
    batch_key = next((k for k in batch_key_candidates if k in adata_temp.obs.columns), None)
    label_key_candidates = ['celltype', 'CellType', 'cell_type', 'cell_type_annotation']
    label_key = next((k for k in label_key_candidates if k in adata_temp.obs.columns), None)

    print(f"Detected batch_key: {batch_key}")
    print(f"Detected label_key: {label_key}")
    del adata_temp

    # 1. Raw Problem (PCA baseline)
    print("\n" + "=" * 80)
    print("CONDITION 1: RAW PROBLEM (Uncorrected)")
    print("=" * 80)

    adata_raw = sc.read_h5ad(RAW_PROBLEM_PATH)

    # Compute PCA for raw
    if 'X_pca' not in adata_raw.obsm:
        print("Computing PCA for raw data...")
        adata_raw_work = adata_raw.copy()
        sc.pp.normalize_total(adata_raw_work, target_sum=1e4)
        sc.pp.log1p(adata_raw_work)
        sc.pp.highly_variable_genes(adata_raw_work, n_top_genes=2000,
                                     batch_key=batch_key, subset=True)
        sc.tl.pca(adata_raw_work)
        adata_raw.obsm['X_pca'] = adata_raw_work.obsm['X_pca'].copy()

    results_raw = benchmark_method(adata_raw, batch_key, label_key, 'X_pca', 'Problem (Uncorrected)')
    all_results.append(results_raw)

    # 2. Gold Standard (scVI + manual curation)
    if ATLAS_PATH.exists():
        print("\n" + "=" * 80)
        print("CONDITION 2: GOLD STANDARD (Atlas)")
        print("=" * 80)

        adata_atlas = sc.read_h5ad(ATLAS_PATH)

        # Use scVI embedding if available
        if 'X_scVI' in adata_atlas.obsm:
            use_rep = 'X_scVI'
        elif 'X_pca' in adata_atlas.obsm:
            use_rep = 'X_pca'
        else:
            print("Computing PCA for atlas...")
            sc.tl.pca(adata_atlas)
            use_rep = 'X_pca'

        results_atlas = benchmark_method(adata_atlas, batch_key, label_key, use_rep, 'Gold Standard (Atlas)')
        all_results.append(results_atlas)
    else:
        print("\n⚠ Atlas file not found, skipping Gold Standard comparison")

    # 3. FM Solution (SCimilarity)
    print("\n" + "=" * 80)
    print("CONDITION 3: FM SOLUTION (SCimilarity)")
    print("=" * 80)

    adata_scim = sc.read_h5ad(SCIM_SOLUTION_PATH)
    results_scim = benchmark_method(adata_scim, batch_key, label_key, 'X_scimilarity', 'FM Solution (SCimilarity)')
    all_results.append(results_scim)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_file = METRICS_DIR / "quantitative_comparison.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")

    # Visualize
    visualize_comparison(results_df, FIGURES_DIR)

    # Print interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if len(all_results) >= 2:
        # Compare FM vs Gold Standard
        if 'Gold Standard (Atlas)' in results_df['Method'].values:
            fm_idx = results_df[results_df['Method'] == 'FM Solution (SCimilarity)'].index[0]
            atlas_idx = results_df[results_df['Method'] == 'Gold Standard (Atlas)'].index[0]

            fm_batch = results_df.loc[fm_idx, 'Batch_Mixing_Score']
            atlas_batch = results_df.loc[atlas_idx, 'Batch_Mixing_Score']

            fm_bio = results_df.loc[fm_idx, 'Bio_Conservation_Score']
            atlas_bio = results_df.loc[atlas_idx, 'Bio_Conservation_Score']

            print(f"\nBatch Mixing:")
            print(f"  FM: {fm_batch:.3f}  vs  Atlas: {atlas_batch:.3f}")
            if fm_batch >= atlas_batch:
                print(f"  ✓ FM achieves comparable/better batch mixing!")
            else:
                print(f"  - FM has lower batch mixing")

            print(f"\nBiology Conservation:")
            print(f"  FM: {fm_bio:.3f}  vs  Atlas: {atlas_bio:.3f}")
            if fm_bio > atlas_bio:
                print(f"  ✓✓ FM BETTER preserves biology!")
                print(f"     → This suggests the atlas may have 'over-corrected'")
                print(f"     → FM preserves finer biological distinctions")
            else:
                print(f"  - FM has lower biology conservation")

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  Metrics:")
    print(f"    - {results_file}")
    print(f"  Figures:")
    print(f"    - {FIGURES_DIR / 'fig2_quantitative_comparison.pdf'}")
    print(f"    - {FIGURES_DIR / 'fig2_metrics_heatmap.pdf'}")

    print(f"\nNext steps:")
    print(f"  Run Phase 4: python phase4_biological_discovery.py")

    return results_df


if __name__ == "__main__":
    results_df = main()
