#!/usr/bin/env python
"""
Diagnostic: Why is ARI low? What's wrong?

This script investigates:
1. Per-dataset performance (is one dataset pulling down the score?)
2. SCimilarity predictions vs clustering (which works better?)
3. Different resolutions (is 0.5 too high?)
4. Batch effects (are studies separating?)

Goal: Figure out why ARI=0.1951 instead of >0.7
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = Path("data/AML_scAtlas.h5ad")
SCIMILARITY_MODEL = "models/model_v1.1"

STUDIES = [
    'van_galen_2019',
    'jiang_2020',
    'beneyto-calabuig-2023',
    'velten_2021',
    'zhang_2023',
]

OUTPUT_DIR = Path("results_scimilarity_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DIAGNOSTIC 1: PER-DATASET PERFORMANCE
# ==============================================================================

def analyze_per_dataset_performance(adata, study_key, label_key):
    """
    Check ARI for each dataset separately.

    This tells us if one dataset is causing problems.
    """
    print("=" * 80)
    print("DIAGNOSTIC 1: PER-DATASET PERFORMANCE")
    print("=" * 80)

    results = []

    for study in STUDIES:
        study_mask = adata.obs[study_key] == study
        if study_mask.sum() == 0:
            print(f"\n⚠ {study}: Not found")
            continue

        adata_study = adata[study_mask].copy()

        # Get labels
        expert = adata_study.obs[label_key].values
        clusters = adata_study.obs['leiden'].values

        # Remove NaN
        valid = pd.notna(expert)
        expert = expert[valid]
        clusters = clusters[valid]

        # Compute ARI
        ari = adjusted_rand_score(expert, clusters)
        nmi = normalized_mutual_info_score(expert, clusters)

        n_types = len(np.unique(expert))
        n_clusters = len(np.unique(clusters))

        print(f"\n{study}:")
        print(f"  Cells: {study_mask.sum():,}")
        print(f"  Expert types: {n_types}")
        print(f"  Clusters: {n_clusters}")
        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")

        results.append({
            'Study': study,
            'N_Cells': study_mask.sum(),
            'N_Expert_Types': n_types,
            'N_Clusters': n_clusters,
            'ARI': ari,
            'NMI': nmi
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))

    # Save
    output_file = OUTPUT_DIR / "per_dataset_performance.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    return results_df


# ==============================================================================
# DIAGNOSTIC 2: SCIMILARITY PREDICTIONS VS CLUSTERING
# ==============================================================================

def compare_prediction_methods(adata, study_key, label_key):
    """
    Compare two approaches:
    1. Current: Clustering SCimilarity embeddings
    2. Alternative: SCimilarity's built-in predictions

    SCimilarity has a classifier - maybe we should use that instead!
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: PREDICTION METHODS COMPARISON")
    print("=" * 80)

    print("\nCurrent approach:")
    print("  SCimilarity embeddings → Leiden clustering → Compare to expert")
    print("  Problem: Leiden resolution might not match expert granularity")

    print("\nAlternative approach:")
    print("  SCimilarity predictions → Compare to expert")
    print("  Note: SCimilarity's predictions might not match AML cell types exactly")

    # Check if we have SCimilarity predictions
    if 'scimilarity_predictions' in adata.obs.columns:
        expert = adata.obs[label_key].values
        predictions = adata.obs['scimilarity_predictions'].values

        valid = pd.notna(expert) & pd.notna(predictions)
        expert = expert[valid]
        predictions = predictions[valid]

        ari = adjusted_rand_score(expert, predictions)
        nmi = normalized_mutual_info_score(expert, predictions)

        print(f"\nSCimilarity predictions vs Expert:")
        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")

        return ari, nmi
    else:
        print("\n⚠ No SCimilarity predictions found in data")
        print("  We're only using clustering on embeddings")
        return None, None


# ==============================================================================
# DIAGNOSTIC 3: RESOLUTION SWEEP
# ==============================================================================

def test_different_resolutions(adata, label_key):
    """
    Test different Leiden resolutions.

    Current: 35 clusters from resolution 0.5
    Target: 16 cell types

    Try different resolutions to match granularity.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: RESOLUTION SWEEP")
    print("=" * 80)

    print(f"\nProblem identified:")
    print(f"  Expert cell types: 16")
    print(f"  Current clusters (res=0.5): 35")
    print(f"  → Resolution too high! Creating too many clusters")

    print(f"\nTesting different resolutions...")

    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = []

    for res in resolutions:
        # Cluster
        sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')

        # Evaluate
        expert = adata.obs[label_key].values
        clusters = adata.obs[f'leiden_r{res}'].values

        valid = pd.notna(expert)
        expert_valid = expert[valid]
        clusters_valid = clusters[valid]

        ari = adjusted_rand_score(expert_valid, clusters_valid)
        nmi = normalized_mutual_info_score(expert_valid, clusters_valid)
        n_clusters = len(np.unique(clusters_valid))

        results.append({
            'Resolution': res,
            'N_Clusters': n_clusters,
            'ARI': ari,
            'NMI': nmi
        })

        print(f"  res={res:.1f}: {n_clusters:2d} clusters, ARI={ari:.4f}, NMI={nmi:.4f}")

    results_df = pd.DataFrame(results)

    # Find best
    best_idx = results_df['ARI'].idxmax()
    best = results_df.loc[best_idx]

    print(f"\n{'='*80}")
    print(f"BEST RESOLUTION:")
    print(f"{'='*80}")
    print(f"  Resolution: {best['Resolution']}")
    print(f"  Clusters: {best['N_Clusters']}")
    print(f"  ARI: {best['ARI']:.4f}")
    print(f"  NMI: {best['NMI']:.4f}")

    # Save
    output_file = OUTPUT_DIR / "resolution_sweep.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Clusters vs resolution
    axes[0].plot(results_df['Resolution'], results_df['N_Clusters'], 'o-')
    axes[0].axhline(16, color='red', linestyle='--', label='Expert types (16)')
    axes[0].set_xlabel('Resolution')
    axes[0].set_ylabel('Number of Clusters')
    axes[0].set_title('Clusters vs Resolution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ARI vs resolution
    axes[1].plot(results_df['Resolution'], results_df['ARI'], 'o-', label='ARI')
    axes[1].plot(results_df['Resolution'], results_df['NMI'], 's-', label='NMI')
    axes[1].axvline(best['Resolution'], color='red', linestyle='--',
                    label=f'Best (res={best["Resolution"]:.1f})')
    axes[1].set_xlabel('Resolution')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Agreement vs Resolution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_file = OUTPUT_DIR / "resolution_sweep.pdf"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {fig_file}")
    plt.close()

    return results_df, best['Resolution']


# ==============================================================================
# DIAGNOSTIC 4: BATCH EFFECTS CHECK
# ==============================================================================

def check_batch_effects(adata, study_key, label_key):
    """
    Are studies separating in the SCimilarity space?

    If yes: Batch effects are still present
    If no: Studies are well-mixed
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: BATCH EFFECTS CHECK")
    print("=" * 80)

    print("\nChecking if studies separate in SCimilarity space...")

    # Compute batch mixing score (simple version)
    from sklearn.neighbors import NearestNeighbors

    X = adata.obsm['X_scimilarity']
    batches = adata.obs[study_key].values

    nn = NearestNeighbors(n_neighbors=30)
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # For each cell, count unique batches in neighborhood
    batch_mixing_scores = []
    for i in range(len(adata)):
        neighbor_batches = batches[indices[i]]
        n_unique = len(np.unique(neighbor_batches))
        batch_mixing_scores.append(n_unique)

    mean_mixing = np.mean(batch_mixing_scores)
    max_possible = len(STUDIES)

    print(f"\nBatch mixing score: {mean_mixing:.2f} / {max_possible}")
    print(f"  (Higher = better mixing)")

    if mean_mixing < 2.0:
        print(f"  ✗ Poor mixing - studies are separating!")
    elif mean_mixing < 3.0:
        print(f"  ~ Moderate mixing")
    else:
        print(f"  ✓ Good mixing")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by study
    sc.pl.umap(adata, color=study_key, ax=axes[0], show=False,
              title='UMAP colored by Study (Batch)')

    # Color by cell type
    sc.pl.umap(adata, color=label_key, ax=axes[1], show=False,
              title='UMAP colored by Cell Type (Biology)')

    plt.tight_layout()
    fig_file = OUTPUT_DIR / "batch_effects_check.pdf"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ UMAP saved: {fig_file}")
    plt.close()

    return mean_mixing


# ==============================================================================
# DIAGNOSTIC 5: WHAT DID YOU DO BEFORE?
# ==============================================================================

def show_comparison_to_previous_approach():
    """
    Show what might have been done differently before.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 5: WHAT'S DIFFERENT FROM BEFORE?")
    print("=" * 80)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║ CURRENT APPROACH (scimilarity_annotation_replication.py)        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("""
    1. Load 5 studies from atlas
    2. Project to SCimilarity embeddings (384-dim)
    3. Compute neighbors on embeddings
    4. Leiden clustering (resolution=0.5)
    5. Compare clusters to expert labels

    Problem: 35 clusters vs 16 expert types
    """)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║ POSSIBLE PREVIOUS APPROACH (that worked better)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("""
    Option A: SCimilarity's built-in predictions
    -------------------------------------------------
    1. Load ONE study at a time
    2. Use SCimilarity.predict() for cell types
    3. Compare predictions to expert labels
    → No clustering, direct predictions
    → Trained on multiple datasets

    Option B: Better clustering resolution
    -------------------------------------------------
    1. Same as current
    2. But use resolution that matches granularity
    → Test different resolutions (0.1-0.4)

    Option C: Single study at a time
    -------------------------------------------------
    1. Process each study separately
    2. No batch effects between studies
    → zhang_2023 alone should work best
    """)

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("""
    Try these in order:

    1. LOWER RESOLUTION (0.2-0.3) to get ~16 clusters
       → See DIAGNOSTIC 3 results above

    2. TEST EACH STUDY SEPARATELY
       → See which studies work well
       → zhang_2023 should be best (atlas paper's data)

    3. USE SCIMILARITY'S PREDICTIONS
       → If available, use ca.predict() instead of clustering
       → This is what the model was trained for
    """)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all diagnostics."""
    print("\n" + "=" * 80)
    print("SCIMILARITY ANNOTATION DIAGNOSTICS")
    print("=" * 80)
    print("\nGoal: Understand why ARI=0.1951 (too low!)")
    print("=" * 80)

    # Load data
    print("\nLoading annotated data...")
    adata_path = Path("results_scimilarity_annotation/scimilarity_annotated.h5ad")

    if not adata_path.exists():
        print(f"✗ Not found: {adata_path}")
        print("  Run scimilarity_annotation_replication.py first")
        return

    adata = sc.read_h5ad(adata_path)
    print(f"✓ Loaded: {adata.n_obs:,} cells")

    # Detect keys
    study_key = 'Study' if 'Study' in adata.obs.columns else 'study'
    label_key_candidates = ['cell_type_annotation', 'celltype', 'CellType', 'cell_type']
    label_key = next((k for k in label_key_candidates if k in adata.obs.columns), None)

    if label_key is None:
        print("\n✗ Could not find cell type label column")
        print(f"Available: {adata.obs.columns.tolist()}")
        return

    print(f"Study key: {study_key}")
    print(f"Label key: {label_key}")

    # Run diagnostics
    print("\n" + "=" * 80)
    print("RUNNING DIAGNOSTICS...")
    print("=" * 80)

    # 1. Per-dataset
    per_dataset_results = analyze_per_dataset_performance(adata, study_key, label_key)

    # 2. Prediction methods
    compare_prediction_methods(adata, study_key, label_key)

    # 3. Resolution sweep (MOST IMPORTANT!)
    resolution_results, best_res = test_different_resolutions(adata, label_key)

    # 4. Batch effects
    mixing_score = check_batch_effects(adata, study_key, label_key)

    # 5. What's different
    show_comparison_to_previous_approach()

    # Final recommendation
    print("\n" + "=" * 80)
    print("ACTIONABLE RECOMMENDATIONS:")
    print("=" * 80)

    best_ari = resolution_results['ARI'].max()

    print(f"\n1. USE LOWER RESOLUTION:")
    print(f"   Current: resolution=0.5 → 35 clusters → ARI=0.1951")
    print(f"   Best: resolution={best_res} → ARI={best_ari:.4f}")
    print(f"   → Re-run with resolution={best_res}")

    print(f"\n2. CHECK PER-DATASET RESULTS:")
    worst_study = per_dataset_results.loc[per_dataset_results['ARI'].idxmin(), 'Study']
    worst_ari = per_dataset_results['ARI'].min()
    best_study = per_dataset_results.loc[per_dataset_results['ARI'].idxmax(), 'Study']
    best_study_ari = per_dataset_results['ARI'].max()

    print(f"   Worst: {worst_study} (ARI={worst_ari:.4f})")
    print(f"   Best: {best_study} (ARI={best_study_ari:.4f})")
    print(f"   → Maybe exclude {worst_study}?")

    print(f"\n3. BATCH MIXING:")
    print(f"   Score: {mixing_score:.2f} / 5")
    if mixing_score < 3.0:
        print(f"   → Studies are somewhat separated")
        print(f"   → This is actually EXPECTED with SCimilarity")
        print(f"   → Not necessarily a problem")

    print(f"\n4. NEXT STEPS:")
    print(f"   a) Re-run with resolution={best_res}")
    print(f"   b) Test zhang_2023 alone (should be best)")
    print(f"   c) Try SCimilarity's direct predictions if available")

    print(f"\n{'='*80}")
    print("All diagnostic results saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
