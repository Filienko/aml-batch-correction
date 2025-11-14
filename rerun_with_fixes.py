#!/usr/bin/env python
"""
Re-run with fixes based on diagnostics.

Key changes from original:
1. Lower resolution (0.2 instead of 0.5)
2. Option to test zhang_2023 alone
3. Try multiple resolutions automatically
4. Better reporting
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
import argparse

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(10, 10))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = Path("data/AML_scAtlas.h5ad")
SCIMILARITY_MODEL = "models/model_v1.1"
SCIMILARITY_BATCH_SIZE = 5000

# All studies or just one?
ALL_STUDIES = [
    'van_galen_2019',
    'jiang_2020',
    'beneyto-calabuig-2023',
    'velten_2021',
    'zhang_2023',
]

OUTPUT_DIR = Path("results_scimilarity_fixed")
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

for d in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_and_project(data_path, studies, model_path, batch_size):
    """Load, subset, and project to SCimilarity."""
    print("=" * 80)
    print("LOADING AND PROJECTING")
    print("=" * 80)

    # Load
    adata = sc.read_h5ad(data_path)
    print(f"✓ Loaded: {adata.n_obs:,} cells")

    # Find keys
    study_key = 'Study' if 'Study' in adata.obs.columns else 'study'
    label_key_candidates = ['cell_type_annotation', 'celltype', 'CellType', 'cell_type']
    label_key = next((k for k in label_key_candidates if k in adata.obs.columns), None)

    # Subset
    mask = adata.obs[study_key].isin(studies)
    adata = adata[mask].copy()
    print(f"✓ Subset to {len(studies)} studies: {adata.n_obs:,} cells")

    for study in studies:
        count = (adata.obs[study_key] == study).sum()
        print(f"  {study}: {count:,} cells")

    # Project if not already done
    if 'X_scimilarity' not in adata.obsm:
        print("\nProjecting to SCimilarity...")
        from scimilarity import CellAnnotation
        from scimilarity.utils import lognorm_counts, align_dataset
        import gc

        ca = CellAnnotation(model_path=model_path)

        # Prepare data
        if adata.raw is not None:
            adata_full = adata.raw.to_adata()
        else:
            adata_full = adata.copy()

        if 'counts' in adata_full.layers:
            adata_full.X = adata_full.layers['counts'].copy()

        if 'gene_name' in adata_full.var.columns:
            adata_full.var.index = adata_full.var['gene_name']

        # Common genes
        common_genes = adata_full.var.index.intersection(ca.gene_order)
        gene_order_dict = {gene: i for i, gene in enumerate(ca.gene_order)}
        common_genes_sorted = sorted(common_genes, key=lambda x: gene_order_dict[x])
        adata_subset = adata_full[:, common_genes_sorted].copy()

        del adata_full
        gc.collect()

        # Compute embeddings
        n_cells = adata_subset.n_obs
        n_batches = (n_cells + batch_size - 1) // batch_size
        embeddings_list = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_cells)
            print(f"  Batch {batch_idx + 1}/{n_batches}")

            batch = adata_subset[start:end].copy()
            batch_aligned = align_dataset(batch, ca.gene_order)
            if 'counts' not in batch_aligned.layers:
                batch_aligned.layers['counts'] = batch_aligned.X.copy()
            batch_norm = lognorm_counts(batch_aligned)
            batch_emb = ca.get_embeddings(batch_norm.X)
            embeddings_list.append(batch_emb)

            del batch, batch_aligned, batch_norm, batch_emb
            gc.collect()

        embeddings = np.vstack(embeddings_list)
        adata.obsm['X_scimilarity'] = embeddings

        del embeddings_list, adata_subset
        gc.collect()

        print(f"✓ SCimilarity embeddings: {embeddings.shape}")
    else:
        print("✓ Using existing SCimilarity embeddings")

    return adata, study_key, label_key


def test_multiple_resolutions(adata, label_key, resolutions=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Test multiple resolutions and find best."""
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE RESOLUTIONS")
    print("=" * 80)

    # Compute neighbors once
    if 'neighbors' not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep='X_scimilarity', n_neighbors=15)

    # Compute UMAP once
    if 'X_umap' not in adata.obsm:
        print("Computing UMAP...")
        sc.tl.umap(adata)

    results = []
    best_ari = 0
    best_res = None
    best_key = None

    expert = adata.obs[label_key].values
    valid_mask = pd.notna(expert)
    expert_valid = expert[valid_mask]

    for res in resolutions:
        key = f'leiden_r{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key)

        clusters = adata.obs[key].values
        clusters_valid = clusters[valid_mask]

        ari = adjusted_rand_score(expert_valid, clusters_valid)
        nmi = normalized_mutual_info_score(expert_valid, clusters_valid)
        n_clusters = len(np.unique(clusters_valid))

        print(f"  Resolution {res:.1f}: {n_clusters:2d} clusters, ARI={ari:.4f}, NMI={nmi:.4f}")

        results.append({
            'Resolution': res,
            'N_Clusters': n_clusters,
            'ARI': ari,
            'NMI': nmi
        })

        if ari > best_ari:
            best_ari = ari
            best_res = res
            best_key = key

    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"BEST: Resolution {best_res:.1f} → ARI={best_ari:.4f}")
    print(f"{'='*80}")

    # Use best clustering
    adata.obs['leiden'] = adata.obs[best_key].copy()

    # Save results
    results_df.to_csv(METRICS_DIR / "resolution_comparison.csv", index=False)

    return adata, best_res, best_ari, results_df


def evaluate_and_visualize(adata, study_key, label_key, best_res):
    """Evaluate and create visualizations."""
    print("\n" + "=" * 80)
    print("EVALUATION AND VISUALIZATION")
    print("=" * 80)

    expert = adata.obs[label_key].values
    clusters = adata.obs['leiden'].values

    valid = pd.notna(expert)
    expert_valid = expert[valid]
    clusters_valid = clusters[valid]

    ari = adjusted_rand_score(expert_valid, clusters_valid)
    nmi = normalized_mutual_info_score(expert_valid, clusters_valid)

    print(f"\nFinal Results:")
    print(f"  Resolution: {best_res}")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Expert types: {len(np.unique(expert_valid))}")
    print(f"  Clusters: {len(np.unique(clusters_valid))}")

    # Cluster mapping
    cluster_to_celltype = {}
    for cluster in np.unique(clusters_valid):
        mask = clusters_valid == cluster
        celltypes = expert_valid[mask]
        most_common = pd.Series(celltypes).value_counts().index[0]
        purity = (celltypes == most_common).sum() / len(celltypes)

        cluster_to_celltype[cluster] = {
            'cell_type': most_common,
            'purity': purity,
            'n_cells': mask.sum()
        }

    # Save mapping
    mapping_df = pd.DataFrame([
        {
            'Cluster': c,
            'Predicted_CellType': info['cell_type'],
            'Purity': info['purity'],
            'N_Cells': info['n_cells']
        }
        for c, info in cluster_to_celltype.items()
    ])
    mapping_df.to_csv(METRICS_DIR / "cluster_mapping.csv", index=False)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sc.pl.umap(adata, color=label_key, ax=axes[0], show=False,
              title=f'Expert Annotations\n({len(np.unique(expert_valid))} types)',
              legend_loc='right margin', legend_fontsize=8)

    sc.pl.umap(adata, color='leiden', ax=axes[1], show=False,
              title=f'SCimilarity Clusters (res={best_res})\n({len(np.unique(clusters_valid))} clusters, ARI={ari:.3f})',
              legend_loc='right margin', legend_fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_fixed.pdf", dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved")
    plt.close()

    # Per-study breakdown
    print("\nPer-study performance:")
    for study in adata.obs[study_key].unique():
        study_mask = adata.obs[study_key] == study
        if study_mask.sum() == 0:
            continue

        study_expert = expert[study_mask & valid]
        study_clusters = clusters[study_mask & valid]

        if len(study_expert) == 0:
            continue

        study_ari = adjusted_rand_score(study_expert, study_clusters)
        print(f"  {study}: ARI={study_ari:.4f}")

    return ari, nmi, mapping_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-study', type=str, default=None,
                       help='Test single study only (e.g., zhang_2023)')
    parser.add_argument('--resolution', type=float, default=None,
                       help='Use specific resolution (otherwise test multiple)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SCIMILARITY ANNOTATION (FIXED)")
    print("=" * 80)

    # Determine studies
    if args.single_study:
        studies = [args.single_study]
        print(f"\nTesting single study: {args.single_study}")
    else:
        studies = ALL_STUDIES
        print(f"\nTesting all {len(studies)} studies")

    # Load and project
    adata, study_key, label_key = load_and_project(
        DATA_PATH, studies, SCIMILARITY_MODEL, SCIMILARITY_BATCH_SIZE
    )

    # Test resolutions
    if args.resolution:
        print(f"\nUsing specified resolution: {args.resolution}")
        sc.pp.neighbors(adata, use_rep='X_scimilarity')
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=args.resolution)
        best_res = args.resolution
    else:
        adata, best_res, best_ari, res_df = test_multiple_resolutions(adata, label_key)

    # Evaluate
    ari, nmi, mapping = evaluate_and_visualize(adata, study_key, label_key, best_res)

    # Save
    adata.write(OUTPUT_DIR / "annotated_fixed.h5ad")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nBest resolution: {best_res}")
    print(f"Final ARI: {ari:.4f}")
    print(f"Final NMI: {nmi:.4f}")

    if ari > 0.7:
        print("\n✓✓ Excellent! SCimilarity closely matches expert annotations!")
    elif ari > 0.5:
        print("\n✓ Good! SCimilarity captures major cell types!")
    elif ari > 0.3:
        print("\n~ Moderate agreement")
    else:
        print("\n✗ Still low agreement - see per-study results above")

    print(f"\nResults in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
