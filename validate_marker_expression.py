#!/usr/bin/env python3
"""
Alternative validation: Marker gene-based evaluation (no label dependency)

This approach validates batch correction WITHOUT requiring matching cell type labels.
Instead, we use marker gene expression to:
1. Identify cell populations by their molecular signatures
2. Test if batch correction preserves these signatures
3. Check if marker relationships are consistent across studies

This is more robust than label transfer when:
- Studies use different annotation schemes
- Labels are at different granularities
- Some studies lack detailed annotations

Based on van Galen's published marker genes for AML subtypes.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# Van Galen AML subtype marker genes (published in Cell 2019)
AML_SUBTYPE_MARKERS = {
    'HSC-like': {
        'markers': ['AVP', 'CD34', 'HOPX', 'SPINK2', 'PROM1'],
        'description': 'Stem-like, quiescent, poor prognosis'
    },
    'Progenitor-like': {
        'markers': ['CD34', 'KIT', 'CEBPA', 'MPO'],
        'description': 'Early progenitor state'
    },
    'GMP-like': {
        'markers': ['MPO', 'ELANE', 'AZU1', 'CTSG', 'CSF3R'],
        'description': 'Granulocyte-monocyte progenitor, better prognosis'
    },
    'Promonocyte-like': {
        'markers': ['CEBPB', 'CEBPD', 'CD14', 'VCAN'],
        'description': 'Transitioning to monocyte'
    },
    'Monocyte-like': {
        'markers': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCN1'],
        'description': 'Mature monocytic phenotype'
    },
    'cDC-like': {
        'markers': ['IRF8', 'IRF4', 'CD1C', 'CLEC10A'],
        'description': 'Conventional dendritic cell-like'
    },
}

# General AML markers
GENERAL_MARKERS = {
    'Stemness': ['CD34', 'THY1', 'ENG', 'PROM1'],
    'Myeloid': ['MPO', 'ELANE', 'AZU1', 'PRTN3'],
    'Monocytic': ['CD14', 'CD68', 'CSF1R', 'LYZ'],
    'Proliferation': ['MKI67', 'TOP2A', 'PCNA'],
}

OUTPUT_DIR = "results_marker_validation"
N_HVGS = 2000
N_JOBS = 8

STUDIES_OF_INTEREST = [
    'van_galen_2019',
    'setty_2019',
    'pei_2020',
    'velten_2021',
    'oetjen_2018',
]


def compute_marker_scores(adata, marker_dict):
    """
    Compute average expression of marker gene sets for each cell.

    Returns: DataFrame with marker scores (cells × marker sets)
    """
    print("\n" + "="*80)
    print("COMPUTING MARKER GENE SCORES")
    print("="*80)

    scores = {}

    for marker_set, marker_info in marker_dict.items():
        markers = marker_info['markers'] if isinstance(marker_info, dict) else marker_info

        # Find which markers are in the dataset
        markers_present = [m for m in markers if m in adata.var_names]
        markers_missing = [m for m in markers if m not in adata.var_names]

        print(f"\n{marker_set}:")
        print(f"  Total markers: {len(markers)}")
        print(f"  Present: {len(markers_present)} - {markers_present}")
        if markers_missing:
            print(f"  Missing: {len(markers_missing)} - {markers_missing}")

        if len(markers_present) == 0:
            print(f"  ✗ No markers found - skipping")
            continue

        # Compute average expression
        marker_expr = adata[:, markers_present].X
        if hasattr(marker_expr, 'toarray'):
            marker_expr = marker_expr.toarray()

        scores[marker_set] = marker_expr.mean(axis=1)

    return pd.DataFrame(scores, index=adata.obs_names)


def test_marker_preservation(adata, embedding_key, marker_scores, batch_key='Study'):
    """
    Test if marker gene expression patterns are preserved after batch correction.

    Measures:
    1. Correlation between marker scores and embedding dimensions
    2. Consistency of marker patterns across batches
    """
    print(f"\n{'='*80}")
    print(f"TESTING MARKER PRESERVATION: {embedding_key}")
    print(f"{'='*80}")

    if embedding_key not in adata.obsm:
        print(f"✗ Embedding not found")
        return None

    embedding = adata.obsm[embedding_key]
    results = []

    # For each marker set, compute correlation with embedding space
    for marker_set in marker_scores.columns:
        marker_values = marker_scores[marker_set].values

        # Overall correlation (all studies together)
        pca = PCA(n_components=10)
        pca_scores = pca.fit_transform(embedding)

        # Find PC most correlated with this marker
        pc_corrs = [pearsonr(pca_scores[:, i], marker_values)[0] for i in range(10)]
        max_corr = max(pc_corrs, key=abs)

        # Per-study correlation (consistency check)
        study_corrs = []
        for study in adata.obs[batch_key].unique():
            study_mask = adata.obs[batch_key] == study
            if study_mask.sum() < 50:  # Skip small studies
                continue

            study_embedding = embedding[study_mask]
            study_markers = marker_values[study_mask]

            study_pca = PCA(n_components=min(5, study_embedding.shape[1]))
            study_pca_scores = study_pca.fit_transform(study_embedding)

            # Correlation with PC1
            corr, _ = pearsonr(study_pca_scores[:, 0], study_markers)
            study_corrs.append(corr)

        consistency = np.std(study_corrs) if len(study_corrs) > 1 else 0

        results.append({
            'marker_set': marker_set,
            'overall_correlation': abs(max_corr),
            'cross_study_consistency': 1 - consistency,  # Lower std = higher consistency
            'n_studies': len(study_corrs),
        })

        print(f"\n{marker_set}:")
        print(f"  Overall correlation: {abs(max_corr):.3f}")
        print(f"  Cross-study consistency: {1-consistency:.3f}")

    return pd.DataFrame(results)


def identify_populations_by_markers(marker_scores, top_percentile=20):
    """
    Identify cell populations by their marker expression (unsupervised).

    For each marker set, identify top-expressing cells.
    """
    print("\n" + "="*80)
    print("IDENTIFYING POPULATIONS BY MARKER EXPRESSION")
    print("="*80)

    populations = {}

    for marker_set in marker_scores.columns:
        threshold = np.percentile(marker_scores[marker_set], 100 - top_percentile)
        high_expressing = marker_scores[marker_set] > threshold

        populations[f'{marker_set}_high'] = high_expressing

        n_cells = high_expressing.sum()
        print(f"\n{marker_set}:")
        print(f"  High-expressing cells (top {top_percentile}%): {n_cells:,}")

    return pd.DataFrame(populations, index=marker_scores.index)


def test_population_separation(adata, embedding_key, populations, batch_key='Study'):
    """
    Test if marker-defined populations are well-separated in embedding space.

    Good batch correction should:
    - Separate different marker-defined populations (e.g., HSC vs Monocyte)
    - Mix same population across batches
    """
    from sklearn.metrics import silhouette_score

    print(f"\n{'='*80}")
    print(f"TESTING POPULATION SEPARATION: {embedding_key}")
    print(f"{'='*80}")

    if embedding_key not in adata.obsm:
        print(f"✗ Embedding not found")
        return None

    embedding = adata.obsm[embedding_key]
    results = []

    for pop_name in populations.columns:
        pop_mask = populations[pop_name].values

        if pop_mask.sum() < 100:  # Need enough cells
            continue

        # Create labels: 0 = background, 1 = this population
        labels = pop_mask.astype(int)

        # Silhouette score (how well-separated is this population?)
        score = silhouette_score(embedding, labels, sample_size=10000)

        # Within-population batch mixing
        pop_embedding = embedding[pop_mask]
        pop_batches = adata.obs[batch_key][pop_mask].values

        if len(np.unique(pop_batches)) > 1:
            batch_silhouette = silhouette_score(pop_embedding, pop_batches, sample_size=min(5000, len(pop_batches)))
            batch_mixing = 1 - batch_silhouette  # Lower silhouette = better mixing
        else:
            batch_mixing = np.nan

        results.append({
            'population': pop_name,
            'n_cells': pop_mask.sum(),
            'separation_score': score,
            'batch_mixing': batch_mixing,
        })

        print(f"\n{pop_name} ({pop_mask.sum():,} cells):")
        print(f"  Separation from background: {score:.3f}")
        print(f"  Batch mixing within population: {batch_mixing:.3f}")

    return pd.DataFrame(results)


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("MARKER GENE-BASED VALIDATION (LABEL-FREE)")
    print("="*80)
    print("\nThis validation does NOT require matching cell type labels.")
    print("Instead, it uses marker gene expression to validate batch correction.")

    # Load data
    print(f"\nLoading: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Subset to studies of interest
    if 'Study' in adata.obs.columns:
        mask = adata.obs['Study'].isin(STUDIES_OF_INTEREST)
        adata = adata[mask].copy()
        print(f"Filtered to {len(STUDIES_OF_INTEREST)} studies: {adata.n_obs:,} cells")

    # Compute marker scores
    marker_scores = compute_marker_scores(adata, AML_SUBTYPE_MARKERS)
    general_marker_scores = compute_marker_scores(adata, GENERAL_MARKERS)

    all_markers = pd.concat([marker_scores, general_marker_scores], axis=1)

    # Save marker scores
    marker_scores_path = f"{OUTPUT_DIR}/marker_scores.csv"
    all_markers.to_csv(marker_scores_path)
    print(f"\n✓ Saved marker scores: {marker_scores_path}")

    # Identify populations by markers
    populations = identify_populations_by_markers(marker_scores, top_percentile=20)

    # Test each embedding
    embedding_keys = ['X_pca', 'X_scVI', 'X_scimilarity', 'X_harmony']

    preservation_results = []
    separation_results = []

    for emb_key in embedding_keys:
        if emb_key not in adata.obsm:
            print(f"\n⚠ Skipping {emb_key} (not found)")
            continue

        # Test marker preservation
        pres_df = test_marker_preservation(adata, emb_key, all_markers)
        if pres_df is not None:
            pres_df['method'] = emb_key
            preservation_results.append(pres_df)

        # Test population separation
        sep_df = test_population_separation(adata, emb_key, populations)
        if sep_df is not None:
            sep_df['method'] = emb_key
            separation_results.append(sep_df)

    # Combine and save results
    if preservation_results:
        all_preservation = pd.concat(preservation_results, ignore_index=True)
        pres_path = f"{OUTPUT_DIR}/marker_preservation.csv"
        all_preservation.to_csv(pres_path, index=False)
        print(f"\n✓ Saved preservation results: {pres_path}")

    if separation_results:
        all_separation = pd.concat(separation_results, ignore_index=True)
        sep_path = f"{OUTPUT_DIR}/population_separation.csv"
        all_separation.to_csv(sep_path, index=False)
        print(f"\n✓ Saved separation results: {sep_path}")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    if preservation_results:
        print("\nMarker Preservation (average correlation):")
        for method in all_preservation['method'].unique():
            method_data = all_preservation[all_preservation['method'] == method]
            avg_corr = method_data['overall_correlation'].mean()
            avg_consistency = method_data['cross_study_consistency'].mean()
            print(f"  {method:20s}: corr={avg_corr:.3f}, consistency={avg_consistency:.3f}")

    if separation_results:
        print("\nPopulation Separation (average):")
        for method in all_separation['method'].unique():
            method_data = all_separation[all_separation['method'] == method]
            avg_sep = method_data['separation_score'].mean()
            avg_mixing = method_data['batch_mixing'].mean()
            print(f"  {method:20s}: separation={avg_sep:.3f}, batch_mixing={avg_mixing:.3f}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nGood batch correction should show:")
    print("  ✓ High marker preservation (>0.6)")
    print("  ✓ High cross-study consistency (>0.7)")
    print("  ✓ High population separation (>0.3)")
    print("  ✓ High batch mixing within populations (>0.6)")
    print("\nSCimilarity may excel at marker preservation and population separation")
    print("(preserving biology) even if batch mixing is lower.")

    print(f"\n✓ Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
