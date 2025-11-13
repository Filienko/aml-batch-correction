#!/usr/bin/env python
"""
Detect sequencing technology from AML atlas metadata and QC metrics

Different technologies have characteristic signatures:
- 10X Chromium: ~1000-5000 genes/cell, ~10k-50k UMIs, droplet-based
- SMART-seq2: ~5000-10000 genes/cell, ~1M+ UMIs, plate-based
- Fluidigm C1: ~3000-7000 genes/cell, ~100k-500k UMIs
- Drop-seq: Similar to 10X but slightly lower depth
- inDrops: Similar to 10X
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_technology_signatures(adata, batch_key='Study'):
    """
    Analyze QC metrics per study to infer sequencing technology

    Returns:
        DataFrame with technology predictions per study
    """

    print("="*80)
    print("TECHNOLOGY DETECTION ANALYSIS")
    print("="*80)

    # Calculate QC metrics if not already present
    print("\n1. Computing QC metrics...")
    if 'n_genes_by_counts' not in adata.obs.columns:
        sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Group by study
    print(f"\n2. Analyzing {adata.obs[batch_key].nunique()} studies...")

    results = []

    for study in adata.obs[batch_key].cat.categories:
        study_data = adata[adata.obs[batch_key] == study]

        # Calculate metrics
        n_cells = study_data.n_obs
        median_genes = study_data.obs['n_genes_by_counts'].median()
        mean_genes = study_data.obs['n_genes_by_counts'].mean()
        median_counts = study_data.obs['total_counts'].median()
        mean_counts = study_data.obs['total_counts'].mean()

        # Calculate gene detection rate (what % of genes are detected in typical cell)
        gene_detection_rate = median_genes / adata.n_vars

        # Predict technology based on signatures
        technology = predict_technology(
            median_genes=median_genes,
            median_counts=median_counts,
            n_cells=n_cells,
            gene_detection_rate=gene_detection_rate
        )

        results.append({
            'Study': study,
            'N_cells': n_cells,
            'Median_genes': median_genes,
            'Mean_genes': mean_genes,
            'Median_UMIs': median_counts,
            'Mean_UMIs': mean_counts,
            'Gene_detection_rate': gene_detection_rate,
            'Predicted_technology': technology['technology'],
            'Confidence': technology['confidence']
        })

    df = pd.DataFrame(results)
    df = df.sort_values('Predicted_technology')

    return df


def predict_technology(median_genes, median_counts, n_cells, gene_detection_rate):
    """
    Predict technology based on characteristic signatures

    Rules of thumb:
    - 10X: 1000-3000 genes, 5k-50k UMIs, >1000 cells per sample
    - SMART-seq2: 5000-10000 genes, >500k UMIs, <100 cells per sample
    - Fluidigm C1: 3000-7000 genes, 100k-500k UMIs, 50-800 cells
    - Drop-seq: 800-2000 genes, 2k-20k UMIs, >500 cells
    """

    # SMART-seq signature (plate-based, deep sequencing)
    if median_genes > 5000 and median_counts > 500000:
        return {'technology': 'SMART-seq2', 'confidence': 'high'}
    elif median_genes > 4000 and median_counts > 300000:
        return {'technology': 'SMART-seq2', 'confidence': 'medium'}

    # Fluidigm C1 signature (plate-based, medium depth)
    elif median_genes > 3500 and 100000 < median_counts < 500000:
        return {'technology': 'Fluidigm C1', 'confidence': 'medium'}

    # 10X Chromium signature (droplet-based, medium depth, high throughput)
    elif 1500 < median_genes < 4000 and 5000 < median_counts < 100000 and n_cells > 500:
        return {'technology': '10X Chromium', 'confidence': 'high'}
    elif 1000 < median_genes < 4000 and 3000 < median_counts < 100000:
        return {'technology': '10X Chromium', 'confidence': 'medium'}

    # Drop-seq signature (droplet-based, lower depth)
    elif 800 < median_genes < 2500 and 2000 < median_counts < 30000 and n_cells > 300:
        return {'technology': 'Drop-seq', 'confidence': 'medium'}

    # inDrops signature (similar to Drop-seq)
    elif 1000 < median_genes < 3000 and 5000 < median_counts < 50000 and n_cells > 300:
        return {'technology': 'inDrops', 'confidence': 'low'}

    # Low depth (possibly degraded or low-quality)
    elif median_genes < 1000:
        return {'technology': 'Low-depth/Unknown', 'confidence': 'low'}

    # Unknown
    else:
        return {'technology': 'Unknown', 'confidence': 'low'}


def check_study_names_for_technology(adata, batch_key='Study'):
    """
    Check if study names contain technology keywords
    """

    print("\n" + "="*80)
    print("CHECKING STUDY NAMES FOR TECHNOLOGY KEYWORDS")
    print("="*80)

    keywords = {
        '10X': ['10x', '10X', 'chromium', 'Chromium', 'CellRanger', 'cellranger'],
        'SMART-seq': ['smart', 'Smart', 'SMART', 'smartseq', 'SmartSeq'],
        'Fluidigm': ['fluidigm', 'Fluidigm', 'C1'],
        'Drop-seq': ['drop', 'Drop', 'dropseq', 'Dropseq'],
        'inDrops': ['indrop', 'inDrop', 'inDrops']
    }

    found = {}

    for study in adata.obs[batch_key].cat.categories:
        study_lower = str(study).lower()

        for tech, keywords_list in keywords.items():
            if any(kw.lower() in study_lower for kw in keywords_list):
                found[study] = tech
                print(f"  ✓ {study} → {tech}")

    if not found:
        print("  ℹ No technology keywords found in study names")

    return found


def plot_technology_distribution(adata, tech_df, batch_key='Study', output_dir='technology_analysis'):
    """
    Create visualizations of technology signatures
    """

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # 1. Scatter plot: Genes vs UMIs colored by predicted technology
    fig, ax = plt.subplots(figsize=(10, 6))

    for tech in tech_df['Predicted_technology'].unique():
        mask = tech_df['Predicted_technology'] == tech
        subset = tech_df[mask]

        ax.scatter(
            subset['Median_genes'],
            subset['Median_UMIs'],
            s=subset['N_cells'] / 10,  # Size by cell count
            alpha=0.6,
            label=f"{tech} (n={len(subset)})"
        )

    ax.set_xlabel('Median genes per cell', fontsize=12)
    ax.set_ylabel('Median UMIs per cell', fontsize=12)
    ax.set_title('Technology Signatures: Genes vs UMIs', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add technology regions
    ax.axhspan(500000, 2000000, alpha=0.1, color='red', label='SMART-seq range')
    ax.axhspan(5000, 100000, alpha=0.1, color='blue', label='10X range')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/technology_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/technology_scatter.png")
    plt.close()

    # 2. Bar plot: Study distribution by technology
    fig, ax = plt.subplots(figsize=(12, 6))

    tech_counts = tech_df['Predicted_technology'].value_counts()
    colors = sns.color_palette('Set2', len(tech_counts))

    tech_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_xlabel('Technology', fontsize=12)
    ax.set_ylabel('Number of studies', fontsize=12)
    ax.set_title('Distribution of Studies by Technology', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/technology_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/technology_distribution.png")
    plt.close()

    # 3. Violin plot: QC metrics by technology
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Merge technology info back to adata
    study_to_tech = dict(zip(tech_df['Study'], tech_df['Predicted_technology']))
    adata.obs['Predicted_technology'] = adata.obs[batch_key].map(study_to_tech)

    # Plot genes
    tech_order = tech_df.groupby('Predicted_technology')['Median_genes'].median().sort_values(ascending=False).index

    from matplotlib import patches
    parts = axes[0].violinplot(
        [adata.obs[adata.obs['Predicted_technology'] == tech]['n_genes_by_counts'].values
         for tech in tech_order],
        positions=range(len(tech_order)),
        showmeans=True,
        showmedians=True
    )
    axes[0].set_xticks(range(len(tech_order)))
    axes[0].set_xticklabels(tech_order, rotation=45, ha='right')
    axes[0].set_ylabel('Genes per cell', fontsize=12)
    axes[0].set_title('Gene Detection by Technology', fontsize=12, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot UMIs
    parts = axes[1].violinplot(
        [adata.obs[adata.obs['Predicted_technology'] == tech]['total_counts'].values
         for tech in tech_order],
        positions=range(len(tech_order)),
        showmeans=True,
        showmedians=True
    )
    axes[1].set_xticks(range(len(tech_order)))
    axes[1].set_xticklabels(tech_order, rotation=45, ha='right')
    axes[1].set_ylabel('UMIs per cell', fontsize=12)
    axes[1].set_title('UMI Counts by Technology', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/qc_by_technology.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/qc_by_technology.png")
    plt.close()

    # Clean up
    adata.obs.drop(columns=['Predicted_technology'], inplace=True)


def main():
    """
    Main analysis pipeline
    """

    print("\n" + "="*80)
    print("AML ATLAS TECHNOLOGY DETECTION")
    print("="*80)

    # Load data
    data_path = "data/AML_scAtlas.h5ad"
    print(f"\nLoading: {data_path}")

    adata = sc.read_h5ad(data_path)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    batch_key = 'Study'

    # 1. Check study names
    study_tech_keywords = check_study_names_for_technology(adata, batch_key)

    # 2. Detect from QC metrics
    tech_df = detect_technology_signatures(adata, batch_key)

    # 3. Display results
    print("\n" + "="*80)
    print("TECHNOLOGY PREDICTIONS")
    print("="*80)
    print()
    print(tech_df.to_string(index=False))

    # 4. Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for tech in tech_df['Predicted_technology'].unique():
        subset = tech_df[tech_df['Predicted_technology'] == tech]
        n_studies = len(subset)
        n_cells = subset['N_cells'].sum()
        confidence = subset['Confidence'].value_counts().to_dict()

        print(f"\n{tech}:")
        print(f"  Studies: {n_studies}")
        print(f"  Total cells: {n_cells:,}")
        print(f"  Confidence: {confidence}")
        print(f"  Median genes: {subset['Median_genes'].median():.0f}")
        print(f"  Median UMIs: {subset['Median_UMIs'].median():.0f}")

    # 5. Create visualizations
    plot_technology_distribution(adata, tech_df, batch_key)

    # 6. Save results
    output_file = "technology_analysis/technology_predictions.csv"
    tech_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved: {output_file}")

    # 7. Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR CROSS-TECHNOLOGY ANALYSIS")
    print("="*80)

    techs = tech_df['Predicted_technology'].unique()

    if len(techs) >= 3:
        print("\n✓ Great! You have 3+ technologies - ideal for cross-platform paper!")
        print("\nSuggested comparisons:")

        # Find representatives of each major tech
        for tech in ['10X Chromium', 'SMART-seq2', 'Fluidigm C1']:
            if tech in techs:
                studies = tech_df[tech_df['Predicted_technology'] == tech]['Study'].tolist()
                print(f"\n  {tech} studies ({len(studies)}):")
                for study in studies[:3]:  # Show first 3
                    print(f"    - {study}")
                if len(studies) > 3:
                    print(f"    ... and {len(studies) - 3} more")

        print("\n  Suggested analysis:")
        print("  1. Select 1-2 representative studies per technology")
        print("  2. Run batch correction comparison")
        print("  3. Evaluate preservation of platform-specific genes")
        print("  4. Check if cell type annotations are consistent")
        print("  5. Build cross-platform cell type classifier")

    elif len(techs) == 2:
        print("\n✓ You have 2 technologies - good for pairwise comparison!")
        print(f"\n  Technologies: {', '.join(techs)}")
        print("\n  Suggested analysis:")
        print("  1. Deep comparison of these two platforms")
        print("  2. Technology transfer evaluation")
        print("  3. Benchmark sensitivity to depth differences")

    else:
        print("\n⚠ Appears to be mostly one technology")
        print("  Consider:")
        print("  1. Focus on within-platform batch effects")
        print("  2. Or obtain additional data with different technologies")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review technology_analysis/technology_predictions.csv")
    print("2. Check plots in technology_analysis/")
    print("3. Manually verify predictions for key studies")
    print("4. Look up original papers if needed (study names might have DOIs)")


if __name__ == "__main__":
    main()
