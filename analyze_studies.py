#!/usr/bin/env python
"""
Analyze AML atlas studies to determine sizes and select datasets for experiments.

This script:
1. Loads the AML atlas data
2. Maps studies to their sequencing technologies
3. Determines cell counts per study
4. Recommends study selections for two experiments:
   - Cross-mechanism: microwell vs well-based vs droplet
   - Within-mechanism: all droplet-based studies
"""

import scanpy as sc
import pandas as pd
import numpy as np

# Technology mapping based on literature
# Note: These keys must match the exact study names in adata.obs['Study']
TECHNOLOGY_MAP = {
    # Non-droplet technologies
    'van_galen_2019': {
        'tech': 'Seq-Well',
        'category': 'non-droplet',
        'subcategory': 'microwell',
        'description': 'Seq-Well (barcoded bead-based, physical nanowells)'
    },
    'zhai_2022': {
        'tech': 'SORT-Seq',
        'category': 'non-droplet',
        'subcategory': 'FACS-based',
        'description': 'SORT-Seq (FACS-based single-cell RNA-seq)'
    },
    'pei_2020': {
        'tech': '10X Genomics CITEseq',
        'category': 'non-droplet',
        'subcategory': 'multimodal',
        'description': '10X Genomics CITEseq (droplet platform + protein measurements)'
    },
    'velten_2021': {
        'tech': 'Muta-Seq',
        'category': 'non-droplet',
        'subcategory': 'mutation-tracking',
        'description': 'Muta-Seq (mutation tracking + transcriptomics)'
    },

    # Droplet-based technologies (10x Genomics Chromium)
    'naldini': {
        'tech': '10x Genomics Chromium',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Chromium (regular droplet-based)'
    },
    'oetjen_2018': {
        'tech': '10x Genomics Single Cell 3′',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Single Cell 3′ Solution (~80k cells)'
    },
    'beneyto-calabuig-2023': {
        'tech': '10x Genomics Chromium Single Cell 3′',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Chromium Single Cell 3′ Solution'
    },
    'jiang_2020': {
        'tech': '10x Genomics Chromium Single Cell 3′',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Chromium Single Cell 3′'
    },
    'zheng_2017': {
        'tech': '10x Genomics GemCode Single-Cell 3′',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics GemCode Single-Cell 3′ solution'
    },
    'setty_2019': {
        'tech': '10x Chromium',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Chromium'
    },
    'petti_2019': {
        'tech': '10x Genomics Chromium Single Cell 5′',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Chromium Single Cell 5′ Gene Expression'
    },
    'mumme_2023': {
        'tech': '10x Genomics Chromium',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Chromium (3′ v3 and 5′ v1); NovaSeq'
    },
    'zhang_2023': {
        'tech': '10x Genomics Chromium',
        'category': 'droplet',
        'subcategory': '10x-chromium',
        'description': '10x Genomics Chromium Controller; NovaSeq (~80k cells)'
    }
}


def find_matching_studies(adata, study_keywords):
    """
    Find studies in the dataset that match the keywords.

    Args:
        adata: AnnData object
        study_keywords: Dictionary mapping study IDs to technology info

    Returns:
        Dictionary mapping study IDs to actual study names in the dataset
    """
    study_col = 'Study'
    all_studies = adata.obs[study_col].cat.categories.tolist()

    matches = {}

    for study_id, info in study_keywords.items():
        # Try exact match first
        if study_id in all_studies:
            matches[study_id] = study_id
            continue

        # Try case-insensitive match
        for actual_study in all_studies:
            if study_id.lower() in actual_study.lower() or actual_study.lower() in study_id.lower():
                matches[study_id] = actual_study
                break

    return matches


def analyze_dataset(data_path="data/AML_scAtlas.h5ad"):
    """
    Analyze the AML atlas dataset and report on study sizes and technologies.
    """
    print("="*80)
    print("AML ATLAS STUDY ANALYSIS")
    print("="*80)

    # Load data
    print(f"\nLoading: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Get all studies
    study_col = 'Study'
    all_studies = adata.obs[study_col].value_counts()

    print(f"\nTotal studies in dataset: {len(all_studies)}")
    print("\nAll studies:")
    for study, count in all_studies.items():
        print(f"  {study}: {count:,} cells")

    # Find matching studies
    print("\n" + "="*80)
    print("MATCHING STUDIES TO TECHNOLOGIES")
    print("="*80)

    matched_studies = find_matching_studies(adata, TECHNOLOGY_MAP)

    results = []

    for study_id, actual_study in matched_studies.items():
        tech_info = TECHNOLOGY_MAP[study_id]
        n_cells = adata.obs[adata.obs[study_col] == actual_study].shape[0]

        results.append({
            'Study_ID': study_id,
            'Actual_Study_Name': actual_study,
            'Technology': tech_info['tech'],
            'Category': tech_info['category'],
            'N_cells': n_cells,
            'Description': tech_info['description']
        })

        print(f"\n✓ {study_id}")
        print(f"  Matched: {actual_study}")
        print(f"  Technology: {tech_info['tech']}")
        print(f"  Category: {tech_info['category']}")
        print(f"  Cells: {n_cells:,}")

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('N_cells', ascending=False)

    # Report unmatched studies
    print("\n" + "="*80)
    print("UNMATCHED STUDIES")
    print("="*80)

    matched_actual_names = set(matched_studies.values())
    unmatched = [s for s in all_studies.index if s not in matched_actual_names]

    if unmatched:
        print(f"\n{len(unmatched)} studies not in technology map:")
        for study in unmatched:
            print(f"  {study}: {all_studies[study]:,} cells")
    else:
        print("\nAll studies matched!")

    # Experiment recommendations
    print("\n" + "="*80)
    print("EXPERIMENT RECOMMENDATIONS")
    print("="*80)

    # Experiment 1: Cross-mechanism
    print("\n1. CROSS-MECHANISM BATCH CORRECTION")
    print("   Compare batch correction across different technologies")
    print("   (Non-droplet vs Droplet-based)\n")

    non_droplet = df[df['Category'] == 'non-droplet'].sort_values('N_cells', ascending=False)
    droplet = df[df['Category'] == 'droplet'].sort_values('N_cells', ascending=False)

    print("   Selected studies:")

    # Select all non-droplet studies (we don't have many)
    if not non_droplet.empty:
        print(f"\n   Non-droplet technologies ({len(non_droplet)} studies):")
        for idx, row in non_droplet.iterrows():
            print(f"   • {row['Study_ID']}: {row['Technology']} - {row['N_cells']:,} cells")

    # Select largest 2-3 droplet studies for balance
    if not droplet.empty:
        print(f"\n   Droplet technologies (selecting largest 2-3 from {len(droplet)} studies):")
        for idx, row in droplet.head(3).iterrows():
            print(f"   • {row['Study_ID']}: {row['N_cells']:,} cells")

    total_cells_cross = (
        non_droplet['N_cells'].sum() if not non_droplet.empty else 0
    ) + (
        droplet.head(3)['N_cells'].sum() if not droplet.empty else 0
    )
    print(f"\n   Total cells (approx): {total_cells_cross:,}")

    # Experiment 2: Within-mechanism
    print("\n2. WITHIN-MECHANISM BATCH CORRECTION (Droplet-based only)")
    print("   Compare batch correction within same technology\n")

    if not droplet.empty:
        print(f"   Selected studies ({len(droplet)} total):")
        for idx, row in droplet.iterrows():
            print(f"   • {row['Study_ID']}: {row['N_cells']:,} cells")

        total_cells_within = droplet['N_cells'].sum()
        print(f"\n   Total cells: {total_cells_within:,}")
    else:
        print("   ⚠ No droplet-based studies found!")

    # Save results
    output_file = "study_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Generate configuration for experiments
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)

    # Cross-mechanism config: all non-droplet + largest 3 droplet
    cross_mechanism_studies = []
    if not non_droplet.empty:
        cross_mechanism_studies.extend(non_droplet['Actual_Study_Name'].tolist())
    if not droplet.empty:
        # Add largest 3 droplet studies to balance with non-droplet
        cross_mechanism_studies.extend(droplet.head(3)['Actual_Study_Name'].tolist())

    # Within-mechanism config: all droplet studies
    within_mechanism_studies = droplet['Actual_Study_Name'].tolist() if not droplet.empty else []

    print("\nCross-mechanism experiment studies:")
    print(f"CROSS_MECHANISM_STUDIES = {cross_mechanism_studies}")

    print("\nWithin-mechanism experiment studies:")
    print(f"WITHIN_MECHANISM_STUDIES = {within_mechanism_studies}")

    # Check for size imbalances
    print("\n" + "="*80)
    print("SIZE BALANCE WARNING")
    print("="*80)

    if not non_droplet.empty and not droplet.empty:
        non_droplet_total = non_droplet['N_cells'].sum()
        droplet_top3_total = droplet.head(3)['N_cells'].sum()

        print(f"\nCross-mechanism experiment:")
        print(f"  Non-droplet total: {non_droplet_total:,} cells ({len(non_droplet)} studies)")
        print(f"  Droplet total (top 3): {droplet_top3_total:,} cells")

        ratio = max(non_droplet_total, droplet_top3_total) / min(non_droplet_total, droplet_top3_total)
        if ratio > 3:
            print(f"  ⚠ WARNING: {ratio:.1f}x imbalance!")
            print(f"     Consider downsampling the larger group or upweighting smaller studies")
            print(f"     in the batch correction evaluation.")

    return df, cross_mechanism_studies, within_mechanism_studies


if __name__ == "__main__":
    df, cross_studies, within_studies = analyze_dataset()

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nData notes:")
    print("  • Raw counts available in adata.layers['counts']")
    print("  • This will be used for uncorrected baseline and SCimilarity")
    print("\nNext steps:")
    print("1. Review study_analysis_results.csv")
    print("2. Run experiment_cross_mechanism.py")
    print("3. Run experiment_within_mechanism.py")
