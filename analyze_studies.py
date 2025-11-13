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
TECHNOLOGY_MAP = {
    'van_galen': {
        'tech': 'Seq-Well',
        'category': 'microwell',
        'description': 'Seq-Well method (barcoded bead-based, physical wells)'
    },
    'naldini_2023': {
        'tech': 'SMART-Seq v4',
        'category': 'well-based',
        'description': 'SMART-Seq v4 (plate-based, full-length)'
    },
    'oetjen_2018': {
        'tech': '10x Genomics Single Cell 3′',
        'category': 'droplet',
        'description': 'Droplet-based, 10x Genomics Single Cell 3′ Solution'
    },
    'beneyto-calabuig-2023': {
        'tech': '10x Genomics Chromium Single Cell 3′',
        'category': 'droplet',
        'description': '10x Genomics Chromium Single Cell 3′ Solution'
    },
    'jiang_2020': {
        'tech': '10x Genomics Chromium Single Cell 3′',
        'category': 'droplet',
        'description': '10x Genomics Chromium Single Cell 3′ (droplet-based)'
    },
    'zheng_2017': {
        'tech': '10x Genomics GemCode Single-Cell 3′',
        'category': 'droplet',
        'description': '10x Genomics GemCode Single-Cell 3′ solution'
    },
    'setty_2019': {
        'tech': '10x Chromium',
        'category': 'droplet',
        'description': '10x Chromium'
    },
    'petti_2019': {
        'tech': '10x Genomics Chromium Single Cell 5′',
        'category': 'droplet',
        'description': '10x Genomics Chromium Single Cell 5′ Gene Expression workflow'
    },
    'mumme_2023': {
        'tech': '10x Genomics Chromium',
        'category': 'droplet',
        'description': '10x Genomics Chromium (3′ v3 and 5′ v1); NovaSeq'
    },
    'zhang_2023': {
        'tech': '10x Genomics Chromium',
        'category': 'droplet',
        'description': '10x Genomics Chromium Controller; NovaSeq'
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
    print("   Compare batch correction across different technologies\n")

    microwell = df[df['Category'] == 'microwell']
    wellbased = df[df['Category'] == 'well-based']
    droplet = df[df['Category'] == 'droplet']

    print("   Selected studies:")

    if not microwell.empty:
        study = microwell.iloc[0]
        print(f"   • Microwell: {study['Study_ID']} ({study['N_cells']:,} cells)")

    if not wellbased.empty:
        study = wellbased.iloc[0]
        print(f"   • Well-based: {study['Study_ID']} ({study['N_cells']:,} cells)")

    if not droplet.empty:
        # Pick the largest droplet-based study
        study = droplet.iloc[0]
        print(f"   • Droplet: {study['Study_ID']} ({study['N_cells']:,} cells) [LARGEST]")

    total_cells_cross = (
        (microwell.iloc[0]['N_cells'] if not microwell.empty else 0) +
        (wellbased.iloc[0]['N_cells'] if not wellbased.empty else 0) +
        (droplet.iloc[0]['N_cells'] if not droplet.empty else 0)
    )
    print(f"\n   Total cells: {total_cells_cross:,}")

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

    # Cross-mechanism config
    cross_mechanism_studies = []
    if not microwell.empty:
        cross_mechanism_studies.append(microwell.iloc[0]['Actual_Study_Name'])
    if not wellbased.empty:
        cross_mechanism_studies.append(wellbased.iloc[0]['Actual_Study_Name'])
    if not droplet.empty:
        cross_mechanism_studies.append(droplet.iloc[0]['Actual_Study_Name'])

    # Within-mechanism config
    within_mechanism_studies = droplet['Actual_Study_Name'].tolist() if not droplet.empty else []

    print("\nCross-mechanism experiment studies:")
    print(f"CROSS_MECHANISM_STUDIES = {cross_mechanism_studies}")

    print("\nWithin-mechanism experiment studies:")
    print(f"WITHIN_MECHANISM_STUDIES = {within_mechanism_studies}")

    return df, cross_mechanism_studies, within_mechanism_studies


if __name__ == "__main__":
    df, cross_studies, within_studies = analyze_dataset()

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review study_analysis_results.csv")
    print("2. Run experiment_cross_mechanism.py")
    print("3. Run experiment_within_mechanism.py")
