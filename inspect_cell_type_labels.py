#!/usr/bin/env python3
"""
Inspect cell type labels across AML studies to determine if van Galen
subtype validation is feasible.

This script checks:
1. What cell type annotation columns exist (CellType, cell_type, annotation, etc.)
2. What labels are present in each study
3. Whether studies have malignant/AML cell annotations
4. If labels can be mapped to van Galen's 6 subtypes

Run this BEFORE validate_aml_subtypes.py to verify feasibility.
"""

import scanpy as sc
import pandas as pd
import numpy as np

# Configuration
DATA_PATH = "data/AML_scAtlas.h5ad"
VAN_GALEN_STUDY = 'van_galen_2019'

# Studies we want to validate
STUDIES_OF_INTEREST = [
    'van_galen_2019',   # Reference
    'setty_2019',
    'pei_2020',
    'velten_2021',
    'oetjen_2018',
]

# Van Galen's published 6 malignant subtypes
VAN_GALEN_SUBTYPES = [
    'HSC-like',
    'Progenitor-like',
    'GMP-like',
    'Promonocyte-like',
    'Monocyte-like',
    'cDC-like',
]

print("="*80)
print("INSPECTING CELL TYPE LABELS FOR VAN GALEN VALIDATION")
print("="*80)

# Load data
print(f"\nLoading: {DATA_PATH}")
adata = sc.read_h5ad(DATA_PATH)
print(f"Total: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Check what annotation columns exist
print("\n" + "="*80)
print("ANNOTATION COLUMNS IN DATASET")
print("="*80)

annotation_columns = []
for col in adata.obs.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['cell', 'type', 'label', 'annot', 'cluster']):
        annotation_columns.append(col)
        n_unique = adata.obs[col].nunique()
        print(f"  {col}: {n_unique} unique values")

if not annotation_columns:
    print("  ✗ No obvious annotation columns found!")
    print(f"  Available columns: {list(adata.obs.columns)}")

# Check Study column
if 'Study' not in adata.obs.columns:
    print("\n✗ ERROR: No 'Study' column found!")
    print(f"Available columns: {list(adata.obs.columns)}")
    exit(1)

print(f"\nStudies in dataset: {adata.obs['Study'].nunique()}")
print(f"Studies of interest available:")
for study in STUDIES_OF_INTEREST:
    if study in adata.obs['Study'].values:
        n_cells = (adata.obs['Study'] == study).sum()
        print(f"  ✓ {study}: {n_cells:,} cells")
    else:
        print(f"  ✗ {study}: NOT FOUND")

# For each annotation column, check labels per study
for col in annotation_columns:
    print("\n" + "="*80)
    print(f"LABELS IN: {col}")
    print("="*80)

    for study in STUDIES_OF_INTEREST:
        if study not in adata.obs['Study'].values:
            continue

        study_mask = adata.obs['Study'] == study
        labels = adata.obs.loc[study_mask, col].value_counts()

        print(f"\n{study} ({study_mask.sum():,} cells):")
        print(f"  Unique labels: {len(labels)}")

        # Show top 15 labels
        for label, count in labels.head(15).items():
            pct = 100 * count / study_mask.sum()
            print(f"    {label}: {count:,} ({pct:.1f}%)")

        if len(labels) > 15:
            print(f"    ... and {len(labels) - 15} more labels")

# Check if we can identify malignant cells
print("\n" + "="*80)
print("CHECKING FOR MALIGNANT CELL ANNOTATIONS")
print("="*80)

for col in annotation_columns:
    malignant_keywords = ['malignant', 'aml', 'blast', 'leukemia', 'tumor', 'cancer']
    labels = adata.obs[col].unique()

    malignant_labels = []
    for label in labels:
        label_str = str(label).lower()
        if any(keyword in label_str for keyword in malignant_keywords):
            malignant_labels.append(label)

    if malignant_labels:
        print(f"\n{col} - Potential malignant labels:")
        for label in malignant_labels[:20]:
            n = (adata.obs[col] == label).sum()
            print(f"  {label}: {n:,} cells")

# Try to match van Galen subtypes
print("\n" + "="*80)
print("SEARCHING FOR VAN GALEN SUBTYPE MATCHES")
print("="*80)

van_galen_keywords = {
    'HSC': ['hsc', 'stem', 'hspc', 'cd34+cd38-'],
    'Progenitors': ['prog', 'progenitor', 'mpp', 'cd34+'],
    'GMP': ['gmp', 'granulocyte', 'myeloid prog'],
    'Promono': ['promono', 'promonocyte'],
    'Monocyte': ['monocyte', 'mono', 'cd14'],
    'cDC': ['dc', 'dendritic', 'cdc'],
}

for col in annotation_columns:
    print(f"\n{col}:")
    labels = adata.obs[col].unique()

    found_matches = False
    for vg_subtype, keywords in van_galen_keywords.items():
        matches = []
        for label in labels:
            label_str = str(label).lower()
            if any(keyword in label_str for keyword in keywords):
                matches.append(label)

        if matches:
            found_matches = True
            print(f"  {vg_subtype} matches:")
            for match in matches[:5]:
                n = (adata.obs[col] == match).sum()
                print(f"    {match}: {n:,} cells")

    if not found_matches:
        print(f"  ✗ No obvious van Galen subtype matches")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n1. BEST ANNOTATION COLUMN:")
if annotation_columns:
    # Prefer columns with moderate number of categories (not too few, not too many)
    col_scores = []
    for col in annotation_columns:
        n_unique = adata.obs[col].nunique()
        # Ideal range: 10-50 cell types
        if 10 <= n_unique <= 50:
            score = 100
        elif 5 <= n_unique < 10:
            score = 80
        elif 50 < n_unique <= 100:
            score = 60
        else:
            score = 20
        col_scores.append((col, n_unique, score))

    col_scores.sort(key=lambda x: x[2], reverse=True)
    best_col = col_scores[0][0]
    print(f"   Recommended: '{best_col}'")
    print(f"   (Has {col_scores[0][1]} unique labels - good granularity)")
else:
    print("   ✗ Cannot determine - no annotation columns found")

print("\n2. VALIDATION FEASIBILITY:")
print("   To validate against van Galen, you need:")
print("   - ✓ Van Galen study present? ", end="")
print("YES" if VAN_GALEN_STUDY in adata.obs['Study'].values else "NO")

if annotation_columns:
    print(f"   - ✓ Cell type annotations? YES ({len(annotation_columns)} columns)")
else:
    print("   - ✗ Cell type annotations? NO")

print("\n3. NEXT STEPS:")
if VAN_GALEN_STUDY in adata.obs['Study'].values and annotation_columns:
    print("   ✓ Van Galen validation is FEASIBLE")
    print("   1. Choose annotation column (see above)")
    print("   2. Manually create label mapping based on labels shown above")
    print("   3. Update harmonize_cell_type_labels() in validate_aml_subtypes.py")
    print("   4. Run validation")
else:
    print("   ✗ Van Galen validation NOT feasible with current data")
    print("   Reasons:")
    if VAN_GALEN_STUDY not in adata.obs['Study'].values:
        print("     - van_galen_2019 study not found in dataset")
    if not annotation_columns:
        print("     - No cell type annotations found")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
