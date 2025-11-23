#!/usr/bin/env python
"""
Inspect Target Study Annotations

This script checks which studies in the AML scAtlas have annotations
compatible with van Galen's cell type framework.

Use this BEFORE running label_transfer_benchmark.py to ensure you're
using appropriate target studies!
"""

import sys
import scanpy as sc
import pandas as pd
from pathlib import Path

DATA_PATH = "data/AML_scAtlas.h5ad"

def main():
    print("\n" + "="*80)
    print("TARGET STUDY ANNOTATION INSPECTOR")
    print("="*80)

    # Check if data exists
    if not Path(DATA_PATH).exists():
        print(f"\n❌ Data file not found: {DATA_PATH}")
        print("   Please download AML_scAtlas.h5ad and place in data/")
        return

    # Load data
    print(f"\nLoading: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"✓ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Detect annotation column
    label_key_candidates = [
        'cell_type_annotation',
        'celltype',
        'CellType',
        'cell_type',
        'annotation',
        'Annotation',
        'celltype_annotation'
    ]

    label_key = None
    for key in label_key_candidates:
        if key in adata.obs.columns:
            label_key = key
            break

    if label_key is None:
        print(f"\n❌ Could not find cell type annotation column!")
        print(f"   Available columns: {adata.obs.columns.tolist()}")
        return

    print(f"✓ Using annotation column: '{label_key}'")

    # Detect study column
    study_key = 'Study' if 'Study' in adata.obs.columns else 'study'
    print(f"✓ Using study column: '{study_key}'")

    # Reference study
    ref_study = 'van_galen_2019'

    if ref_study not in adata.obs[study_key].values:
        print(f"\n❌ Reference study '{ref_study}' not found!")
        print(f"   Available studies:")
        for study in sorted(adata.obs[study_key].unique()):
            print(f"     - {study}")
        return

    # Get reference labels
    ref_mask = adata.obs[study_key] == ref_study
    ref_subset = adata[ref_mask]
    ref_labels = ref_subset.obs[label_key].unique()
    ref_labels = [l for l in ref_labels if pd.notna(l)]

    print("\n" + "="*80)
    print("VAN GALEN REFERENCE LABELS")
    print("="*80)
    print(f"Study: {ref_study}")
    print(f"Cells: {ref_mask.sum():,}")
    print(f"Cell types: {len(ref_labels)}")
    print("\nLabel distribution:")

    label_counts = ref_subset.obs[label_key].value_counts()
    for label, count in label_counts.items():
        if pd.notna(label):
            print(f"  {label:30s}: {count:6,} cells")

    # Candidate target studies
    candidate_targets = [
        'zhang_2023',
        'beneyto-calabuig-2023',
        'jiang_2020',
        'velten_2021',
        'zhai_2022',
        'pei_2020',
        'setty_2019',
        'oetjen_2018',
        'abbas_2021',
    ]

    print("\n" + "="*80)
    print("TARGET STUDIES ANALYSIS")
    print("="*80)

    compatible_studies = []
    partially_compatible = []
    incompatible_studies = []
    missing_studies = []

    for target in candidate_targets:
        if target not in adata.obs[study_key].values:
            missing_studies.append(target)
            continue

        target_mask = adata.obs[study_key] == target
        target_subset = adata[target_mask]
        target_labels = target_subset.obs[label_key].unique()
        target_labels = [l for l in target_labels if pd.notna(l)]

        print(f"\n{'─'*80}")
        print(f"{target}")
        print(f"{'─'*80}")
        print(f"Cells: {target_mask.sum():,}")
        print(f"Cell types: {len(target_labels)}")

        # Show top labels
        print(f"\nTop 10 labels:")
        label_counts = target_subset.obs[label_key].value_counts().head(10)
        for label, count in label_counts.items():
            if pd.notna(label):
                in_ref = "✓" if label in ref_labels else "✗"
                print(f"  [{in_ref}] {label:30s}: {count:6,} cells")

        # Check overlap with van Galen
        overlap = set(target_labels) & set(ref_labels)
        overlap_pct = len(overlap) / len(ref_labels) * 100 if len(ref_labels) > 0 else 0

        print(f"\nOverlap with van Galen:")
        print(f"  Matching labels: {len(overlap)}/{len(ref_labels)} ({overlap_pct:.0f}%)")

        if len(overlap) > 0:
            print(f"  Shared labels: {', '.join(sorted(overlap)[:5])}")
            if len(overlap) > 5:
                print(f"                 ... and {len(overlap) - 5} more")

        # Categorize compatibility
        if overlap_pct >= 70:
            print(f"\n  ✅ HIGHLY COMPATIBLE (≥70% overlap)")
            print(f"     → Excellent for label transfer validation!")
            compatible_studies.append(target)
        elif overlap_pct >= 40:
            print(f"\n  ⚠️  PARTIALLY COMPATIBLE (40-70% overlap)")
            print(f"     → May work but some labels won't match")
            partially_compatible.append(target)
        else:
            print(f"\n  ❌ INCOMPATIBLE (<40% overlap)")
            print(f"     → Not suitable for label-based validation")
            incompatible_studies.append(target)

    # Missing studies
    if len(missing_studies) > 0:
        print(f"\n{'─'*80}")
        print(f"Studies NOT FOUND in atlas:")
        print(f"{'─'*80}")
        for study in missing_studies:
            print(f"  ❌ {study}")

    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if len(compatible_studies) > 0:
        print("\n✅ RECOMMENDED: Use highly compatible studies")
        print("\nAdd to label_transfer_benchmark.py:")
        print("```python")
        print("REFERENCE_STUDY = 'van_galen_2019'")
        print("TARGET_STUDIES = [")
        for study in compatible_studies:
            print(f"    '{study}',  # ✅ {len(set(adata[adata.obs[study_key] == study].obs[label_key].unique()) & set(ref_labels))}/{len(ref_labels)} matching labels")
        print("]")
        print("```")

    if len(partially_compatible) > 0:
        print("\n⚠️  OPTIONAL: Partially compatible studies (use with caution)")
        print("\nCould add if needed:")
        print("```python")
        print("TARGET_STUDIES = [")
        for study in compatible_studies:
            print(f"    '{study}',  # ✅ Highly compatible")
        for study in partially_compatible:
            print(f"    '{study}',  # ⚠️ Partially compatible")
        print("]")
        print("```")

    if len(incompatible_studies) > 0:
        print(f"\n❌ NOT RECOMMENDED: Incompatible studies")
        print(f"\nThese studies have different annotation schemes:")
        for study in incompatible_studies:
            print(f"  - {study}")
        print("\nFor these, consider using marker-based validation instead:")
        print("  python validate_marker_expression.py")

    if len(compatible_studies) == 0 and len(partially_compatible) == 0:
        print("\n⚠️  WARNING: No compatible studies found!")
        print("\nPossible reasons:")
        print("  1. Atlas uses different annotation column")
        print("  2. Studies use different cell type naming conventions")
        print("  3. Need to create label mapping function")
        print("\nRecommended alternatives:")
        print("  1. Use marker-based validation (no labels needed)")
        print("  2. Manually map labels between naming schemes")
        print("  3. Check atlas documentation for annotation details")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Category':<30} {'Count':<10} {'Studies'}")
    print(f"{'-'*80}")
    print(f"{'Highly compatible (≥70%)':<30} {len(compatible_studies):<10} {', '.join(compatible_studies)}")
    print(f"{'Partially compatible (40-70%)':<30} {len(partially_compatible):<10} {', '.join(partially_compatible)}")
    print(f"{'Incompatible (<40%)':<30} {len(incompatible_studies):<10} {', '.join(incompatible_studies)}")
    print(f"{'Not found in atlas':<30} {len(missing_studies):<10} {', '.join(missing_studies)}")

    print("\n" + "="*80)
    print("✓ INSPECTION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Update TARGET_STUDIES in label_transfer_benchmark.py")
    print("2. Run: python label_transfer_benchmark.py")
    print("3. Or use marker-based validation if no compatible studies found")


if __name__ == "__main__":
    main()
