#!/usr/bin/env python3
"""
Verify the converted NPM1 h5ad file has everything needed for validation
"""

import scanpy as sc
import numpy as np
import os

h5ad_file = "data/NPM1_AML.h5ad"

print("=" * 80)
print("VERIFYING CONVERTED NPM1 H5AD FILE")
print("=" * 80)

if not os.path.exists(h5ad_file):
    print(f"\n✗ File not found: {h5ad_file}")
    print("\nRun the conversion first:")
    print("  Rscript convert_npm1_to_h5ad.R")
    exit(1)

print(f"\nLoading: {h5ad_file}")
adata = sc.read_h5ad(h5ad_file)

print(f"✓ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Check structure
print("\n" + "=" * 80)
print("DATA STRUCTURE")
print("=" * 80)

print("\n1. Main matrix (.X):")
print(f"   Type: {type(adata.X)}")
print(f"   Max: {adata.X.max():.2f}")
print(f"   Min: {adata.X.min():.2f}")

print("\n2. Layers:")
if len(adata.layers) == 0:
    print("   ✗ No layers found!")
else:
    for layer_name in adata.layers.keys():
        layer = adata.layers[layer_name]
        print(f"   '{layer_name}':")
        print(f"      Max: {layer.max():.2f}")

# Check for raw counts
print("\n" + "=" * 80)
print("RAW COUNTS CHECK")
print("=" * 80)

has_raw_counts = False

if 'counts' in adata.layers:
    max_val = adata.layers['counts'].max()
    print(f"\n✓ Found layers['counts']")
    print(f"  Max value: {max_val:.0f}")

    if max_val > 100:
        has_raw_counts = True
        print("  ✓ This looks like raw counts!")

        # Check if integers
        if hasattr(adata.layers['counts'], 'toarray'):
            sample = adata.layers['counts'][0].toarray().flatten()
        else:
            sample = adata.layers['counts'][0].flatten()

        nonzero = sample[sample > 0][:100]
        are_ints = all(abs(x - round(x)) < 0.01 for x in nonzero)

        if are_ints:
            print("  ✓ Values are integers (raw counts confirmed)")
        else:
            print("  ⚠ Values are not integers (might be normalized)")
    else:
        print("  ✗ Max value too low - not raw counts")

elif adata.X.max() > 100:
    has_raw_counts = True
    print(f"\n⚠ Raw counts in .X (max = {adata.X.max():.0f})")
    print("  Consider moving to layers['counts']")

else:
    print("\n✗ No raw counts found!")
    has_raw_counts = False

# Check metadata
print("\n" + "=" * 80)
print("METADATA")
print("=" * 80)

print(f"\nColumns: {len(adata.obs.columns)}")

# Look for cell type columns
cell_type_cols = [col for col in adata.obs.columns
                  if any(x in col.lower() for x in ['cell', 'type', 'cluster', 'annotation'])]

print(f"\nPotential cell type columns ({len(cell_type_cols)}):")
for col in cell_type_cols[:10]:
    n_unique = adata.obs[col].nunique()
    print(f"  {col}: {n_unique} unique values")

# Check for Study column
if 'Study' in adata.obs.columns:
    print(f"\n✓ Study column found")
    studies = adata.obs['Study'].unique()
    print(f"  Studies: {list(studies)}")
else:
    print("\n✗ No 'Study' column - add it for validation")

# Check for van Galen subtypes
print("\n" + "=" * 80)
print("VAN GALEN SUBTYPE SEARCH")
print("=" * 80)

van_galen_subtypes = ['HSPC', 'CMP', 'GMP', 'ProMono', 'CD14+ Mono', 'cDC']
found_subtypes = []

for col in adata.obs.columns:
    labels = adata.obs[col].astype(str).unique()
    for vg in van_galen_subtypes:
        if any(vg.lower() in label.lower() for label in labels):
            found_subtypes.append(vg)
            print(f"  ✓ Found '{vg}' in column '{col}'")
            break

if len(found_subtypes) == 0:
    print("  ✗ No van Galen subtypes found")
    print("  Manual inspection needed")

# Final summary
print("\n" + "=" * 80)
print("VALIDATION READINESS")
print("=" * 80)

issues = []

if not has_raw_counts:
    issues.append("✗ No raw counts")
if 'Study' not in adata.obs.columns:
    issues.append("⚠ No Study column")
if len(found_subtypes) == 0:
    issues.append("⚠ No van Galen subtype labels found")

if len(issues) == 0:
    print("\n✓✓✓ FILE IS READY FOR VALIDATION! ✓✓✓")
    print("\nNext steps:")
    print("  1. Edit validate_aml_subtypes.py:")
    print("     TEST_STUDIES = ['velten_2021', 'npm1_2024']")
    print("  2. Run validation:")
    print("     python validate_aml_subtypes.py")
else:
    print("\n⚠ Issues found:")
    for issue in issues:
        print(f"  {issue}")

    if not has_raw_counts:
        print("\n✗ CRITICAL: No raw counts - cannot use for SCimilarity")
        print("  Get raw data from EGA: EGAS50000000332")

print()
