#!/usr/bin/env python3
"""
Check if AML atlas has raw counts in layers['counts']
"""

import scanpy as sc
import os

DATA_PATH = "data/AML_scAtlas.h5ad"

if not os.path.exists(DATA_PATH):
    print(f"✗ Data file not found: {DATA_PATH}")
    print("\nThis script checks if your data has raw counts preserved.")
    print("Without raw counts, SCimilarity won't work properly!")
    exit(1)

print("="*80)
print("CHECKING RAW COUNTS IN AML ATLAS")
print("="*80)

adata = sc.read_h5ad(DATA_PATH)

print(f"\nLoaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Check what's in the AnnData
print("\n" + "="*80)
print("ANNDATA STRUCTURE")
print("="*80)

print("\n1. Main matrix (.X):")
print(f"   Type: {type(adata.X)}")
print(f"   Shape: {adata.X.shape}")
print(f"   Max value: {adata.X.max():.2f}")
print(f"   Min value: {adata.X.min():.2f}")

if adata.X.max() > 100:
    print("   → Looks like RAW COUNTS (max > 100)")
elif adata.X.max() < 20:
    print("   → Looks like LOG-NORMALIZED (max < 20)")
else:
    print("   → Unclear - could be normalized counts")

print("\n2. Layers:")
if len(adata.layers) == 0:
    print("   ✗ No layers found!")
else:
    for layer_name in adata.layers.keys():
        layer = adata.layers[layer_name]
        print(f"\n   '{layer_name}':")
        print(f"      Type: {type(layer)}")
        print(f"      Shape: {layer.shape}")
        print(f"      Max: {layer.max():.2f}")
        print(f"      Min: {layer.min():.2f}")

        if layer_name == 'counts':
            if layer.max() > 100:
                print("      ✓ This is RAW COUNTS! SCimilarity will use this.")
            else:
                print("      ⚠ WARNING: 'counts' layer has low max value - might not be raw?")

print("\n3. .raw:")
if adata.raw is None:
    print("   ✗ No .raw attribute")
else:
    print(f"   ✓ .raw exists")
    print(f"      Shape: {adata.raw.X.shape}")
    print(f"      Max: {adata.raw.X.max():.2f}")

# Summary
print("\n" + "="*80)
print("SUMMARY FOR SCIMILARITY")
print("="*80)

has_raw_counts = False
source = None

if 'counts' in adata.layers:
    if adata.layers['counts'].max() > 100:
        has_raw_counts = True
        source = "layers['counts']"
        print("\n✓ RAW COUNTS FOUND in layers['counts']")
        print("  SCimilarity will use this (correct!)")
elif adata.raw is not None:
    if adata.raw.X.max() > 100:
        has_raw_counts = True
        source = ".raw.X"
        print("\n⚠ RAW COUNTS FOUND in .raw.X")
        print("  SCimilarity will use this as fallback")
elif adata.X.max() > 100:
    has_raw_counts = True
    source = ".X"
    print("\n⚠ RAW COUNTS FOUND in .X")
    print("  SCimilarity will use this (but .X might get overwritten!)")

if not has_raw_counts:
    print("\n✗ NO RAW COUNTS FOUND!")
    print("\nThis is a CRITICAL PROBLEM:")
    print("  - SCimilarity REQUIRES raw counts")
    print("  - Without them, embeddings will be wrong")
    print("  - Results will not be valid")
    print("\nYou need to get the original data with raw counts preserved.")

if has_raw_counts:
    print(f"\nSource: {source}")
    print("\n" + "="*80)
    print("VALIDATION CHECK")
    print("="*80)

    # Check a few cells
    if source == "layers['counts']":
        counts = adata.layers['counts']
    elif source == ".raw.X":
        counts = adata.raw.X
    else:
        counts = adata.X

    # Get first cell
    if hasattr(counts, 'toarray'):
        first_cell = counts[0].toarray().flatten()
    else:
        first_cell = counts[0].flatten()

    nonzero = first_cell[first_cell > 0]

    print(f"\nFirst cell statistics:")
    print(f"  Total genes: {len(first_cell)}")
    print(f"  Expressed genes: {len(nonzero)}")
    print(f"  Total counts: {nonzero.sum():.0f}")
    print(f"  Mean (non-zero): {nonzero.mean():.2f}")
    print(f"  Max: {nonzero.max():.0f}")

    # Check if values look like counts (integers or close to integers)
    if len(nonzero) > 0:
        sample = nonzero[:100]  # First 100 non-zero values
        are_integers = all(abs(x - round(x)) < 0.01 for x in sample)

        if are_integers:
            print("\n✓ Values look like integers (raw counts)")
        else:
            print("\n⚠ WARNING: Values are not integers!")
            print("  This might be TPM, CPM, or other normalized data")
            print("  SCimilarity expects RAW COUNTS (integers from sequencing)")

print("\n" + "="*80)
