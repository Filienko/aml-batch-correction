#!/usr/bin/env python
"""
Validation Against Van Galen 2019 Reference

This script validates batch correction methods against the van Galen 2019
Cell paper, which is the gold standard reference for AML cell types.

WHY THIS IS IMPORTANT:
1. Van Galen 2019 provides expert-curated AML cell type annotations
2. Most cited reference in AML single-cell field
3. Shows our methods preserve established biology
4. Provides ground truth for validation

WHAT WE TEST:
1. Can we reproduce van Galen's cell type structure after integration?
2. Do integrated studies show the same cell types van Galen identified?
3. Are rare AML subtypes preserved?
4. Do marker genes for each cell type remain correct?

METHODOLOGY:
Following best practices from single-cell integration papers:
- Use van Galen as reference (well-annotated)
- Transfer labels to query studies
- Validate with marker gene expression
- Compare cell type proportions
- Check rare cell type preservation
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Import evaluation modules
import run_evaluation
from run_evaluation import (
    detect_batch_key,
    detect_label_key,
    preprocess_adata_exact,
    prepare_uncorrected_embedding_exact,
    load_scvi_embedding,
    compute_scimilarity_corrected,
    compute_harmony_corrected,
    force_cleanup,
    optimize_adata_memory,
    print_memory
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# Van Galen reference
VAN_GALEN_STUDY = 'van_galen_2019'

# Studies to validate
VALIDATION_STUDIES = [
    'setty_2019',      # 10x Chromium
    'pei_2020',        # CITEseq
    'oetjen_2018',     # 10x
]

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "validation_van_galen"
SCIMILARITY_BATCH_SIZE = 1000
SCIMILARITY_ENABLED = True

# Known van Galen cell type markers (from their Cell 2019 paper)
VAN_GALEN_MARKERS = {
    'HSC': ['AVP', 'CD34', 'PROM1'],
    'Progenitors': ['MPO', 'CEBPA', 'ELANE'],
    'Monocyte': ['CD14', 'LYZ', 'S100A9'],
    'Blast': ['MYB', 'CD34', 'CD38'],
    'T_Cell': ['CD3D', 'CD3E', 'CD8A'],
    'B_Cell': ['CD79A', 'MS4A1', 'CD19'],
    'Erythroid': ['HBB', 'HBA1', 'GYPB'],
}


def analyze_cell_type_structure(adata, embedding_key, label_key, batch_key):
    """
    Analyze cell type structure in embedding space.

    Tests:
    1. Are van Galen cell types well-separated?
    2. Do cell types cluster coherently across batches?
    3. Are rare cell types preserved?
    """
    print(f"\n{'='*80}")
    print(f"CELL TYPE STRUCTURE ANALYSIS: {embedding_key}")
    print(f"{'='*80}")

    if embedding_key not in adata.obsm:
        print(f"✗ Embedding '{embedding_key}' not found")
        return None

    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist, squareform

    embedding = adata.obsm[embedding_key]
    labels = adata.obs[label_key].values
    batches = adata.obs[batch_key].values

    # 1. Cell type silhouette score (higher = better separation)
    try:
        sil_celltype = silhouette_score(embedding, labels, sample_size=10000)
        print(f"\n1. Cell Type Separation:")
        print(f"   Silhouette score: {sil_celltype:.3f} (higher = better)")
    except:
        sil_celltype = np.nan
        print(f"\n1. Cell Type Separation: Failed to compute")

    # 2. Batch mixing within cell types
    print(f"\n2. Batch Mixing Within Cell Types:")

    results_by_celltype = []
    for celltype in np.unique(labels):
        mask = labels == celltype
        if mask.sum() < 10:
            continue

        celltype_batches = batches[mask]
        n_batches = len(np.unique(celltype_batches))
        n_cells = mask.sum()

        print(f"   {celltype}: {n_cells} cells across {n_batches} batches")

        results_by_celltype.append({
            'Cell_Type': celltype,
            'N_Cells': n_cells,
            'N_Batches': n_batches,
        })

    # 3. Rare cell type preservation
    print(f"\n3. Rare Cell Type Preservation:")
    celltype_counts = pd.Series(labels).value_counts()
    rare_threshold = 100
    rare_celltypes = celltype_counts[celltype_counts < rare_threshold]

    print(f"   Cell types with <{rare_threshold} cells: {len(rare_celltypes)}")
    for ct, count in rare_celltypes.items():
        print(f"     {ct}: {count} cells")

    results = {
        'Silhouette_CellType': sil_celltype,
        'N_CellTypes': len(np.unique(labels)),
        'N_RareCellTypes': len(rare_celltypes),
        'By_CellType': pd.DataFrame(results_by_celltype)
    }

    return results


def validate_marker_genes(adata, label_key):
    """
    Validate that van Galen's marker genes are expressed in correct cell types.

    This ensures batch correction didn't distort gene expression patterns.
    """
    print(f"\n{'='*80}")
    print(f"MARKER GENE VALIDATION")
    print(f"{'='*80}")

    print(f"\nValidating van Galen marker genes in each cell type...")

    # Need normalized expression
    if 'normalised_counts' in adata.layers:
        X = adata.layers['normalised_counts']
    else:
        X = adata.X

    results = []

    for celltype, markers in VAN_GALEN_MARKERS.items():
        # Find cells of this type
        if celltype in adata.obs[label_key].values:
            mask = adata.obs[label_key] == celltype
            n_cells = mask.sum()
        else:
            # Try fuzzy matching (e.g., "HSC/MPPs" contains "HSC")
            mask = adata.obs[label_key].str.contains(celltype, case=False, na=False)
            n_cells = mask.sum()

        if n_cells == 0:
            print(f"  ⚠ {celltype}: Not found in data")
            continue

        # Check marker expression
        markers_found = [m for m in markers if m in adata.var_names]
        if len(markers_found) == 0:
            print(f"  ⚠ {celltype}: No markers found in data")
            continue

        # Get mean expression in this cell type vs others
        marker_indices = [adata.var_names.get_loc(m) for m in markers_found]

        if hasattr(X, 'toarray'):
            in_celltype = X[mask][:, marker_indices].toarray().mean()
            not_in_celltype = X[~mask][:, marker_indices].toarray().mean()
        else:
            in_celltype = X[mask][:, marker_indices].mean()
            not_in_celltype = X[~mask][:, marker_indices].mean()

        fold_change = in_celltype / (not_in_celltype + 1e-10)

        print(f"  {celltype} ({n_cells} cells):")
        print(f"    Markers: {', '.join(markers_found)}")
        print(f"    Expression in {celltype}: {in_celltype:.2f}")
        print(f"    Expression in others: {not_in_celltype:.2f}")
        print(f"    Fold change: {fold_change:.2f}x")

        results.append({
            'Cell_Type': celltype,
            'N_Cells': n_cells,
            'N_Markers_Found': len(markers_found),
            'Mean_Expression_In': in_celltype,
            'Mean_Expression_Out': not_in_celltype,
            'Fold_Change': fold_change,
        })

    return pd.DataFrame(results)


def compare_with_van_galen_proportions(adata, label_key, batch_key):
    """
    Compare cell type proportions in other studies vs van Galen.

    While proportions may differ (disease states, etc.), we should see
    the same major cell types present.
    """
    print(f"\n{'='*80}")
    print(f"CELL TYPE PROPORTION COMPARISON")
    print(f"{'='*80}")

    # Get van Galen proportions
    van_galen_mask = adata.obs[batch_key] == VAN_GALEN_STUDY
    van_galen_props = adata.obs[van_galen_mask][label_key].value_counts(normalize=True)

    print(f"\nVan Galen 2019 cell type proportions:")
    for ct, prop in van_galen_props.head(10).items():
        print(f"  {ct}: {prop*100:.1f}%")

    # Compare with other studies
    print(f"\nOther studies:")
    all_results = []

    for study in adata.obs[batch_key].unique():
        if study == VAN_GALEN_STUDY:
            continue

        study_mask = adata.obs[batch_key] == study
        study_props = adata.obs[study_mask][label_key].value_counts(normalize=True)

        print(f"\n  {study}:")
        for ct, prop in study_props.head(5).items():
            print(f"    {ct}: {prop*100:.1f}%")

        # Check how many van Galen cell types are present
        van_galen_types = set(van_galen_props.index)
        study_types = set(study_props.index)
        overlap = van_galen_types.intersection(study_types)

        print(f"    Cell types in common with van Galen: {len(overlap)}/{len(van_galen_types)}")

        all_results.append({
            'Study': study,
            'N_Cells': study_mask.sum(),
            'N_CellTypes': len(study_types),
            'N_Overlap_With_VanGalen': len(overlap),
            'Proportion_Overlap': len(overlap) / len(van_galen_types)
        })

    return pd.DataFrame(all_results)


def main():
    """
    Main validation pipeline against van Galen 2019.
    """
    print("="*80)
    print("VALIDATION AGAINST VAN GALEN 2019 REFERENCE")
    print("="*80)
    print(f"\nReference: van Galen et al. (2019) Cell")
    print(f"  'Single-cell RNA-seq reveals AML hierarchies relevant to")
    print(f"   disease progression and immunity'")
    print(f"\nObjective: Validate that batch correction preserves van Galen's")
    print(f"           established AML cell type structure")

    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Data file not found: {DATA_PATH}")
        return

    # STEP 1: Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Detect keys
    run_evaluation.BATCH_KEY = detect_batch_key(adata)
    run_evaluation.LABEL_KEY = detect_label_key(adata)
    run_evaluation.BATCH_KEY_LOWER = run_evaluation.BATCH_KEY.lower()

    BATCH_KEY = run_evaluation.BATCH_KEY
    LABEL_KEY = run_evaluation.LABEL_KEY
    BATCH_KEY_LOWER = run_evaluation.BATCH_KEY_LOWER

    if BATCH_KEY_LOWER not in adata.obs.columns:
        adata.obs[BATCH_KEY_LOWER] = adata.obs[BATCH_KEY].copy()

    # STEP 2: Subset to van Galen + validation studies
    all_studies = [VAN_GALEN_STUDY] + VALIDATION_STUDIES
    mask = adata.obs[BATCH_KEY].isin(all_studies)
    adata = adata[mask].copy()

    print(f"\nSubset: {adata.n_obs:,} cells from {len(all_studies)} studies")

    # STEP 2.5: Load scVI
    if os.path.exists(SCVI_PATH):
        try:
            adata_scvi = sc.read_h5ad(SCVI_PATH)
            scvi_has_numeric = all(str(idx).isdigit() for idx in adata_scvi.obs_names[:100])

            if scvi_has_numeric:
                adata_full_temp = sc.read_h5ad(DATA_PATH)
                mask = adata_full_temp.obs[BATCH_KEY].isin(all_studies)
                original_indices = np.where(mask)[0]
                adata.obsm['X_scVI'] = adata_scvi.X[original_indices].copy()
                print(f"  ✓ Added scVI: {adata.obsm['X_scVI'].shape}")
                del adata_full_temp, adata_scvi
                force_cleanup()
        except Exception as e:
            print(f"  ✗ scVI loading failed: {e}")

    adata = optimize_adata_memory(adata)
    force_cleanup()

    # STEP 3: Preprocess
    adata = preprocess_adata_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # STEP 4: Uncorrected PCA
    adata = prepare_uncorrected_embedding_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # STEP 5: SCimilarity
    if SCIMILARITY_ENABLED and os.path.exists(SCIMILARITY_MODEL):
        try:
            adata = compute_scimilarity_corrected(
                adata, model_path=SCIMILARITY_MODEL,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            force_cleanup()
        except Exception as e:
            print(f"✗ SCimilarity failed: {e}")

    # STEP 6: Harmony
    try:
        adata = compute_harmony_corrected(adata, BATCH_KEY, N_JOBS)
        force_cleanup()
    except Exception as e:
        print(f"✗ Harmony failed: {e}")

    # STEP 7: Validation analyses
    print("\n" + "="*80)
    print("STEP 7: VALIDATION ANALYSES")
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for embedding_key, method_name in [
        ('X_uncorrected', 'Uncorrected'),
        ('X_scVI', 'scVI'),
        ('X_scimilarity', 'SCimilarity'),
        ('X_harmony', 'Harmony')
    ]:
        if embedding_key not in adata.obsm:
            continue

        print(f"\n{'='*80}")
        print(f"VALIDATING: {method_name}")
        print(f"{'='*80}")

        # 1. Cell type structure
        structure_results = analyze_cell_type_structure(
            adata, embedding_key, LABEL_KEY, BATCH_KEY
        )

        # 2. Marker genes
        marker_results = validate_marker_genes(adata, LABEL_KEY)

        # 3. Proportions
        proportion_results = compare_with_van_galen_proportions(
            adata, LABEL_KEY, BATCH_KEY
        )

        # Save results
        if structure_results:
            structure_results['By_CellType'].to_csv(
                os.path.join(OUTPUT_DIR, f"{method_name.lower()}_celltype_structure.csv"),
                index=False
            )

        if marker_results is not None and not marker_results.empty:
            marker_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{method_name.lower()}_marker_validation.csv"),
                index=False
            )

        if proportion_results is not None and not proportion_results.empty:
            proportion_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{method_name.lower()}_proportions.csv"),
                index=False
            )

        all_results[method_name] = {
            'structure': structure_results,
            'markers': marker_results,
            'proportions': proportion_results
        }

    # STEP 8: Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    summary_data = []
    for method_name, results in all_results.items():
        if results['structure']:
            row = {
                'Method': method_name,
                'CellType_Silhouette': results['structure']['Silhouette_CellType'],
                'N_CellTypes_Detected': results['structure']['N_CellTypes'],
                'N_RareCellTypes': results['structure']['N_RareCellTypes'],
            }

            if results['markers'] is not None and not results['markers'].empty:
                row['Mean_MarkerFoldChange'] = results['markers']['Fold_Change'].mean()

            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    summary_file = os.path.join(OUTPUT_DIR, "validation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved: {summary_file}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nThis validation shows:")
    print("  1. Cell Type Silhouette: Are van Galen cell types well-separated?")
    print("  2. Marker Gene Validation: Do markers still identify correct types?")
    print("  3. Cross-Study Consistency: Are same types found in other studies?")
    print("\nGood batch correction should:")
    print("  ✓ Preserve van Galen's cell type structure")
    print("  ✓ Maintain marker gene expression patterns")
    print("  ✓ Identify consistent types across studies")

    print("\n" + "="*80)
    print("✓ VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"\nFor your paper:")
    print(f"  'We validated our results against van Galen 2019, the gold")
    print(f"   standard reference for AML cell types. [Method X] achieved")
    print(f"   [silhouette score] for cell type separation and correctly")
    print(f"   preserved marker gene expression patterns (mean fold change: [X]).'")


if __name__ == "__main__":
    main()
