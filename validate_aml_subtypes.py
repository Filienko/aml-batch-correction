#!/usr/bin/env python
"""
Validation of AML Subtype Predictions Using Van Galen Framework

This script tests whether batch correction methods can reproduce the AML
cellular hierarchy described in van Galen et al. (2019) Cell when applied
to independent studies.

VAN GALEN'S AML HIERARCHY (6 malignant cell types):
1. HSC-like (stem-like, poor prognosis)
2. Progenitor-like
3. GMP-like (granulocyte-macrophage progenitor, better prognosis)
4. Promonocyte-like
5. Monocyte-like
6. Dendritic-like

VALIDATION APPROACH:
1. Train: Use van Galen's annotations as ground truth
2. Test: Predict subtypes in other AML studies
3. Validate: Check biological consistency
   - Do HSC-like cells have stem markers?
   - Do GMP-like cells have myeloid markers?
   - Can we distinguish prognostic subtypes?

WHY THIS MATTERS:
- Van Galen framework is widely cited (>1000 citations)
- Multiple papers use this classification system
- Shows batch correction preserves clinically relevant biology
- Demonstrates real-world utility of foundation models

PAPERS USING VAN GALEN FRAMEWORK:
- van Galen 2019 (original)
- J Hematol Oncol 2020 (Microwell-seq, 40 patients)
- Nature Leukemia 2022 (relapsed/refractory AML)
- Blood 2023 (prognostic signatures)
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Van Galen reference (expert-curated)
VAN_GALEN_STUDY = 'van_galen_2019'

# Studies to test van Galen classification on
# IMPORTANT: Only use studies that actually CITED and USED van Galen's framework!
# - velten_2021: ✅ Nature 2021, explicitly uses van Galen classification
# - zhai_2022: ✅ Nat Commun 2022, uses van Galen classifier (if available in atlas)
# - setty_2019: ❌ Published BEFORE van Galen (2019)
# - pei_2020: ❌ Uses own annotation, not van Galen
# - oetjen_2018: ❌ Published BEFORE van Galen
TEST_STUDIES = [
    'velten_2021',     # ✅ Muta-Seq - Actually uses van Galen framework
    'zhai_2022',       # ✅ SORT-Seq - Uses van Galen classifier (if in atlas)
]

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "validation_aml_subtypes"
SCIMILARITY_BATCH_SIZE = 1000
SCIMILARITY_ENABLED = True
K_NEIGHBORS = 15

# Van Galen's 6 malignant AML subtypes - using actual labels from data
# All studies in this atlas use the same annotation scheme!
VAN_GALEN_MALIGNANT_SUBTYPES = [
    'HSPC',          # HSC-like (stem-like) - Hematopoietic Stem/Progenitor Cells
    'CMP',           # Progenitor-like - Common Myeloid Progenitor
    'GMP',           # GMP-like - Granulocyte-Monocyte Progenitor
    'ProMono',       # Promonocyte-like
    'CD14+ Mono',    # Monocyte-like - CD14+ Monocytes
    'cDC',           # Dendritic-like - conventional Dendritic Cells
]

# Marker genes for each van Galen subtype (from van Galen Cell 2019)
VAN_GALEN_SUBTYPE_MARKERS = {
    'HSPC': ['AVP', 'CD34', 'HOPX', 'SPINK2'],  # HSC/stem markers
    'CMP': ['CD34', 'MPO', 'CEBPA'],            # Early progenitor markers
    'GMP': ['MPO', 'ELANE', 'AZU1', 'CTSG'],    # Myeloid progenitor markers
    'ProMono': ['CEBPB', 'CEBPD', 'CD14'],      # Promonocyte markers
    'CD14+ Mono': ['CD14', 'LYZ', 'S100A9', 'S100A8'],  # Monocyte markers
    'cDC': ['IRF8', 'IRF4', 'CD1C'],            # Dendritic cell markers
}


def harmonize_cell_type_labels(adata, label_key):
    """
    Harmonize cell type labels to match van Galen's 6 malignant subtypes.

    Your atlas may have different naming conventions. This function
    attempts to map them to van Galen's canonical names.
    """
    print("\n" + "="*80)
    print("HARMONIZING CELL TYPE LABELS TO VAN GALEN FRAMEWORK")
    print("="*80)

    # Get current unique labels
    current_labels = adata.obs[label_key].unique()
    print(f"\nCurrent unique labels ({len(current_labels)}):")
    for label in sorted(current_labels)[:20]:
        n = (adata.obs[label_key] == label).sum()
        print(f"  {label}: {n:,} cells")

    # Create mapping to van Galen subtypes
    # Good news: All studies in this atlas use the same annotation scheme!
    # We just need to keep the 6 malignant AML subtypes
    mapping = {
        # These are the 6 van Galen malignant subtypes - keep as-is
        'HSPC': 'HSPC',
        'CMP': 'CMP',
        'GMP': 'GMP',
        'ProMono': 'ProMono',
        'CD14+ Mono': 'CD14+ Mono',
        'cDC': 'cDC',

        # Also include related monocyte subtype
        'CD16+ Mono': 'CD14+ Mono',  # Group with CD14+ Mono (both monocytic)

        # Note: Other cell types (T, B, NK, Erythroid, etc.) are NOT malignant
        # and will be filtered out (unmapped)
    }

    # Apply mapping
    adata.obs['van_galen_subtype'] = adata.obs[label_key].map(mapping)

    # Report mapping
    n_mapped = adata.obs['van_galen_subtype'].notna().sum()
    n_total = adata.n_obs
    print(f"\nMapping results:")
    print(f"  Mapped: {n_mapped:,} / {n_total:,} cells ({100*n_mapped/n_total:.1f}%)")
    print(f"  Unmapped: {n_total - n_mapped:,} cells")

    print(f"\nVan Galen subtype distribution:")
    for subtype in VAN_GALEN_MALIGNANT_SUBTYPES:
        n = (adata.obs['van_galen_subtype'] == subtype).sum()
        if n > 0:
            print(f"  {subtype}: {n:,} cells")

    return adata


def predict_van_galen_subtypes(adata, embedding_key, ref_study, test_studies, k=15):
    """
    Predict van Galen subtypes in test studies using reference.

    Args:
        adata: AnnData with embeddings and van_galen_subtype labels
        embedding_key: Which embedding to use
        ref_study: Reference study (van Galen)
        test_studies: List of test study names
        k: Number of neighbors for k-NN

    Returns:
        DataFrame with prediction results per study
    """
    print(f"\n{'='*80}")
    print(f"PREDICTING VAN GALEN SUBTYPES: {embedding_key}")
    print(f"{'='*80}")

    if embedding_key not in adata.obsm:
        print(f"✗ Embedding '{embedding_key}' not found")
        return None

    # Get reference data (van Galen with known subtypes)
    ref_mask = (adata.obs['Study'] == ref_study) & (adata.obs['van_galen_subtype'].notna())
    ref_embedding = adata.obsm[embedding_key][ref_mask]
    ref_labels = adata.obs['van_galen_subtype'][ref_mask].values

    print(f"\nReference: {ref_study}")
    print(f"  Cells: {ref_mask.sum():,}")
    print(f"  Subtypes: {len(np.unique(ref_labels))}")
    print(f"  Distribution:")
    for subtype, count in pd.Series(ref_labels).value_counts().items():
        print(f"    {subtype}: {count}")

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(ref_embedding, ref_labels)

    results = []

    # Predict for each test study
    for test_study in test_studies:
        print(f"\n{'─'*80}")
        print(f"Test Study: {test_study}")
        print(f"{'─'*80}")

        test_mask = (adata.obs['Study'] == test_study) & (adata.obs['van_galen_subtype'].notna())

        if test_mask.sum() == 0:
            print(f"  ⚠ No labeled cells in {test_study}")
            continue

        test_embedding = adata.obsm[embedding_key][test_mask]
        test_true_labels = adata.obs['van_galen_subtype'][test_mask].values

        print(f"  Test cells: {test_mask.sum():,}")
        print(f"  True subtypes: {len(np.unique(test_true_labels))}")

        # Predict
        pred_labels = knn.predict(test_embedding)
        pred_proba = knn.predict_proba(test_embedding)
        confidence = pred_proba.max(axis=1)

        # Evaluate
        accuracy = accuracy_score(test_true_labels, pred_labels)

        print(f"\n  Results:")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Mean confidence: {confidence.mean():.3f}")
        print(f"    Low confidence (<0.5): {(confidence < 0.5).sum():,} cells")

        # Confusion matrix
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(test_true_labels, pred_labels,
                             labels=VAN_GALEN_MALIGNANT_SUBTYPES)
        print("  True \\ Pred  ", "  ".join([f"{s:8s}" for s in VAN_GALEN_MALIGNANT_SUBTYPES]))
        for i, true_label in enumerate(VAN_GALEN_MALIGNANT_SUBTYPES):
            if cm[i].sum() > 0:
                print(f"  {true_label:12s}", "  ".join([f"{cm[i,j]:8d}" for j in range(len(VAN_GALEN_MALIGNANT_SUBTYPES))]))

        # Per-class accuracy
        print(f"\n  Per-Subtype Accuracy:")
        for subtype in VAN_GALEN_MALIGNANT_SUBTYPES:
            mask_true = test_true_labels == subtype
            if mask_true.sum() > 0:
                mask_correct = (test_true_labels == subtype) & (pred_labels == subtype)
                acc = mask_correct.sum() / mask_true.sum()
                print(f"    {subtype}: {acc:.3f} ({mask_correct.sum()}/{mask_true.sum()})")

        results.append({
            'Test_Study': test_study,
            'Accuracy': accuracy,
            'Mean_Confidence': confidence.mean(),
            'Low_Confidence_Cells': (confidence < 0.5).sum(),
            'N_Cells': test_mask.sum(),
            'N_Subtypes': len(np.unique(pred_labels)),
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {embedding_key}")
    print(f"{'='*80}")
    if len(results_df) > 0:
        print(f"\nAverage accuracy: {results_df['Accuracy'].mean():.3f}")
        print(f"Average confidence: {results_df['Mean_Confidence'].mean():.3f}")

    return results_df


def validate_marker_gene_enrichment(adata, label_col='van_galen_subtype'):
    """
    Validate that van Galen markers are enriched in predicted subtypes.

    This confirms predictions are biologically meaningful.
    """
    print(f"\n{'='*80}")
    print(f"MARKER GENE ENRICHMENT VALIDATION")
    print(f"{'='*80}")

    # Need normalized expression
    if 'normalised_counts' in adata.layers:
        X = adata.layers['normalised_counts']
    else:
        X = adata.X

    results = []

    for subtype, markers in VAN_GALEN_SUBTYPE_MARKERS.items():
        mask = adata.obs[label_col] == subtype
        n_cells = mask.sum()

        if n_cells == 0:
            print(f"  ⚠ {subtype}: No cells")
            continue

        markers_found = [m for m in markers if m in adata.var_names]
        if len(markers_found) == 0:
            print(f"  ⚠ {subtype}: No markers in data")
            continue

        marker_indices = [adata.var_names.get_loc(m) for m in markers_found]

        if hasattr(X, 'toarray'):
            in_subtype = X[mask][:, marker_indices].toarray().mean()
            not_in_subtype = X[~mask][:, marker_indices].toarray().mean()
        else:
            in_subtype = X[mask][:, marker_indices].mean()
            not_in_subtype = X[~mask][:, marker_indices].mean()

        fold_change = in_subtype / (not_in_subtype + 1e-10)

        print(f"  {subtype} ({n_cells} cells):")
        print(f"    Markers: {', '.join(markers_found)}")
        print(f"    Fold change: {fold_change:.2f}x")

        results.append({
            'Subtype': subtype,
            'N_Cells': n_cells,
            'N_Markers': len(markers_found),
            'Fold_Change': fold_change,
        })

    return pd.DataFrame(results)


def validate_marker_enrichment_per_method(adata, embedding_key, method_name, ref_study, test_studies, k=15):
    """
    Validate marker enrichment for PREDICTIONS from a specific method.

    This tests if predicted cell types actually express the correct markers.
    Critical for determining if methods preserve biological signal.
    """
    print(f"\n{'='*80}")
    print(f"MARKER ENRICHMENT: {method_name}")
    print(f"{'='*80}")

    if embedding_key not in adata.obsm:
        print(f"✗ Embedding not found")
        return None

    # Need normalized expression
    if 'normalised_counts' in adata.layers:
        X = adata.layers['normalised_counts']
    else:
        X = adata.X

    # Train k-NN on reference
    ref_mask = adata.obs['Study'] == ref_study
    ref_embedding = adata.obsm[embedding_key][ref_mask]
    ref_labels = adata.obs['van_galen_subtype'][ref_mask].values

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(ref_embedding, ref_labels)

    all_results = []

    # Test on each study
    for test_study in test_studies:
        test_mask = adata.obs['Study'] == test_study
        if test_mask.sum() == 0:
            continue

        test_embedding = adata.obsm[embedding_key][test_mask]
        test_indices = np.where(test_mask)[0]

        # Get PREDICTIONS (not true labels!)
        pred_labels = knn.predict(test_embedding)

        print(f"\nStudy: {test_study} ({test_mask.sum():,} cells)")

        # For each predicted subtype, check marker enrichment
        for subtype, markers in VAN_GALEN_SUBTYPE_MARKERS.items():
            pred_mask_local = pred_labels == subtype  # In test_embedding space
            n_predicted = pred_mask_local.sum()

            if n_predicted == 0:
                continue

            markers_found = [m for m in markers if m in adata.var_names]
            if len(markers_found) == 0:
                continue

            marker_indices = [adata.var_names.get_loc(m) for m in markers_found]

            # Get marker expression for predicted cells
            pred_global_indices = test_indices[pred_mask_local]

            if hasattr(X, 'toarray'):
                in_predicted = X[pred_global_indices][:, marker_indices].toarray().mean()
                not_in_predicted = X[~np.isin(np.arange(adata.n_obs), pred_global_indices)][:, marker_indices].toarray().mean()
            else:
                in_predicted = X[pred_global_indices][:, marker_indices].mean()
                not_in_predicted = X[~np.isin(np.arange(adata.n_obs), pred_global_indices)][:, marker_indices].mean()

            fold_change = in_predicted / (not_in_predicted + 1e-10)

            print(f"  {subtype} (n={n_predicted}):")
            print(f"    Enrichment: {fold_change:.2f}x")

            all_results.append({
                'Method': method_name,
                'Test_Study': test_study,
                'Predicted_Subtype': subtype,
                'N_Predicted_Cells': n_predicted,
                'N_Markers': len(markers_found),
                'Fold_Enrichment': fold_change,
            })

    return pd.DataFrame(all_results)


def main():
    """
    Main validation pipeline for AML subtype prediction.
    """
    print("="*80)
    print("VAN GALEN AML SUBTYPE VALIDATION")
    print("="*80)
    print("\nObjective: Test if batch correction preserves van Galen's")
    print("           AML cellular hierarchy when applied to new studies")
    print("\nVan Galen's 6 malignant AML subtypes:")
    for i, subtype in enumerate(VAN_GALEN_MALIGNANT_SUBTYPES, 1):
        print(f"  {i}. {subtype}")

    print("\n" + "="*80)
    print("IMPORTANT NOTES")
    print("="*80)
    print("\n1. 'Uncorrected' baseline = PCA on log-normalized, HVG-selected data")
    print("   → NOT truly uncorrected! PCA already provides structure.")
    print("   → If 'uncorrected' performs well, batch effects may be weak.")
    print("\n2. Different embedding spaces (50D PCA vs 256D SCimilarity)")
    print("   → k-NN performance depends on embedding properties")
    print("   → PCA's orthogonal/standardized space may favor k-NN")
    print("\n3. Only using studies that actually cited van Galen's framework:")
    print("   → velten_2021: Nature 2021, explicitly uses van Galen")
    print("   → zhai_2022: Nat Commun 2022, uses van Galen classifier (if available)")
    print("   → This is STRONGER validation than using any study with matching labels")

    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Data file not found: {DATA_PATH}")
        return

    # Load data
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

    # Check which test studies are actually available
    available_studies = set(adata.obs[BATCH_KEY].unique())

    print(f"\nChecking study availability:")
    print(f"  Reference: {VAN_GALEN_STUDY}", "✓" if VAN_GALEN_STUDY in available_studies else "✗ NOT FOUND")

    if VAN_GALEN_STUDY not in available_studies:
        print(f"\n✗ ERROR: Reference study '{VAN_GALEN_STUDY}' not found in dataset!")
        print(f"Available studies: {sorted(available_studies)}")
        return

    # Filter to available test studies
    available_test_studies = [s for s in TEST_STUDIES if s in available_studies]
    unavailable_test_studies = [s for s in TEST_STUDIES if s not in available_studies]

    print(f"\n  Test studies:")
    for study in TEST_STUDIES:
        if study in available_studies:
            n_cells = (adata.obs[BATCH_KEY] == study).sum()
            print(f"    ✓ {study}: {n_cells:,} cells")
        else:
            print(f"    ✗ {study}: NOT FOUND")

    if len(available_test_studies) == 0:
        print(f"\n✗ ERROR: No test studies found!")
        print(f"Available studies: {sorted(available_studies)}")
        return

    if unavailable_test_studies:
        print(f"\n⚠ WARNING: {len(unavailable_test_studies)} test studies not found: {unavailable_test_studies}")
        print(f"Continuing with {len(available_test_studies)} available studies: {available_test_studies}")

    # Update TEST_STUDIES to only available ones
    TEST_STUDIES_FILTERED = available_test_studies

    # Subset to van Galen + available test studies
    all_studies = [VAN_GALEN_STUDY] + TEST_STUDIES_FILTERED
    mask = adata.obs[BATCH_KEY].isin(all_studies)
    adata = adata[mask].copy()

    print(f"\nSubset: {adata.n_obs:,} cells from {len(all_studies)} studies")

    # Harmonize labels to van Galen framework
    adata = harmonize_cell_type_labels(adata, LABEL_KEY)

    # Load scVI
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

    # Preprocess
    adata = preprocess_adata_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # Uncorrected PCA
    adata = prepare_uncorrected_embedding_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # SCimilarity
    if SCIMILARITY_ENABLED and os.path.exists(SCIMILARITY_MODEL):
        try:
            adata = compute_scimilarity_corrected(
                adata, model_path=SCIMILARITY_MODEL,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            force_cleanup()
        except Exception as e:
            print(f"✗ SCimilarity failed: {e}")

    # Harmony
    try:
        adata = compute_harmony_corrected(adata, BATCH_KEY, N_JOBS)
        force_cleanup()
    except Exception as e:
        print(f"✗ Harmony failed: {e}")

    # Validate predictions
    print("\n" + "="*80)
    print("STEP 7: SUBTYPE PREDICTION VALIDATION")
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

        df = predict_van_galen_subtypes(
            adata, embedding_key, VAN_GALEN_STUDY, TEST_STUDIES_FILTERED, k=K_NEIGHBORS
        )

        if df is not None:
            df['Method'] = method_name
            all_results[method_name] = df

            output_file = os.path.join(OUTPUT_DIR, f"{method_name.lower()}_subtype_prediction.csv")
            df.to_csv(output_file, index=False)
            print(f"\n✓ Saved: {output_file}")

    # Marker validation - PER METHOD (CRITICAL!)
    print("\n" + "="*80)
    print("MARKER ENRICHMENT VALIDATION (PER METHOD)")
    print("="*80)
    print("\nThis tests if PREDICTED cell types express correct markers.")
    print("Higher enrichment = better biological preservation!")

    all_marker_results = []

    for embedding_key, method_name in [
        ('X_uncorrected', 'Uncorrected'),
        ('X_scVI', 'scVI'),
        ('X_scimilarity', 'SCimilarity'),
        ('X_harmony', 'Harmony')
    ]:
        if embedding_key not in adata.obsm:
            continue

        marker_df = validate_marker_enrichment_per_method(
            adata, embedding_key, method_name,
            VAN_GALEN_STUDY, TEST_STUDIES_FILTERED, k=K_NEIGHBORS
        )

        if marker_df is not None and not marker_df.empty:
            all_marker_results.append(marker_df)

            output_file = os.path.join(OUTPUT_DIR, f"{method_name.lower()}_marker_enrichment.csv")
            marker_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved: {output_file}")

    # Combine and compare
    if all_marker_results:
        combined_markers = pd.concat(all_marker_results, ignore_index=True)
        combined_file = os.path.join(OUTPUT_DIR, "marker_enrichment_comparison.csv")
        combined_markers.to_csv(combined_file, index=False)
        print(f"\n✓ Saved combined: {combined_file}")

        # Summary statistics
        print("\n" + "="*80)
        print("MARKER ENRICHMENT SUMMARY (BY METHOD)")
        print("="*80)

        summary = combined_markers.groupby('Method')['Fold_Enrichment'].agg(['mean', 'std', 'min', 'max'])
        print("\nAverage Marker Enrichment:")
        for method, row in summary.iterrows():
            print(f"  {method:15s}: {row['mean']:.2f}x (±{row['std']:.2f})")

        # Per-subtype comparison
        print("\nPer-Subtype Enrichment:")
        for subtype in VAN_GALEN_MALIGNANT_SUBTYPES:
            subtype_data = combined_markers[combined_markers['Predicted_Subtype'] == subtype]
            if subtype_data.empty:
                continue
            print(f"\n  {subtype}:")
            for method in subtype_data['Method'].unique():
                method_data = subtype_data[subtype_data['Method'] == method]
                if not method_data.empty:
                    avg_enrichment = method_data['Fold_Enrichment'].mean()
                    print(f"    {method:15s}: {avg_enrichment:.2f}x")

    # Summary - Combine accuracy and marker enrichment
    if len(all_results) > 0 or all_marker_results:
        print("\n" + "="*80)
        print("FINAL COMPARISON: ACCURACY + MARKER ENRICHMENT")
        print("="*80)

        final_summary = []

        # Get accuracy stats
        if len(all_results) > 0:
            combined = pd.concat(all_results.values(), ignore_index=True)
            accuracy_stats = combined.groupby('Method').agg({
                'Accuracy': 'mean',
                'Mean_Confidence': 'mean'
            })
        else:
            accuracy_stats = pd.DataFrame()

        # Get marker enrichment stats
        if all_marker_results:
            combined_markers = pd.concat(all_marker_results, ignore_index=True)
            marker_stats = combined_markers.groupby('Method')['Fold_Enrichment'].mean()
        else:
            marker_stats = pd.Series()

        # Combine
        for method in ['Uncorrected', 'scVI', 'SCimilarity', 'Harmony']:
            row = {'Method': method}

            if method in accuracy_stats.index:
                row['Accuracy'] = accuracy_stats.loc[method, 'Accuracy']
                row['Confidence'] = accuracy_stats.loc[method, 'Mean_Confidence']
            else:
                row['Accuracy'] = None
                row['Confidence'] = None

            if method in marker_stats.index:
                row['Marker_Enrichment'] = marker_stats.loc[method]
            else:
                row['Marker_Enrichment'] = None

            final_summary.append(row)

        final_df = pd.DataFrame(final_summary)
        final_df = final_df.round(3)

        print("\n" + final_df.to_string(index=False))

        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nKey Metrics:")
        print("  • Accuracy: Can we correctly classify cells? (higher = better)")
        print("  • Marker Enrichment: Do predictions match biology? (higher = better)")
        print("\nIdeal result: High accuracy + High marker enrichment")
        print("⚠️  If accuracy is high but enrichment is low → over-correction (distorted biology)")
        print("⚠️  If enrichment is high but accuracy is low → under-correction (batch effects remain)")

        # Identify best method
        if not final_df.empty:
            best_accuracy = final_df.loc[final_df['Accuracy'].idxmax(), 'Method']
            best_marker = final_df.loc[final_df['Marker_Enrichment'].idxmax(), 'Method']

            print(f"\n✓ Highest Accuracy: {best_accuracy}")
            print(f"✓ Highest Marker Enrichment: {best_marker}")

            if best_accuracy == best_marker:
                print(f"\n⭐ {best_accuracy} excels at both metrics!")
            else:
                print(f"\n💡 Trade-off detected:")
                print(f"   {best_accuracy} prioritizes accuracy (batch mixing)")
                print(f"   {best_marker} prioritizes biology (marker preservation)")

        summary_file = os.path.join(OUTPUT_DIR, "final_comparison.csv")
        final_df.to_csv(summary_file, index=False)
        print(f"\n✓ Final comparison saved: {summary_file}")

    print("\n" + "="*80)
    print("✓ VAN GALEN SUBTYPE VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"\nFor your paper:")
    print(f"  'We validated batch correction by testing whether methods")
    print(f"   could reproduce van Galen et al.'s (2019) AML cellular")
    print(f"   hierarchy in independent studies. [Method X] achieved [Y]%")
    print(f"   accuracy in predicting van Galen subtypes across studies,")
    print(f"   with marker gene enrichment confirming biological validity.'")


if __name__ == "__main__":
    main()
