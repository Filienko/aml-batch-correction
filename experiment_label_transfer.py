#!/usr/bin/env python
"""
Label Transfer Evaluation for Batch Correction

Uses van Galen 2019 (well-annotated AML reference) to transfer cell type
labels to other studies, then evaluates how consistent the labels are
across batches.

This tests:
1. Can we identify cell types consistently across studies?
2. Which batch correction method best preserves biological identity?
3. Are rare AML subtypes preserved after correction?

This is more interpretable than abstract metrics!
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Reference study (well-annotated)
REFERENCE_STUDY = 'van_galen_2019'  # Has expert AML annotations

# Query studies (to annotate)
QUERY_STUDIES = [
    'setty_2019',      # 10x Chromium
    'pei_2020',        # CITEseq
    'velten_2021',     # Muta-Seq
]

N_HVGS = 2000
N_JOBS = 8
OUTPUT_DIR = "results_label_transfer"
SCIMILARITY_BATCH_SIZE = 1000
SCIMILARITY_ENABLED = True

# Label transfer parameters
K_NEIGHBORS = 15  # Number of neighbors for label transfer


def transfer_labels_knn(ref_embedding, ref_labels, query_embedding, k=15):
    """
    Transfer labels from reference to query using k-NN.

    Args:
        ref_embedding: Reference cell embeddings (n_ref × n_dims)
        ref_labels: Reference cell labels (n_ref,)
        query_embedding: Query cell embeddings (n_query × n_dims)
        k: Number of neighbors

    Returns:
        predicted_labels: Predicted labels for query cells
        confidence: Prediction confidence (fraction of neighbors agreeing)
    """
    # Train k-NN classifier on reference
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(ref_embedding, ref_labels)

    # Predict query labels
    predicted_labels = knn.predict(query_embedding)

    # Get confidence (probability of most likely class)
    probabilities = knn.predict_proba(query_embedding)
    confidence = probabilities.max(axis=1)

    return predicted_labels, confidence


def evaluate_label_transfer(adata, embedding_key, ref_study, query_studies,
                            batch_key, label_key, k=15):
    """
    Evaluate label transfer for a given embedding.

    Args:
        adata: AnnData with embeddings
        embedding_key: Which embedding to use (e.g., 'X_scimilarity')
        ref_study: Reference study name
        query_studies: List of query study names
        batch_key: Batch column name
        label_key: Cell type label column name
        k: Number of neighbors for k-NN

    Returns:
        results_df: DataFrame with transfer results per query study
    """
    print(f"\n{'='*80}")
    print(f"LABEL TRANSFER: {embedding_key}")
    print(f"{'='*80}")

    if embedding_key not in adata.obsm:
        print(f"✗ Embedding '{embedding_key}' not found")
        return None

    # Get reference data
    ref_mask = adata.obs[batch_key] == ref_study
    ref_embedding = adata.obsm[embedding_key][ref_mask]
    ref_labels = adata.obs[label_key][ref_mask].values

    print(f"\nReference: {ref_study}")
    print(f"  Cells: {ref_mask.sum():,}")
    print(f"  Cell types: {len(np.unique(ref_labels))}")
    print(f"  Distribution:")
    for celltype, count in pd.Series(ref_labels).value_counts().head(10).items():
        print(f"    {celltype}: {count}")

    results = []

    # Transfer to each query study
    for query_study in query_studies:
        print(f"\n{'─'*80}")
        print(f"Query: {query_study}")
        print(f"{'─'*80}")

        query_mask = adata.obs[batch_key] == query_study
        query_embedding = adata.obsm[embedding_key][query_mask]
        query_true_labels = adata.obs[label_key][query_mask].values

        print(f"  Cells: {query_mask.sum():,}")
        print(f"  True cell types: {len(np.unique(query_true_labels))}")

        # Transfer labels
        predicted_labels, confidence = transfer_labels_knn(
            ref_embedding, ref_labels, query_embedding, k=k
        )

        # Compute accuracy (if query has labels)
        accuracy = accuracy_score(query_true_labels, predicted_labels)

        print(f"\n  Results:")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Mean confidence: {confidence.mean():.3f}")
        print(f"    Low confidence (<0.5): {(confidence < 0.5).sum():,} cells")

        # Show predicted distribution
        print(f"\n  Predicted distribution:")
        for celltype, count in pd.Series(predicted_labels).value_counts().head(10).items():
            print(f"    {celltype}: {count}")

        # Store results
        results.append({
            'Query_Study': query_study,
            'Accuracy': accuracy,
            'Mean_Confidence': confidence.mean(),
            'Low_Confidence_Cells': (confidence < 0.5).sum(),
            'N_Cells': query_mask.sum(),
            'N_Predicted_Types': len(np.unique(predicted_labels)),
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {embedding_key}")
    print(f"{'='*80}")
    print(f"\nAverage accuracy: {results_df['Accuracy'].mean():.3f}")
    print(f"Average confidence: {results_df['Mean_Confidence'].mean():.3f}")

    return results_df


def main():
    """
    Main pipeline for label transfer evaluation.
    """
    print("="*80)
    print("LABEL TRANSFER EVALUATION")
    print("="*80)
    print(f"\nReference: {REFERENCE_STUDY} (expert-curated AML annotations)")
    print(f"Query studies: {', '.join(QUERY_STUDIES)}")
    print("\nGoal: Transfer van Galen cell types to other studies,")
    print("      evaluate which batch correction method preserves identity best")

    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Data file not found: {DATA_PATH}")
        return

    # STEP 1: Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    print(f"\nLoading: {DATA_PATH}")
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

    # STEP 2: Subset to reference + query studies
    print("\n" + "="*80)
    print("STEP 2: SUBSETTING TO REFERENCE + QUERY STUDIES")
    print("="*80)

    all_studies = [REFERENCE_STUDY] + QUERY_STUDIES
    mask = adata.obs[BATCH_KEY].isin(all_studies)
    adata = adata[mask].copy()

    print(f"\nSubset: {adata.n_obs:,} cells")
    print(f"Studies: {adata.obs[BATCH_KEY].nunique()}")
    for study in all_studies:
        n = (adata.obs[BATCH_KEY] == study).sum()
        print(f"  {study}: {n:,} cells")

    # STEP 2.5: Load scVI
    print("\n" + "="*80)
    print("STEP 2.5: SCVI EMBEDDINGS")
    print("="*80)

    if os.path.exists(SCVI_PATH):
        try:
            adata_scvi = sc.read_h5ad(SCVI_PATH)
            scvi_has_numeric = all(str(idx).isdigit() for idx in adata_scvi.obs_names[:100])

            if scvi_has_numeric and adata_scvi.n_obs >= adata.n_obs:
                adata_full_temp = sc.read_h5ad(DATA_PATH)
                mask = adata_full_temp.obs[BATCH_KEY].isin(all_studies)
                original_indices = np.where(mask)[0]

                adata.obsm['X_scVI'] = adata_scvi.X[original_indices].copy()
                print(f"  ✓ Added scVI: {adata.obsm['X_scVI'].shape}")

                del adata_full_temp, adata_scvi
                force_cleanup()
        except Exception as e:
            print(f"  ✗ scVI loading failed: {e}")

    # Optimize memory
    adata = optimize_adata_memory(adata)
    force_cleanup()

    # STEP 3: Preprocess
    print("\n" + "="*80)
    print("STEP 3: PREPROCESSING")
    print("="*80)

    adata = preprocess_adata_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # STEP 4: Uncorrected PCA
    print("\n" + "="*80)
    print("STEP 4: UNCORRECTED PCA")
    print("="*80)

    adata = prepare_uncorrected_embedding_exact(adata, BATCH_KEY_LOWER)
    force_cleanup()

    # STEP 5: SCimilarity
    if SCIMILARITY_ENABLED and os.path.exists(SCIMILARITY_MODEL):
        print("\n" + "="*80)
        print("STEP 5: SCIMILARITY")
        print("="*80)

        try:
            adata = compute_scimilarity_corrected(
                adata,
                model_path=SCIMILARITY_MODEL,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
            force_cleanup()
        except Exception as e:
            print(f"✗ SCimilarity failed: {e}")

    # STEP 6: Harmony
    print("\n" + "="*80)
    print("STEP 6: HARMONY")
    print("="*80)

    try:
        adata = compute_harmony_corrected(adata, BATCH_KEY, N_JOBS)
        force_cleanup()
    except Exception as e:
        print(f"✗ Harmony failed: {e}")

    # STEP 7: Label Transfer Evaluation
    print("\n" + "="*80)
    print("STEP 7: LABEL TRANSFER EVALUATION")
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    # Evaluate each method
    for embedding_key, method_name in [
        ('X_uncorrected', 'Uncorrected'),
        ('X_scVI', 'scVI'),
        ('X_scimilarity', 'SCimilarity'),
        ('X_harmony', 'Harmony')
    ]:
        if embedding_key in adata.obsm:
            df = evaluate_label_transfer(
                adata, embedding_key, REFERENCE_STUDY, QUERY_STUDIES,
                BATCH_KEY, LABEL_KEY, k=K_NEIGHBORS
            )
            if df is not None:
                df['Method'] = method_name
                all_results[method_name] = df

                # Save individual results
                output_file = os.path.join(OUTPUT_DIR, f"{method_name.lower()}_transfer.csv")
                df.to_csv(output_file, index=False)
                print(f"\n✓ Saved: {output_file}")

    # STEP 8: Compare methods
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)

        combined = pd.concat(all_results.values(), ignore_index=True)

        # Average by method
        summary = combined.groupby('Method').agg({
            'Accuracy': 'mean',
            'Mean_Confidence': 'mean',
            'Low_Confidence_Cells': 'sum',
            'N_Cells': 'sum'
        }).round(3)

        print("\n" + summary.to_string())

        # Save summary
        summary_file = os.path.join(OUTPUT_DIR, "label_transfer_summary.csv")
        summary.to_csv(summary_file)
        print(f"\n✓ Summary saved: {summary_file}")

        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nLabel transfer measures:")
        print("  • Accuracy: Can we correctly identify cell types across studies?")
        print("  • Confidence: How certain are the predictions?")
        print("  • Low confidence: Cells that are hard to classify")
        print("\nBetter batch correction should:")
        print("  ✓ Higher accuracy (cells cluster by type, not batch)")
        print("  ✓ Higher confidence (clear biological clusters)")
        print("  ✓ Fewer low-confidence cells (consistent identity)")

    print("\n" + "="*80)
    print("✓ LABEL TRANSFER EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
