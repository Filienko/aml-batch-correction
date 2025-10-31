#!/usr/bin/env python
"""
Extended Batch Correction Evaluation
Includes: Uncorrected, Harmony, scVI, scANVI, SCimilarity

This matches the full evaluation from the AML Atlas paper with corrected implementations.
"""

import os
from batch_correction_evaluation import (
    prepare_uncorrected_embedding,
    load_scvi_embedding,
    train_scvi_model,
    compute_scimilarity_embedding,
    compute_harmony_embedding,
    train_scanvi_model,
    run_scib_benchmark,
    compare_methods
)
import scanpy as sc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad"
SCIMILARITY_MODEL = "models/model_v1.1"

# Metadata keys
BATCH_KEY = "Sample"  # Use "sample" or "study" depending on your data
LABEL_KEY = "Cell Type"  # Cell type annotation column

# Analysis parameters
N_HVGS = 2000
N_JOBS = 16
OUTPUT_DIR = "batch_correction_extended"
MODEL_SAVE_DIR = "models"  # Directory to save trained models
SCIMILARITY_BATCH_SIZE = 500  # Cells per batch for SCimilarity (reduce if OOM)

# Which methods to evaluate
EVAL_UNCORRECTED = True
EVAL_HARMONY = False
EVAL_SCVI = False
EVAL_SCANVI = False
EVAL_SCIMILARITY = True

# Training options
TRAIN_SCVI_FROM_SCRATCH = False  # Set to True to train scVI instead of loading
SCVI_MODEL_PATH = None  # Path to pre-trained scVI model for scANVI (optional)

SAVE_COMBINED_ADATA = False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("EXTENDED BATCH CORRECTION EVALUATION")
    print("Methods: Uncorrected, Harmony, scVI, scANVI, SCimilarity")
    print("Matching AML Atlas paper implementation")
    print("="*80)

    # Create local copies of evaluation flags to avoid scoping issues
    eval_uncorrected = EVAL_UNCORRECTED
    eval_harmony = EVAL_HARMONY
    eval_scvi = EVAL_SCVI
    eval_scanvi = EVAL_SCANVI
    eval_scimilarity = EVAL_SCIMILARITY

    # Check files
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Error: Data file not found: {DATA_PATH}")
        return

    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"✓ Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Verify metadata - try common variations
    batch_key = BATCH_KEY
    label_key = LABEL_KEY
    
    # Check for batch key variations
    if batch_key not in adata.obs.columns:
        # Try lowercase
        if batch_key.lower() in adata.obs.columns:
            batch_key = batch_key.lower()
            print(f"  Using batch key: '{batch_key}' (lowercase)")
        else:
            print(f"\n❌ Error: Batch key '{BATCH_KEY}' not found!")
            print(f"Available columns: {adata.obs.columns.tolist()}")
            return
    
    # Check for label key variations
    if label_key not in adata.obs.columns:
        # Try lowercase
        if label_key.lower() in adata.obs.columns:
            label_key = label_key.lower()
            print(f"  Using label key: '{label_key}' (lowercase)")
        # Try common alternatives
        elif "celltype" in adata.obs.columns:
            label_key = "celltype"
            print(f"  Using label key: '{label_key}'")
        elif "cell_type" in adata.obs.columns:
            label_key = "cell_type"
            print(f"  Using label key: '{label_key}'")
        else:
            print(f"\n❌ Error: Label key '{LABEL_KEY}' not found!")
            print(f"Available columns: {adata.obs.columns.tolist()}")
            return

    print(f"\n✓ Batch key: {batch_key} ({adata.obs[batch_key].nunique()} batches)")
    print(f"✓ Label key: {label_key} ({adata.obs[label_key].nunique()} types)")

    # Check for required data layers
    if 'counts' not in adata.layers:
        print("\n⚠ Warning: No 'counts' layer found.")
        print("  Attempting to use .X as counts...")
        adata.layers['counts'] = adata.X.copy()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Store results
    results = {}

    # ========================================================================
    # PREPARE EMBEDDINGS
    # ========================================================================
    print("\n" + "="*80)
    print("PREPARING EMBEDDINGS")
    print("="*80)

    # 1. Uncorrected (PCA)
    if eval_uncorrected:
        print("\n1. Computing Uncorrected PCA...")
        try:
            adata = prepare_uncorrected_embedding(adata, n_hvgs=N_HVGS)
        except Exception as e:
            print(f"⚠ Warning: Uncorrected PCA failed: {e}")
            eval_uncorrected = False

    # 2. Harmony
    if eval_harmony:
        print("\n2. Computing Harmony...")
        try:
            adata = compute_harmony_embedding(
                adata,
                batch_key=batch_key
            )
        except ImportError as e:
            print(f"⚠ Warning: {e}")
            print("Skipping Harmony. Install with: pip install scib")
            eval_harmony = False
        except Exception as e:
            print(f"⚠ Warning: Harmony failed: {e}")
            eval_harmony = False

    # 3. scVI
    if eval_scvi:
        if TRAIN_SCVI_FROM_SCRATCH:
            print("\n3. Training scVI model from scratch...")
            try:
                adata = train_scvi_model(
                    adata,
                    batch_key=batch_key,
                    n_layers=2,
                    n_latent=30,
                    save_dir=MODEL_SAVE_DIR
                )
            except ImportError as e:
                print(f"⚠ Warning: {e}")
                print("Skipping scVI. Install with: pip install scvi-tools")
                eval_scvi = False
            except Exception as e:
                print(f"⚠ Warning: scVI training failed: {e}")
                eval_scvi = False
        else:
            print("\n3. Loading pre-computed scVI embeddings...")
            if os.path.exists(SCVI_PATH):
                try:
                    adata = load_scvi_embedding(adata, scvi_path=SCVI_PATH)
                except Exception as e:
                    print(f"⚠ Warning: Failed to load scVI embeddings: {e}")
                    eval_scvi = False
            else:
                print(f"⚠ Warning: scVI file not found: {SCVI_PATH}")
                print("  Set TRAIN_SCVI_FROM_SCRATCH=True to train from scratch")
                eval_scvi = False

    # 4. scANVI (semi-supervised scVI)
    if eval_scanvi:
        print("\n4. Training scANVI model...")
        try:
            # Prepare labels for scANVI
            # The original implementation uses "Unknown" for unlabeled cells
            # You can modify this based on your needs
            adata_scanvi = adata.copy()
            
            # Ensure celltype is categorical
            if not pd.api.types.is_categorical_dtype(adata_scanvi.obs[label_key]):
                adata_scanvi.obs[label_key] = pd.Categorical(adata_scanvi.obs[label_key])
            
            # Add "Unknown" as a category if not present
            if "Unknown" not in adata_scanvi.obs[label_key].cat.categories:
                adata_scanvi.obs[label_key] = adata_scanvi.obs[label_key].cat.add_categories("Unknown")
            
            # Train scANVI
            adata_scanvi = train_scanvi_model(
                adata_scanvi,
                batch_key=batch_key,
                label_key=label_key,
                scvi_model_path=SCVI_MODEL_PATH,
                n_layers=2,
                n_latent=30,
                max_epochs=20,
                save_dir=MODEL_SAVE_DIR
            )
            
            # Transfer embeddings back to main adata
            adata.obsm['X_scANVI'] = adata_scanvi.obsm['X_scANVI']
            if 'scANVI_annotations' in adata_scanvi.obs.columns:
                adata.obs['scANVI_annotations'] = adata_scanvi.obs['scANVI_annotations']
            
        except ImportError as e:
            print(f"⚠ Warning: {e}")
            print("Skipping scANVI. Install with: pip install scvi-tools")
            eval_scanvi = False
        except Exception as e:
            print(f"⚠ Warning: scANVI failed: {e}")
            import traceback
            traceback.print_exc()
            eval_scanvi = False

    # 5. SCimilarity
    if eval_scimilarity:
        print("\n5. Computing SCimilarity embeddings...")
        try:
            adata = compute_scimilarity_embedding(
                adata,
                model_path=SCIMILARITY_MODEL,
                use_full_gene_set=False,
                batch_size=SCIMILARITY_BATCH_SIZE
            )
        except Exception as e:
            print(f"⚠ Warning: SCimilarity failed: {e}")
            import traceback
            traceback.print_exc()
            eval_scimilarity = False

    # ========================================================================
    # RUN BENCHMARKING
    # ========================================================================
    print("\n" + "="*80)
    print("RUNNING SCIB BENCHMARKING")
    print("="*80)

    # Import pandas for scANVI
    import pandas as pd

    # 1. Uncorrected
    if eval_uncorrected:
        print("\n" + "-"*80)
        print("1. Evaluating: Uncorrected (PCA)")
        print("-"*80)
        try:
            results['Uncorrected'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_uncorrected',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
        except Exception as e:
            print(f"⚠ Warning: Uncorrected benchmarking failed: {e}")

    # 2. Harmony
    if eval_harmony:
        print("\n" + "-"*80)
        print("2. Evaluating: Harmony")
        print("-"*80)
        try:
            results['Harmony'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_harmony',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
        except Exception as e:
            print(f"⚠ Warning: Harmony benchmarking failed: {e}")

    # 3. scVI
    if eval_scvi:
        print("\n" + "-"*80)
        print("3. Evaluating: scVI")
        print("-"*80)
        try:
            results['scVI'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_scVI',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
        except Exception as e:
            print(f"⚠ Warning: scVI benchmarking failed: {e}")

    # 4. scANVI
    if eval_scanvi:
        print("\n" + "-"*80)
        print("4. Evaluating: scANVI")
        print("-"*80)
        try:
            results['scANVI'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_scANVI',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
        except Exception as e:
            print(f"⚠ Warning: scANVI benchmarking failed: {e}")

    # 5. SCimilarity
    if eval_scimilarity:
        print("\n" + "-"*80)
        print("5. Evaluating: SCimilarity")
        print("-"*80)
        try:
            results['SCimilarity'] = run_scib_benchmark(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key='X_scimilarity',
                output_dir=OUTPUT_DIR,
                n_jobs=N_JOBS
            )
        except Exception as e:
            print(f"⚠ Warning: SCimilarity benchmarking failed: {e}")

    # ========================================================================
    # COMPARE METHODS
    # ========================================================================
    if len(results) > 1:
        print("\n" + "="*80)
        print("COMPARING ALL METHODS")
        print("="*80)

        try:
            combined = compare_methods(results, output_dir=OUTPUT_DIR)

            # Print final summary
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)

            print("\nMethod Performance (sorted by Total score):")
            print("-" * 80)
            for method in combined.sort_values('Total', ascending=False).index:
                total = combined.loc[method, 'Total']
                batch = combined.loc[method, 'Batch correction']
                bio = combined.loc[method, 'Bio conservation']
                print(f"  {method:20s}: Total={total:.3f}, Batch={batch:.3f}, Bio={bio:.3f}")

            # Winner
            winner = combined['Total'].idxmax()
            winner_score = combined.loc[winner, 'Total']

            print(f"\n🏆 Best method: {winner}")
            print(f"   Total score: {winner_score:.3f}")
            print(f"   Batch correction: {combined.loc[winner, 'Batch correction']:.3f}")
            print(f"   Bio conservation: {combined.loc[winner, 'Bio conservation']:.3f}")
        except Exception as e:
            print(f"⚠ Warning: Method comparison failed: {e}")
    elif len(results) == 0:
        print("\n⚠ Warning: No methods were successfully evaluated!")
    else:
        print(f"\n✓ Only one method evaluated: {list(results.keys())[0]}")

    # Save processed data
    if SAVE_COMBINED_ADATA and len(results) > 0:
        output_file = os.path.join(OUTPUT_DIR, "adata_with_all_embeddings.h5ad")
        try:
            adata.write_h5ad(output_file)
            print(f"\n✓ Saved data with all embeddings to: {output_file}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save combined adata: {e}")

    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")
    print("="*80)

    if len(results) > 1:
        print("\nKey files:")
        print(f"  - {OUTPUT_DIR}/combined_metrics.csv")
        print(f"  - {OUTPUT_DIR}/batch_vs_bio_scatter.png")
        print(f"  - {OUTPUT_DIR}/metrics_radar_plot.png")
        print(f"  - {OUTPUT_DIR}/metrics_heatmap.png")


if __name__ == "__main__":
    main()
