#!/usr/bin/env python
"""
Example: How SCimilarity from your original code is integrated

This shows how the run_scimilarity_pipeline function from your original code
is adapted for batch correction evaluation.
"""

import scanpy as sc
import numpy as np
from scimilarity import CellAnnotation
from scimilarity.utils import lognorm_counts, align_dataset

# ==============================================================================
# YOUR ORIGINAL CODE (simplified)
# ==============================================================================

def run_scimilarity_pipeline_original(adata_train, adata_test):
    """
    Your original SCimilarity pipeline for classification.
    This is designed for train/test splits.
    """
    config = {'MODEL_PATH': 'model_v1.1'}
    
    # Load model
    ca = CellAnnotation(model_path=config['MODEL_PATH'])
    
    # Get full data
    adata_train_full = adata_train.raw.to_adata()
    adata_test_full = adata_test.raw.to_adata()
    
    # Ensure gene symbols
    for adata, name in [(adata_train_full, "train"), (adata_test_full, "test")]:
        if 'gene_name' in adata.var.columns:
            adata.var.index = adata.var['gene_name']
    
    # Prepare for SCimilarity
    adata_train_full.layers['counts'] = adata_train_full.X.copy()
    adata_test_full.layers['counts'] = adata_test_full.X.copy()
    
    # Align and normalize
    adata_train_scim = align_dataset(adata_train_full, ca.gene_order)
    adata_train_scim = lognorm_counts(adata_train_scim)
    
    adata_test_scim = align_dataset(adata_test_full, ca.gene_order)
    adata_test_scim = lognorm_counts(adata_test_scim)
    
    # Get embeddings
    train_embeddings = ca.get_embeddings(adata_train_scim.X)
    test_embeddings = ca.get_embeddings(adata_test_scim.X)
    
    return train_embeddings, test_embeddings


# ==============================================================================
# ADAPTED FOR BATCH CORRECTION EVALUATION
# ==============================================================================

def compute_scimilarity_for_batch_correction(adata):
    """
    Adapted version for batch correction evaluation.
    This processes ALL data at once (no train/test split).
    
    Key differences:
    1. No train/test split - processes entire dataset
    2. Returns embeddings in-place
    3. Handles gene symbol alignment more robustly
    4. Better error handling
    """
    
    print("Computing SCimilarity embeddings for batch correction...")
    config = {'MODEL_PATH': 'model_v1.1'}
    
    # Load model
    try:
        ca = CellAnnotation(model_path=config['MODEL_PATH'])
        print(f"✓ Model loaded from {config['MODEL_PATH']}")
    except Exception as e:
        raise RuntimeError(f"Failed to load SCimilarity model: {e}")
    
    # Get full gene set (same as your original code)
    if adata.raw is None:
        print("  No .raw found, using main object")
        adata_full = adata.copy()
    else:
        print("  Using .raw for full gene set")
        adata_full = adata.raw.to_adata()
    
    # Ensure gene symbols are in index (same as your original code)
    possible_cols = ['gene_name', 'gene_symbols']
    gene_col = next((col for col in possible_cols if col in adata_full.var.columns), None)
    
    if gene_col:
        print(f"  Setting gene symbols from '{gene_col}'")
        adata_full.var.index = adata_full.var[gene_col]
    else:
        print("  Warning: Assuming .var.index already contains gene symbols")
    
    # Prepare counts layer (same as your original code)
    if 'counts' in adata_full.layers:
        print("  Using .layers['counts']")
        adata_full.X = adata_full.layers['counts'].copy()
    
    adata_full.layers['counts'] = adata_full.X.copy()
    
    # Align with SCimilarity gene order (same as your original code)
    print("  Aligning with SCimilarity gene order...")
    try:
        adata_scim = align_dataset(adata_full, ca.gene_order)
        adata_scim = lognorm_counts(adata_scim)
        print(f"  ✓ Aligned to {adata_scim.shape[1]} genes")
    except Exception as e:
        raise RuntimeError(f"Failed to align dataset: {e}")
    
    # Compute embeddings (same as your original code)
    print("  Computing embeddings...")
    try:
        embeddings = ca.get_embeddings(adata_scim.X)
        print(f"  ✓ Embeddings: {embeddings.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to compute embeddings: {e}")
    
    # Add to original object
    adata.obsm['X_scimilarity'] = embeddings
    print(f"✓ SCimilarity embeddings added to .obsm['X_scimilarity']")
    
    return adata


# ==============================================================================
# USAGE COMPARISON
# ==============================================================================

def example_usage():
    """
    Shows how to use both versions.
    """
    
    # Load data
    adata = sc.read_h5ad("data/AML_scAtlas.h5ad")
    
    print("="*80)
    print("OPTION 1: Original approach (train/test split for classification)")
    print("="*80)
    
    # Your original approach for classification
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(adata)), 
        test_size=0.2, 
        random_state=42
    )
    
    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()
    
    # Get embeddings
    train_emb, test_emb = run_scimilarity_pipeline_original(adata_train, adata_test)
    
    print(f"Train embeddings: {train_emb.shape}")
    print(f"Test embeddings: {test_emb.shape}")
    print("→ Use these for downstream classification")
    
    print("\n" + "="*80)
    print("OPTION 2: Batch correction approach (entire dataset)")
    print("="*80)
    
    # Adapted approach for batch correction evaluation
    adata = compute_scimilarity_for_batch_correction(adata)
    
    print(f"All embeddings: {adata.obsm['X_scimilarity'].shape}")
    print("→ Use these for batch correction evaluation with scIB")
    
    # Now you can evaluate with scIB
    from batch_correction_evaluation import run_scib_benchmark
    
    results = run_scib_benchmark(
        adata,
        batch_key='sample',
        label_key='celltype',
        embedding_key='X_scimilarity',
        output_dir='results',
        n_jobs=8
    )
    
    print("\n✓ Batch correction evaluation complete!")


# ==============================================================================
# KEY POINTS
# ==============================================================================

"""
SUMMARY OF DIFFERENCES:

1. YOUR ORIGINAL CODE:
   - Designed for train/test classification
   - Processes train and test sets separately
   - Returns embeddings as arrays
   - Used for: Cell type prediction/classification

2. BATCH CORRECTION ADAPTATION:
   - Designed for batch correction evaluation
   - Processes entire dataset at once
   - Adds embeddings to .obsm
   - Used for: Evaluating batch mixing quality with scIB metrics

WHAT'S THE SAME:
   - Uses same SCimilarity model
   - Same gene alignment procedure
   - Same normalization (lognorm_counts)
   - Same embedding computation
   - Handles full gene set the same way

WHEN TO USE WHICH:
   - Use original: When doing cell type classification (train/test)
   - Use adapted: When evaluating batch correction quality
   
INTEGRATION IN THIS PIPELINE:
   The batch_correction_evaluation.py file uses the adapted version in the
   compute_scimilarity_embedding() function, which follows the same logic
   as your original code but processes the entire dataset for batch evaluation.
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("To see this in action, run: python run_evaluation.py")
    print("="*80)

