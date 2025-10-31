
#!/usr/bin/env python
"""
Data Inspection Script
Run this first to understand your data structure before running the evaluation.
"""

import scanpy as sc
import pandas as pd
import os

def inspect_anndata(file_path: str):
    """Inspect AnnData structure and print useful information."""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print("="*80)
    print(f"INSPECTING: {file_path}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(file_path)
    print("names", adata.obs_names[:5])

    # Basic info
    print("\n" + "-"*80)
    print("BASIC INFORMATION")
    print("-"*80)
    print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    # print(f"Memory: {adata.X.nbytes / 1e9:.2f} GB")
    
    # Observations (cells)
    print("\n" + "-"*80)
    print("CELL METADATA (.obs)")
    print("-"*80)
    print(f"Number of metadata columns: {len(adata.obs.columns)}")
    print("\nAvailable columns:")
    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        dtype = adata.obs[col].dtype
        print(f"  • {col:30s} ({n_unique:6d} unique values, {dtype})")
    
    # Check for common batch/label columns
    print("\n" + "-"*80)
    print("SUGGESTED KEYS FOR EVALUATION")
    print("-"*80)
    
    # Batch keys
    batch_candidates = ['sample', 'batch', 'study', 'donor', 'patient', 'replicate']
    print("\nPossible BATCH_KEY options (for batch correction):")
    found_batch = False
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(bc in col_lower for bc in batch_candidates):
            n_batches = adata.obs[col].nunique()
            print(f"  ✓ '{col}' - {n_batches} unique values")
            found_batch = True
    if not found_batch:
        print("  ⚠ No obvious batch column found. You may need to inspect manually.")
    
    # Label keys
    label_candidates = ['celltype', 'cell_type', 'cluster', 'annotation', 'label', 'type']
    print("\nPossible LABEL_KEY options (for cell type labels):")
    found_label = False
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(lc in col_lower for lc in label_candidates):
            n_types = adata.obs[col].nunique()
            print(f"  ✓ '{col}' - {n_types} unique values")
            found_label = True
    if not found_label:
        print("  ⚠ No obvious cell type column found. You may need to inspect manually.")
    
    # Variables (genes)
    print("\n" + "-"*80)
    print("GENE INFORMATION (.var)")
    print("-"*80)
    print(f"Number of gene metadata columns: {len(adata.var.columns)}")
    print("\nAvailable columns:")
    for col in adata.var.columns:
        print(f"  • {col}")
    print(f"\nGene index type: {type(adata.var.index[0])}")
    print(f"First 5 gene names: {adata.var.index[:5].tolist()}")
    
    # Layers
    print("\n" + "-"*80)
    print("DATA LAYERS")
    print("-"*80)
    if len(adata.layers) > 0:
        print("Available layers:")
        for layer_name, layer_data in adata.layers.items():
            print(f"  • {layer_name:20s} - Shape: {layer_data.shape}, Type: {type(layer_data)}")
        
        # Check for raw counts
        if 'counts' in adata.layers:
            print("\n✓ Raw counts found in .layers['counts'] (needed for SCimilarity)")
        else:
            print("\n⚠ No 'counts' layer found. SCimilarity may need raw counts.")
    else:
        print("No layers found.")
    
    # Check main matrix
    print(f"\nMain matrix (.X):")
    print(f"  Type: {type(adata.X)}")
    print(f"  Shape: {adata.X.shape}")
    print(f"  Data type: {adata.X.dtype if hasattr(adata.X, 'dtype') else 'N/A'}")
    
    # Embeddings
    print("\n" + "-"*80)
    print("EMBEDDINGS (.obsm)")
    print("-"*80)
    if len(adata.obsm) > 0:
        print("Available embeddings:")
        for emb_name, emb_data in adata.obsm.items():
            print(f"  • {emb_name:30s} - Shape: {emb_data.shape}")
    else:
        print("No embeddings found.")
    
    # Raw data
    print("\n" + "-"*80)
    print("RAW DATA (.raw)")
    print("-"*80)
    if adata.raw is not None:
        print(f"Raw data available: {adata.raw.shape}")
        print(f"  Variables: {adata.raw.n_vars} genes")
        print("  ✓ Good! Can use full gene set for SCimilarity")
    else:
        print("No raw data stored.")
        print("  ⚠ SCimilarity works best with full gene set")
    
    # Show example batch/label distribution
    print("\n" + "-"*80)
    print("EXAMPLE DISTRIBUTIONS")
    print("-"*80)
    
    # Find likely batch column
    batch_col = None
    for col in ['sample', 'batch', 'study']:
        if col in adata.obs.columns:
            batch_col = col
            break
    
    if batch_col:
        print(f"\nDistribution of '{batch_col}':")
        counts = adata.obs[batch_col].value_counts()
        print(counts.head(10).to_string())
        if len(counts) > 10:
            print(f"  ... and {len(counts) - 10} more")
    
    # Find likely label column
    label_col = None
    for col in ['celltype', 'cell_type', 'cluster', 'annotation']:
        if col in adata.obs.columns:
            label_col = col
            break
    
    if label_col:
        print(f"\nDistribution of '{label_col}':")
        counts = adata.obs[label_col].value_counts()
        print(counts.head(10).to_string())
        if len(counts) > 10:
            print(f"  ... and {len(counts) - 10} more")
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    print("-"*80)
    
    if found_batch and found_label:
        print("✓ Your data appears ready for evaluation!")
        print("\nSuggested settings for run_evaluation.py:")
        if batch_col:
            print(f"  BATCH_KEY = '{batch_col}'")
        if label_col:
            print(f"  LABEL_KEY = '{label_col}'")
    else:
        print("⚠ Please review the metadata columns above")
        print("  and manually specify BATCH_KEY and LABEL_KEY")
    
    if 'counts' not in adata.layers and adata.raw is None:
        print("\n⚠ WARNING: No raw counts detected!")
        print("  SCimilarity requires raw counts.")
        print("  If your data is already normalized, you may need to reload from raw counts.")
    
    print()


def main():
    """Main function to inspect data files."""
    
    print("\n" + "="*80)
    print("AnnData INSPECTION TOOL")
    print("="*80)
    
    # Check main data file
    main_file = "data/AML_scAtlas.h5ad"
    if os.path.exists(main_file):
        inspect_anndata(main_file)
    else:
        print(f"\n❌ Main data file not found: {main_file}")
        print("Please update the path in this script or move your data file.")
    
    # Check scVI file if exists
    scvi_file = "data/AML_scAtlas_X_scVI.h5ad"
    if os.path.exists(scvi_file):
        print("\n\n")
        inspect_anndata(scvi_file)
    else:
        print(f"\nℹ scVI embedding file not found: {scvi_file}")
        print("  (This is OK if you only want to evaluate SCimilarity)")


if __name__ == "__main__":
    main()

