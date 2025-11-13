#!/usr/bin/env python
"""
Standalone scVI Embedding Loader with Robust Error Handling

This utility loads pre-computed scVI embeddings and handles common issues:
- Cell count mismatches
- Cell order differences
- Missing files
- Format errors

Can be imported as a module or run standalone to test loading.
"""

import os
import scanpy as sc
import numpy as np
import pandas as pd
import gc


def load_scvi_embedding(
    adata,
    scvi_path="data/AML_scAtlas_X_scVI.h5ad",
    embedding_key="X_scVI",
    return_subset=False,
    verbose=True
):
    """
    Load pre-computed scVI embeddings with comprehensive error handling.
    
    Args:
        adata: Main AnnData object
        scvi_path: Path to scVI embedding file (.h5ad)
        embedding_key: Key to use in adata.obsm (default: 'X_scVI')
        return_subset: If True and cells don't match, return subset of common cells
                      If False and cells don't match, raises error
        verbose: Print detailed progress messages
    
    Returns:
        adata: Original AnnData with scVI embeddings added in .obsm[embedding_key]
               OR subset of adata with common cells (if return_subset=True)
    
    Raises:
        FileNotFoundError: If scvi_path doesn't exist and not in graceful mode
        ValueError: If cell matching fails and return_subset=False
    """
    
    if verbose:
        print("\n" + "="*80)
        print("LOADING SCVI EMBEDDINGS")
        print("="*80)
    
    # Check if file exists
    if not os.path.exists(scvi_path):
        if verbose:
            print(f"  ✗ File not found: {scvi_path}")
        raise FileNotFoundError(f"scVI embedding file not found: {scvi_path}")
    
    try:
        # Load scVI file
        if verbose:
            print(f"  Loading from: {scvi_path}")
        
        adata_scvi = sc.read_h5ad(scvi_path)
        
        if verbose:
            print(f"  scVI file loaded:")
            print(f"    Shape: {adata_scvi.shape}")
            print(f"    Embedding dim: {adata_scvi.X.shape[1]}")
        
        # Case 1: Perfect match (same cells, same order)
        if adata_scvi.n_obs == adata.n_obs:
            if verbose:
                print(f"\n  Checking cell alignment...")
                print(f"    Main data: {adata.n_obs:,} cells")
                print(f"    scVI data: {adata_scvi.n_obs:,} cells")
            
            # Check if order matches
            if (adata.obs_names == adata_scvi.obs_names).all():
                if verbose:
                    print(f"    ✓ Perfect match (same cells, same order)")
                
                adata.obsm[embedding_key] = adata_scvi.X.copy()
                
                if verbose:
                    print(f"\n  ✓ Added '{embedding_key}' to adata.obsm")
                    print(f"    Shape: {adata.obsm[embedding_key].shape}")
                
                del adata_scvi
                gc.collect()
                
                return adata
            
            else:
                # Same count but different order - try to reorder
                if verbose:
                    print(f"    ⚠ Same cell count but different order")
                    print(f"    Attempting to reorder...")
                
                try:
                    # Reorder scVI to match main data
                    adata_scvi_reordered = adata_scvi[adata.obs_names]
                    
                    adata.obsm[embedding_key] = adata_scvi_reordered.X.copy()
                    
                    if verbose:
                        print(f"    ✓ Successfully reordered")
                        print(f"\n  ✓ Added '{embedding_key}' to adata.obsm")
                        print(f"    Shape: {adata.obsm[embedding_key].shape}")
                    
                    del adata_scvi, adata_scvi_reordered
                    gc.collect()
                    
                    return adata
                
                except KeyError as e:
                    if verbose:
                        print(f"    ✗ Cannot reorder: cell IDs don't match")
                        print(f"    ℹ Checking if IDs are just numeric indices...")

                    # Case 1b: Same count, IDs don't match, but might be in same order
                    # This happens when scVI file has numeric indices ('0', '1', '2')
                    # but main data has cell barcodes
                    try:
                        # Check if scVI IDs are just numeric indices
                        scvi_ids_numeric = all(str(idx).isdigit() for idx in adata_scvi.obs_names[:100])

                        if scvi_ids_numeric:
                            if verbose:
                                print(f"    ✓ scVI file uses numeric indices")
                                print(f"    ℹ Assuming embeddings are in same order as main data")
                                print(f"    ⚠ WARNING: This assumes scVI embeddings were computed on same data in same order!")

                            # Trust the order and copy directly
                            adata.obsm[embedding_key] = adata_scvi.X.copy()

                            if verbose:
                                print(f"\n  ✓ Added '{embedding_key}' to adata.obsm")
                                print(f"    Shape: {adata.obsm[embedding_key].shape}")

                            del adata_scvi
                            gc.collect()

                            return adata

                    except Exception as ex:
                        if verbose:
                            print(f"    ℹ Could not verify numeric indices: {ex}")
                        pass
                    # Fall through to mismatch handling below

        # Case 2: Different cell counts - try to find common cells
        if verbose:
            print(f"\n  ⚠ Cell count mismatch:")
            print(f"    Main data: {adata.n_obs:,} cells")
            print(f"    scVI data: {adata_scvi.n_obs:,} cells")
            print(f"    Searching for common cells...")
        
        # Find intersection
        common_cells = adata.obs_names.intersection(adata_scvi.obs_names)
        
        if len(common_cells) == 0:
            error_msg = (
                f"No common cells found between main data and scVI embeddings!\n"
                f"  Main data has {adata.n_obs:,} cells\n"
                f"  scVI file has {adata_scvi.n_obs:,} cells\n"
                f"  Check that cell IDs match between files"
            )
            if verbose:
                print(f"  ✗ {error_msg}")
            raise ValueError(error_msg)
        
        if verbose:
            print(f"    Found {len(common_cells):,} common cells")
            print(f"    Coverage: {100*len(common_cells)/adata.n_obs:.1f}% of main data")
        
        if return_subset:
            # Return subset with common cells
            if verbose:
                print(f"\n  Creating subset with common cells...")
            
            adata_subset = adata[common_cells].copy()
            adata_scvi_subset = adata_scvi[common_cells]
            
            adata_subset.obsm[embedding_key] = adata_scvi_subset.X.copy()
            
            if verbose:
                print(f"  ✓ Subset created:")
                print(f"    Cells: {adata_subset.n_obs:,}")
                print(f"    Added '{embedding_key}': {adata_subset.obsm[embedding_key].shape}")
                print(f"\n  ⚠ Returning SUBSET of original data!")
            
            del adata_scvi
            gc.collect()
            
            return adata_subset
        
        else:
            # Strict mode - raise error
            error_msg = (
                f"Cell count mismatch!\n"
                f"  Main: {adata.n_obs:,}, scVI: {adata_scvi.n_obs:,}\n"
                f"  Use return_subset=True to return common cells only"
            )
            if verbose:
                print(f"  ✗ {error_msg}")
            raise ValueError(error_msg)
    
    except Exception as e:
        if verbose:
            print(f"\n  ✗ Error loading scVI embeddings:")
            print(f"    {type(e).__name__}: {e}")
            print(f"\n  Traceback:")
            import traceback
            traceback.print_exc()
        
        raise


def load_scvi_graceful(adata, scvi_path="data/AML_scAtlas_X_scVI.h5ad", verbose=True):
    """
    Load scVI embeddings with graceful error handling.
    Returns original adata unchanged if loading fails.
    
    This is useful for pipelines where scVI is optional.
    
    Args:
        adata: Main AnnData object
        scvi_path: Path to scVI embedding file
        verbose: Print messages
    
    Returns:
        adata: With scVI added if successful, unchanged if failed
    """
    
    try:
        return load_scvi_embedding(
            adata,
            scvi_path=scvi_path,
            return_subset=True,  # Allow subset on mismatch
            verbose=verbose
        )
    
    except FileNotFoundError:
        if verbose:
            print(f"  ℹ scVI file not found, skipping")
        return adata
    
    except Exception as e:
        if verbose:
            print(f"  ℹ Could not load scVI embeddings, skipping")
            print(f"    Reason: {type(e).__name__}: {e}")
        return adata


def verify_scvi_file(scvi_path="data/AML_scAtlas_X_scVI.h5ad"):
    """
    Quick check of scVI file without loading main data.
    
    Args:
        scvi_path: Path to scVI embedding file
    
    Returns:
        dict with file info, or None if file invalid
    """
    
    print("="*80)
    print("SCVI FILE VERIFICATION")
    print("="*80)
    
    if not os.path.exists(scvi_path):
        print(f"\n✗ File not found: {scvi_path}")
        return None
    
    print(f"\nFile: {scvi_path}")
    print(f"Size: {os.path.getsize(scvi_path) / 1024**2:.1f} MB")
    
    try:
        adata_scvi = sc.read_h5ad(scvi_path)
        
        info = {
            'n_cells': adata_scvi.n_obs,
            'n_features': adata_scvi.n_vars,
            'embedding_dim': adata_scvi.X.shape[1] if hasattr(adata_scvi.X, 'shape') else None,
            'cell_ids': adata_scvi.obs_names[:5].tolist(),
            'sparse': hasattr(adata_scvi.X, 'nnz'),
        }
        
        print(f"\n✓ File loaded successfully:")
        print(f"  Cells: {info['n_cells']:,}")
        print(f"  Features: {info['n_features']:,}")
        print(f"  Embedding dim: {info['embedding_dim']}")
        print(f"  Format: {'Sparse' if info['sparse'] else 'Dense'}")
        print(f"\n  First 5 cell IDs:")
        for cell_id in info['cell_ids']:
            print(f"    {cell_id}")
        
        # Check for metadata
        if len(adata_scvi.obs.columns) > 0:
            print(f"\n  Metadata columns: {adata_scvi.obs.columns.tolist()}")
        
        print(f"\n✓ scVI file appears valid")
        
        return info
    
    except Exception as e:
        print(f"\n✗ Error reading file:")
        print(f"  {type(e).__name__}: {e}")
        return None


def test_scvi_loading(
    main_data_path="data/AML_scAtlas.h5ad",
    scvi_path="data/AML_scAtlas_X_scVI.h5ad"
):
    """
    Test scVI loading with your actual files.
    
    Args:
        main_data_path: Path to main AnnData file
        scvi_path: Path to scVI embedding file
    """
    
    print("="*80)
    print("TESTING SCVI LOADING")
    print("="*80)
    
    # Step 1: Verify scVI file
    print("\nStep 1: Verify scVI file exists and is valid...")
    scvi_info = verify_scvi_file(scvi_path)
    
    if scvi_info is None:
        print("\n✗ scVI file invalid, cannot proceed")
        return
    
    # Step 2: Load main data
    print("\n" + "="*80)
    print("Step 2: Load main data...")
    
    if not os.path.exists(main_data_path):
        print(f"✗ Main data not found: {main_data_path}")
        return
    
    adata = sc.read_h5ad(main_data_path)
    print(f"✓ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  First 5 cell IDs:")
    for cell_id in adata.obs_names[:5]:
        print(f"    {cell_id}")
    
    # Step 3: Try loading scVI
    print("\n" + "="*80)
    print("Step 3: Attempt scVI loading (graceful mode)...")
    
    adata_with_scvi = load_scvi_graceful(adata, scvi_path=scvi_path, verbose=True)
    
    # Step 4: Check results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if 'X_scVI' in adata_with_scvi.obsm:
        print(f"\n✓ SUCCESS!")
        print(f"  scVI embeddings loaded: {adata_with_scvi.obsm['X_scVI'].shape}")
        print(f"  Final data: {adata_with_scvi.n_obs:,} cells")
        
        if adata_with_scvi.n_obs < adata.n_obs:
            print(f"  ⚠ Note: Data was subset to {adata_with_scvi.n_obs:,} common cells")
            print(f"    (Original had {adata.n_obs:,} cells)")
    else:
        print(f"\n✗ FAILED")
        print(f"  scVI embeddings not loaded")
        print(f"  Check error messages above")


if __name__ == "__main__":
    """
    Run standalone test
    """
    
    import sys
    
    print("\n" + "="*80)
    print("SCVI EMBEDDING LOADER - STANDALONE TEST")
    print("="*80)
    
    # Default paths
    main_path = "data/AML_scAtlas.h5ad"
    scvi_path = "data/AML_scAtlas_X_scVI.h5ad"
    
    # Allow command line override
    if len(sys.argv) > 1:
        scvi_path = sys.argv[1]
    if len(sys.argv) > 2:
        main_path = sys.argv[2]
    
    print(f"\nUsing:")
    print(f"  Main data: {main_path}")
    print(f"  scVI file: {scvi_path}")
    print()
    
    # Run test
    test_scvi_loading(main_path, scvi_path)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

