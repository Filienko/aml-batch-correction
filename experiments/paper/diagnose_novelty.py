#!/usr/bin/env python3
"""
Diagnostic: Find Natural & Synthetic Discovery Candidates (Memory Optimized)
===========================================================================
This version uses h5ad backed mode and AnnData Views to prevent loading 
large matrices into RAM.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl.data import get_study_column, get_cell_type_column

# --- CONFIGURATION ---
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
QUERY_SUBSET_SIZE = 1000  # Size of the test subset for quick diagnostic

SCENARIOS = [
    {
        'name': 'Same-Platform beneyto vs jiang',
        'ref_study': 'beneyto-calabuig-2023',
        'query_study': 'veltern_2021',
    },
    {
        'name': 'Cross-Platform van galen vs beneyto',
        'ref_study': 'van_galen_2019',
        'query_study': 'beneyto-calabuig-2023',
    },
    {
        'name': 'Cross-Platform zhai vs zhang',
        'ref_study': 'zhai_2022',
        'query_study': 'zhang_2023',
    },
]

def analyze_scenario(adata, scenario):
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"Reference: {scenario['ref_study']}")
    print(f"Query:     {scenario['query_study']} (Subsampled to {QUERY_SUBSET_SIZE})")
    print(f"{'='*80}")

    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    # 1. Reference Data
    # Get boolean mask and slice once from the root object
    idx_ref_mask = adata.obs[study_col] == scenario['ref_study']
    adata_ref = adata[idx_ref_mask]
    
    ref_types = set(adata_ref.obs[cell_type_col].dropna().unique())
    print(f"\nReference contains {len(ref_types)} unique cell types.")

    # 2. Query Data - THE FIX IS HERE
    # Find the integer locations (indices) of all cells in this study
    query_indices = np.where(adata.obs[study_col] == scenario['query_study'])[0]

    if len(query_indices) > QUERY_SUBSET_SIZE:
        # Sample from the absolute integer positions
        sampled_indices = np.random.choice(query_indices, QUERY_SUBSET_SIZE, replace=False)
        # Slice once from the original 'adata' to avoid "view of a view"
        adata_query = adata[sampled_indices]
    else:
        adata_query = adata[query_indices]

    # 3. Analyze Labels
    query_counts = adata_query.obs[cell_type_col].value_counts()
    query_types = set(query_counts.index)

    print(f"Query (subset) contains {len(query_types)} unique cell types.")

    # Analyze "Natural Novelty"
    natural_novelty = query_types - ref_types

    print("\n" + "-"*60)
    print("A. NATURAL NOVELTY (Types in Query but MISSING in Ref)")
    print("-" * 60)

    if natural_novelty:
        print(f"Found {len(natural_novelty)} naturally novel cell types!")
        print(f"{'Cell Type':<30} | {'Count in Query (5k)':<20}")
        print("-" * 55)
        for ct in natural_novelty:
            print(f"{ct:<30} | {query_counts[ct]:<20}")
        print("\nRECOMMENDATION: Use one of these! No need to 'hide' anything manually.")
    else:
        print("NONE. All cell types in the query subset already exist in the reference.")

    # 4. Analyze "Synthetic Candidates"
    shared_types = query_types.intersection(ref_types)
    candidates = []
    for ct in shared_types:
        count = query_counts[ct]
        if count >= 50:
            candidates.append((ct, count))

    candidates.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "-"*60)
    print("B. SYNTHETIC CANDIDATES (For Leave-One-Out Experiment)")
    print("-" * 60)
    print(f"{'Cell Type':<30} | {'Count in Query (5k)':<20}")
    print("-" * 55)

    if candidates:
        for ct, count in candidates:
            print(f"{ct:<30} | {count:<20}")
        top_pick = candidates[0][0]
        print(f"\nRECOMMENDATION: Hide '{top_pick}' (abundant) or '{candidates[-1][0]}' (rare).")
    else:
        print("No robust candidates found (>50 cells).")


def main():
    print(f"Opening {DATA_PATH} in backed mode...")
    adata = sc.read_h5ad(DATA_PATH, backed='r')

    # Load ONLY the metadata into memory to avoid repeated disk reads 
    # and "view of a view" issues with the obs dataframe itself.
    print("Loading metadata into memory...")
    adata.obs = adata.obs.copy() 

    cell_type_col = get_cell_type_column(adata)
    adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str)

    for scenario in SCENARIOS:
        analyze_scenario(adata, scenario)


if __name__ == "__main__":
    main()

