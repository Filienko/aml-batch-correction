#!/usr/bin/env python3
"""
Diagnostic: Find Natural & Synthetic Discovery Candidates
=========================================================

Checks if the Test (Query) dataset contains cell types that are 
NATURALLY missing from the Reference dataset.

If "Natural Novelty" exists, you don't need to hide anything—the models 
should detect these automatically.

If "Natural Novelty" does NOT exist (perfectly overlapping labels), 
it identifies the best "Synthetic Candidates" (shared types with enough cells) 
to hide for the Leave-One-Out experiment.

Scenarios checked:
1. Same-Platform (Beneyto -> Jiang)
2. Cross-Platform (Van Galen -> Jiang)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl.data import subset_data, get_study_column, get_cell_type_column

# --- CONFIGURATION ---
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
QUERY_SUBSET_SIZE = 5000  # As requested: check within a 5k subset of test data

SCENARIOS = [
    {
        'name': 'Same-Platform',
        'ref_study': 'beneyto-calabuig-2023',
        'query_study': 'jiang_2020',
    },
    {
        'name': 'Cross-Platform',
        'ref_study': 'van_galen_2019',
        'query_study': 'jiang_2020',
    },
]

def analyze_scenario(adata, scenario):
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"Reference: {scenario['ref_study']}")
    print(f"Query:     {scenario['query_study']} (Subsampled to {QUERY_SUBSET_SIZE})")
    print(f"{'='*80}")

    # 1. Get Reference Data (Full)
    adata_ref = subset_data(adata, studies=[scenario['ref_study']], copy=True)
    cell_type_col = get_cell_type_column(adata_ref)
    
    # Get Reference Types
    ref_types = set(adata_ref.obs[cell_type_col].dropna().unique())
    print(f"\nReference contains {len(ref_types)} unique cell types.")

    # 2. Get Query Data (Subsampled)
    adata_query = subset_data(adata, studies=[scenario['query_study']], copy=True)
    
    # Subsample Query to 5k
    if adata_query.n_obs > QUERY_SUBSET_SIZE:
        indices = np.random.choice(adata_query.n_obs, QUERY_SUBSET_SIZE, replace=False)
        adata_query = adata_query[indices].copy()
    
    # Get Query Types & Counts
    query_counts = adata_query.obs[cell_type_col].value_counts()
    query_types = set(query_counts.index)
    
    print(f"Query (subset) contains {len(query_types)} unique cell types.")

    # 3. Analyze "Natural Novelty" (In Query, NOT in Ref)
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
        print("NONE. All cell types in the 5k query subset already exist in the reference.")
        print("(This means labels are perfectly harmonized/overlapping).")

    # 4. Analyze "Synthetic Candidates" (In Both, Robust enough to hide)
    # Ideally, we want a type that has >50 cells in Query so we have good stats
    shared_types = query_types.intersection(ref_types)
    
    candidates = []
    for ct in shared_types:
        count = query_counts[ct]
        if count >= 50:
            candidates.append((ct, count))
    
    # Sort by count
    candidates.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "-"*60)
    print("B. SYNTHETIC CANDIDATES (For Leave-One-Out Experiment)")
    print("-" * 60)
    print("Types present in BOTH that are abundant enough (>50 cells) to hide:")
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
    print("Loading Atlas data...")
    adata = sc.read_h5ad(DATA_PATH)
    
    # Normalize labels if needed (e.g., ensure strings)
    cell_type_col = get_cell_type_column(adata)
    adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str)

    for scenario in SCENARIOS:
        analyze_scenario(adata, scenario)

if __name__ == "__main__":
    main()

