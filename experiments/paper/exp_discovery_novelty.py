#!/usr/bin/env python3
"""
Experiment: Novelty Detection / Out-of-Distribution (OOD) Discovery
====================================================================

Hypothesis:
    While CellTypist (parametric) forces novel cell types into known classes 
    (silent failure), SCimilarity (non-parametric) will map them to low-density 
    regions of the embedding space, allowing for "flagged failure" (discovery).

Methodology: "Leave-One-Class-Out"
    1. Remove a distinct cell type (e.g., 'Erythroid' or 'HSC') from the Reference.
    2. Train models (or use pre-trained) on this restricted Reference.
    3. Predict on a Query that CONTAINS the hidden cell type.
    4. Metric: AUROC of the "Uncertainty Score" or "Distance Score" in detecting the hidden class.

"""

import sys
import warnings
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.models.celltypist import CellTypistModel

# --- CONFIGURATION ---
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results" / "discovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We use the 'beneyto' dataset as it is high quality, splitting it for this test
# Or we can use beneyto -> jiang. Let's stick to a controlled split of one dataset 
# to ensure batch effects don't confound the "novelty" signal.
STUDY_TO_USE = 'beneyto-calabuig-2023' 

# The class to HIDE from the reference (Simulating a "New" cell type)
# Candidates: 'HSC', 'Erythroid', 'Plasma cell', 'T cell'
HIDDEN_CLASS = 'HSC' 

MAX_CELLS = 10000

def get_scimilarity_embeddings(adata, model_path):
    """Helper to get embeddings efficiently."""
    pipeline = Pipeline(
        model='scimilarity',
        model_params={'model_path': model_path}
    )
    # Ensure log1p (preprocessing might have done it, but SCimilarity checks)
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return pipeline.model.get_embedding(adata)

def compute_novelty_scores(
    adata_ref, 
    adata_query, 
    embeddings_ref, 
    embeddings_query, 
    cell_type_col,
    hidden_class
):
    """
    Runs both models and calculates their 'Novelty/Uncertainty' scores.
    """
    results = {}
    
    # --- 1. SCimilarity: Distance to Nearest Neighbor ---
    print(f"  Computing SCimilarity distances...")
    # We look at the distance to the nearest neighbor in the REFERENCE
    nn = NearestNeighbors(n_neighbors=5, n_jobs=-1, metric='euclidean')
    nn.fit(embeddings_ref)
    
    # Get distance to nearest neighbor (k=1) and average of k=5 for robustness
    dists, _ = nn.kneighbors(embeddings_query)
    
    # Score = Mean distance to 5 nearest reference neighbors
    # Higher distance = More likely to be novel
    scim_scores = dists.mean(axis=1)
    results['SCimilarity'] = scim_scores

    # --- 2. CellTypist: 1 - Prediction Confidence ---
    print(f"  Training CellTypist on restricted reference...")
    
    # CellTypist needs to be trained on the Reference (which lacks the hidden class)
    # We use a custom model to ensure it doesn't use pretrained weights that "know" the hidden class
    ct_model = CellTypistModel(model=None) # Start fresh
    ct_model.fit(adata_ref, target_column=cell_type_col)
    
    print(f"  Predicting with CellTypist...")
    # Access underlying predict_proba logic
    # We need to use the internal celltypist functionality to get probabilities
    import celltypist
    
    # Normalize query for CellTypist
    adata_query_norm = adata_query.copy()
    if 'log1p' not in adata_query_norm.uns:
        sc.pp.normalize_total(adata_query_norm, target_sum=1e4)
        sc.pp.log1p(adata_query_norm)
        
    predictions = celltypist.annotate(
        adata_query_norm,
        model=ct_model._model,
        majority_voting=False # Raw probabilities are better for uncertainty
    )
    
    # Score = 1 - Max Probability (Probability of "None of the above")
    # Higher score = More uncertain = More likely to be novel
    probs = predictions.probability_matrix.values
    max_probs = probs.max(axis=1)
    ct_scores = 1.0 - max_probs
    results['CellTypist'] = ct_scores
    
    return results

def plot_distributions(df, hidden_class, output_path):
    """Plot score distributions for Known vs Novel cells."""
    plt.figure(figsize=(12, 5))
    
    for i, model in enumerate(['SCimilarity', 'CellTypist']):
        plt.subplot(1, 2, i+1)
        sns.kdeplot(
            data=df, x=f'{model}_Score', hue='Is_Novel', 
            fill=True, common_norm=False, palette=['gray', 'red']
        )
        plt.title(f"{model}\nScore Distribution")
        plt.xlabel("Novelty Score (Distance / Uncertainty)")
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("="*80)
    print(f"EXPERIMENT: Discovery of '{HIDDEN_CLASS}' via Novelty Detection")
    print("="*80)

    # 1. Load Data
    print("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)
    
    # Subset to single study to control for batch effects
    adata = subset_data(adata, studies=[STUDY_TO_USE])
    
    # Filter out small classes that might disturb the test
    counts = adata.obs[cell_type_col].value_counts()
    valid_types = counts[counts > 50].index
    adata = adata[adata.obs[cell_type_col].isin(valid_types)].copy()
    
    print(f"Using study: {STUDY_TO_USE} ({adata.n_obs} cells)")
    print(f"Target hidden class: {HIDDEN_CLASS} ({counts.get(HIDDEN_CLASS, 0)} cells)")

    if HIDDEN_CLASS not in adata.obs[cell_type_col].values:
        print(f"ERROR: Hidden class '{HIDDEN_CLASS}' not found in dataset.")
        return

    # 2. Split into Reference (NO Hidden Class) and Query (Mixed)
    # Query gets ALL cells (to see if we can pick out the hidden ones)
    # Reference gets only the "Known" cells
    
    # Shuffle indices
    indices = np.random.permutation(adata.n_obs)
    split = int(0.6 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    
    # Initial Split
    adata_train_full = adata[train_idx].copy()
    adata_query = adata[test_idx].copy()
    
    # REMOVE Hidden Class from Reference
    print(f"\nRemoving '{HIDDEN_CLASS}' from Reference...")
    ref_mask = adata_train_full.obs[cell_type_col] != HIDDEN_CLASS
    adata_ref = adata_train_full[ref_mask].copy()
    
    print(f"Reference: {adata_ref.n_obs} cells (Known types only)")
    print(f"Query:     {adata_query.n_obs} cells (Mixed types)")
    
    # Ground Truth: Is the cell the hidden class?
    y_true = (adata_query.obs[cell_type_col] == HIDDEN_CLASS).astype(int).values
    n_novel = y_true.sum()
    print(f"Query contains {n_novel} '{HIDDEN_CLASS}' cells ({n_novel/len(y_true)*100:.1f}%)")

    # 3. Preprocess and Embed
    print("\nComputing SCimilarity embeddings...")
    # Preprocess
    adata_ref_prep = preprocess_data(adata_ref.copy(), n_top_genes=2000)
    adata_query_prep = preprocess_data(adata_query.copy(), n_top_genes=2000)
    
    # Get Embeddings
    emb_ref = get_scimilarity_embeddings(adata_ref_prep, MODEL_PATH)
    emb_query = get_scimilarity_embeddings(adata_query_prep, MODEL_PATH)
    
    # 4. Run Experiment
    print("\nCalculating Novelty Scores...")
    scores = compute_novelty_scores(
        adata_ref, adata_query, 
        emb_ref, emb_query, 
        cell_type_col, HIDDEN_CLASS
    )
    
    # 5. Evaluate
    print("\n" + "="*80)
    print("RESULTS: AUROC (1.0 = Perfect separation of novel cells)")
    print("="*80)
    
    results_df = pd.DataFrame({
        'Is_Novel': y_true,
        'Cell_Type': adata_query.obs[cell_type_col].values
    })
    
    for model_name, score_values in scores.items():
        # Calculate AUROC
        auroc = roc_auc_score(y_true, score_values)
        
        # Calculate AUPRC (Area Under Precision-Recall Curve) - better for imbalanced data
        precision, recall, _ = precision_recall_curve(y_true, score_values)
        auprc = auc(recall, precision)
        
        results_df[f'{model_name}_Score'] = score_values
        
        print(f"{model_name:12s} | AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")
        
        if model_name == 'SCimilarity':
            scim_auc = auroc
        else:
            ct_auc = auroc

    # 6. Conclusion
    print("-" * 80)
    if scim_auc > ct_auc:
        diff = scim_auc - ct_auc
        print(f"✓ CONCLUSION: SCimilarity outperforms CellTypist by {diff:.3f}")
        print("  SCimilarity successfully mapped the novel cells to 'empty' space,")
        print("  while CellTypist likely forced them into existing classes.")
    else:
        print("x CONCLUSION: CellTypist Uncertainty was more informative.")
        
    # Save results
    csv_path = OUTPUT_DIR / f"novelty_results_{HIDDEN_CLASS}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed scores saved to: {csv_path}")
    
    # Plot
    plot_path = OUTPUT_DIR / f"novelty_dist_{HIDDEN_CLASS}.png"
    plot_distributions(results_df, HIDDEN_CLASS, plot_path)
    print(f"Distributions plotted to: {plot_path}")

if __name__ == "__main__":
    main()
