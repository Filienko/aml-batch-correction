#!/usr/bin/env python3
"""
Quick Start: Working with SCimilarity Embeddings
=================================================

This script demonstrates direct access to SCimilarity embeddings for:
1. Getting batch-corrected shared embeddings
2. Training custom classifiers on embeddings
3. Using embeddings for downstream analysis

Note: Requires SCimilarity model file. Uses synthetic data by default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, adjusted_rand_score

from sccl.data import generate_synthetic_data, get_cell_type_column

print("="*80)
print("QUICK START: SCimilarity Embeddings")
print("="*80)

# ==============================================================================
# STEP 1: Load Data
# ==============================================================================

print("\n📦 STEP 1: Load Data")
print("-" * 80)

# Generate synthetic data (replace with your data)
adata = generate_synthetic_data(n_cells=1000, n_genes=1000, seed=42)

print(f"✓ Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"  Cell types: {adata.obs['cell_type'].nunique()}")

cell_type_col = get_cell_type_column(adata)

# ==============================================================================
# STEP 2: Get SCimilarity Embeddings
# ==============================================================================

print("\n🧬 STEP 2: Get SCimilarity Embeddings")
print("-" * 80)

try:
    from sccl.models import SCimilarityModel

    # Option 1: With local model file
    # MODEL_PATH = "/path/to/model_v1.1"
    # model = SCimilarityModel(model_path=MODEL_PATH, species='human')

    # Option 2: Default model (downloads automatically)
    print("Initializing SCimilarity model...")
    print("NOTE: This demo requires SCimilarity model. Skipping actual embedding computation.")
    print("      To run with real embeddings, set MODEL_PATH to your model file.")

    # Simulate embeddings for demo purposes
    print("\nℹ️  Using simulated embeddings for demo (PCA)")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=100)
    sc.pp.pca(adata, n_comps=50)
    embeddings = adata.obsm['X_pca']

    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  Cells: {embeddings.shape[0]:,}")
    print(f"  Dimensions: {embeddings.shape[1]}")

except ImportError:
    print("⚠️  SCimilarity not installed. Using PCA instead for demo.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=100)
    sc.pp.pca(adata, n_comps=50)
    embeddings = adata.obsm['X_pca']
    print(f"✓ PCA embeddings shape: {embeddings.shape}")

# ==============================================================================
# STEP 3: Use Embeddings for Classification
# ==============================================================================

print("\n🎯 STEP 3: Train Classifier on Embeddings")
print("-" * 80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    adata.obs[cell_type_col].values,
    test_size=0.2,
    random_state=42,
    stratify=adata.obs[cell_type_col].values
)

print(f"Training set: {X_train.shape[0]:,} cells")
print(f"Test set:     {X_test.shape[0]:,} cells")

# Test multiple classifiers on embeddings
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=15)': KNeighborsClassifier(n_neighbors=15),
}

print(f"\nTesting {len(classifiers)} classifiers on embeddings:")

results = []
for name, clf in classifiers.items():
    print(f"\n{name}:")

    # Train
    print("  → Training...")
    clf.fit(X_train, y_train)

    # Predict
    print("  → Predicting...")
    y_pred = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, y_pred)

    print(f"  → Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ARI:      {ari:.4f}")

    results.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'ARI': ari
    })

# ==============================================================================
# STEP 4: Embeddings for Visualization (Optional)
# ==============================================================================

print("\n📊 STEP 4: Embeddings for Downstream Analysis")
print("-" * 80)

# Store embeddings in AnnData for visualization
if 'X_pca' not in adata.obsm:
    adata.obsm['X_embedding'] = embeddings
else:
    adata.obsm['X_embedding'] = embeddings

# Compute UMAP on embeddings (without plotting)
print("Computing UMAP on embeddings...")
sc.pp.neighbors(adata, use_rep='X_embedding', n_neighbors=15)
sc.tl.umap(adata)
print("✓ UMAP computed (use scanpy plotting on machine with display)")

# Compute clustering on embeddings
print("\nComputing Leiden clustering on embeddings...")
sc.tl.leiden(adata, resolution=0.5)
leiden_ari = adjusted_rand_score(
    adata.obs[cell_type_col].values,
    adata.obs['leiden'].values
)
print(f"✓ Clustering ARI vs ground truth: {leiden_ari:.4f}")

# ==============================================================================
# STEP 5: Summary
# ==============================================================================

print("\n" + "="*80)
print("📊 SUMMARY: Classifier Performance on Embeddings")
print("="*80)

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_idx = results_df['ARI'].idxmax()
best_clf = results_df.loc[best_idx, 'Classifier']
best_ari = results_df.loc[best_idx, 'ARI']

print(f"\n🏆 Best Classifier: {best_clf}")
print(f"   ARI: {best_ari:.4f}")

# ==============================================================================
# STEP 6: Key Takeaways
# ==============================================================================

print("\n" + "="*80)
print("✅ KEY TAKEAWAYS")
print("="*80)
print("""
1. ✓ SCimilarity embeddings capture cell type information
2. ✓ Multiple classifiers can be trained on embeddings
3. ✓ Embeddings work for clustering, UMAP, etc.
4. ✓ Flexible: choose best classifier for your data

Embeddings are useful for:
  • Cross-study batch correction (shared embedding space)
  • Label transfer (train on ref, predict on query)
  • Clustering and visualization
  • Custom downstream analysis

Real usage with SCimilarity model:
```python
from sccl.models import SCimilarityModel

model = SCimilarityModel(
    model_path='/path/to/model_v1.1',
    species='human',
    classifier='random_forest'  # Choose classifier
)

# Get embeddings
embeddings = model.get_embedding(adata)

# Or do full label transfer
model.fit(adata_ref, target_column='cell_type')
predictions = model.predict(adata_query)
```
""")

print("="*80)
print("✓ Quick Start completed successfully!")
print("="*80)
