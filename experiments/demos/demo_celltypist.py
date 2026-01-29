#!/usr/bin/env python3
"""
Demo: Using CellTypist for Cell Type Annotation
===============================================

CellTypist is an automated cell type annotation tool that uses
pre-trained logistic regression models. NO manual marker definition needed!

This demo shows:
1. Using pre-trained CellTypist models
2. Comparing CellTypist vs other methods
3. Training custom CellTypist models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEMO: CellTypist for Automated Annotation")
print("="*80)

# Check if CellTypist is installed
try:
    import celltypist
    print("\n✓ CellTypist is installed")
    print(f"  Version: {celltypist.__version__}")
except ImportError:
    print("\n❌ CellTypist not installed!")
    print("\nInstall with:")
    print("  pip install celltypist")
    print("\nThen re-run this script.")
    sys.exit(1)

from sccl import Pipeline
from sccl.data import generate_synthetic_data
from sccl.evaluation import compute_metrics

# ==============================================================================
# Example 1: Using Pre-trained CellTypist Model
# ==============================================================================

print("\n" + "="*80)
print("Example 1: Pre-trained CellTypist Model")
print("="*80)

print("\nGenerating synthetic data...")
adata = generate_synthetic_data(n_cells=1000, n_genes=1000, seed=42)
print(f"✓ Generated {adata.n_obs:,} cells")

print("\nAvailable pre-trained models:")
print("  • Immune_All_Low.pkl  - All immune cells (27 types)")
print("  • Immune_All_High.pkl - All immune cells (59 types)")
print("  • Healthy_COVID19_PBMC.pkl - PBMC focused")
print("  • See: https://www.celltypist.org/models")

# Note: For demo with synthetic data, we'll use a simple approach
# For real data, you would use:
# pipeline = Pipeline(model='celltypist', model_params={'model': 'Immune_All_Low.pkl'})

print("\n(Skipping actual CellTypist prediction on synthetic data)")
print("(CellTypist models are trained on real biological data)")

# ==============================================================================
# Example 2: Training Custom CellTypist Model
# ==============================================================================

print("\n" + "="*80)
print("Example 2: Training Custom CellTypist Model")
print("="*80)

print("\nThis shows how to train CellTypist on YOUR reference data:")
print("""
from sccl import Pipeline

# Train on your labeled reference data
pipeline = Pipeline(model='celltypist')
pipeline.model.fit(adata_reference, target_column='cell_type')

# Predict on new data
predictions = pipeline.model.predict(adata_new)
""")

# ==============================================================================
# Example 3: Comparison with Other Methods
# ==============================================================================

print("\n" + "="*80)
print("Example 3: CellTypist vs Other Methods")
print("="*80)

print("\nFor label transfer comparison:")
print("""
models = {
    'CellTypist (pre-trained)': ('celltypist', {'model': 'Immune_All_Low.pkl'}),
    'CellTypist (custom)': ('celltypist', {}),  # Train on reference
    'SCimilarity': ('scimilarity', {'classifier': 'knn'}),
    'Random Forest': ('random_forest', {}),
}

# Test each model
for name, (model_type, params) in models.items():
    pipeline = Pipeline(model=model_type, model_params=params)

    # Train if needed
    if hasattr(pipeline.model, 'fit'):
        pipeline.model.fit(adata_ref, target_column='cell_type')

    # Predict
    predictions = pipeline.model.predict(adata_query)

    # Evaluate
    metrics = compute_metrics(y_true=adata_query.obs['cell_type'], y_pred=predictions)
    print(f"{name}: ARI = {metrics['ari']:.3f}")
""")

# ==============================================================================
# Example 4: Real Usage Example
# ==============================================================================

print("\n" + "="*80)
print("Example 4: Real Usage Pattern")
print("="*80)

print("""
# Scenario 1: Using pre-trained model (no training needed)
from sccl import Pipeline
import scanpy as sc

adata = sc.read_h5ad("your_data.h5ad")

pipeline = Pipeline(
    model='celltypist',
    model_params={
        'model': 'Immune_All_Low.pkl',  # Pre-trained model
        'majority_voting': True         # Refine predictions
    }
)

# Just predict - no training needed!
predictions = pipeline.model.predict(adata)
adata.obs['celltypist_prediction'] = predictions


# Scenario 2: Training custom model (for label transfer)
adata_ref = sc.read_h5ad("reference.h5ad")  # Your labeled data
adata_query = sc.read_h5ad("query.h5ad")    # New unlabeled data

# Train on reference
pipeline = Pipeline(model='celltypist')
pipeline.model.fit(adata_ref, target_column='cell_type')

# Predict on query
predictions = pipeline.model.predict(adata_query)
""")

# ==============================================================================
# Key Takeaways
# ==============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. NO MARKERS NEEDED
   ✓ CellTypist uses pre-trained models (no manual marker definition)
   ✓ Models learned from large reference datasets
   ✓ Works out-of-the-box for common cell types

2. TWO USAGE MODES
   a) Pre-trained models: Just load and predict
   b) Custom training: Train on your reference data

3. ADVANTAGES
   ✓ Fast: Logistic regression is very efficient
   ✓ Interpretable: Can see feature importances
   ✓ Well-validated: Widely used in literature

4. COMPARISON IN YOUR PIPELINE
   - CellTypist: Supervised ML with pre-trained models
   - SCimilarity: Foundation model with learned embeddings
   - Traditional ML: Train-from-scratch supervised learning

5. WHEN TO USE CELLTYPIST
   ✓ When you have standard cell types (immune, PBMC, etc.)
   ✓ When you want fast annotation without training
   ✓ When you want to replicate published analyses
   ✓ For your consensus pipeline (CellTypist + SingleR + scType)
""")

print("="*80)
print("✓ Demo completed!")
print("="*80)
print("\nNext steps:")
print("  1. Install: pip install celltypist")
print("  2. See available models: https://www.celltypist.org/models")
print("  3. Add to your experiments for comparison")
print("  4. Use in consensus pipeline as you mentioned in paper")
