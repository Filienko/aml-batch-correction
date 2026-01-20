#!/usr/bin/env python3
"""Example 2: Compare multiple models.

This example demonstrates:
1. Comparing different classification models
2. Visualizing comparison results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sccl import Pipeline
from sccl.data import generate_synthetic_data
from sccl.evaluation import plot_comparison

print("="*60)
print("EXAMPLE 2: Model Comparison")
print("="*60)

# Load data
print("\nLoading data...")
adata = generate_synthetic_data(n_cells=1500, n_genes=1000, seed=42)

# Create pipeline
print("\nComparing models...")
pipeline = Pipeline(model="random_forest")  # Dummy, will be replaced

# Compare multiple models
models = ['random_forest', 'svm', 'logistic_regression', 'knn']

comparison = pipeline.compare_models(
    adata=adata,
    target_column="cell_type",
    models=models,
    test_size=0.2,
)

# Print results
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
print(comparison)
print("="*60)

# Find best model
best_model = comparison['accuracy'].idxmax()
print(f"\nBest model by accuracy: {best_model}")
print(f"Accuracy: {comparison.loc[best_model, 'accuracy']:.4f}")

# Plot comparison
try:
    import matplotlib.pyplot as plt
    fig = plot_comparison(comparison, metrics=['accuracy', 'ari', 'nmi'])
    plt.savefig('model_comparison.pdf')
    print("\n✓ Saved comparison plot to model_comparison.pdf")
    plt.close()
except ImportError:
    print("\n(Install matplotlib to generate plots)")

print("\n✓ Example completed successfully!")
print("\nKey takeaways:")
print("  - compare_models() makes it easy to test multiple approaches")
print("  - Results are returned as a pandas DataFrame for easy analysis")
print("  - Different models have different trade-offs")
