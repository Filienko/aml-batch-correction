# SCCL Tool Overview

Complete guide to understanding and using the Single Cell Classification Library.

## What is SCCL?

SCCL is a comprehensive tool that simplifies single-cell RNA-seq classification by providing:

1. **Unified Interface** - One API for multiple classification methods
2. **Multiple Models** - From traditional ML to foundation models
3. **Easy Evaluation** - Automatic metrics and visualization
4. **Flexible Usage** - CLI, Python API, or config files
5. **Batch Correction** - Built-in handling of technical effects

## Who Should Use SCCL?

### Researchers
- Quick cell type annotation
- Method comparison for publications
- Exploratory analysis

### Bioinformaticians
- Pipeline integration
- Batch processing
- Custom model development

### Data Scientists
- Benchmarking ML approaches
- Feature engineering
- Model evaluation

## Tool Structure

### Directory Organization

```
aml-batch-correction/
├── sccl/                    # Main package
│   ├── __init__.py
│   ├── pipeline.py          # Core Pipeline class
│   ├── models/              # Model implementations
│   ├── data/                # Data utilities
│   ├── evaluation/          # Metrics & visualization
│   └── cli/                 # Command-line interface
├── examples/                # Example scripts
├── docs/                    # Documentation
├── data/                    # Data directory (created)
├── setup.py                 # Installation script
├── requirements.txt         # Dependencies
└── SCCL_README.md          # Main README

Your existing research scripts remain in the root directory.
```

### Core Components

#### 1. Pipeline (`sccl/pipeline.py`)
**What it does**: Orchestrates the entire workflow

**Key methods**:
- `predict()` - Run prediction
- `evaluate()` - Train/test evaluation
- `compare_models()` - Compare multiple models

**Example**:
```python
pipeline = Pipeline(model="random_forest")
metrics = pipeline.evaluate(adata, target_column="cell_type")
```

#### 2. Models (`sccl/models/`)
**What they do**: Implement different classification algorithms

**Available models**:
- `SCimilarityModel` - Foundation model
- `ScVIModel` - Deep generative model
- `RandomForestModel` - Ensemble classifier
- `SVMModel` - Support vector machine
- `LogisticRegressionModel` - Linear classifier
- `KNNModel` - Nearest neighbors

**How to add a new model**:
1. Create class inheriting from `BaseModel`
2. Implement `predict()` method
3. Optionally implement `fit()` and `get_embedding()`
4. Register in `models/__init__.py`

#### 3. Data (`sccl/data/`)
**What it does**: Handle data loading, preprocessing, generation

**Key functions**:
- `load_data()` - Load .h5ad files
- `preprocess_data()` - Standard preprocessing
- `subset_data()` - Filter by studies/cell types
- `generate_synthetic_data()` - Create test data

#### 4. Evaluation (`sccl/evaluation/`)
**What it does**: Compute metrics and create visualizations

**Key functions**:
- `compute_metrics()` - Calculate accuracy, ARI, NMI, F1, etc.
- `plot_umap()` - UMAP visualization
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_comparison()` - Model comparison bar charts

#### 5. CLI (`sccl/cli/`)
**What it does**: Provide command-line interface

**Commands**:
- `sccl predict` - Run predictions
- `sccl evaluate` - Evaluate model
- `sccl compare` - Compare models
- `sccl generate` - Create synthetic data
- `sccl run` - Run from config file

## Usage Patterns

### Pattern 1: Quick Cell Type Prediction

**Use case**: You have data, want quick cell type annotations

```bash
# CLI
sccl predict --data data.h5ad --model scimilarity --output predictions.csv

# Python
from sccl import Pipeline
pipeline = Pipeline(model="scimilarity")
predictions = pipeline.predict(adata)
```

**When to use**:
- New datasets
- Need fast results
- Don't want to train

### Pattern 2: Model Comparison

**Use case**: Determine best approach for your data

```python
from sccl import Pipeline

pipeline = Pipeline(model="random_forest")
comparison = pipeline.compare_models(
    adata=adata,
    target_column="cell_type",
    models=['random_forest', 'svm', 'scimilarity'],
    test_size=0.2
)
```

**When to use**:
- Research/publication
- Benchmarking
- Method selection

### Pattern 3: Batch Correction

**Use case**: Multi-study integration with batch effects

```python
pipeline = Pipeline(
    model="scimilarity",  # or "scvi"
    batch_key="study"
)
predictions = pipeline.predict(adata, target_column="cell_type")
```

**When to use**:
- Multiple studies
- Visible batch effects
- Need integrated analysis

### Pattern 4: Subset Analysis

**Use case**: Focus on specific studies or cell types

```python
from sccl.data import subset_data

# Train on one study
adata_train = subset_data(adata, studies=['study1'])

# Test on another
adata_test = subset_data(adata, studies=['study2'])

# Evaluate generalization
pipeline.fit(adata_train, target_column='cell_type')
predictions = pipeline.predict(adata_test)
```

**When to use**:
- Cross-study validation
- Cell type-specific analysis
- Development/debugging

### Pattern 5: Config-Based Pipeline

**Use case**: Reproducible, documented workflows

```yaml
# config.yaml
data:
  path: "data.h5ad"
  target_column: "cell_type"
  batch_column: "study"

models:
  - random_forest
  - scimilarity

evaluation:
  test_size: 0.2
```

```bash
sccl run --config config.yaml --output results.csv
```

**When to use**:
- Production pipelines
- Reproducibility
- Team collaboration

## Workflow Examples

### Workflow 1: New Dataset Exploration

```python
# 1. Load data
import scanpy as sc
adata = sc.read_h5ad("new_data.h5ad")

# 2. Quick check with SCimilarity
from sccl import Pipeline
pipeline = Pipeline(model="scimilarity")
predictions = pipeline.predict(adata)

# 3. Visualize
from sccl.evaluation import plot_umap
adata.obs['predicted'] = predictions
plot_umap(adata, color='predicted', save='initial_predictions.pdf')

# 4. If results look good, evaluate properly
metrics = pipeline.evaluate(adata, target_column='cell_type', test_size=0.2)
print(metrics)
```

### Workflow 2: Method Paper

```python
# Compare all available methods
from sccl import Pipeline

models = [
    'scimilarity',
    'scvi',
    'random_forest',
    'svm',
    'logistic_regression',
    'knn'
]

pipeline = Pipeline(model=models[0])
comparison = pipeline.compare_models(
    adata=adata,
    target_column='cell_type',
    models=models,
    test_size=0.2
)

# Save results
comparison.to_csv('method_comparison.csv')

# Plot
from sccl.evaluation import plot_comparison
plot_comparison(comparison, save='comparison.pdf')
```

### Workflow 3: Production Pipeline

```python
# 1. Load and preprocess
from sccl.data import load_data, preprocess_data
adata = load_data("data.h5ad")
adata = preprocess_data(adata, batch_key="study")

# 2. Create pipeline
pipeline = Pipeline(model="random_forest")

# 3. Train on reference
pipeline.model.fit(adata_reference, target_column='cell_type')

# 4. Predict on new data
predictions = pipeline.model.predict(adata_new)

# 5. Save results
import pandas as pd
results = pd.DataFrame({
    'cell_id': adata_new.obs_names,
    'prediction': predictions
})
results.to_csv('predictions.csv', index=False)
```

## Integration with Your Existing Code

SCCL is designed to complement your existing research code:

### Option 1: Side-by-Side
Keep your research scripts as is, use SCCL for specific tasks:

```python
# Your existing code
import scimilarity
adata = load_your_data()
embeddings = scimilarity.get_embeddings(adata)

# Use SCCL for evaluation
from sccl.evaluation import compute_metrics
metrics = compute_metrics(y_true, y_pred)
```

### Option 2: Gradual Migration
Move specific functions to SCCL over time:

```python
# Before
embeddings = your_custom_scimilarity_code(adata)

# After
from sccl.models import SCimilarityModel
model = SCimilarityModel()
embeddings = model.get_embedding(adata)
```

### Option 3: Full Integration
Use SCCL as your main interface:

```python
from sccl import Pipeline

# Replace multiple custom functions
pipeline = Pipeline(model="scimilarity", batch_key="study")
results = pipeline.evaluate(adata, target_column="cell_type")
```

## Extending SCCL

### Add a New Model

1. Create model file:

```python
# sccl/models/my_model.py
from .base import BaseModel
import numpy as np

class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model

    def predict(self, adata, target_column=None, batch_key=None):
        # Your prediction logic
        predictions = your_algorithm(adata)
        return predictions
```

2. Register model:

```python
# sccl/models/__init__.py
from .my_model import MyModel

AVAILABLE_MODELS = {
    # ... existing models
    'my_model': MyModel,
}
```

3. Use it:

```python
pipeline = Pipeline(model="my_model")
```

### Add Custom Metrics

```python
# sccl/evaluation/metrics.py
def compute_my_metric(y_true, y_pred):
    # Your metric logic
    return score

# Use in compute_metrics()
results['my_metric'] = compute_my_metric(y_true, y_pred)
```

### Add Custom Preprocessing

```python
# sccl/data/preprocessing.py
def my_preprocessing(adata, **kwargs):
    # Your preprocessing logic
    return adata

# Use it
from sccl.data import my_preprocessing
adata = my_preprocessing(adata)
```

## Best Practices

### 1. Data Preparation
- ✅ Use AnnData format (.h5ad)
- ✅ Include raw counts (for some models)
- ✅ Annotate cell types consistently
- ✅ Document batch/study information

### 2. Model Selection
- ✅ Start with SCimilarity for quick assessment
- ✅ Compare multiple models
- ✅ Consider your specific requirements (speed, accuracy, interpretability)
- ✅ Use appropriate batch correction

### 3. Evaluation
- ✅ Use proper train/test splits
- ✅ Report multiple metrics (accuracy, ARI, NMI)
- ✅ Visualize results (UMAP, confusion matrix)
- ✅ Check per-class performance

### 4. Reproducibility
- ✅ Set random seeds
- ✅ Document preprocessing steps
- ✅ Save configuration files
- ✅ Version your data

## Common Patterns

### Load Data Once, Try Multiple Approaches

```python
import scanpy as sc
from sccl import Pipeline

# Load once
adata = sc.read_h5ad("data.h5ad")

# Try multiple models
for model_name in ['random_forest', 'svm', 'scimilarity']:
    pipeline = Pipeline(model=model_name)
    metrics = pipeline.evaluate(adata, target_column='cell_type')
    print(f"{model_name}: {metrics['accuracy']:.3f}")
```

### Train on Subset, Test on Full

```python
from sccl.data import subset_data
from sccl import Pipeline

# Use subset for development
adata_sub = subset_data(adata, n_cells=5000)

# Quick iteration
pipeline = Pipeline(model="random_forest")
metrics = pipeline.evaluate(adata_sub, target_column='cell_type')

# Final evaluation on full data
if metrics['accuracy'] > 0.8:
    final_metrics = pipeline.evaluate(adata, target_column='cell_type')
```

### Batch Processing

```bash
# Process multiple files
for file in data/*.h5ad; do
    sccl evaluate \
        --data $file \
        --model random_forest \
        --target cell_type \
        --output results/$(basename $file .h5ad)_results.json
done
```

## Troubleshooting

See [Installation Guide](INSTALLATION.md) for installation issues.

### Low Performance

**Problem**: Model accuracy is poor

**Solutions**:
1. Check data quality (missing values, low counts)
2. Try different models
3. Use batch correction if needed
4. Increase training data
5. Check if labels are consistent

### Slow Execution

**Problem**: Pipeline is too slow

**Solutions**:
1. Use simpler models (Logistic Regression, KNN)
2. Work with data subsets during development
3. Use SCimilarity (no training needed)
4. Reduce preprocessing steps

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Downsample data
2. Use backed mode for h5ad files
3. Process in batches
4. Use simpler models

## Next Steps

1. **Get Started**: [Quick Start Guide](QUICKSTART.md)
2. **Learn More**: [User Guide](USER_GUIDE.md)
3. **Choose Models**: [Model Guide](MODELS.md)
4. **See Examples**: Check `examples/` directory
5. **Integrate**: Use with your existing code

## Support

- 📖 Documentation in `docs/`
- 💡 Examples in `examples/`
- 🐛 Issues: https://github.com/Filienko/aml-batch-correction/issues
