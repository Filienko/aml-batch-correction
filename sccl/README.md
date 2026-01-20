# SCCL: Single Cell Classification Library

A flexible, easy-to-use tool for single cell RNA-seq classification and batch correction.

## Overview

SCCL provides a unified interface for:
- Running multiple classification models (SCimilarity, scVI, traditional ML)
- Predicting any column from single cell data
- Batch correction and integration
- Easy model comparison and evaluation

## Quick Start

### Installation

```bash
# Install from the repository
pip install -e .
```

### Basic Usage

#### 1. CLI Interface (Easiest)

```bash
# Predict cell types using SCimilarity
sccl predict --data data/example.h5ad --model scimilarity --target cell_type

# Train and predict with Random Forest
sccl predict --data data/example.h5ad --model random_forest --target cell_type --train-split 0.8

# Compare multiple models
sccl compare --data data/example.h5ad --target cell_type --models scimilarity,scvi,random_forest
```

#### 2. Python API

```python
from sccl import Pipeline
import scanpy as sc

# Load your data
adata = sc.read_h5ad("data/example.h5ad")

# Create pipeline
pipeline = Pipeline(model="scimilarity")

# Predict
predictions = pipeline.predict(adata, target_column="cell_type")

# Evaluate
metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ARI: {metrics['ari']:.3f}")
```

#### 3. Configuration File

```yaml
# config.yaml
data:
  path: "data/AML_scAtlas.h5ad"
  target_column: "cell_type"
  batch_column: "study"
  subset:
    studies: ["van_galen_2019", "zhang_2023"]

models:
  - scimilarity
  - random_forest:
      n_estimators: 200
      max_depth: 10

evaluation:
  metrics: ["accuracy", "ari", "nmi", "f1"]
  test_size: 0.2
```

```bash
sccl run --config config.yaml
```

## Features

### Supported Models

1. **SCimilarity** - Foundation model (no training needed)
2. **scVI** - Deep generative model for batch correction
3. **Random Forest** - Traditional ML classifier
4. **SVM** - Support Vector Machine
5. **Logistic Regression** - Simple linear classifier
6. **KNN** - K-Nearest Neighbors

### Key Capabilities

- **Flexible Input**: Works with AnnData (.h5ad) files
- **Any Target Column**: Predict cell types, cell states, disease status, etc.
- **Batch Correction**: Built-in batch effect removal
- **Subset Support**: Train/test on specific studies or conditions
- **Evaluation Metrics**: ARI, NMI, accuracy, F1, silhouette scores
- **Visualization**: Automatic UMAP plots, confusion matrices
- **Synthetic Data**: Generate test datasets for development

## Directory Structure

```
sccl/
├── __init__.py           # Package initialization
├── pipeline.py           # Main Pipeline class
├── models/               # Model implementations
│   ├── __init__.py
│   ├── base.py          # Base model interface
│   ├── scimilarity.py   # SCimilarity wrapper
│   ├── scvi.py          # scVI wrapper
│   └── sklearn.py       # Traditional ML models
├── data/                 # Data utilities
│   ├── __init__.py
│   ├── loader.py        # Data loading
│   ├── preprocessing.py # Preprocessing functions
│   └── synthetic.py     # Synthetic data generation
├── evaluation/          # Evaluation utilities
│   ├── __init__.py
│   ├── metrics.py       # Metric computation
│   └── visualization.py # Plotting functions
└── cli/                 # Command-line interface
    ├── __init__.py
    └── main.py          # CLI entry point
```

## Examples

See the `examples/` directory for detailed tutorials:

- `examples/01_basic_prediction.py` - Simple prediction workflow
- `examples/02_model_comparison.py` - Compare multiple models
- `examples/03_batch_correction.py` - Handle batch effects
- `examples/04_subset_analysis.py` - Work with data subsets
- `examples/05_synthetic_data.py` - Generate and use synthetic data

## Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [Model Guide](docs/models.md)
- [API Reference](docs/api_reference.md)
- [Configuration Reference](docs/configuration.md)

## Requirements

- Python >= 3.8
- scanpy
- anndata
- scikit-learn
- scimilarity (optional, for foundation model)
- scvi-tools (optional, for scVI model)
- numpy, pandas, matplotlib, seaborn

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{sccl2026,
  title = {SCCL: Single Cell Classification Library},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/Filienko/aml-batch-correction}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
