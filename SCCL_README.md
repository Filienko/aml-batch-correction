# SCCL: Single Cell Classification Library

**A flexible, easy-to-use tool for single cell RNA-seq classification and batch correction.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SCCL provides a unified interface for classifying single-cell RNA-seq data using multiple machine learning approaches, from traditional classifiers to state-of-the-art foundation models.

### Key Features

✨ **Multiple Models**: SCimilarity, scVI, Random Forest, SVM, Logistic Regression, KNN
🔄 **Batch Correction**: Built-in support for handling batch effects
📊 **Easy Evaluation**: Comprehensive metrics (accuracy, ARI, NMI, F1, etc.)
🎯 **Flexible**: Predict any column, work with subsets, custom configs
🚀 **Fast**: Quick predictions with foundation models, no training needed
💻 **Three interfaces**: CLI, Python API, and configuration files

## Quick Start

### Installation

```bash
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction
pip install -e .
```

### 30-Second Demo

```bash
# Generate test data
python examples/generate_synthetic_data.py

# Run prediction
sccl evaluate \
  --data data/synthetic_example.h5ad \
  --model random_forest \
  --target cell_type
```

### Python API

```python
from sccl import Pipeline
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Create pipeline
pipeline = Pipeline(model="random_forest")

# Evaluate
metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ARI: {metrics['ari']:.3f}")
```

## Supported Models

| Model | Type | Training | Batch Correction | Best For |
|-------|------|----------|-----------------|----------|
| **SCimilarity** | Foundation | ❌ None | ✅ | General-purpose, quick results |
| **scVI** | Deep Learning | ✅ Required | ✅ | Multi-study integration |
| **Random Forest** | Traditional ML | ✅ Required | ❌ | Baseline, interpretable |
| **SVM** | Traditional ML | ✅ Required | ❌ | Small datasets |
| **Logistic Regression** | Traditional ML | ✅ Required | ❌ | Fast, simple |
| **KNN** | Instance-based | ❌ None | ❌ | Reference mapping |

## Documentation

### Getting Started
- 📚 **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running in 5 minutes
- 🔧 **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation instructions
- 📖 **[User Guide](docs/USER_GUIDE.md)** - Comprehensive usage guide
- 🤖 **[Model Guide](docs/MODELS.md)** - Choose the right model

### Examples
- `examples/01_basic_prediction.py` - Simple prediction workflow
- `examples/02_model_comparison.py` - Compare multiple models
- `examples/03_batch_correction.py` - Handle batch effects
- `examples/04_subset_analysis.py` - Work with data subsets
- `examples/example_config.yaml` - Configuration file example

## Usage Examples

### CLI: Predict Cell Types

```bash
# With Random Forest
sccl predict \
  --data data.h5ad \
  --model random_forest \
  --target cell_type \
  --output predictions.csv

# With SCimilarity (foundation model)
sccl predict \
  --data data.h5ad \
  --model scimilarity \
  --batch-key study
```

### CLI: Compare Models

```bash
sccl compare \
  --data data.h5ad \
  --models random_forest,svm,logistic_regression \
  --target cell_type \
  --output comparison.csv
```

### Python: Batch Correction

```python
from sccl import Pipeline

# Use SCimilarity for batch correction
pipeline = Pipeline(
    model="scimilarity",
    batch_key="study"
)

predictions = pipeline.predict(adata, target_column="cell_type")
```

### Python: Model Comparison

```python
from sccl import Pipeline

pipeline = Pipeline(model="random_forest")

comparison = pipeline.compare_models(
    adata=adata,
    target_column="cell_type",
    models=['random_forest', 'svm', 'scimilarity'],
    test_size=0.2
)

print(comparison)
```

### Config File

Create `config.yaml`:

```yaml
data:
  path: "data/your_data.h5ad"
  target_column: "cell_type"
  batch_column: "study"

models:
  - random_forest
  - scimilarity

evaluation:
  test_size: 0.2
  metrics: [accuracy, ari, nmi, f1]
```

Run:

```bash
sccl run --config config.yaml --output results.csv
```

## Use Cases

### 1. Cell Type Annotation
Predict cell types in new datasets using pre-trained models or train custom classifiers.

### 2. Batch Effect Correction
Integrate data from multiple studies while preserving biological variation.

### 3. Model Benchmarking
Compare different classification approaches on your data.

### 4. Label Transfer
Transfer annotations from well-annotated reference to new query data.

### 5. Quality Control
Validate cell type assignments across different methods.

## Architecture

```
sccl/
├── __init__.py           # Package entry point
├── pipeline.py           # Main Pipeline class
├── models/               # Model implementations
│   ├── base.py          # Base model interface
│   ├── scimilarity.py   # SCimilarity wrapper
│   ├── scvi.py          # scVI wrapper
│   └── sklearn.py       # Traditional ML models
├── data/                 # Data utilities
│   ├── loader.py        # Data loading
│   ├── preprocessing.py # Preprocessing
│   └── synthetic.py     # Synthetic data generation
├── evaluation/          # Evaluation utilities
│   ├── metrics.py       # Metrics computation
│   └── visualization.py # Plotting
└── cli/                 # Command-line interface
    └── main.py          # CLI entry point
```

## Requirements

### Core
- Python >= 3.8
- numpy, pandas, scipy
- scikit-learn
- anndata, scanpy
- matplotlib, seaborn

### Optional
- `scimilarity` - For foundation model
- `scvi-tools` - For scVI batch correction

Install optional dependencies:
```bash
pip install scimilarity scvi-tools
# or
pip install -e ".[all]"
```

## Data Format

SCCL works with AnnData (`.h5ad`) files:

**Required**:
- Expression matrix in `.X`
- Cell metadata in `.obs` with target column

**Optional**:
- Batch/study column for batch correction
- Raw counts in `.raw` (for some models)

## Performance Tips

1. **Start with SCimilarity** for quick assessment
2. **Use batch_key** if you have batch effects
3. **Compare models** - different datasets favor different approaches
4. **Check preprocessing** - ensure data quality
5. **Use subsets** for faster iteration during development

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use SCCL in your research, please cite:

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

## Acknowledgments

SCCL builds on excellent work from:
- [SCimilarity](https://github.com/Genentech/scimilarity) - Foundation model for cell type annotation
- [scVI](https://scvi-tools.org/) - Deep generative models for single-cell
- [scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis in Python
- [scikit-learn](https://scikit-learn.org/) - Machine learning library

## Support

- 📖 [Documentation](docs/)
- 💬 [Issues](https://github.com/Filienko/aml-batch-correction/issues)
- 📧 Contact: your.email@example.com

## Roadmap

- [ ] Add more foundation models
- [ ] Support for spatial transcriptomics
- [ ] GUI interface
- [ ] Pre-trained models for common tissues
- [ ] Integration with more single-cell tools

---

**Made with ❤️ for the single-cell community**
