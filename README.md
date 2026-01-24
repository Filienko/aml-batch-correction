# SCCL: Single Cell Classification Library

**A unified tool for single-cell RNA-seq classification with multiple models.** Easily classify cells, handle batch effects, and compare different methods through a simple interface.

## What It Does

SCCL provides a unified pipeline for:
- **Cell type classification** using foundation models (SCimilarity, scVI) or traditional ML (Random Forest, SVM, KNN, Logistic Regression)
- **Batch correction** for multi-study integration
- **Label transfer** across datasets
- **Model comparison** to find the best approach for your data

---

## 🚀 Quick Start

**New to SCCL?** Start here: **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step guide with complete examples.

**Quick examples:**

```bash
# Run interactive demos (work on headless VM)
python experiments/demos/quickstart_annotation.py     # Full annotation workflow
python experiments/demos/quickstart_embeddings.py     # Work with embeddings
python experiments/demos/01_basic_prediction.py       # Basic usage

# Test all demos
bash experiments/demos/test_all_demos.sh
```

---

## Installation & Usage

### Installation

```bash
# Clone and install
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction
pip install -e .

# Optional: Install foundation models
pip install scimilarity scvi-tools
```

### Basic Usage (3 ways)

#### 1. Command Line

```bash
# Predict cell types with Random Forest
sccl predict --data data.h5ad --model random_forest --target cell_type

# Evaluate model performance
sccl evaluate --data data.h5ad --model scimilarity --target cell_type --test-size 0.2

# Compare multiple models
sccl compare --data data.h5ad --models random_forest,svm,scimilarity --target cell_type
```

#### 2. Python API

```python
from sccl import Pipeline
import scanpy as sc

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Predict with SCimilarity
pipeline = Pipeline(model="scimilarity")
predictions = pipeline.predict(adata)

# Evaluate
metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)
print(f"Accuracy: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")
```

#### 3. Configuration File

```yaml
# config.yaml
data:
  path: "data.h5ad"
  target_column: "cell_type"
  batch_column: "study"
models:
  - scimilarity
  - random_forest
evaluation:
  test_size: 0.2
```

```bash
sccl run --config config.yaml --output results.csv
```

## Available Models

| Model | Type | Training | Batch Correction | Best For |
|-------|------|----------|-----------------|----------|
| `scimilarity` | Foundation | None | ✅ Yes | Quick predictions, new datasets |
| `scvi` | Deep Learning | Required | ✅ Yes | Multi-study integration |
| `random_forest` | Traditional ML | Required | ❌ No | Baseline, interpretable |
| `svm` | Traditional ML | Required | ❌ No | Small datasets |
| `logistic_regression` | Traditional ML | Required | ❌ No | Fast, simple |
| `knn` | Instance-based | None | ❌ No | Reference mapping |

## Common Tasks

### Cell Type Prediction

```python
from sccl import Pipeline

# With SCimilarity (no training needed)
pipeline = Pipeline(model="scimilarity")
predictions = pipeline.predict(adata)

# With Random Forest (supervised)
pipeline = Pipeline(model="random_forest")
metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)
```

### Batch Correction

```python
# SCimilarity handles batch effects automatically
pipeline = Pipeline(model="scimilarity", batch_key="study")
predictions = pipeline.predict(adata, target_column="cell_type")

# Or use scVI
pipeline = Pipeline(model="scvi", batch_key="study")
predictions = pipeline.predict(adata)
```

### Label Transfer (Cross-Study)

```python
from sccl.data import subset_data

# Train on one study
adata_ref = subset_data(adata, studies=['study1'])

# Test on another
adata_query = subset_data(adata, studies=['study2'])

# Transfer labels
pipeline = Pipeline(model="random_forest")
pipeline.model.fit(adata_ref, target_column='cell_type')
predictions = pipeline.model.predict(adata_query)
```

### Model Comparison

```python
# Compare all models
comparison = pipeline.compare_models(
    adata=adata,
    target_column="cell_type",
    models=['scimilarity', 'random_forest', 'svm', 'knn'],
    test_size=0.2
)
print(comparison)
```

## Project Structure

```
aml-batch-correction/
├── sccl/                          # Main package
│   ├── pipeline.py                # Pipeline class
│   ├── models/                    # Model implementations
│   ├── data/                      # Data utilities
│   ├── evaluation/                # Metrics & visualization
│   └── cli/                       # Command-line interface
├── experiments/
│   ├── paper/                     # Paper experiments
│   │   ├── exp1_annotation_replication.py
│   │   ├── exp2_label_transfer.py
│   │   ├── exp3_computational_efficiency.py
│   │   ├── exp4_cross_study_generalization.py
│   │   └── run_all_experiments.py
│   ├── notebooks/                 # Jupyter notebooks
│   └── demos/                     # Simple examples
│       ├── 01_basic_prediction.py
│       ├── 02_model_comparison.py
│       └── generate_synthetic_data.py
├── setup.py
├── requirements.txt
└── README.md
```

## Running Paper Experiments

### All Experiments at Once

```bash
cd experiments/paper
python run_all_experiments.py
```

This runs all 4 experiments and generates a summary report in `experiments/paper/results/`.

### Individual Experiments

```bash
# Experiment 1: Can SCimilarity replicate expert annotations?
python experiments/paper/exp1_annotation_replication.py

# Experiment 2: SCimilarity vs traditional ML for label transfer
python experiments/paper/exp2_label_transfer.py

# Experiment 3: How fast is SCimilarity?
python experiments/paper/exp3_computational_efficiency.py

# Experiment 4: Cross-study robustness
python experiments/paper/exp4_cross_study_generalization.py
```

**Requirements**: Update `DATA_PATH` in each script to point to your AML_scAtlas.h5ad file.

## Demo Examples

```bash
# Generate synthetic test data
python experiments/demos/generate_synthetic_data.py

# Run basic examples
python experiments/demos/01_basic_prediction.py
python experiments/demos/02_model_comparison.py
python experiments/demos/03_batch_correction.py
```

## Data Format

SCCL works with **AnnData** (`.h5ad`) files containing:

**Required:**
- Expression matrix in `.X`
- Cell metadata in `.obs` with target column (e.g., `cell_type`)

**Optional:**
- Batch/study column for batch correction (e.g., `batch`, `study`)
- Raw counts in `.raw` (for some models)

## CLI Reference

### Commands

```bash
# Predict cell types
sccl predict --data <file> --model <model> [--target <column>] [--output <file>]

# Evaluate performance with train/test split
sccl evaluate --data <file> --model <model> --target <column> [--test-size 0.2]

# Compare multiple models
sccl compare --data <file> --models <model1,model2,...> --target <column>

# Generate synthetic data for testing
sccl generate --output <file> [--n-cells 1000] [--n-genes 2000]

# Run from configuration file
sccl run --config <file.yaml> [--output <file>]
```

### Common Options

- `--data PATH` - Path to .h5ad file
- `--model NAME` - Model to use (scimilarity, random_forest, svm, etc.)
- `--target COLUMN` - Target column name in .obs
- `--batch-key COLUMN` - Batch column for batch correction
- `--test-size FLOAT` - Test set fraction (default: 0.2)
- `--output PATH` - Output file path

### Examples

```bash
# Quick prediction
sccl predict --data data.h5ad --model scimilarity --output predictions.csv

# Evaluate with batch correction
sccl evaluate --data data.h5ad --model scimilarity --target cell_type --batch-key study

# Compare models, save results
sccl compare \
  --data data.h5ad \
  --models scimilarity,random_forest,svm \
  --target cell_type \
  --output comparison.csv

# Generate test data
sccl generate --output test_data.h5ad --n-cells 2000 --n-cell-types 5
```

## Python API Reference

### Pipeline Class

```python
from sccl import Pipeline

pipeline = Pipeline(
    model="scimilarity",           # Model name
    batch_key="study",             # Optional: batch column
    preprocess=True,               # Apply standard preprocessing
    model_params={}                # Optional: model-specific parameters
)
```

**Methods:**

- `predict(adata, target_column=None)` - Predict cell types
- `evaluate(adata, target_column, test_size=0.2)` - Train/test evaluation
- `compare_models(adata, target_column, models, test_size=0.2)` - Compare multiple models

### Data Utilities

```python
from sccl.data import load_data, subset_data, preprocess_data, generate_synthetic_data

# Load data
adata = load_data("data.h5ad")

# Subset
adata_subset = subset_data(adata, studies=['study1', 'study2'])
adata_subset = subset_data(adata, cell_types=['T_cell', 'B_cell'])
adata_subset = subset_data(adata, n_cells=5000)

# Preprocess
adata = preprocess_data(adata, batch_key="study", n_top_genes=2000)

# Generate synthetic data
adata = generate_synthetic_data(n_cells=1000, n_genes=2000, n_cell_types=5)
```

### Evaluation

```python
from sccl.evaluation import compute_metrics, plot_confusion_matrix, plot_umap

# Compute metrics
metrics = compute_metrics(y_true, y_pred, adata, metrics=['accuracy', 'ari', 'nmi', 'f1'])

# Plot confusion matrix
fig = plot_confusion_matrix(y_true, y_pred, normalize=True)

# Plot UMAP
plot_umap(adata, color='cell_type', save='umap.pdf')
```

## Requirements

- Python >= 3.8
- numpy, pandas, scipy
- scikit-learn
- anndata, scanpy
- matplotlib, seaborn

**Optional:**
- `scimilarity` - For foundation model
- `scvi-tools` - For scVI model

## Tips

### Model Selection

- **New dataset?** Start with `scimilarity` (fast, no training)
- **Batch effects?** Use `scimilarity` or `scvi` with `batch_key`
- **Need interpretability?** Try `random_forest` or `logistic_regression`
- **Small dataset (<10k cells)?** Use `svm` or `logistic_regression`
- **Large dataset (>50k cells)?** Use `random_forest` or `scimilarity`

### Performance

- Use `subset_data(n_cells=5000)` for faster iteration during development
- SCimilarity and KNN don't require training (faster)
- Random Forest is a good balance of speed and accuracy
- SVM can be slow on large datasets

### Troubleshooting

- **Low accuracy?** Try different models, check data quality, use batch correction
- **Out of memory?** Downsample with `subset_data(n_cells=10000)`
- **Slow execution?** Use simpler models or preprocess once and save
- **"Column not found"?** Check column names with `adata.obs.columns`

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

## Acknowledgments

Built on excellent tools:
- [SCimilarity](https://github.com/Genentech/scimilarity) - Foundation model for cell annotation
- [scVI](https://scvi-tools.org/) - Deep generative models for single-cell
- [scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis in Python

## License

MIT License - See LICENSE file for details.

## Support

- 📖 Check this README for usage
- 💡 See `experiments/demos/` for examples
- 🐛 Report issues: https://github.com/Filienko/aml-batch-correction/issues
