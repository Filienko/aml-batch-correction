# SCCL Quick Start Guide

Get started with SCCL in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction

# Install
pip install -e .

# Verify installation
sccl --help
```

## Generate Test Data

```bash
# Generate synthetic data for testing
python examples/generate_synthetic_data.py
```

This creates `data/synthetic_example.h5ad` with:
- 2,000 cells
- 6 cell types
- 3 batches

## Run Your First Prediction

### Option 1: Command Line (Easiest)

```bash
# Predict with Random Forest
sccl evaluate \
  --data data/synthetic_example.h5ad \
  --model random_forest \
  --target cell_type

# Compare multiple models
sccl compare \
  --data data/synthetic_example.h5ad \
  --models random_forest,svm,knn \
  --target cell_type \
  --output comparison.csv
```

### Option 2: Python API

```python
from sccl import Pipeline
import scanpy as sc

# Load data
adata = sc.read_h5ad("data/synthetic_example.h5ad")

# Create pipeline
pipeline = Pipeline(model="random_forest")

# Evaluate
metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ARI: {metrics['ari']:.3f}")
```

### Option 3: Configuration File

Create `config.yaml`:

```yaml
data:
  path: "data/synthetic_example.h5ad"
  target_column: "cell_type"

models:
  - random_forest
  - svm
  - knn

evaluation:
  test_size: 0.2
```

Run:

```bash
sccl run --config config.yaml --output results.csv
```

## Using Your Own Data

### Required Format

Your data should be in `.h5ad` (AnnData) format with:

- **Required**: Expression matrix in `.X`
- **Required**: Cell metadata in `.obs` with at least:
  - A column for cell type/label (e.g., `cell_type`, `annotation`)
- **Optional**: Batch/study column (e.g., `batch`, `study`)

### Example with Your Data

```bash
# Evaluate on your data
sccl evaluate \
  --data /path/to/your_data.h5ad \
  --model random_forest \
  --target cell_type \
  --batch-key study \
  --test-size 0.2
```

## Next Steps

1. **Try different models**: `scimilarity`, `scvi`, `random_forest`, `svm`, `knn`, `logistic_regression`

2. **Run examples**:
   ```bash
   python examples/01_basic_prediction.py
   python examples/02_model_comparison.py
   python examples/03_batch_correction.py
   python examples/04_subset_analysis.py
   ```

3. **Read the guides**:
   - [User Guide](USER_GUIDE.md) - Detailed usage
   - [Model Guide](MODELS.md) - Model selection
   - [API Reference](API_REFERENCE.md) - Full API

## Common Use Cases

### 1. Cell Type Prediction

```bash
sccl predict \
  --data data.h5ad \
  --model scimilarity \
  --target cell_type \
  --output predictions.csv
```

### 2. Model Comparison

```bash
sccl compare \
  --data data.h5ad \
  --models random_forest,svm,scimilarity \
  --target cell_type
```

### 3. Batch Correction

```python
from sccl import Pipeline

pipeline = Pipeline(model="scimilarity", batch_key="study")
predictions = pipeline.predict(adata, target_column="cell_type")
```

### 4. Subset Analysis

```python
from sccl.data import subset_data

# Keep only specific studies
adata_subset = subset_data(
    adata,
    studies=['study1', 'study2']
)

# Analyze subset
pipeline.evaluate(adata_subset, target_column="cell_type")
```

## Troubleshooting

### "SCimilarity not installed"

```bash
pip install scimilarity
```

### "scVI not installed"

```bash
pip install scvi-tools
```

### "File not found"

Make sure you're using absolute paths or running from the correct directory.

### Low performance?

- Try different models
- Check for batch effects (use batch_key parameter)
- Ensure you have enough training data
- Check data quality (missing values, low counts)

## Getting Help

- Check the [User Guide](USER_GUIDE.md)
- Look at [examples/](../examples/)
- Report issues: https://github.com/Filienko/aml-batch-correction/issues

---

**Congratulations!** You're ready to use SCCL for single cell classification.
