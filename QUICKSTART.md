# SCCL Quick Start Guide

**SCCL (Single Cell Classification Library)** - Easy-to-use tool for single-cell RNA-seq cell type classification using foundation models (SCimilarity) and traditional ML.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction

# Install in development mode
pip install -e .

# Install optional dependencies
pip install scimilarity  # For foundation model support
pip install scvi-tools   # For scVI batch correction
```

## 🚀 Quick Start: 3 Usage Patterns

### 1️⃣ Simple Prediction (High-Level API)

For quick cell type prediction on your data:

```python
from sccl import Pipeline
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Create pipeline and predict
pipeline = Pipeline(model="random_forest")
metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ARI: {metrics['ari']:.3f}")
```

**Run:** `python experiments/demos/01_basic_prediction.py`

---

### 2️⃣ Label Transfer (Cross-Study)

Train on one study, predict on another:

```python
from sccl import Pipeline
from sccl.data import subset_data, preprocess_data
from sccl.evaluation import compute_metrics
import scanpy as sc

# Load data with multiple studies
adata = sc.read_h5ad("atlas.h5ad")

# Split into reference and query
adata_ref = subset_data(adata, studies=['study_1'])
adata_query = subset_data(adata, studies=['study_2'])

# Option A: SCimilarity (foundation model)
pipeline = Pipeline(
    model="scimilarity",
    model_params={'model_path': 'path/to/model', 'classifier': 'knn'}
)

# Option B: Traditional ML
# pipeline = Pipeline(model="random_forest")

# Train on reference
adata_ref_prep = preprocess_data(adata_ref)
pipeline.model.fit(adata_ref_prep, target_column='cell_type')

# Predict on query
adata_query_prep = preprocess_data(adata_query)
predictions = pipeline.model.predict(adata_query_prep)

# Evaluate
metrics = compute_metrics(
    y_true=adata_query.obs['cell_type'].values,
    y_pred=predictions,
    metrics=['accuracy', 'ari', 'nmi']
)

print(f"Label Transfer ARI: {metrics['ari']:.3f}")
```

**Run:** `python experiments/paper/exp2_label_transfer.py`

---

### 3️⃣ Advanced: Direct Embedding Access

For custom downstream analysis on SCimilarity embeddings:

```python
from sccl.models import SCimilarityModel
import scanpy as sc
import numpy as np

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize SCimilarity
model = SCimilarityModel(
    model_path='path/to/model',
    species='human',
    classifier='random_forest'  # Classifier for label transfer
)

# Get embeddings (batch-corrected shared space)
embeddings = model.get_embedding(adata)
print(f"Embeddings shape: {embeddings.shape}")  # (n_cells, embedding_dim)

# Use embeddings for custom analysis
# Option 1: Add to AnnData for visualization
adata.obsm['X_scimilarity'] = embeddings
sc.pp.neighbors(adata, use_rep='X_scimilarity')
sc.tl.umap(adata)
# sc.pl.umap(adata, color='cell_type')  # Requires display

# Option 2: Train custom classifier on embeddings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    adata.obs['cell_type'].values,
    test_size=0.2,
    random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"Custom classifier accuracy: {accuracy:.3f}")
```

**Run:** `python experiments/demos/quickstart_embeddings.py`

---

## 📊 Available Models

| Model | Type | Use Case | Training Required |
|-------|------|----------|-------------------|
| `scimilarity` | Foundation | Cross-study transfer, batch correction | Optional (pre-trained) |
| `random_forest` | Traditional | High accuracy, interpretable | Yes |
| `svm` | Traditional | Small datasets, kernel methods | Yes |
| `knn` | Traditional | Simple baseline | Yes |
| `logistic_regression` | Traditional | Fast, linear | Yes |
| `scvi` | Deep Learning | Batch correction | Yes |

## 🔧 SCimilarity Classifiers

SCimilarity can use different classifiers on its embeddings:

```python
# KNN (default, fast)
model = SCimilarityModel(classifier='knn', n_neighbors=15)

# Random Forest (often best performance)
model = SCimilarityModel(classifier='random_forest',
                        classifier_params={'n_estimators': 200})

# SVM (good for small datasets)
model = SCimilarityModel(classifier='svm')

# Logistic Regression (fast, interpretable)
model = SCimilarityModel(classifier='logistic_regression')
```

## 📁 Example Workflows

### Workflow 1: Annotate New Dataset

```bash
# Train on reference, predict on query
python experiments/demos/quickstart_annotation.py \
    --reference data/reference.h5ad \
    --query data/new_data.h5ad \
    --model scimilarity \
    --output results/predictions.csv
```

### Workflow 2: Compare Models

```bash
# Compare multiple models
python experiments/demos/02_model_comparison.py
```

### Workflow 3: Benchmark Performance

```bash
# Run all paper experiments
cd experiments/paper
python exp1_annotation_replication.py  # Clustering quality
python exp2_label_transfer.py          # Cross-study transfer
python exp3_computational_efficiency.py # Speed comparison
python exp4_cross_study_generalization.py # Robustness test
```

## 🔍 Column Auto-Detection

SCCL automatically detects common column names:

```python
from sccl.data import get_study_column, get_cell_type_column

# Automatically finds: 'study', 'Study', 'dataset', 'batch', 'sample'
study_col = get_study_column(adata)

# Automatically finds: 'cell_type', 'Cell Type', 'celltype', 'annotation'
cell_type_col = get_cell_type_column(adata)

# Or specify custom columns
study_col = get_study_column(adata, study_col='my_study_column')
```

## 🧪 Testing on Headless VM

All demos work without display (no plotting):

```bash
# Run all demos
for demo in experiments/demos/*.py; do
    echo "Running $demo"
    python "$demo" || echo "Failed: $demo"
done

# Run paper experiments (saves results to CSV)
cd experiments/paper
python exp2_label_transfer.py  # Results in results/exp2_*.csv
python exp3_computational_efficiency.py
python exp4_cross_study_generalization.py
```

## 📖 Next Steps

1. **Start with synthetic data:**
   ```bash
   python experiments/demos/01_basic_prediction.py
   ```

2. **Try label transfer:**
   ```bash
   python experiments/demos/quickstart_annotation.py
   ```

3. **Explore embeddings:**
   ```bash
   python experiments/demos/quickstart_embeddings.py
   ```

4. **Read full documentation:**
   - See `README.md` for detailed API reference
   - Check `experiments/paper/` for research-grade experiments
   - Look at `experiments/demos/` for more examples

## ❓ Common Issues

**Issue:** "ModuleNotFoundError: No module named 'scimilarity'"
```bash
pip install scimilarity
```

**Issue:** "Column 'cell_type' not found"
```python
# Check available columns
print(adata.obs.columns.tolist())

# Use custom column name
from sccl.data import get_cell_type_column
cell_type_col = get_cell_type_column(adata, cell_type_col='your_column')
```

**Issue:** "SCimilarity model not found"
```python
# Download or specify model path
pipeline = Pipeline(
    model="scimilarity",
    model_params={'model_path': '/path/to/model_v1.1'}
)
```

## 📬 Support

- GitHub Issues: https://github.com/Filienko/aml-batch-correction/issues
- Documentation: See `README.md`
- Examples: See `experiments/demos/`
