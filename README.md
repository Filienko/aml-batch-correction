# SCCL: Single Cell Classification Library

**A unified tool for single-cell RNA-seq cell type annotation with multiple models.**
Easily classify cells, handle batch effects, and compare methods through a simple interface.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Python API](#python-api)
4. [Available Models](#available-models)
5. [Demos](#demos)
6. [Data Format](#data-format)
7. [Project Structure](#project-structure)
8. [Benchmark Experiment](#benchmark-experiment)

---

## Installation

```bash
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction
pip install -e .

# Optional: install foundation model backends
pip install scimilarity          # SCimilarity
pip install scvi-tools           # scVI
pip install celltypist           # CellTypist
```

---

## Quick Start

Run any demo to try the library immediately — no real data required, synthetic data is generated automatically:

```bash
# Compare sklearn models on synthetic data (no extra installs needed)
python experiments/demos/02_model_comparison.py

# Full annotation workflow: reference → query label transfer
python experiments/demos/quickstart_annotation.py

# All numbered demos in sequence
python experiments/demos/01_basic_prediction.py
python experiments/demos/02_model_comparison.py
python experiments/demos/03_batch_correction.py
python experiments/demos/04_subset_analysis.py
```

---

## Python API

### Label Transfer (Reference → Query)

The core workflow: train on a labeled reference dataset, predict on an unlabeled query dataset.

```python
import scanpy as sc
from sccl import Pipeline
from sccl.data import preprocess_data
from sccl.evaluation import compute_metrics

# Load your data
adata_ref = sc.read_h5ad("reference.h5ad")
adata_query = sc.read_h5ad("query.h5ad")

# Preprocess (normalise, log-transform, HVG, PCA)
adata_ref_prep = preprocess_data(adata_ref.copy())
adata_query_prep = preprocess_data(adata_query.copy())

# Train on reference, predict on query
pipeline = Pipeline(model="random_forest")
pipeline.model.fit(adata_ref_prep, target_column="cell_type")
predictions = pipeline.model.predict(adata_query_prep)

# Evaluate
metrics = compute_metrics(y_true=adata_query.obs["cell_type"].values, y_pred=predictions)
print(metrics)
```

### Model Comparison

```python
from sccl import Pipeline
from sccl.data import generate_synthetic_data

adata = generate_synthetic_data(n_cells=2000, n_genes=1000)

pipeline = Pipeline(model="random_forest")
comparison = pipeline.compare_models(
    adata=adata,
    target_column="cell_type",
    models=["random_forest", "svm", "logistic_regression", "knn"],
    test_size=0.2,
)
print(comparison)
```

### SCimilarity (Foundation Model)

SCimilarity works in two stages: project cells into a shared embedding space, then train a classifier on those embeddings.

```python
from sccl import Pipeline
from sccl.data import preprocess_data

pipeline = Pipeline(
    model="scimilarity",
    model_params={
        "model_path": "/path/to/model_v1.1",
        "classifier": "mlp",          # knn | logistic_regression | random_forest | mlp
        "label_propagation": True,    # k-NN majority-vote refinement
    },
)

adata_ref_prep = preprocess_data(adata_ref.copy())
adata_query_prep = preprocess_data(adata_query.copy())

pipeline.model.fit(adata_ref_prep, target_column="cell_type")
predictions = pipeline.model.predict(adata_query_prep)
```

You can also access the raw embeddings directly:

```python
emb_ref = pipeline.model.get_embedding(adata_ref_prep)    # (n_cells, latent_dim)
emb_query = pipeline.model.get_embedding(adata_query_prep)
```

### Subsetting Data

```python
from sccl.data import subset_data

# Split by study
adata_ref = subset_data(adata, studies=["study_A", "study_B"])
adata_query = subset_data(adata, studies=["study_C"])

# Filter to specific cell types
adata_immune = subset_data(adata, cell_types=["T cell", "B cell", "NK cell"])
```

---

## Available Models

| Model key | Type | Needs training | Batch correction |
|-----------|------|:--------------:|:----------------:|
| `random_forest` | Traditional ML | Yes | No |
| `svm` | Traditional ML | Yes | No |
| `logistic_regression` | Traditional ML | Yes | No |
| `knn` | Instance-based | No | No |
| `scimilarity` | Foundation model | No (embedding) | Yes |
| `scvi` | Deep learning | Yes | Yes |
| `celltypist` | Pre-trained LR | Optional | No |

---

## Demos

All demos live in `experiments/demos/` and use synthetic data, so they run without any real datasets.

| Script | What it shows |
|--------|---------------|
| `01_basic_prediction.py` | SCimilarity embedding + KNN label transfer (falls back to Random Forest) |
| `02_model_comparison.py` | Compare RF, SVM, LR, KNN side-by-side |
| `03_batch_correction.py` | UMAP before/after SCimilarity batch correction |
| `04_subset_analysis.py` | Subsetting by batch/cell-type, cross-study eval |
| `quickstart_annotation.py` | Full multi-model annotation workflow |
| `quickstart_embeddings.py` | Using embeddings for clustering and classification |
| `demo_celltypist.py` | CellTypist pre-trained and custom model usage |
| `demo_run.ipynb` | Interactive Jupyter notebook walkthrough |

Run all demos automatically:

```bash
bash experiments/demos/test_all_demos.sh
```

---

## Data Format

SCCL works with **AnnData** (`.h5ad`) files:

**Required**
- Expression matrix in `.X`
- Cell metadata in `.obs` with a target column (e.g., `cell_type`)

**Optional**
- Batch/study column in `.obs` (e.g., `batch`, `study`)
- Raw counts in `.raw` (used by SCimilarity; `preprocess_data` stores them there automatically)

Column names are detected automatically from common variants (`cell_type`, `Cell Type`, `label`, etc.).

---

## Project Structure

```
aml-batch-correction/
├── sccl/                          # Main package
│   ├── pipeline.py                # Pipeline class (entry point)
│   ├── models/                    # Model implementations
│   │   ├── scimilarity.py
│   │   ├── sklearn.py             # RF, SVM, LR, KNN
│   │   ├── scvi.py
│   │   └── celltypist.py
│   ├── data/                      # Loading, preprocessing, synthetic generation
│   └── evaluation/                # Metrics and visualisation
├── experiments/
│   ├── paper/
│   │   ├── exp_ensemble_embeddings.py   <- Active benchmark (generates paper figures)
│   │   ├── benchmark_data/              <- zheng_train.h5ad, zheng_test.h5ad
│   │   └── results/                     <- Output CSVs and figures
│   ├── demos/                     # Quick-start examples (see table above)
│   ├── scvi_loader.py             # Utility: load pre-computed scVI embeddings
│   └── inspect_data.py            # Utility: data inspection helper
├── setup.py
├── requirements.txt
└── README.md
```

---

## Benchmark Experiment

The experiment used to generate all figures in the paper is:

```bash
python experiments/paper/exp_ensemble_embeddings.py
```

It benchmarks four methods (CellTypist, SCimilarity+classifiers, SingleR, scTab) on the AML scAtlas and Zheng 68k PBMC datasets, with five repeated runs per scenario for statistical robustness.

### Configuration (top of `exp_ensemble_embeddings.py`)

```python
MODEL_PATH         = "..."   # SCimilarity model weights
BENCHMARK_DATA_DIR = ...     # Directory with zheng_train.h5ad / zheng_test.h5ad
SCTAB_CHECKPOINT   = ...     # scTab .ckpt file
MERLIN_DIR         = ...     # scTab Merlin metadata directory

MAX_CELLS_PER_STUDY = 15000   # Subsample cap (None = use all)
N_RUNS = 5                    # Repeated runs for box-whisker variance

RUN_CELLTYPIST  = True
RUN_SCIMILARITY = True
RUN_SINGLER     = True
RUN_SCTAB       = True
```

### Methods Compared

| Method | Type | Training step |
|--------|------|---------------|
| CellTypist | Reference-based logistic regression | Fit on reference |
| SCimilarity + classifiers | Embedding-based (KNN, LogReg, RF, MLP, Ensemble) | Embed reference -> fit classifier |
| SingleR | Correlation-based label transfer | None |
| scTab | Zero-shot foundation model | None (pre-trained checkpoint) |

### Outputs (`experiments/paper/results/`)

| File | Description |
|------|-------------|
| `comprehensive_benchmark_results.csv` | Full per-run results |
| `benchmark_summary.csv` | Mean +/- std by scenario and method |
| `percelltype_f1_results.csv` | Per-cell-type F1 for SCimilarity-MLP |
| `figures/methods_f1_macro_*.png` | Box-whisker: F1 per method |
| `figures/methods_time_sec_*.png` | Stacked bar: training vs inference time |
| `figures/accumulative_f1_all_datasets.png` | F1 aggregated across all scenarios |

---

## Requirements

- Python >= 3.8
- numpy, pandas, scipy, scikit-learn, anndata, scanpy, matplotlib, seaborn

Optional (for specific models):
- `scimilarity` — SCimilarity foundation model
- `scvi-tools` — scVI deep learning model
- `celltypist` — CellTypist pre-trained models
- `singler` — SingleR label transfer
- `torch` — scTab zero-shot model

---

## Acknowledgments

- [SCimilarity](https://github.com/Genentech/scimilarity)
- [scVI-tools](https://scvi-tools.org/)
- [CellTypist](https://www.celltypist.org/)
- [scanpy](https://scanpy.readthedocs.io/)

## License

MIT License — see [LICENSE](LICENSE) for details.
