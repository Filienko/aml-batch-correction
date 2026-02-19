# SCCL: Single Cell Classification Library

**A unified tool for single-cell RNA-seq classification with multiple models.** Easily classify cells, handle batch effects, and compare different methods through a simple interface.

---

## Active Experiment: Comprehensive Cell Type Annotation Benchmark

The primary experiment currently running is:

```
experiments/paper/exp_ensemble_embeddings.py
```

This script benchmarks four cell type annotation methods head-to-head on the same reference/query dataset pairs. Results are saved to `experiments/paper/results/`.

### What It Does

#### 1. Methods Compared

| Method | Type | Training Step |
|---|---|---|
| **CellTypist** | Reference-based logistic regression | Fit on reference cells |
| **SCimilarity + Classifiers** | Embedding-based (KNN, LogReg, RF, MLP, Ensemble) | Embed reference → fit classifier |
| **SingleR** | Correlation-based label transfer | None (correlation at inference time) |
| **scTab** | Zero-shot foundation model | None (pre-trained checkpoint) |

Each method is trained (where applicable) on a **reference** dataset and evaluated on a held-out **query** dataset. No method sees query labels during training.

#### 2. Dataset Configuration

The benchmark currently runs on the **Zheng 68k PBMC** dataset, split into pre-defined train/test files:

```
experiments/paper/benchmark_data/zheng_train.h5ad   ← reference (train)
experiments/paper/benchmark_data/zheng_test.h5ad    ← query (test)
```

The `SCENARIOS` list in the script controls which dataset pairs are run. To switch to AML Atlas cross-study scenarios, replace `SCENARIOS = ZHENG_SCENARIOS` with AML-style entries using `reference`/`query` study name keys.

#### 3. Repeated Runs for Statistical Robustness

Each scenario is repeated `N_RUNS = 5` times with different random seeds (for subsampling and classifier initialization), producing distributions rather than point estimates. This enables box-whisker plots with meaningful variance.

#### 4. Timing Measurement

Each method's timing is measured and separated into two phases:
- **Training time**: reference preprocessing + embedding (SCimilarity) or model fitting
- **Inference time**: query preprocessing + embedding (SCimilarity) or prediction + neighbor refinement

Timing for each SCimilarity variant (e.g., `SCimilarity-mlp`) reflects the full cost of deploying that specific variant from scratch, including the shared embedding step.

#### 5. Outputs

All figures and CSVs are saved to `experiments/paper/results/`:

| Output file | Description |
|---|---|
| `figures/umap_SCimilarity_mlp_*.png` | UMAP: ground truth vs MLP predictions (run 1 only) |
| `figures/methods_f1_macro_*.png` | Box-whisker: F1 per method, per scenario |
| `figures/methods_time_sec_*.png` | Stacked bar: training vs inference time, per scenario |
| `figures/accumulative_f1_all_datasets.png` | Box-whisker: F1 aggregated across all scenarios |
| `figures/accumulative_time_all_datasets.png` | Stacked bar: average runtime across all scenarios |
| `figures/percelltype_SCimilarity_mlp_*.png` | Box-whisker: per-cell-type F1 for SCimilarity-MLP |
| `comprehensive_benchmark_results.csv` | Full per-run results table |
| `percelltype_f1_results.csv` | Per-cell-type F1 per run for SCimilarity-MLP |
| `benchmark_summary.csv` | Mean ± std aggregated by scenario and method |

#### 6. Configuration

Key settings at the top of `exp_ensemble_embeddings.py`:

```python
MODEL_PATH = "..."          # Path to SCimilarity model weights
BENCHMARK_DATA_DIR = ...    # Directory containing zheng_train.h5ad / zheng_test.h5ad
SCTAB_CHECKPOINT = ...      # Path to scTab .ckpt file
MERLIN_DIR = ...            # Path to scTab Merlin gene/cell-type metadata directory

MAX_CELLS_PER_STUDY = 15000 # Subsample cap per dataset (None = use all)
N_RUNS = 5                  # Repeated runs for box-whisker variance

RUN_CELLTYPIST = True       # Toggle individual methods on/off
RUN_SCIMILARITY = True
RUN_SINGLER = True
RUN_SCTAB = True
```

#### 7. Running the Experiment

```bash
cd /path/to/aml-batch-correction
python experiments/paper/exp_ensemble_embeddings.py
```

---

## Library Overview

SCCL provides a unified pipeline for:
- **Cell type classification** using foundation models (SCimilarity, scVI) or traditional ML (Random Forest, SVM, KNN, Logistic Regression)
- **Batch correction** for multi-study integration
- **Label transfer** across datasets
- **Model comparison** to find the best approach for your data

---

## Installation

```bash
git clone https://github.com/Filienko/aml-batch-correction.git
cd aml-batch-correction
pip install -e .

# Optional: Install foundation models
pip install scimilarity scvi-tools
```

---

## Quick Start

```bash
# Run the active benchmark experiment
python experiments/paper/exp_ensemble_embeddings.py

# Run interactive demos
python experiments/demos/quickstart_annotation.py
python experiments/demos/quickstart_embeddings.py
python experiments/demos/01_basic_prediction.py
```

---

## Usage

### Command Line

```bash
sccl predict --data data.h5ad --model random_forest --target cell_type
sccl evaluate --data data.h5ad --model scimilarity --target cell_type --test-size 0.2
sccl compare --data data.h5ad --models random_forest,svm,scimilarity --target cell_type
```

### Python API

```python
from sccl import Pipeline
import scanpy as sc

adata = sc.read_h5ad("your_data.h5ad")

pipeline = Pipeline(model="scimilarity")
predictions = pipeline.predict(adata)

metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)
print(f"Accuracy: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")
```

### Label Transfer (Cross-Study)

```python
from sccl.data import subset_data

adata_ref = subset_data(adata, studies=['study1'])
adata_query = subset_data(adata, studies=['study2'])

pipeline = Pipeline(model="random_forest")
pipeline.model.fit(adata_ref, target_column='cell_type')
predictions = pipeline.model.predict(adata_query)
```

---

## Available Models

| Model | Type | Training | Batch Correction |
|-------|------|----------|-----------------|
| `scimilarity` | Foundation | None | Yes |
| `scvi` | Deep Learning | Required | Yes |
| `random_forest` | Traditional ML | Required | No |
| `svm` | Traditional ML | Required | No |
| `logistic_regression` | Traditional ML | Required | No |
| `knn` | Instance-based | None | No |

---

## Project Structure

```
aml-batch-correction/
├── sccl/                          # Main package
│   ├── pipeline.py
│   ├── models/
│   ├── data/
│   ├── evaluation/
│   └── cli/
├── experiments/
│   ├── paper/
│   │   ├── exp_ensemble_embeddings.py   ← Active benchmark experiment
│   │   ├── benchmark_data/              ← zheng_train.h5ad, zheng_test.h5ad
│   │   └── results/                     ← Output CSVs and figures
│   └── demos/
│       ├── 01_basic_prediction.py
│       ├── 02_model_comparison.py
│       └── generate_synthetic_data.py
├── setup.py
└── README.md
```

---

## Data Format

SCCL works with **AnnData** (`.h5ad`) files:

**Required:**
- Expression matrix in `.X`
- Cell metadata in `.obs` with target column (e.g., `cell_type`)

**Optional:**
- Batch/study column in `.obs` (e.g., `batch`, `study`)
- Raw counts in `.raw`

---

## Requirements

- Python >= 3.8
- numpy, pandas, scipy, scikit-learn
- anndata, scanpy
- matplotlib, seaborn

**Optional (for specific methods):**
- `scimilarity` — SCimilarity foundation model
- `scvi-tools` — scVI model
- `singler` — SingleR annotation
- `torch`, `cellnet` — scTab zero-shot model

---

## Acknowledgments

- [SCimilarity](https://github.com/Genentech/scimilarity)
- [scVI](https://scvi-tools.org/)
- [scanpy](https://scanpy.readthedocs.io/)

## License

MIT License — See LICENSE file for details.
