# SCCL Demos

Quick-start examples that run entirely on synthetic data — no real datasets or model files required (unless noted).

## Run order for newcomers

```bash
# 1. Simplest possible demo: compare sklearn models
python 02_model_comparison.py

# 2. Full annotation workflow (ref → query label transfer)
python quickstart_annotation.py

# 3. Numbered deep-dives
python 01_basic_prediction.py    # SCimilarity + KNN (falls back to RF)
python 03_batch_correction.py    # UMAP comparison raw vs embedded
python 04_subset_analysis.py     # Subsetting, cross-study eval
```

## Demo index

| Script | Dependencies | What it shows |
|--------|-------------|---------------|
| `02_model_comparison.py` | core only | RF / SVM / LR / KNN comparison |
| `quickstart_annotation.py` | core only | Multi-model label transfer, auto column detection |
| `quickstart_embeddings.py` | core only | Embeddings for clustering and classification |
| `01_basic_prediction.py` | scimilarity (optional) | SCimilarity embedding + KNN; RF fallback |
| `03_batch_correction.py` | scimilarity (optional) | UMAP before/after batch correction |
| `04_subset_analysis.py` | core only | subset_data(), cross-study generalisation |
| `demo_celltypist.py` | celltypist | CellTypist pre-trained and custom models |
| `demo_run.ipynb` | core only | Interactive Jupyter walkthrough |
| `generate_synthetic_data.py` | core only | Save a synthetic .h5ad for other scripts |

## Run all demos

```bash
bash test_all_demos.sh
```

## Using your own data

Replace the `generate_synthetic_data(...)` call with:

```python
import scanpy as sc
adata = sc.read_h5ad("/path/to/your_data.h5ad")
```

Column names (`cell_type`, `batch`, `study`, etc.) are detected automatically.
