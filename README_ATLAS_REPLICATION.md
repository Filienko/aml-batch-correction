# Foundation Models for Automated AML Atlas Annotation

## 🎯 Research Question

**Can SCimilarity, a foundation model, automatically replicate the complex, multi-tool, manually-curated annotation pipeline used in the AML scAtlas — without any manual intervention?**

## 🔬 Background

The AML scAtlas (159 patients) represents **months of expert curation**:
- ✅ scVI batch correction
- ✅ CellTypist + SingleR + scType (3-way consensus)
- ✅ Manual curation with marker genes
- ✅ Custom LSC annotation using Zeng et al. reference
- ✅ LSC6/LSC17 score correlation

**Our Hypothesis**: A pre-trained foundation model (SCimilarity) can achieve **comparable or better results automatically**.

---

## 📊 Project Structure

```
aml-batch-correction/
├── README_ATLAS_REPLICATION.md        ← You are here
├── ATLAS_REPLICATION_PROJECT.md       ← Detailed project plan
├── DATA_SOURCES.md                    ← How to get the data
│
├── phase1_ground_truth.py             ← Load atlas + raw data
├── phase2_scimilarity_projection.py   ← Project to SCimilarity
├── phase3_quantitative_benchmark.py   ← LISI, kBET, ARI, NMI
├── phase4_biological_discovery.py     ← Hierarchy analysis
├── run_full_pipeline.py               ← Run all phases
│
├── data/                              ← Data directory (create this)
│   ├── AML_scAtlas.h5ad              ← Ground truth (download)
│   └── raw/                          ← Raw datasets (optional)
│
└── results_atlas_replication/        ← Generated outputs
    ├── figures/                      ← All publication figures
    ├── metrics/                      ← Quantitative results
    ├── hierarchy/                    ← Hierarchy analysis
    └── data/                         ← Intermediate files
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create conda environment (recommended)
conda create -n aml-atlas python=3.10
conda activate aml-atlas

# Install packages
pip install scanpy scimilarity scvi-tools
pip install scikit-learn pandas numpy matplotlib seaborn
pip install anndata h5py

# Optional (for additional metrics)
pip install harmonypy scib-metrics
```

### 2. Download Data

See `DATA_SOURCES.md` for detailed instructions. You need:
- **AML_scAtlas.h5ad** (159 patients) — Ground truth
- Optionally: Raw datasets (van Galen, Abbas, Wang)

Place in `data/` directory:
```bash
mkdir -p data
# Download AML_scAtlas.h5ad to data/
```

### 3. Run the Complete Pipeline

```bash
# Run all 4 phases
python run_full_pipeline.py

# Or run phases individually
python phase1_ground_truth.py           # ~10-15 min
python phase2_scimilarity_projection.py # ~20-30 min
python phase3_quantitative_benchmark.py # ~15-20 min
python phase4_biological_discovery.py   # ~10 min
```

### 4. View Results

```bash
ls -lh results_atlas_replication/figures/
# fig1a_ground_truth_atlas.pdf
# fig1b_raw_problem.pdf
# fig1c_scimilarity_solution.pdf
# fig2_quantitative_comparison.pdf
# fig3_hierarchy_dendrogram.pdf
# fig3_principal_axes.pdf
```

---

## 📈 The 4 Phases

### Phase 1: Data and Model Setup (Figure 1A & 1B)

**Goal**: Establish ground truth and demonstrate the problem

**What it does**:
1. Loads the fully processed AML scAtlas (159 patients)
2. Visualizes the "Gold Standard" UMAP (Figure 1A)
   - Cells clustered by **biology** (cell types)
   - **Not** by batch (good integration)
3. Creates the "raw problem" by merging unintegrated data
4. Visualizes massive batch effects (Figure 1B)
   - Cells clustered by **batch** (study)
   - Biology obscured

**Output**:
- `fig1a_ground_truth_atlas.pdf` — What good integration looks like
- `fig1b_raw_problem.pdf` — The problem to solve
- `merged_raw_problem.h5ad` — Raw data for Phase 2

---

### Phase 2: SCimilarity Projection (Figure 1C)

**Goal**: Show that a foundation model can automatically solve batch effects

**What it does**:
1. Takes raw, unintegrated data from Phase 1
2. Projects every cell into SCimilarity latent space
   - **No training or fine-tuning**
   - Just inference using pre-trained model
3. Computes UMAP from SCimilarity embeddings
4. Visualizes the "FM Solution" (Figure 1C)

**Key point**: This is **fully automated** — no manual work!

**Output**:
- `fig1c_scimilarity_solution.pdf` — FM-corrected atlas
- `fig_comparison_before_after.pdf` — Side-by-side comparison
- `scimilarity_solution.h5ad` — For Phase 3 analysis

---

### Phase 3: Quantitative Benchmarking (Figure 2)

**Goal**: Quantitatively prove FM ≥ Expert curation

**Metrics**:

**Batch Mixing** (higher = better):
- **LISI** (Local Inverse Simpson's Index)
  - Perfect mixing = # of batches
  - No mixing = 1
- **kBET** (k-Nearest Neighbor Batch-Effect Test)
  - Acceptance rate (higher = better mixing)

**Biology Conservation** (higher = better):
- **ARI** (Adjusted Rand Index)
  - Agreement between clusters and true labels
  - 1.0 = perfect, 0.0 = random
- **NMI** (Normalized Mutual Information)
  - Information preserved
- **ASW** (Average Silhouette Width)
  - Cell type separation

**Hypothesis to test**:
1. **Batch mixing**: FM ≥ Gold Standard
2. **Biology conservation**: FM > Gold Standard (!)
   - If true → suggests expert curation "over-corrected"
   - FM preserves finer biological distinctions

**Output**:
- `quantitative_comparison.csv` — All metrics
- `fig2_quantitative_comparison.pdf` — Bar charts + table
- `fig2_metrics_heatmap.pdf` — Heatmap visualization

---

### Phase 4: Biological Discovery (Figure 3)

**Goal**: Show SCimilarity can approximate the manual hierarchy work

**What the AML scAtlas team did manually**:
1. scVI integration
2. UMAP + Leiden clustering
3. CellTypist + SingleR + scType consensus
4. Manual marker gene curation
5. Custom LSC annotation
6. Identified 12 aberrant differentiation patterns
7. Defined PC1 (Primitive vs GMP) and PC2 (Primitive vs Mature) axes

**What we do**:
1. ✅ Skip all of that
2. ✅ Just use SCimilarity embeddings
3. ✅ Compute cell type centroids
4. ✅ Hierarchical clustering → cell type relationships
5. ✅ PCA on centroids → identify principal axes
6. ✅ Validate marker gene enrichment

**Key findings**:
- Does SCimilarity recover the same PC1/PC2 axes?
- Are cell type hierarchies preserved?
- Do clusters show correct marker enrichment?

**Output**:
- `fig3_hierarchy_dendrogram.pdf` — Cell type relationships
- `fig3_principal_axes.pdf` — PC1/PC2 visualization
- `marker_gene_validation.csv` — Marker enrichment scores

---

## 🎯 Expected Results

### Best Case Scenario

| Metric | Problem | Gold Standard | FM Solution | Interpretation |
|--------|---------|---------------|-------------|----------------|
| Batch Mixing | 0.2 | 0.85 | **≥0.85** | ✓ FM matches expert |
| Bio Conservation | 0.7 | 0.80 | **>0.80** | ✓✓ FM better! |
| PC1 Correlation | — | 1.0 | **>0.7** | ✓ Recovers hierarchy |

**If biology conservation is higher**: This suggests the manual curation process may have "over-corrected" and merged biologically distinct subtypes, while the FM's generalized latent space preserved them.

---

## 📊 Interpretation Guide

### Good Result

```
Method              Batch    Bio     Overall
Problem             0.20     0.70    0.45     ← Baseline (bad)
Gold Standard       0.85     0.80    0.83     ← Expert work
FM Solution         0.87     0.83    0.85     ✓ Better!
```

**Conclusion**: Foundation model **exceeds** expert curation

### Moderate Result

```
Method              Batch    Bio     Overall
Gold Standard       0.85     0.80    0.83
FM Solution         0.82     0.81    0.82     ✓ Comparable
```

**Conclusion**: Foundation model **matches** expert curation (still impressive!)

### What Each Metric Tells You

- **LISI score close to # of batches**: Good batch mixing
- **kBET acceptance rate >0.7**: Batches well integrated
- **ARI/NMI >0.7**: Cell types preserved
- **ASW >0.5**: Cell types well-separated
- **PC1 correlation >0.7**: Hierarchy recovered

---

## 🔧 Troubleshooting

### Problem: `FileNotFoundError: AML_scAtlas.h5ad`

**Solution**: Download the atlas data
```bash
# See DATA_SOURCES.md for instructions
# Place in data/AML_scAtlas.h5ad
```

### Problem: `ImportError: scimilarity`

**Solution**: Install SCimilarity
```bash
pip install scimilarity
```

### Problem: `MemoryError` during SCimilarity projection

**Solution**: Reduce batch size in `phase2_scimilarity_projection.py`
```python
SCIMILARITY_BATCH_SIZE = 2500  # Reduce from 5000
```

### Problem: Missing cell type labels

**Solution**: Check your atlas metadata
```python
import scanpy as sc
adata = sc.read_h5ad('data/AML_scAtlas.h5ad')
print(adata.obs.columns.tolist())  # Find the correct column name
```

### Problem: No raw counts in atlas

**Solution**: You may need to download raw datasets separately
- See `DATA_SOURCES.md` for GEO accessions
- Or Phase 1 will use current data (may be normalized)

---

## 📝 For Your Manuscript

### Key Claims to Make

1. **Batch Correction**:
   > "SCimilarity achieved batch mixing scores of X.XX, comparable to
   > expert-curated scVI integration (X.XX), without any manual work."

2. **Biology Preservation**:
   > "Notably, SCimilarity showed higher biological conservation (ARI=X.XX)
   > than the manually curated atlas (ARI=X.XX), suggesting the foundation
   > model preserves finer biological distinctions that may be lost during
   > expert curation."

3. **Hierarchy**:
   > "SCimilarity-derived cell type hierarchies showed strong correlation
   > (r=X.XX) with established AML differentiation axes, confirming the
   > biological validity of automated embeddings."

4. **Efficiency**:
   > "The entire pipeline required ~1 hour of compute time, compared to
   > weeks of expert manual curation for the original atlas."

### Suggested Figure Titles

- **Figure 1**: Foundation Model Batch Correction Overview
  - (A) Ground truth: Expert-curated AML scAtlas
  - (B) Problem: Raw merged data with batch effects
  - (C) Solution: SCimilarity-corrected atlas

- **Figure 2**: Quantitative Comparison of Batch Correction Methods
  - Bar charts showing batch mixing and biology conservation scores

- **Figure 3**: Biological Hierarchy Recovery
  - (A) Dendrogram of cell type relationships
  - (B) Principal axes of cell type variation

---

## 🔗 References

**To cite this work** (when published):
```
[Your publication citation here]
```

**Dependencies to cite**:
1. **SCimilarity**: Heimberg et al. (2023) bioRxiv
2. **scVI**: Lopez et al. (2018) Nature Methods
3. **AML scAtlas**: [Citation from the atlas paper]
4. **van Galen et al.**: van Galen et al. (2019) Cell

---

## 💡 Tips for Success

1. **Start small**: Test on a subset first
   ```python
   # In phase1_ground_truth.py, add:
   adata = adata[adata.obs['Study'].isin(['van_galen_2019', 'abbas_2021'])].copy()
   ```

2. **Check intermediate outputs**: After each phase, examine the figures

3. **Save checkpoints**: Each phase saves its output for later phases

4. **Parameter tuning**:
   - Leiden resolution for clustering
   - Number of neighbors for UMAP
   - Batch size for SCimilarity

5. **Interpretation**: Always compare against biological expectations

---

## 🤝 Contributing

Found a bug or have suggestions? Open an issue or PR!

---

## 📧 Contact

For questions about this pipeline, please contact [your email].

For SCimilarity-specific issues, see: https://github.com/Genentech/scimilarity

---

## ⚖️ License

[Specify your license]

---

## 🎓 Acknowledgments

- AML scAtlas authors for the ground truth data
- SCimilarity team for the pre-trained model
- All the single-cell community

---

**Last updated**: 2025

**Status**: ✅ Ready to use
