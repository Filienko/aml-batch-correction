# Batch Correction Experiments: Cross-Mechanism vs Within-Mechanism

This document describes the experimental design for evaluating batch correction methods across different scRNA-seq technologies.

## Overview

We conduct **two experiments** to understand how batch correction methods perform in different scenarios:

### Experiment 1: Cross-Mechanism Batch Correction
**Goal**: Test batch correction across different scRNA-seq technologies.

**Selected Technologies**:

*Non-droplet technologies:*
- **van_galen_2019**: Seq-Well (microwell-based) - ~20k cells
- **zhai_2022**: SORT-Seq (FACS-based)
- **pei_2020**: 10X CITEseq (multimodal) - ~2k cells
- **velten_2021**: Muta-Seq (mutation tracking)

*Droplet-based technologies:*
- **Top 3 largest 10x Genomics studies** (e.g., oetjen_2018, zhang_2023 with ~80k cells each)

**Why this is harder**:
- Different technologies have fundamentally different biases
- Gene detection rates vary significantly across platforms
- Library preparation differences create systematic variations
- Technical variation can be as large as biological variation
- **Study size imbalance**: 2k to 80k cells per study

### Experiment 2: Within-Mechanism Batch Correction
**Goal**: Test batch correction within the same technology (droplet-based only).

**Selected Studies** (all 10x Genomics Chromium or similar):
- naldini: 10x Genomics Chromium
- oetjen_2018: 10x Genomics Single Cell 3′ (~80k cells)
- beneyto-calabuig-2023: 10x Genomics Chromium Single Cell 3′
- jiang_2020: 10x Genomics Chromium Single Cell 3′
- zheng_2017: 10x Genomics GemCode Single-Cell 3′
- setty_2019: 10x Chromium
- petti_2019: 10x Genomics Chromium Single Cell 5′
- mumme_2023: 10x Genomics Chromium (3′ v3 and 5′ v1)
- zhang_2023: 10x Genomics Chromium (~80k cells)

**Why this is easier**:
- Same core technology (droplet encapsulation)
- Similar gene detection characteristics
- Batch effects are primarily experimental/biological, not technological
- More common use case in practice

---

## Technology Details

### Non-Droplet Technologies

#### Seq-Well (van_galen_2019)
- **Mechanism**: Barcoded beads in physical nanowells
- **Throughput**: 10,000s of cells
- **Depth**: ~5,000 UMIs/cell, ~2,000 genes/cell
- **Advantages**: Lower cost than droplet, simpler setup
- **Size**: ~20k cells

#### SORT-Seq (zhai_2022)
- **Mechanism**: FACS-based single-cell RNA-seq
- **Throughput**: Low-medium
- **Advantages**: Can sort based on surface markers
- **Size**: TBD from data

#### CITEseq (pei_2020)
- **Mechanism**: 10x platform + antibody-derived tags for protein measurement
- **Throughput**: High (10x-based)
- **Advantages**: Multimodal (RNA + protein)
- **Size**: ~2k cells (smallest study)

#### Muta-Seq (velten_2021)
- **Mechanism**: Mutation tracking + transcriptomics
- **Throughput**: Medium
- **Advantages**: Links mutations to transcriptional states
- **Size**: TBD from data

### Droplet-based (10x Genomics)
- **Mechanism**: Microfluidic droplet encapsulation
- **Throughput**: 10,000s-100,000s of cells
- **Depth**: Medium (~10,000 UMIs/cell, ~2,000 genes/cell)
- **Advantages**: High throughput, standard protocol, widely used
- **Studies**: 9 studies included (naldini + 8 others)
- **Size range**: Variable (some up to ~80k cells per study)

---

## File Structure

```
aml-batch-correction/
├── analyze_studies.py              # Analyze dataset and determine study sizes
├── experiment_cross_mechanism.py   # Experiment 1: Cross-mechanism
├── experiment_within_mechanism.py  # Experiment 2: Within-mechanism
├── compare_experiments.py          # Compare results from both experiments
│
├── results_cross_mechanism/        # Results from Experiment 1
│   ├── cross_mechanism_results.csv
│   ├── uncorrected_metrics.csv
│   ├── scvi_metrics.csv
│   └── scimilarity_metrics.csv
│
├── results_within_mechanism/       # Results from Experiment 2
│   ├── within_mechanism_results.csv
│   ├── uncorrected_metrics.csv
│   ├── scvi_metrics.csv
│   └── scimilarity_metrics.csv
│
└── results_comparison/             # Comparison of both experiments
    ├── experiment_comparison.csv
    ├── total_score_comparison.png
    ├── metrics_heatmap.png
    └── performance_difference.png
```

---

## How to Run

### Step 1: Analyze Studies
First, run the analysis script to determine which studies are available and their sizes:

```bash
python analyze_studies.py
```

**Output**:
- `study_analysis_results.csv`: Study sizes and technology mappings
- Console output showing selected studies for each experiment

### Step 2: Run Experiments

#### Experiment 1: Cross-Mechanism
```bash
python experiment_cross_mechanism.py
```

**What it does**:
1. Subsets data to 3 studies (one per technology)
2. Preprocesses data
3. Computes embeddings:
   - Uncorrected (PCA baseline)
   - scVI (if available)
   - SCimilarity
4. Evaluates batch correction using scIB metrics
5. Saves results to `results_cross_mechanism/`

**Expected runtime**: 30-60 minutes (depends on dataset size)

#### Experiment 2: Within-Mechanism
```bash
python experiment_within_mechanism.py
```

**What it does**:
1. Subsets data to droplet-based studies only (~8 studies)
2. Same pipeline as Experiment 1
3. Saves results to `results_within_mechanism/`

**Expected runtime**: 1-2 hours (more studies = larger dataset)

### Step 3: Compare Results
```bash
python compare_experiments.py
```

**What it does**:
1. Loads results from both experiments
2. Creates comparison tables
3. Generates visualizations:
   - Bar chart comparing total scores
   - Heatmaps for each experiment
   - Difference plot (Within - Cross)
4. Saves to `results_comparison/`

---

## Evaluation Metrics

All experiments use **scIB metrics** (Luecken et al., 2022):

### Batch Correction Metrics
- **Graph connectivity**: Do same cell types connect across batches?
- **kBET**: k-nearest neighbor batch effect test
- Higher = better batch mixing

### Biological Conservation Metrics
- **Cell type ASW**: Average silhouette width for cell types
- **Isolated label scores**: Are cell types preserved?
- Higher = better biology preservation

### Overall Score
- **Total**: Average of batch correction and bio conservation
- **Best scenario**: High batch correction + high bio conservation

---

## Expected Results

### Cross-Mechanism
- **Lower batch correction scores** (harder to mix different technologies)
- **Variable bio conservation** (depends on method)
- Challenge: Distinguishing technical from biological differences

### Within-Mechanism
- **Higher batch correction scores** (easier, same technology)
- **Higher bio conservation** (less technical noise)
- More representative of typical use cases

### Comparison
We expect to see:
```
Within-Mechanism Scores > Cross-Mechanism Scores
```

Methods that maintain high performance in **cross-mechanism** are more robust.

---

## Methods Evaluated

### 1. Uncorrected (PCA)
- **Purpose**: Baseline to show batch effects without correction
- **Expected**: Low batch correction, high bio conservation
- Standard PCA on log-normalized, HVG-selected data

### 2. scVI
- **Type**: Deep learning-based batch correction
- **Expected**: High batch correction, moderate-high bio conservation
- Pre-computed embeddings (30 dimensions)
- Specifically designed for batch correction

### 3. SCimilarity
- **Type**: Cell annotation model (not primarily batch correction)
- **Expected**: Variable batch correction, high bio conservation
- Focus: Preserve cell type identity
- May show lower batch correction than scVI (by design)

---

## Interpreting Results

### Good Batch Correction
```
Method          Total    Batch    Bio
scVI            0.85     0.90     0.80
```
- High batch mixing (0.90)
- Preserved biology (0.80)
- Balanced performance

### Poor Batch Correction
```
Method          Total    Batch    Bio
Uncorrected     0.45     0.20     0.70
```
- Low batch mixing (0.20) - batches stay separated
- Biology preserved (0.70) but not integrated

### Technology-Specific
If cross-mechanism shows much lower scores than within-mechanism for a method:
- Method struggles with technology differences
- May be overfitting to specific technology characteristics

---

## Troubleshooting

### Problem: "No studies configured"
**Solution**: Run `analyze_studies.py` first to generate configuration.

### Problem: "scVI embeddings not available"
**Solution**: This is optional. The analysis will run without scVI, comparing only Uncorrected vs SCimilarity.

### Problem: "Out of memory"
**Solution**: Reduce `SCIMILARITY_BATCH_SIZE` in the experiment script:
```python
SCIMILARITY_BATCH_SIZE = 500  # Reduce from 1000
```

### Problem: Study names don't match
**Solution**: Check the actual study names in your dataset:
```python
import scanpy as sc
adata = sc.read_h5ad("data/AML_scAtlas.h5ad")
print(adata.obs['Study'].unique())
```

Then manually edit `TECHNOLOGY_MAP` in `analyze_studies.py` to match your study names.

---

## References

1. **scIB**: Luecken et al. (2022) "Benchmarking atlas-level data integration in single-cell genomics" *Nature Methods*
2. **scVI**: Lopez et al. (2018) "Deep generative modeling for single-cell transcriptomics" *Nature Methods*
3. **SCimilarity**: Heimberg et al. (2023) "Jointly learning cell types and their regulation with SCimilarity" *bioRxiv*
4. **Seq-Well**: Gierahn et al. (2017) "Seq-Well: portable, low-cost RNA sequencing" *Nature Methods*
5. **SMART-Seq**: Picelli et al. (2014) "Full-length RNA-seq from single cells" *Nature Protocols*

---

## Citation

If you use these experiments in your research, please cite:
- The AML atlas data source
- scIB metrics (Luecken et al., 2022)
- The specific batch correction methods used (scVI, SCimilarity)

---

## Questions?

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are installed (`scanpy`, `scvi-tools`, `scimilarity`, `scib-metrics`)
