# Van Galen Subtype Transfer Validation - Implementation Summary

## What We're Validating

**Research Question**: Can batch correction methods (especially SCimilarity) successfully transfer van Galen's AML cell type classifications to independent studies that used the same annotation framework?

## The Setup

### Papers Using Van Galen's AML Framework

All studies in this atlas use **the same cell type annotation scheme** based on van Galen et al. 2019:

1. **van Galen et al. 2019** (Cell) - Reference study (23,344 cells)
   - Technology: Seq-Well (microwell-based)
   - 40 AML patients
   - Established 6 malignant AML subtypes

2. **Setty et al. 2019** (30,558 cells)
   - Technology: 10x Chromium (droplet-based)
   - Uses same annotation framework

3. **Pei et al. 2020** (1,735 cells)
   - Technology: CITEseq (10x + protein)
   - Uses same annotation framework

4. **Velten et al. 2021** (4,191 cells)
   - Technology: Muta-Seq (mutation tracking)
   - Uses same annotation framework

5. **Oetjen et al. 2018** (79,822 cells)
   - Technology: 10x Genomics
   - Uses same annotation framework

### The 6 Malignant AML Subtypes (Van Galen Framework)

All studies share these cell type labels:

| Label | Van Galen Subtype | Biological Meaning | Key Markers |
|-------|-------------------|-------------------|-------------|
| **HSPC** | HSC-like | Stem-like, quiescent, poor prognosis | AVP, CD34, HOPX, SPINK2 |
| **CMP** | Progenitor-like | Common Myeloid Progenitor, early stage | CD34, MPO, CEBPA |
| **GMP** | GMP-like | Granulocyte-Monocyte Progenitor, better prognosis | MPO, ELANE, AZU1, CTSG |
| **ProMono** | Promonocyte-like | Transitioning to monocyte | CEBPB, CEBPD, CD14 |
| **CD14+ Mono** | Monocyte-like | Mature monocytic phenotype | CD14, LYZ, S100A8, S100A9 |
| **cDC** | Dendritic-like | Conventional dendritic cells | IRF8, IRF4, CD1C |

### Cell Type Distribution Across Studies

```
van_galen_2019 (Seq-Well):
  HSPC: 10,226 (43.8%) - Large stem/progenitor population
  CD14+ Mono: 4,387 (18.8%)
  ProMono: 2,266 (9.7%)
  cDC: 1,024 (4.4%)
  CMP: 846 (3.6%)
  GMP: 195 (0.8%)

setty_2019 (10x Chromium):
  HSPC: 11,206 (36.7%)
  ProMono: 2,513 (8.2%)
  CMP: 2,210 (7.2%)
  CD14+ Mono: 60 (0.2%) - Much fewer monocytes
  cDC: 40 (0.1%)
  GMP: 8 (0.0%) - Very rare

pei_2020 (CITEseq):
  HSPC: 958 (55.2%)
  CD14+ Mono: 614 (35.4%)
  CMP: 10 (0.6%)
  ProMono: 10 (0.6%)
  cDC: 1 (0.1%)
  GMP: 0 (0.0%) - None!

velten_2021 (Muta-Seq):
  CD14+ Mono: 1,172 (28.0%)
  HSPC: 1,127 (26.9%)
  ProMono: 325 (7.8%)
  CMP: 134 (3.2%)
  cDC: 61 (1.5%)
  GMP: 9 (0.2%)

oetjen_2018 (10x):
  HSPC: 1,581 (2.0%) - Much lower proportion
  ProMono: 1,419 (1.8%)
  CD14+ Mono: 6,701 (8.4%)
  CMP: 284 (0.4%)
  cDC: 737 (0.9%)
  GMP: 0 (0.0%)
```

**Key Observations**:
- All studies have HSPC, CD14+ Mono, ProMono (common subtypes)
- GMP is very rare or absent in some studies (challenging to transfer)
- Cell type proportions differ dramatically (biological or technical variation?)
- Technologies differ: Seq-Well vs 10x vs CITEseq vs Muta-Seq

## The Validation Approach

### Task: Label Transfer via k-NN

For each batch correction method:

1. **Train**: Use van Galen as reference
   - Embedding: Batch-corrected embedding (X_scimilarity, X_scVI, etc.)
   - Labels: 6 malignant AML subtypes
   - Method: k-NN classifier (k=15)

2. **Predict**: Transfer labels to test studies
   - setty_2019, pei_2020, velten_2021, oetjen_2018
   - Compute prediction confidence

3. **Validate**: Compare predictions to ground truth
   - Accuracy: % correct predictions
   - Confidence: Average k-NN confidence
   - Per-subtype performance

4. **Marker validation**: Check biological consistency
   - Do predicted HSPC cells express HSPC markers (AVP, CD34)?
   - Do predicted Monocyte cells express monocyte markers (CD14, LYZ)?
   - Marker enrichment scores

### What Good Results Look Like

**SCimilarity should excel at**:
- ✅ High accuracy (>0.75) transferring common subtypes (HSPC, CD14+ Mono)
- ✅ High marker enrichment (>0.80) - predicted cells express correct markers
- ✅ Preserving rare subtypes (cDC) even if less frequent
- ✅ Consistent performance across technologies (Seq-Well → 10x → CITEseq)

**Comparison with other methods**:
- **scVI**: May show high accuracy due to aggressive batch mixing, but potentially lower marker enrichment (over-correction)
- **Harmony**: Balanced performance, good for technical batch correction
- **Uncorrected**: Low accuracy due to batch effects preventing transfer

### Expected Challenges

1. **Technology differences**: Seq-Well (van Galen) → 10x/CITEseq/Muta-Seq (test studies)
   - Different gene detection rates
   - Different library preparation biases
   - This is a HARD task (cross-mechanism transfer)

2. **Rare subtypes**: GMP, cDC are rare in some studies
   - May have low accuracy due to small sample size
   - Important test: Can methods preserve these rare populations?

3. **Biological variation**: AML is heterogeneous
   - Different patients may have different subtype distributions
   - Not all "wrong" predictions are method failures (real biology)

## Running the Validation

```bash
# The validation script is ready to run!
python validate_aml_subtypes.py
```

**What it does**:
1. Loads data and filters to 5 studies (van Galen + 4 test studies)
2. Harmonizes labels (keeps 6 malignant subtypes)
3. Preprocesses data (log-normalization, HVG selection, PCA)
4. Computes batch-corrected embeddings:
   - Uncorrected (PCA baseline)
   - scVI (if available)
   - SCimilarity
   - Harmony
5. For each method:
   - Trains k-NN on van Galen embeddings
   - Predicts subtypes in test studies
   - Validates with ground truth
   - Checks marker gene enrichment
6. Saves results to `validation_aml_subtypes/`

**Output files**:
- `uncorrected_subtype_prediction.csv` - Baseline results
- `scvi_subtype_prediction.csv` - scVI results
- `scimilarity_subtype_prediction.csv` ⭐ - SCimilarity results
- `harmony_subtype_prediction.csv` - Harmony results
- `marker_enrichment.csv` - Marker gene validation
- `validation_summary.csv` - Final comparison

## Interpreting Results

### Example Good Result (SCimilarity)

```
Study: setty_2019
Method: SCimilarity
Accuracy: 0.82
Mean Confidence: 0.78

Per-Subtype:
  HSPC: 0.89 accuracy, 0.85 confidence (11,206 cells)
  CD14+ Mono: 0.91 accuracy, 0.89 confidence (60 cells)
  ProMono: 0.75 accuracy, 0.72 confidence (2,513 cells)

Marker Enrichment:
  HSPC → HSPC markers: 0.88 (strong enrichment)
  CD14+ Mono → Monocyte markers: 0.92 (strong enrichment)
```

**Interpretation**: SCimilarity successfully transferred van Galen's subtypes with high accuracy and strong marker gene support. Preserves biological identity across technologies.

### Example Poor Result (Uncorrected)

```
Study: setty_2019
Method: Uncorrected
Accuracy: 0.48
Mean Confidence: 0.55

Marker Enrichment:
  HSPC → HSPC markers: 0.45 (weak enrichment)
```

**Interpretation**: Batch effects dominate, preventing accurate label transfer. Low marker enrichment suggests predictions are not biologically meaningful.

## Why This Validation Matters

1. **Real biological task**: Not abstract metrics, but actual scientific question - can we classify AML subtypes?

2. **Published framework**: Van Galen's framework is widely used (Zhai 2022, Pei 2020, Velten 2021)

3. **Cross-technology**: Tests robustness across different scRNA-seq platforms

4. **Marker validation**: Ensures predictions are biologically meaningful, not just statistically consistent

5. **Rare subtype preservation**: Tests if methods preserve small but important populations (GMP, cDC)

This directly addresses: **"Can SCimilarity approximate findings from papers that use van Galen's subtypes?"**

The answer will show whether SCimilarity's design (preserving biology over aggressive mixing) helps with real biological discovery tasks.
