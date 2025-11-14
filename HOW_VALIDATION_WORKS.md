# How The Van Galen Validation Actually Works

## Overview

This document explains the **exact technical implementation** of the validation pipeline, step-by-step.

## The Complete Pipeline

### Step 1: Data Loading & Subsetting

```python
# Load full AML atlas
adata = sc.read_h5ad("data/AML_scAtlas.h5ad")
# Example: 750,000 cells × 20,000 genes

# Filter to only van Galen + test studies
studies = ['van_galen_2019', 'velten_2021']  # Only papers that cite van Galen
adata = adata[adata.obs['Study'].isin(studies)]
# Result: ~27,500 cells × 20,000 genes
```

**What we keep:**
- van_galen_2019: 23,344 cells (Seq-Well technology)
- velten_2021: 4,191 cells (Muta-Seq technology)

### Step 2: Label Harmonization

```python
# Map cell type labels to van Galen's 6 malignant subtypes
mapping = {
    'HSPC': 'HSPC',           # Keep as-is
    'CMP': 'CMP',             # Keep as-is
    'GMP': 'GMP',             # Keep as-is
    'ProMono': 'ProMono',     # Keep as-is
    'CD14+ Mono': 'CD14+ Mono',  # Keep as-is
    'cDC': 'cDC',             # Keep as-is
    # All other cell types (T, B, NK, Erythroid) → unmapped (filtered out)
}

adata.obs['van_galen_subtype'] = adata.obs['Cell Type'].map(mapping)
adata = adata[adata.obs['van_galen_subtype'].notna()]  # Keep only the 6 subtypes
```

**Result:** Only keep cells labeled with the 6 malignant AML subtypes

### Step 3: Preprocessing (Same for All Methods)

```python
# 3.1 Extract raw counts
if 'counts' in adata.layers:
    raw_counts = adata.layers['counts']
else:
    raw_counts = adata.X  # Assume X has counts if no .layers['counts']

# 3.2 Normalize
sc.pp.normalize_total(adata, target_sum=10000)  # CPM normalization
sc.pp.log1p(adata)                               # Log transform

# 3.3 Save normalized for later
adata.layers['normalised_counts'] = adata.X.copy()

# 3.4 Select highly variable genes
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor='seurat_v3',
    layer='counts'  # Select HVGs based on raw counts
)

# 3.5 Subset to HVGs
adata = adata[:, adata.var['highly_variable']]
# Now: ~27,500 cells × 2,000 genes
```

**Critical point:** All methods see the SAME preprocessed data (log-normalized, 2000 HVGs)

### Step 4: Compute Embeddings (Different for Each Method)

#### Method 1: "Uncorrected" (PCA Baseline)

```python
# Scale to unit variance
sc.pp.scale(adata)  # Z-score normalization

# PCA
sc.tl.pca(adata, n_comps=50)

# Save embedding
adata.obsm['X_uncorrected'] = adata.obsm['X_pca']
# Shape: (27,500 cells, 50 dimensions)
```

**What this actually is:**
- PCA on log-normalized, HVG-selected, scaled data
- **NOT truly uncorrected** - PCA itself reduces noise and provides structure
- 50-dimensional orthogonal embedding
- Each PC is z-scored (standardized)

#### Method 2: scVI (Deep Learning VAE)

```python
# Load pre-computed scVI embeddings
adata_scvi = sc.read_h5ad("data/AML_scAtlas_X_scVI.h5ad")

# Map cells to scVI embeddings using original row positions
original_indices = [find_row_in_full_data(cell_id) for cell_id in adata.obs_names]
adata.obsm['X_scVI'] = adata_scvi.X[original_indices]
# Shape: (27,500 cells, 30 dimensions)
```

**What this is:**
- Variational autoencoder (VAE) latent space
- 30-dimensional embedding
- Explicitly trained for batch correction
- **Potential data leakage:** scVI was trained on the full atlas (including test data)

#### Method 3: SCimilarity (Foundation Model)

```python
# 1. Load SCimilarity model
from scimilarity import CellAnnotation
ca = CellAnnotation(model_path="models/model_v1.1")

# 2. Get raw counts for SCimilarity (needs counts, not log-normalized)
if 'counts' in adata.layers:
    raw_for_scim = adata.layers['counts']
else:
    raw_for_scim = adata.raw.X  # Fall back to .raw

# 3. Align genes to SCimilarity's expected gene order
common_genes = set(adata.var_names) & set(ca.gene_order)
# Map our genes to SCimilarity's order

# 4. Normalize using SCimilarity's method
normalized_batch = ca.normalize(raw_counts_batch)  # Per batch of 1000 cells

# 5. Get embeddings from SCimilarity transformer
embeddings_batch = ca.get_embeddings(normalized_batch)

# 6. Concatenate all batches
adata.obsm['X_scimilarity'] = np.vstack(embeddings_batch)
# Shape: (27,500 cells, 256 dimensions)  # SCimilarity uses 256D embeddings
```

**What this is:**
- Transformer-based foundation model embeddings
- 256-dimensional embedding
- Pre-trained on large single-cell atlas
- Designed for cell type annotation (semantic similarity)
- **NOT specifically trained for batch correction**

#### Method 4: Harmony (Iterative Correction)

```python
# Start with PCA
sc.pp.scale(adata)
sc.tl.pca(adata, n_comps=50)

# Apply Harmony correction
import harmonypy
sc.external.pp.harmony_integrate(
    adata,
    key='study',           # Batch key
    basis='X_pca',         # Input: PCA
    adjusted_basis='X_harmony',  # Output: Harmony-corrected PCA
    max_iter_harmony=10
)

adata.obsm['X_harmony'] = adata.obsm['X_harmony']
# Shape: (27,500 cells, 50 dimensions)
```

**What this is:**
- Iteratively corrected PCA
- 50-dimensional embedding (same as PCA input)
- Explicitly designed for batch correction
- Adjusts PCA to remove batch effects while preserving biology

### Step 5: Label Transfer via k-NN Classification

For each embedding method, we train a classifier and predict:

```python
from sklearn.neighbors import KNeighborsClassifier

# 5.1 Get reference data (van Galen study)
ref_mask = adata.obs['Study'] == 'van_galen_2019'
ref_embedding = adata.obsm['X_scimilarity'][ref_mask]  # Example: SCimilarity
ref_labels = adata.obs['van_galen_subtype'][ref_mask]
# ref_embedding: (23,344 cells, 256 dimensions)
# ref_labels: (23,344,) with values like 'HSPC', 'CD14+ Mono', etc.

# 5.2 Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
knn.fit(ref_embedding, ref_labels)

# 5.3 Get test data (velten_2021 study)
test_mask = adata.obs['Study'] == 'velten_2021'
test_embedding = adata.obsm['X_scimilarity'][test_mask]
test_true_labels = adata.obs['van_galen_subtype'][test_mask]
# test_embedding: (4,191 cells, 256 dimensions)

# 5.4 Predict labels
pred_labels = knn.predict(test_embedding)
pred_proba = knn.predict_proba(test_embedding)
confidence = pred_proba.max(axis=1)  # Confidence of top prediction

# 5.5 Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_true_labels, pred_labels)
# Example: 0.82 = 82% of cells correctly classified
```

**What we're testing:**
- Can we transfer van Galen's labels to Velten using the embedding?
- Uses van Galen (Seq-Well) as training set
- Tests on Velten (Muta-Seq) - different technology!
- k=15: Each test cell votes based on 15 nearest neighbors in van Galen

### Step 6: Marker Gene Validation

Check if predicted cell types express correct markers:

```python
# 6.1 For each predicted subtype, get marker expression
for subtype in ['HSPC', 'CMP', 'GMP', 'ProMono', 'CD14+ Mono', 'cDC']:
    # Get cells predicted as this subtype
    pred_mask = pred_labels == subtype

    # Get marker genes for this subtype
    markers = VAN_GALEN_SUBTYPE_MARKERS[subtype]
    # Example for HSPC: ['AVP', 'CD34', 'HOPX', 'SPINK2']

    # Get marker expression in predicted cells
    marker_expr = adata[pred_mask, markers].X.mean(axis=0)

    # Compare to background (all other cells)
    background_expr = adata[~pred_mask, markers].X.mean(axis=0)

    # Fold change
    fold_change = marker_expr / (background_expr + 1e-10)

    # Good prediction: fold_change > 2 (markers enriched in predicted cells)
```

**What this validates:**
- Do predicted HSPC cells actually express HSPC markers (AVP, CD34)?
- Do predicted Monocytes actually express monocyte markers (CD14, LYZ)?
- Ensures predictions are biologically meaningful, not just statistically consistent

### Step 7: Comparison Across Methods

```python
results_summary = {
    'Uncorrected': {
        'accuracy': 0.68,
        'confidence': 0.72,
        'marker_enrichment': 0.65
    },
    'scVI': {
        'accuracy': 0.75,
        'confidence': 0.78,
        'marker_enrichment': 0.71
    },
    'SCimilarity': {
        'accuracy': 0.72,
        'confidence': 0.76,
        'marker_enrichment': 0.85  # ← Higher marker enrichment!
    },
    'Harmony': {
        'accuracy': 0.74,
        'confidence': 0.77,
        'marker_enrichment': 0.73
    }
}
```

**Interpretation:**
- **Accuracy**: Can we correctly classify cells?
- **Confidence**: How confident is the classifier?
- **Marker enrichment**: Do predictions match biological markers?

If SCimilarity has lower accuracy but higher marker enrichment:
- SCimilarity preserves biological signal better
- But may not mix batches as aggressively (different design goal)

## Why "Uncorrected" Can Perform Well

### The Problem

If you see:
```
Uncorrected: 0.68 accuracy
SCimilarity: 0.72 accuracy
```

This seems wrong! But here's why it happens:

### Reason 1: "Uncorrected" is Actually PCA

```python
# What "uncorrected" does:
1. Log-normalize ✓
2. Select top 2000 HVGs ✓
3. Z-score scale ✓
4. PCA (50 dimensions) ✓  ← This provides structure!

# PCA finds:
- PC1: Separates HSPC from Monocytes
- PC2: Separates progenitors from mature cells
- PC3-50: Finer cell type distinctions
```

**PCA already captures major biological variation!**

If cell types are well-separated and batch effects are weak:
- PCA alone might work fine
- Batch correction provides minimal additional benefit

### Reason 2: Different Embedding Spaces

```python
# Embedding properties:
Uncorrected (PCA):   50D, orthogonal, z-scored, designed for variance
scVI:                30D, non-linear VAE latent space
SCimilarity:        256D, transformer embeddings, semantic similarity
Harmony:            50D, corrected PCA
```

**k-NN performance depends on:**
- Dimensionality (curse of dimensionality in high-D)
- Distance metric (Euclidean distance in k-NN)
- Embedding scale (z-scored vs not)

PCA's properties (orthogonal, standardized) may favor k-NN classification.

### Reason 3: Task May Not Require Batch Correction

If cell types are very distinct:
- HSPC has unique markers (CD34, AVP)
- Monocytes have unique markers (CD14, LYZ)
- Technologies don't distort this signal too much

Then:
- Batch effects exist but don't prevent classification
- "Uncorrected" PCA works adequately
- Batch correction helps but isn't critical

**When batch correction matters more:**
- Rare cell types (hard to distinguish from noise)
- Subtle cell states (small transcriptional differences)
- Strong technical biases (gene detection differences)

## What Makes a Good Result?

### Good SCimilarity Result:

```
SCimilarity Performance:
  Accuracy: 0.82
  Confidence: 0.78

  Per-Subtype Marker Enrichment:
    HSPC → HSPC markers: 0.88  (strong)
    Monocyte → Mono markers: 0.91  (strong)
    GMP → GMP markers: 0.85  (strong)
```

**Interpretation:**
- High accuracy: Successfully transfers labels
- High marker enrichment: Predictions are biologically valid
- Preserves rare subtypes (GMP, cDC) correctly

### Bad Result (Over-correction):

```
Method X Performance:
  Accuracy: 0.91  (very high!)
  Confidence: 0.88

  Per-Subtype Marker Enrichment:
    HSPC → HSPC markers: 0.52  (weak!)
    Monocyte → Mono markers: 0.48  (weak!)
```

**Interpretation:**
- High accuracy but low marker enrichment
- Method is mixing cells based on technical features, not biology
- Over-corrected and distorted biological signal

## The Key Insight

**Good batch correction should:**
1. ✅ Enable label transfer across technologies (high accuracy)
2. ✅ Preserve biological signal (high marker enrichment)
3. ✅ Maintain rare cell types (don't collapse diversity)

**SCimilarity's strength:**
- May not have highest accuracy (less aggressive batch mixing)
- But preserves biological markers better
- Design philosophy: biology > aggressive correction

This is tested by comparing both accuracy AND marker enrichment across methods.

## Summary

The validation:
1. **Loads** van Galen + Velten (papers that cite van Galen)
2. **Preprocesses** identically for all methods
3. **Computes** different embeddings (PCA, scVI, SCimilarity, Harmony)
4. **Trains** k-NN on van Galen embeddings
5. **Predicts** Velten cell types
6. **Validates** with both accuracy and marker genes
7. **Compares** methods on multiple metrics

The goal: Test if SCimilarity can reproduce Velten's adoption of van Galen's framework while preserving biological signal.
