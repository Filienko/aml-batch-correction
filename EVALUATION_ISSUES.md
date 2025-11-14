# Critical Issues with Current Evaluation

## Issue 1: "Uncorrected" is NOT Actually Uncorrected

### The Problem

Looking at the code in `run_evaluation.py`:

```python
def prepare_uncorrected_embedding_exact(adata, batch_key_lower):
    """Compute uncorrected PCA baseline"""
    # 1. Normalize
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)

    # 2. Select HVGs
    sc.pp.highly_variable_genes(adata_work, n_top_genes=2000)

    # 3. Scale
    sc.pp.scale(adata_work)

    # 4. PCA
    sc.tl.pca(adata_work, n_comps=50)

    adata.obsm['X_uncorrected'] = adata_work.obsm['X_pca']
```

**This is NOT uncorrected!** It's:
- Log-normalized
- HVG-selected (2000 genes)
- Scaled (z-scored)
- **PCA-transformed** (50 dimensions)

**PCA itself provides structure** and can separate cell types reasonably well even without explicit batch correction.

### Why This Performs Well

PCA on normalized, HVG-selected data:
- ✅ Captures major biological variation (cell types)
- ✅ Reduces noise through dimensionality reduction
- ✅ May even reduce some batch effects (if batches don't align with top PCs)

**This is already a form of batch correction** - just not an explicit one.

### Why SCimilarity Might Appear Worse

If "uncorrected" PCA performs well, it means:
1. **Batch effects are not severe** in this particular subset
2. **Biology dominates** the top PCs
3. **Cell types are well-separated** even without explicit correction

In this case:
- SCimilarity's aggressive transformation might be **unnecessary**
- SCimilarity might be **over-correcting** and distorting biological signal
- Or SCimilarity embeddings might not be comparable to PCA embeddings for k-NN

### The Real Comparison

We're actually comparing:
- **"Uncorrected"**: PCA (50D) on normalized, scaled, HVG-selected data
- **SCimilarity**: Transformer embeddings (256D?) from CellAnnotation model
- **scVI**: VAE latent space (30D) from deep learning
- **Harmony**: Iteratively corrected PCA (50D)

These are **different embedding spaces** with different dimensions!

## Issue 2: k-NN Might Not Be Fair for SCimilarity

### The Problem

k-NN classification performance depends on:
1. **Distance metric** (Euclidean by default in sklearn)
2. **Embedding dimensionality** (50D PCA vs 256D SCimilarity)
3. **Embedding scale** (PCA is z-scored, SCimilarity is not)

**SCimilarity embeddings** are designed for:
- Cell type annotation (similarity search)
- Semantic relationships between cells
- **NOT necessarily optimized for Euclidean k-NN**

**PCA embeddings** are:
- Orthogonal, uncorrelated dimensions
- Z-scored (standardized scale)
- **Well-suited for Euclidean k-NN**

### Why This Matters

If we use k-NN on different embedding spaces:
- **PCA might have an advantage** due to orthogonality and standardization
- **SCimilarity might be penalized** if its embeddings have different scales/distributions

## Issue 3: The Task Might Not Test Batch Correction

### What We're Testing

Current validation:
1. Train k-NN on van Galen (Seq-Well)
2. Predict cell types in test study (10x/Muta-Seq)
3. Compare accuracy

**This tests**: Can we transfer labels across technologies?

**This does NOT directly test**: Did batch correction help?

### Why?

If batch effects DON'T dominate biological signal:
- "Uncorrected" PCA will work fine
- Batch correction might not improve (or could hurt)
- This doesn't mean batch correction is bad, just unnecessary for this task

### When Batch Correction Matters

Batch correction should help when:
- ❌ Batch effects are strong
- ❌ Batches separate cells more than biology
- ❌ Without correction, same cell types from different batches don't cluster together

But if:
- ✅ Biology is stronger than batch effects
- ✅ Cell types are very distinct (HSPC vs Monocyte)
- ✅ Studies have similar quality

Then "uncorrected" PCA might already work well!

## Recommended Fixes

### Fix 1: Add a Truly Uncorrected Baseline

```python
# Raw normalized counts without PCA
def prepare_truly_uncorrected(adata):
    # Just log-normalize, no PCA, no HVG selection
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Use all genes (or common subset)
    adata.obsm['X_truly_uncorrected'] = adata.X
```

This shows what happens WITHOUT any dimensionality reduction.

### Fix 2: Normalize Embedding Dimensions

```python
# Standardize all embeddings to same scale
from sklearn.preprocessing import StandardScaler

for key in ['X_pca', 'X_scimilarity', 'X_scVI', 'X_harmony']:
    if key in adata.obsm:
        scaler = StandardScaler()
        adata.obsm[key] = scaler.fit_transform(adata.obsm[key])
```

This makes k-NN comparison fairer.

### Fix 3: Use Multiple Metrics

k-NN accuracy is just one metric. Also check:
- **Silhouette scores** (are cell types well-separated?)
- **Batch mixing** (are same cell types from different studies mixed?)
- **Marker enrichment** (do predicted cells express correct markers?)

### Fix 4: Focus on Velten Only

```python
TEST_STUDIES = ['velten_2021']  # Only study that actually uses van Galen
```

- Cleaner experimental design
- Stronger validation (they actually used van Galen's framework)
- Seq-Well → Muta-Seq (cross-technology transfer)

## Expected Results After Fixes

### If batch effects are weak:
```
Method          Accuracy
Truly Uncorrected: 0.45  (baseline, no structure)
PCA (current):     0.78  (PCA provides structure)
SCimilarity:       0.82  (slight improvement)
Harmony:           0.80  (similar to PCA)
scVI:              0.79  (over-corrects?)
```

### If batch effects are strong:
```
Method          Accuracy
Truly Uncorrected: 0.45  (baseline)
PCA (current):     0.52  (batch effects dominate)
SCimilarity:       0.82  (batch correction helps!)
Harmony:           0.79  (explicit correction needed)
scVI:              0.85  (designed for this)
```

The current results suggest **batch effects are not strong enough** to require aggressive correction for this particular label transfer task.

## Bottom Line

**Why uncorrected performs well**:
1. It's not truly uncorrected (uses PCA)
2. PCA already provides good cell type separation
3. Batch effects may not be severe for this task
4. k-NN might favor PCA's standardized, orthogonal space

**This doesn't mean SCimilarity is bad** - it means:
- The task doesn't require strong batch correction
- Or the evaluation setup isn't fair for comparing different embedding spaces
- Or we should focus on metrics where SCimilarity excels (marker preservation, rare cell detection)
