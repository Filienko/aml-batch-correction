# Exactly What Happens in the Validation

## Your Critical Question

> "Do you take PCA over the training data (one study) and test on PCA on another study? And then train a classifier over the training study and test on the testing study, so these are separate?"

## The Answer: NO! (And This is Correct)

### What Actually Happens

**Embeddings are computed on COMBINED data (all studies together), THEN split for train/test.**

```python
# Step 1: Load and combine studies
van_galen_cells = load_study('van_galen_2019')  # 23,344 cells
velten_cells = load_study('velten_2021')        # 4,191 cells
combined = concatenate(van_galen_cells, velten_cells)  # 27,535 cells TOGETHER

# Step 2: Preprocess TOGETHER
combined = normalize(combined)          # All cells together
combined = select_hvgs(combined)        # All cells together
combined = scale(combined)              # All cells together

# Step 3: Compute embeddings on ALL cells TOGETHER
combined.obsm['X_pca'] = PCA(combined)  # PCA on ALL 27,535 cells
combined.obsm['X_scimilarity'] = SCimilarity(combined)  # SCimilarity on ALL cells
combined.obsm['X_harmony'] = Harmony(combined)  # Harmony on ALL cells

# Step 4: THEN split for train/test
ref_embedding = combined.obsm['X_pca'][combined.obs['Study'] == 'van_galen']
ref_labels = combined.obs['label'][combined.obs['Study'] == 'van_galen']

test_embedding = combined.obsm['X_pca'][combined.obs['Study'] == 'velten']
test_labels = combined.obs['label'][combined.obs['Study'] == 'velten']

# Step 5: Train classifier on reference, predict on test
knn = train(ref_embedding, ref_labels)
predictions = knn.predict(test_embedding)
```

### Why This is the CORRECT Approach

**Batch correction REQUIRES seeing both batches!**

#### For PCA:
```python
# CORRECT (what we do):
pca = PCA()
pca.fit([van_galen_cells + velten_cells])  # Fit on combined data
# Now both studies are in the SAME PCA space

# WRONG (what you thought we might be doing):
pca_vg = PCA().fit(van_galen_cells)  # Fit on van Galen only
pca_velten = PCA().fit(velten_cells)  # Fit on Velten only
# These are DIFFERENT PCA spaces - can't compare!
```

#### For Harmony:
```python
# CORRECT:
combined = [van_galen + velten]  # Combined
pca = PCA(combined)  # PCA on combined
harmony = correct_batches(pca, batch_labels=['van_galen', 'velten'])
# Harmony ADJUSTS the combined PCA to remove batch effects

# WRONG:
harmony_vg = correct_batches(van_galen_only)  # No batch to correct!
# Can't correct batch effects if you only see one batch
```

#### For SCimilarity:
```python
# CORRECT:
scimilarity = SCimilarity_model.embed([van_galen + velten])
# Both studies embedded in same space

# WRONG:
emb_vg = SCimilarity_model.embed(van_galen_only)
emb_velten = SCimilarity_model.embed(velten_only)
# These are separate - can't directly compare
```

### The Embedding Space is SHARED

All cells from both studies exist in the **same embedding space**:

```
         PC1 (dimension 1)
              ↓
    -5  -4  -3  -2  -1   0   1   2   3   4   5
PC2  ┌──────────────────────────────────────────┐
 5   │                                          │
 4   │    🔵 van Galen HSPC                    │
 3   │    🔵🔵                                  │
 2   │    🔵🔵🔴 velten HSPC                   │
 1   │         🔴🔴                             │
 0   ├──────────────────────────────────────────┤
-1   │         🔴🔴                             │
-2   │    🔵🔵🔴 velten Mono                   │
-3   │    🔵🔵                                  │
-4   │    🔵 van Galen Mono                    │
-5   │                                          │
     └──────────────────────────────────────────┘

🔵 = van Galen cells
🔴 = Velten cells
```

**Good batch correction**: Same cell types from different studies are CLOSE in the embedding

**Bad batch correction**: Same cell types from different studies are FAR apart

### What We're Testing

```python
# Train on van Galen's neighborhood structure
knn = KNN(k=15)
knn.fit(
    X=van_galen_embeddings,  # Points in shared space
    y=van_galen_labels       # HSPC, Monocyte, etc.
)

# Test: Can we predict Velten labels using van Galen's structure?
predictions = knn.predict(velten_embeddings)

# If batch correction worked:
# - Velten HSPC cells are near van Galen HSPC cells → predicted as HSPC ✓
# - Velten Monocyte cells are near van Galen Monocytes → predicted as Monocyte ✓

# If batch correction failed:
# - Velten cells cluster by study, not cell type → wrong predictions ✗
```

## Why This is NOT Data Leakage

**Question**: "Isn't this leakage if the test data is used in PCA?"

**Answer**: NO, because:

1. **We're testing batch correction**, not general ML
   - The GOAL is to integrate both studies
   - We WANT both studies in the same space
   - That's what "batch correction" means!

2. **The labels are separate**
   - PCA sees both studies' gene expression (features)
   - k-NN only trains on van Galen labels
   - Velten labels are NEVER seen during training

3. **This is the standard approach**
   - scIB metrics do this
   - All batch correction benchmarks do this
   - It's measuring: "Can we integrate the batches?"

## What WOULD Be Leakage

### Leakage Example 1: scVI in Our Results

```python
# scVI was TRAINED on the full atlas (including velten)
scvi_model = train_scVI(full_atlas)  # Includes velten ✗

# Then we test on velten
predictions = test(scvi_embeddings, velten_subset)  # Leakage!
```

**This is why scVI has 93.8% accuracy** - it's seen velten before!

### Leakage Example 2: Training k-NN on Test Labels

```python
# WRONG:
knn.fit([van_galen + velten], [van_galen_labels + velten_labels])
predictions = knn.predict(velten)
# This would be leakage - test labels used in training

# CORRECT (what we do):
knn.fit(van_galen, van_galen_labels)  # Only van Galen labels
predictions = knn.predict(velten)     # Velten labels never seen
```

## Detailed Step-by-Step

### Input Data
```
van_galen_2019:  23,344 cells × 20,000 genes, labels: [HSPC, Mono, GMP, ...]
velten_2021:      4,191 cells × 20,000 genes, labels: [HSPC, Mono, GMP, ...]
```

### Step 1: Combine & Preprocess (TOGETHER)
```python
adata = concatenate([van_galen, velten])
# Shape: 27,535 cells × 20,000 genes

adata.obs['Study'] = ['van_galen']*23344 + ['velten']*4191  # Keep track of study

# Normalize TOGETHER
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select HVGs TOGETHER (across both studies)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
# Shape: 27,535 cells × 2,000 genes

# Scale TOGETHER
sc.pp.scale(adata)
```

### Step 2: Compute PCA (TOGETHER)
```python
sc.tl.pca(adata, n_comps=50)
# adata.obsm['X_pca']: (27,535 cells, 50 dimensions)

# This PCA captures:
# - PC1: Separates HSPC vs Monocytes (biology)
# - PC2: Might separate studies (batch effect)
# - PC3-50: Other variation
```

### Step 3: Compute Harmony (Corrects Batch Effects)
```python
# Start with PCA
harmony_integrate(
    adata,
    key='Study',           # Correct for study differences
    basis='X_pca',         # Input: PCA
    adjusted_basis='X_harmony'  # Output
)

# Harmony ADJUSTS PCA to reduce study differences while keeping biology
# adata.obsm['X_harmony']: (27,535 cells, 50 dimensions)
```

### Step 4: Compute SCimilarity (TOGETHER)
```python
ca = CellAnnotation(model_path)
embeddings = ca.get_embeddings(adata)  # All 27,535 cells
adata.obsm['X_scimilarity'] = embeddings
# Shape: (27,535 cells, 256 dimensions)
```

### Step 5: Split for Train/Test
```python
# Reference (training)
ref_mask = adata.obs['Study'] == 'van_galen'
ref_X = adata.obsm['X_pca'][ref_mask]      # (23,344, 50)
ref_y = adata.obs['label'][ref_mask]       # (23,344,) labels

# Test
test_mask = adata.obs['Study'] == 'velten'
test_X = adata.obsm['X_pca'][test_mask]    # (4,191, 50)
test_y = adata.obs['label'][test_mask]     # (4,191,) labels - NOT used in training!
```

### Step 6: Train k-NN (Only on van Galen)
```python
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(ref_X, ref_y)  # Train on van Galen only
```

### Step 7: Predict (On Velten)
```python
predictions = knn.predict(test_X)
accuracy = (predictions == test_y).mean()
```

## Key Insight

**The embedding space is shared, but the labels are not.**

- ✅ Gene expression from both studies → PCA/embeddings (features)
- ✅ Labels from van Galen only → k-NN training (supervision)
- ✅ Labels from Velten only → evaluation (held out)

This tests: **"If we integrate the studies (batch correction), can we transfer knowledge from one to another?"**

## Summary

Your question was excellent because it highlighted a potential confusion!

**What we do:**
1. Compute embeddings on COMBINED data (all studies together)
2. Train classifier on van Galen subset only
3. Test on Velten subset

**Why this is correct:**
- Batch correction NEEDS both batches to work
- This is the standard approach for evaluating batch correction
- Not data leakage because test labels are never used in training

**The test:**
- Good batch correction → Velten cells near similar van Galen cells → high accuracy
- Bad batch correction → Velten cells separate from van Galen → low accuracy

This is exactly what we want to measure!
