# Using Seurat Objects for Van Galen Validation

## Question: Seurat Object vs Raw Data?

**Answer: Use the Seurat object!** It contains everything you need.

## What's in a Seurat Object?

A Seurat object typically contains:

```r
# In R:
seurat_obj@assays$RNA@counts        # Raw counts ✓
seurat_obj@assays$RNA@data          # Normalized data ✓
seurat_obj@meta.data                # Cell annotations ✓
seurat_obj@reductions$pca           # PCA embeddings
seurat_obj@reductions$umap          # UMAP embeddings
```

**You need:**
1. ✅ Raw counts (for SCimilarity)
2. ✅ Cell type labels (for validation)
3. ✅ Study/sample metadata

All of this is in the Seurat object!

## How to Convert Seurat to h5ad

### Option 1: Using SeuratDisk (Recommended)

```r
# In R
library(Seurat)
library(SeuratDisk)

# Load Seurat object
seurat_obj <- readRDS("npm1_aml_seurat.rds")

# Convert to h5ad
SaveH5Seurat(seurat_obj, filename = "npm1_aml.h5Seurat")
Convert("npm1_aml.h5Seurat", dest = "h5ad")

# Result: npm1_aml.h5ad
```

### Option 2: Manual Export

```r
# In R
library(Seurat)

# Load Seurat object
seurat_obj <- readRDS("npm1_aml_seurat.rds")

# Export counts
library(Matrix)
writeMM(seurat_obj@assays$RNA@counts, "counts.mtx")

# Export genes
write.csv(rownames(seurat_obj), "genes.csv", row.names=FALSE)

# Export cells
write.csv(colnames(seurat_obj), "cells.csv", row.names=FALSE)

# Export metadata
write.csv(seurat_obj@meta.data, "metadata.csv")
```

Then in Python:
```python
import scanpy as sc
import scipy.io
import pandas as pd

# Load data
counts = scipy.io.mmread("counts.mtx").T.tocsr()
genes = pd.read_csv("genes.csv").iloc[:, 0].values
cells = pd.read_csv("cells.csv").iloc[:, 0].values
metadata = pd.read_csv("metadata.csv", index_col=0)

# Create AnnData
adata = sc.AnnData(X=counts, obs=metadata)
adata.var_names = genes

# Save as h5ad
adata.write_h5ad("npm1_aml.h5ad")
```

## What to Check in the Seurat Object

### 1. Cell Type Labels

Check if they have van Galen's 6 subtypes:

```r
# In R
unique(seurat_obj@meta.data$cell_type)  # Or whatever the column is called

# Should see:
# "HSPC", "CMP", "GMP", "ProMono", "CD14+ Mono", "cDC"
# Or similar van Galen categories
```

### 2. Study/Sample Information

```r
# Check if multiple samples
unique(seurat_obj@meta.data$sample)  # Or "patient", "orig.ident", etc.

# You mentioned 16 samples → perfect for batch correction testing!
```

### 3. Raw Counts

```r
# Verify raw counts exist
max(seurat_obj@assays$RNA@counts)  # Should be > 100 (integers)

# If normalized data:
max(seurat_obj@assays$RNA@data)    # Should be < 10 (log-normalized)
```

## How to Use This Data

### Option 1: Add to Your Atlas

If this data uses van Galen's framework, add it to your validation:

```python
# Load your atlas
adata_atlas = sc.read_h5ad("data/AML_scAtlas.h5ad")

# Load NPM1 study
adata_npm1 = sc.read_h5ad("npm1_aml.h5ad")
adata_npm1.obs['Study'] = 'npm1_2024'  # Or whatever year

# Combine
adata_combined = adata_atlas.concatenate(adata_npm1)

# Now use in validation
TEST_STUDIES = [
    'velten_2021',
    'npm1_2024',  # Your new study!
]
```

### Option 2: Standalone Validation

Use ONLY van Galen + this NPM1 study:

```python
# Just these two studies
TEST_STUDIES = ['npm1_2024']

# Cleaner test:
# - van Galen as reference (Seq-Well)
# - NPM1 study as test (10x Chromium probably)
# - Tests if SCimilarity can reproduce van Galen's classification
#   in a study that explicitly adopted it
```

## Why This Study is Perfect for Your Question

You asked: "Use papers that look at the same latent subtypes as van Galen"

This NPM1 study is **ideal** because:

1. ✅ **Explicitly uses van Galen's framework**
   - They mention following van Galen's cell type classification
   - You can cite this as validation

2. ✅ **Large dataset** (83,162 cells)
   - More power to detect differences
   - Can test rare subtypes (GMP, cDC)

3. ✅ **Multiple samples** (16 patients)
   - Tests batch correction across patients
   - More realistic than single-patient studies

4. ✅ **NPM1-mutated AML** (specific subtype)
   - Well-characterized molecular subtype
   - Good biological control

## Paper Citation Needed

Please share:
- Paper title
- Authors
- Journal/Year
- DOI/PubMed ID

So I can:
1. Verify they actually use van Galen's framework
2. Check their processing pipeline
3. Update validation documentation
4. Properly cite them

## Summary

**Do you need raw data or Seurat object?**
→ **Seurat object is perfect!** It has everything you need.

**Next steps:**
1. Share paper citation (so I can verify it uses van Galen)
2. Get Seurat object (from authors or GEO/SRA)
3. Convert to h5ad format (using SeuratDisk)
4. Add to validation pipeline
5. Run validation with this new study

This would be a **much stronger validation** than velten_2021 alone, because:
- Larger dataset (83k vs 4k cells)
- Multiple patients (16 vs fewer in Velten)
- Explicitly documented use of van Galen framework

Let me know the paper details and I'll help you integrate it!
