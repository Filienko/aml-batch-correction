# Data Sources for AML Atlas Replication Project

## Overview

This project requires two types of data:
1. **Ground Truth**: The fully processed AML scAtlas (159 patients)
2. **Raw Problem Data**: Unintegrated datasets used to build the atlas

---

## 1. AML scAtlas (Ground Truth)

### Publication
- **Paper**: [Insert AML scAtlas publication title]
- **DOI**: [Insert DOI]
- **Year**: 2024 (or appropriate year)

### Data Availability

**Option A: Direct Download from Publication**
- Check the paper's "Data Availability" section
- Usually deposited in:
  - **GEO** (Gene Expression Omnibus)
  - **ArrayExpress**
  - **Zenodo**
  - **FigShare**
  - **Institutional repository**

**Option B: Request from Authors**
- Contact corresponding author
- Request: `AML_scAtlas.h5ad` (processed AnnData object)

**What we need**:
```
File: AML_scAtlas.h5ad
Size: ~X GB
Format: AnnData (h5ad)

Required metadata columns:
- dataset_of_origin (or Study, or Batch): Which study each cell came from
- cell_type_annotation (or celltype): Expert-curated cell type labels
- Sample: Patient ID
- Any LSC-related annotations (LSC6, LSC17 scores if available)

Optional but useful:
- UMAP coordinates (X_umap)
- scVI embeddings (X_scVI or in separate file)
- Raw counts (.raw or .layers['counts'])
```

### Download Instructions (Generic)

```bash
# Create data directory
mkdir -p data/

# Option 1: If hosted on GEO
# Example: GSE123456
wget -O data/AML_scAtlas.h5ad "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE123nnn/GSE123456/suppl/[filename]"

# Option 2: If using SRA toolkit
prefetch GSE123456
fasterq-dump GSE123456

# Option 3: If authors provide direct link
wget -O data/AML_scAtlas.h5ad "https://[author-provided-url]/AML_scAtlas.h5ad"

# Verify file
python -c "import scanpy as sc; adata = sc.read_h5ad('data/AML_scAtlas.h5ad'); print(adata)"
```

---

## 2. Raw Datasets (The "Problem")

### 2A. van Galen et al. 2019

**Publication**:
- **Title**: "A single-cell RNA-seq reveals AML hierarchies relevant to disease progression and immunity"
- **Journal**: Cell (2019)
- **DOI**: 10.1016/j.cell.2019.01.031
- **GEO**: GSE116256

**Why this dataset**:
- Gold standard reference for AML cell types
- Established LSC hierarchy
- Most cited AML scRNA-seq paper
- ~30,000 cells from 16 patients

**Download**:

```bash
# Method 1: Direct GEO download
mkdir -p data/raw/
cd data/raw/

# Download the processed matrix
wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116256/suppl/GSE116256_RAW.tar"
tar -xvf GSE116256_RAW.tar

# Method 2: Using SRA toolkit (if fastq needed)
prefetch GSE116256
fasterq-dump --split-files GSE116256

# Method 3: Using scanpy/anndata if pre-processed available
# Check supplementary files for h5ad or loom format

# The data may come as:
# - .mtx.gz (matrix market format)
# - .h5 (HDF5 format)
# - .loom (loom format)
# - Individual .tsv.gz files per sample

# Convert to h5ad
python << EOF
import scanpy as sc
import pandas as pd
import scipy.io as sio
import os

# Example: If data comes as MTX
# matrix = sio.mmread('matrix.mtx.gz')
# genes = pd.read_csv('genes.tsv.gz', header=None, sep='\t')
# barcodes = pd.read_csv('barcodes.tsv.gz', header=None, sep='\t')
#
# adata = sc.AnnData(X=matrix.T.tocsr())
# adata.var_names = genes[1].values
# adata.obs_names = barcodes[0].values
# adata.write('van_galen_2019_raw.h5ad')

# More likely: Use scanpy's built-in readers
# adata = sc.read_10x_mtx('path/to/filtered_feature_bc_matrix/')
# adata.write('van_galen_2019_raw.h5ad')
EOF
```

**Expected output**:
- File: `data/raw/van_galen_2019_raw.h5ad`
- ~30,000 cells
- Raw count matrix
- Metadata: patient ID, cell type annotations (if provided)

---

### 2B. Abbas et al. 2021

**Publication**:
- **Title**: [Insert full title]
- **Journal**: [Insert journal]
- **DOI**: [Insert DOI]
- **GEO**: GSE198052

**Download**:

```bash
mkdir -p data/raw/
cd data/raw/

# Download from GEO
wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198052/suppl/GSE198052_RAW.tar"
tar -xvf GSE198052_RAW.tar

# Convert to h5ad (similar process as van Galen)
python convert_abbas_to_h5ad.py
```

**Expected output**:
- File: `data/raw/abbas_2021_raw.h5ad`

---

### 2C. Wang et al. 2024

**Publication**:
- **Title**: [Check AML scAtlas paper references]
- **GEO**: [TBD - check atlas paper methods]

**Note**: The specific Wang et al. dataset should be listed in the AML scAtlas methods section under "Data sources" or "Datasets included."

**Download**: Similar process to above once GEO accession identified.

---

## 3. Data Validation

After downloading all datasets, validate them:

```bash
python << EOF
import scanpy as sc
import os

# Check atlas
if os.path.exists('data/AML_scAtlas.h5ad'):
    atlas = sc.read_h5ad('data/AML_scAtlas.h5ad')
    print(f"✓ Atlas: {atlas.n_obs:,} cells × {atlas.n_vars:,} genes")
    print(f"  Studies: {atlas.obs['Study'].nunique() if 'Study' in atlas.obs else 'Unknown'}")
    print(f"  Cell types: {atlas.obs.columns.tolist()}")
else:
    print("✗ AML_scAtlas.h5ad not found")

# Check raw datasets
raw_files = [
    'data/raw/van_galen_2019_raw.h5ad',
    'data/raw/abbas_2021_raw.h5ad',
    'data/raw/wang_2024_raw.h5ad',
]

for file in raw_files:
    if os.path.exists(file):
        adata = sc.read_h5ad(file)
        print(f"✓ {file}: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    else:
        print(f"✗ {file} not found")
EOF
```

---

## 4. Alternative: Use Existing AML scAtlas Subset

If you already have the AML scAtlas loaded and it contains the raw count data:

```python
import scanpy as sc

# Load atlas
atlas = sc.read_h5ad('data/AML_scAtlas.h5ad')

# Extract specific studies back to "raw" state
studies_of_interest = ['van_galen_2019', 'abbas_2021', 'wang_2024']

for study in studies_of_interest:
    # Subset
    study_mask = atlas.obs['Study'] == study
    adata_study = atlas[study_mask].copy()

    # Get raw counts if available
    if adata_study.raw is not None:
        adata_raw = adata_study.raw.to_adata()
    elif 'counts' in adata_study.layers:
        adata_raw = adata_study.copy()
        adata_raw.X = adata_raw.layers['counts']
    else:
        print(f"Warning: No raw counts found for {study}")
        continue

    # Save
    adata_raw.write(f'data/raw/{study}_raw.h5ad')
    print(f"✓ Extracted {study}: {adata_raw.n_obs:,} cells")
```

---

## 5. Directory Structure

After downloading all data:

```
data/
├── AML_scAtlas.h5ad                    # Ground truth (159 patients)
├── AML_scAtlas_X_scVI.h5ad            # Optional: pre-computed scVI (if separate)
│
├── raw/                                # Raw, unintegrated datasets
│   ├── van_galen_2019_raw.h5ad
│   ├── abbas_2021_raw.h5ad
│   └── wang_2024_raw.h5ad
│
└── processed/                          # Our analysis outputs
    ├── merged_raw_problem.h5ad         # Merged raw data (Phase 1)
    ├── scimilarity_solution.h5ad       # SCimilarity-corrected (Phase 2)
    └── [other intermediate files]
```

---

## 6. Troubleshooting

### Problem: Can't find AML scAtlas file
**Solution**:
1. Check the publication's supplementary materials
2. Search GEO for author names + "AML" + year
3. Email corresponding author directly
4. Check preprint servers (bioRxiv) for earlier versions with data links

### Problem: GEO data is in fastq format
**Solution**:
1. Use Cell Ranger (10x) or appropriate tool to generate count matrices
2. Or look for processed files in supplementary materials
3. Authors often provide pre-processed h5ad files directly

### Problem: File formats don't match
**Solution**:
```python
import scanpy as sc

# From loom
adata = sc.read_loom('file.loom')
adata.write('file.h5ad')

# From 10x
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')
adata.write('file.h5ad')

# From MTX
adata = sc.read_mtx('matrix.mtx.gz').T
# Add gene names and barcode names manually
adata.write('file.h5ad')
```

---

## 7. Quick Start

If you want to start immediately with a similar dataset:

```python
# Use scanpy's built-in datasets or public AML datasets
import scanpy as sc

# Example: Load a public AML dataset
# (Note: This won't be the exact AML scAtlas, but similar)
adata = sc.datasets.pbmc3k()  # Replace with actual AML dataset

# Or download from published atlases
# Many labs provide processed h5ad files directly in Zenodo/FigShare
```

---

## 8. Citation Requirements

When using these datasets, cite:

1. **AML scAtlas**: [Full citation]
2. **van Galen et al. 2019**: van Galen et al., Cell 2019
3. **Abbas et al. 2021**: [Full citation]
4. **Wang et al. 2024**: [Full citation]

---

## Need Help?

1. Check the original publication's "Data Availability" statement
2. Look for author-provided code repositories (GitHub) with data download scripts
3. Contact the corresponding author
4. Post in bioinformatics forums (Biostars, SEqanswers)
