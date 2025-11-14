# Using NPM1 AML Data for Van Galen Validation

## Overview

This guide explains how to check and potentially use the NPM1 AML Seurat data (83,162 cells, 16 patients) for validation.

**Paper**: "Resolving inter- and intra-patient heterogeneity in NPM1-mutated AML at single-cell resolution" (Leukemia/Nature 2025)

**Why this data is valuable:**
- ✅ Cites van Galen et al. 2019 framework
- ✅ 83k cells (20× larger than velten_2021)
- ✅ 16 independent patient samples
- ✅ Published in high-impact journal

**The critical question:** Does it have raw counts? (SCimilarity needs them)

## Quick Start

### Option 1: Simple Check (Recommended)

```bash
# Check if the Seurat file has usable data
./check_npm1_data.sh
```

This runs the full inspection and tells you if the data can be used.

### Option 2: Manual Steps

If you prefer step-by-step:

```bash
# 1. Inspect the Seurat object
Rscript inspect_npm1_seurat.R

# 2. If it has raw counts, convert to h5ad
Rscript convert_npm1_to_h5ad.R

# 3. Verify the converted file
python check_npm1_h5ad.py
```

## What Each Script Does

### 1. `inspect_npm1_seurat.R`

**Purpose:** Check if Seurat object has raw counts and van Galen labels

**What it checks:**
- ✅ Does `@assays$RNA@counts` exist?
- ✅ Are counts raw (max > 100, integer values)?
- ✅ What cell type labels are available?
- ✅ Are there van Galen subtype labels?
- ✅ Sample/patient information

**Output example (GOOD):**
```
✓✓✓ THIS LOOKS LIKE RAW COUNTS! ✓✓✓
  - Max value > 100 (406,026)
  - Values are integers
  → SCimilarity can use this!

✓ Found van Galen-like labels: HSPC, GMP, CD14+ Mono, cDC
```

**Output example (BAD):**
```
✗✗✗ WARNING: NOT RAW COUNTS ✗✗✗
  - Max value < 20 (5.3)
  - This looks normalized/log-transformed
  → SCimilarity CANNOT use this
```

### 2. `convert_npm1_to_h5ad.R`

**Purpose:** Convert Seurat object to h5ad format (Python/scanpy)

**Requirements:**
- R packages: `qs`, `Seurat`, `SeuratDisk`
- Raw counts must exist (verified by inspection script)

**What it does:**
1. Loads Seurat object
2. Verifies raw counts
3. Adds `Study = "npm1_2024"` metadata
4. Saves as `.h5Seurat`
5. Converts to `.h5ad` format

**Output:** `data/NPM1_AML.h5ad` (ready for Python)

### 3. `check_npm1_h5ad.py`

**Purpose:** Verify converted h5ad file is ready for validation

**What it checks:**
- ✅ File exists and loads correctly
- ✅ Has raw counts (in `layers['counts']` or `.X`)
- ✅ Has `Study` column
- ✅ Has cell type labels
- ✅ Can find van Galen subtypes

**Output:**
```
✓✓✓ FILE IS READY FOR VALIDATION! ✓✓✓

Next steps:
  1. Edit validate_aml_subtypes.py:
     TEST_STUDIES = ['velten_2021', 'npm1_2024']
  2. Run validation:
     python validate_aml_subtypes.py
```

## Expected Scenarios

### ✅ Scenario A: Data Has Raw Counts

```bash
./check_npm1_data.sh
```

**Output:**
```
✓ Raw counts verified (max = 406,026)
✓ Found van Galen-like labels
✓✓✓ THIS DATA CAN BE USED FOR VALIDATION! ✓✓✓
```

**Next steps:**
```bash
# Convert to h5ad
Rscript convert_npm1_to_h5ad.R

# Verify
python check_npm1_h5ad.py

# Add to validation
# Edit validate_aml_subtypes.py line 84-87:
TEST_STUDIES = [
    'velten_2021',
    'npm1_2024',  # Add this!
]

# Run validation
python validate_aml_subtypes.py
```

**Result:** Much stronger validation with 87k total cells instead of 27k!

### ⚠️ Scenario B: Data Might Have Raw Counts (Unclear)

```bash
./check_npm1_data.sh
```

**Output:**
```
⚠ UNCLEAR
  - Max value: 50
  - Might be raw counts, might not
```

**Action:** Proceed cautiously:
1. Try conversion anyway
2. Check `check_npm1_h5ad.py` output carefully
3. Inspect sample cells manually
4. Compare results with/without this data

### ✗ Scenario C: Data Has NO Raw Counts

```bash
./check_npm1_data.sh
```

**Output:**
```
✗✗✗ WARNING: NOT RAW COUNTS ✗✗✗
  - Max value < 20 (5.3)
  → SCimilarity CANNOT use this
```

**Your options:**

**Option 1: Get Raw Data from EGA** (Recommended for publication)
1. Register at https://ega-archive.org/
2. Apply for access: **EGAS50000000332**
3. Download fastq files
4. Process with CellRanger
5. Create h5ad with raw counts

**Timeline:** 2-4 weeks for access + processing

**Option 2: Contact Authors**
- Email corresponding author
- Ask if they can share raw counts version
- Explain it's for methods comparison study

**Option 3: Continue Without NPM1**
- Your current validation with velten_2021 is **already valid**
- 27k cells is sufficient for demonstrating the approach
- No need for NPM1 data unless writing a paper

## Installing R Dependencies

If the scripts fail due to missing packages:

```r
# In R console
install.packages("qs")
install.packages("Seurat")

# SeuratDisk (for h5ad conversion)
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk")
```

## Troubleshooting

### Problem: "qs package not found"

```bash
Rscript -e "install.packages('qs', repos='https://cloud.r-project.org')"
```

### Problem: "Seurat package not found"

```bash
Rscript -e "install.packages('Seurat', repos='https://cloud.r-project.org')"
```

### Problem: "Cannot convert to h5ad"

```bash
# Install SeuratDisk
Rscript -e "remotes::install_github('mojaveazure/seurat-disk')"
```

### Problem: Conversion succeeds but check fails

The converted file might have issues. Check:
```python
import scanpy as sc
adata = sc.read_h5ad("data/NPM1_AML.h5ad")
print(adata)
print(adata.layers.keys())
print(adata.X.max())  # Should be > 100 for raw counts
```

## Decision Tree

```
Do you have NPM1 Seurat file?
├─ NO → Use velten_2021 only (current validation, sufficient)
│
└─ YES → Run: ./check_npm1_data.sh
    │
    ├─ Has raw counts? YES
    │   └─ Convert → Add to validation → Stronger results!
    │
    ├─ Has raw counts? UNCLEAR
    │   └─ Try conversion → Manual inspection → Decide
    │
    └─ Has raw counts? NO
        ├─ Get EGA raw data (2-4 weeks) → Best for publication
        ├─ Contact authors → Might work quickly
        └─ Skip NPM1 → Current validation is fine
```

## Summary

**For most users:**
- Run `./check_npm1_data.sh`
- See what it says
- If no raw counts, stick with velten_2021 (perfectly valid!)

**For publication:**
- If NPM1 has raw counts: USE IT (much stronger validation)
- If not: Apply for EGA data (worth the wait)

**Bottom line:** Your current validation with velten_2021 is already complete and valid. NPM1 would make it even stronger, but it's optional.
