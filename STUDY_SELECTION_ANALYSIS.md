# Study Selection Analysis for Label Transfer Benchmark

## 🚨 Critical Issue Identified

**The current `label_transfer_benchmark.py` may be using studies that DON'T have van Galen-compatible annotations!**

---

## Current Setup in `label_transfer_benchmark.py`

### Reference Study:
- **van_galen_2019** - Expert-labeled AML cells with 6 malignant subtypes

### Target Studies:
```python
TARGET_STUDIES = [
    'zhang_2023',              # ❓ Unknown annotation scheme
    'beneyto-calabuig-2023',   # ❓ Unknown annotation scheme
    'jiang_2020',              # ❓ Unknown annotation scheme
    'velten_2021',             # ✅ Uses van Galen framework!
]
```

---

## The Problem

**For label transfer evaluation to work, you need:**
1. Reference study with **expert labels** → van_galen_2019 ✅
2. Target studies with **ground truth labels** to compare predictions against
3. Target labels must be **compatible** with reference labels

**The benchmark computes accuracy like this:**
```python
y_true = adata_target.obs[label_key].values  # Ground truth
y_pred = classifier.predict(...)              # Predictions
accuracy = accuracy_score(y_true, y_pred)     # Compare!
```

**This FAILS if:**
- Target study has no cell type labels
- Target study uses different annotation scheme
- Target study labels don't match van Galen categories

---

## Studies That Actually Use Van Galen Framework

Based on `validate_aml_subtypes.py` (which was specifically designed for this):

### ✅ Confirmed Compatible:

**velten_2021**
- Paper: "Tracking leukemia evolution" *Nature* 2021
- Status: ✅ **Explicitly uses van Galen classification**
- Cells: ~27,000
- Technology: Muta-Seq

**zhai_2022** (Note: You wrote "Zheng" but it's "Zhai")
- Paper: "Molecular characterization of AML with inv(16)" *Nature Communications* 2022
- Status: ✅ **Used van Galen cell type classifier**
- Cells: Unknown (check if in atlas)
- Technology: SORT-Seq

### ❌ Incompatible or Unknown:

**pei_2020**
- Status: ❌ **Uses own annotation scheme, NOT van Galen**
- Published: 2020 (contemporary with van Galen)
- Note: May have similar cell types but different labels

**setty_2019**
- Status: ❌ **Published BEFORE van Galen (2019)**
- Cannot have used van Galen framework (didn't exist yet!)

**oetjen_2018**
- Status: ❌ **Published BEFORE van Galen**

**zhang_2023**
- Status: ❓ **Unknown** - need to check
- Published: 2023 (after van Galen)
- Might use van Galen framework OR own annotations

**beneyto-calabuig-2023**
- Status: ❓ **Unknown** - need to check
- Published: 2023 (after van Galen)

**jiang_2020**
- Status: ❓ **Unknown** - need to check
- Published: 2020 (contemporary with van Galen)

---

## What Your Benchmark Currently Does

Looking at the evaluation code:

```python
def evaluate_transfer(y_true, y_pred, method_name):
    """
    Evaluate label transfer quality.
    """
    # Remove NaN from ground truth
    valid_mask = pd.notna(y_true)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Compute metrics
    ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    nmi = normalized_mutual_info_score(y_true_valid, y_pred_valid)
```

**This means:**
1. It loads the target study's cell type labels as "ground truth"
2. Predicts labels using the classifier
3. Compares predictions to ground truth

**Critical assumption:** Target study labels are comparable to van Galen labels!

---

## Scenarios and Outcomes

### Scenario 1: Target has van Galen labels ✅
```
van Galen reference: 'HSC', 'GMP', 'Monocyte'
Target ground truth: 'HSC', 'GMP', 'Monocyte'
Predicted labels:    'HSC', 'GMP', 'Monocyte'
→ ARI = 0.85 ✅ Meaningful!
```

### Scenario 2: Target has different labels ❌
```
van Galen reference: 'HSC', 'GMP', 'Monocyte'
Target ground truth: 'Cluster_1', 'Cluster_2', 'Cluster_3'
Predicted labels:    'HSC', 'GMP', 'Monocyte'
→ ARI = 0.02 ❌ Meaningless comparison!
```

### Scenario 3: Target has similar but different labels ⚠️
```
van Galen reference: 'HSC', 'GMP', 'Monocyte'
Target ground truth: 'Stem_like', 'Progenitor', 'Mature_myeloid'
Predicted labels:    'HSC', 'GMP', 'Monocyte'
→ ARI = ??? ⚠️ Partial match, hard to interpret
```

---

## Which Studies Should You Use?

### Option 1: Use Only Confirmed Van Galen Studies ⭐ (SAFEST)

```python
TARGET_STUDIES = [
    'velten_2021',    # ✅ Confirmed uses van Galen framework
    'zhai_2022',      # ✅ Confirmed uses van Galen classifier (if in atlas)
]
```

**Pros:**
- ✅ Ground truth labels are **directly comparable**
- ✅ Evaluation metrics are **meaningful**
- ✅ Scientifically **rigorous**
- ✅ You're testing: "Can we replicate van Galen annotations in studies that originally used them?"

**Cons:**
- ⚠️ Only 2 target studies (but that's still valid!)
- ⚠️ Smaller sample (but quality > quantity)

---

### Option 2: Check Which Studies Have Compatible Labels

**Before running the benchmark, inspect the actual labels:**

```python
import scanpy as sc

adata = sc.read_h5ad('data/AML_scAtlas.h5ad')

# Check what studies exist
print("Available studies:")
print(adata.obs['Study'].value_counts())

# For each target study, check labels
for study in ['zhang_2023', 'beneyto-calabuig-2023', 'jiang_2020', 'velten_2021']:
    if study in adata.obs['Study'].values:
        subset = adata[adata.obs['Study'] == study]
        print(f"\n{study}:")
        print(f"  Cells: {len(subset)}")
        print(f"  Labels: {subset.obs['cell_type_annotation'].value_counts().head(10)}")
```

**If labels look like:** `'HSC', 'GMP', 'Monocyte', 'ProMono'` → ✅ Compatible!

**If labels look like:** `'Cluster_1', 'Cluster_2'` → ❌ Not compatible!

---

### Option 3: Use All Studies But Accept Lower Accuracy

If you don't care about absolute accuracy values:

```python
TARGET_STUDIES = [
    'zhang_2023',
    'beneyto-calabuig-2023',
    'jiang_2020',
    'velten_2021',
]
```

**Interpretation:**
- ARI = 0.80 for velten_2021 → ✅ Meaningful (has van Galen labels)
- ARI = 0.15 for zhang_2023 → ⚠️ May just mean label mismatch, not poor transfer!

**When this is OK:**
- If you're comparing methods relatively (SCimilarity vs Traditional)
- If both methods get similar (low) scores on zhang_2023, the comparison is still fair
- Your hypothesis is about **relative performance**, not absolute accuracy

---

## Recommendation

### Step 1: Inspect Your Data (REQUIRED!)

**Run this script to see what you actually have:**

```python
#!/usr/bin/env python
"""Check what annotations exist in target studies."""

import scanpy as sc
import pandas as pd

DATA_PATH = "data/AML_scAtlas.h5ad"

# Load data
adata = sc.read_h5ad(DATA_PATH)

# Reference study
ref_study = 'van_galen_2019'
ref_labels = adata[adata.obs['Study'] == ref_study].obs['cell_type_annotation'].unique()

print("="*80)
print("VAN GALEN REFERENCE LABELS")
print("="*80)
print(f"Study: {ref_study}")
print(f"Cell types: {len(ref_labels)}")
for label in sorted(ref_labels):
    count = (adata[adata.obs['Study'] == ref_study].obs['cell_type_annotation'] == label).sum()
    print(f"  {label}: {count:,} cells")

# Target studies
targets = ['zhang_2023', 'beneyto-calabuig-2023', 'jiang_2020', 'velten_2021', 'zhai_2022']

print("\n" + "="*80)
print("TARGET STUDIES")
print("="*80)

compatible_studies = []

for target in targets:
    if target not in adata.obs['Study'].values:
        print(f"\n{target}: ❌ NOT FOUND in atlas")
        continue

    subset = adata[adata.obs['Study'] == target]
    target_labels = subset.obs['cell_type_annotation'].unique()

    print(f"\n{target}:")
    print(f"  Cells: {len(subset):,}")
    print(f"  Cell types: {len(target_labels)}")

    # Show top labels
    print(f"  Top labels:")
    for label, count in subset.obs['cell_type_annotation'].value_counts().head(10).items():
        print(f"    {label}: {count:,} cells")

    # Check overlap with van Galen
    overlap = set(target_labels) & set(ref_labels)
    overlap_pct = len(overlap) / len(ref_labels) * 100

    print(f"  Overlap with van Galen: {len(overlap)}/{len(ref_labels)} labels ({overlap_pct:.0f}%)")

    if overlap_pct >= 50:
        print(f"  ✅ COMPATIBLE (≥50% label overlap)")
        compatible_studies.append(target)
    else:
        print(f"  ❌ INCOMPATIBLE (<50% label overlap)")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
if len(compatible_studies) > 0:
    print(f"\n✅ Use these studies with van Galen-compatible labels:")
    print("TARGET_STUDIES = [")
    for study in compatible_studies:
        print(f"    '{study}',")
    print("]")
else:
    print("\n⚠️ No studies found with van Galen-compatible labels!")
    print("   Consider using marker-based validation instead")
```

**Save this as: `inspect_target_studies.py`**

---

### Step 2: Update TARGET_STUDIES Based on Results

**After running the inspection script:**

**If you find 3+ compatible studies:**
```python
# Use all compatible studies
TARGET_STUDIES = ['velten_2021', 'zhai_2022', 'zhang_2023', ...]
```

**If you find only 2 compatible studies:**
```python
# Use confirmed van Galen studies only
TARGET_STUDIES = ['velten_2021', 'zhai_2022']
```

**If you find 0 compatible studies:**
```python
# Problem! Need to:
# 1. Use marker-based validation instead (validate_marker_expression.py)
# 2. Or manually map labels between annotation schemes
# 3. Or get data with van Galen annotations
```

---

## Why This Matters

### For Label-Based Validation (your current approach):

**You MUST have compatible labels!**

Otherwise:
- ❌ Accuracy scores are meaningless
- ❌ Can't tell if low ARI means poor transfer or label mismatch
- ❌ Can't compare to original study's results
- ❌ Reviewers will question the validity

### For Marker-Based Validation (alternative):

**You DON'T need matching labels!**

Instead, you:
- ✅ Define marker gene sets for each cell type
- ✅ Compute marker scores in target studies
- ✅ Check if predicted cell types have correct markers
- ✅ Works even with "Cluster_1, Cluster_2" labels

See: `validate_marker_expression.py`

---

## Summary

### Current Status: ⚠️ UNCERTAIN

Your `label_transfer_benchmark.py` uses:
- zhang_2023 → ❓ Unknown if compatible
- beneyto-calabuig-2023 → ❓ Unknown if compatible
- jiang_2020 → ❓ Unknown if compatible
- velten_2021 → ✅ Known compatible

### Immediate Action Required:

1. **Run inspection script** to check actual labels
2. **Update TARGET_STUDIES** to only include compatible studies
3. **Or switch to marker-based validation** if labels don't match

### Validated Approach:

Based on `validate_aml_subtypes.py` (which was designed specifically for this):
```python
REFERENCE_STUDY = 'van_galen_2019'
TARGET_STUDIES = [
    'velten_2021',   # ✅ Nature 2021, uses van Galen framework
    'zhai_2022',     # ✅ Nat Commun 2022, uses van Galen classifier
]
```

This is **scientifically rigorous** and **guaranteed to work**.

---

## Next Steps

1. Create `inspect_target_studies.py` (code provided above)
2. Run it: `python inspect_target_studies.py`
3. Look at the output to see which studies have compatible labels
4. Update `label_transfer_benchmark.py` accordingly
5. Or use marker-based validation if labels don't match

**Bottom line:** You were absolutely right to question the study selection! 🎯
