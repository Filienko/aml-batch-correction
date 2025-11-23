# Study Selection Summary - ACTION REQUIRED ⚠️

## 🚨 Critical Issue Found

**Your `label_transfer_benchmark.py` is using target studies that may NOT have van Galen-compatible annotations!**

This means the accuracy metrics could be meaningless.

---

## Current Situation

### Your Current Target Studies:
```python
TARGET_STUDIES = [
    'zhang_2023',              # ❓ Unknown if compatible
    'beneyto-calabuig-2023',   # ❓ Unknown if compatible
    'jiang_2020',              # ❓ Unknown if compatible
    'velten_2021',             # ✅ Known compatible
]
```

**Problem:** Only 1 out of 4 studies is confirmed to have van Galen annotations!

---

## What This Means

### The benchmark computes accuracy by comparing:
```python
y_true = target_study.labels  # Ground truth from atlas
y_pred = classifier.predict() # Your predictions
accuracy = compare(y_true, y_pred)
```

**If ground truth labels don't match van Galen's framework:**
- ❌ Accuracy scores are meaningless
- ❌ Can't interpret ARI/NMI/F1
- ❌ Comparison fails

### Example Problem:
```
van Galen predicts: 'HSC', 'GMP', 'Monocyte'
Target has labels:  'Cluster_1', 'Cluster_2', 'Cluster_3'
Comparison result:  ARI = 0.02 (meaningless!)
```

---

## Studies That Actually Use Van Galen Framework

Based on literature review (from `validate_aml_subtypes.py`):

### ✅ Confirmed Compatible:

**velten_2021**
- Paper: *Nature* 2021 "Tracking leukemia evolution"
- Status: ✅ **Explicitly uses van Galen classification**
- (~27,000 cells)

**zhai_2022**
- Paper: *Nature Communications* 2022 "AML with inv(16)"
- Status: ✅ **Used van Galen cell type classifier**
- (Note: You wrote "Zheng" but it's "Zhai")
- (Check if this study is in your atlas)

### ❌ Known Incompatible:

**pei_2020**
- Status: ❌ Uses own annotation, NOT van Galen

**setty_2019, oetjen_2018**
- Status: ❌ Published BEFORE van Galen (2019)

### ❓ Unknown:

**zhang_2023, beneyto-calabuig-2023, jiang_2020**
- Need to check actual labels in atlas

---

## What You Need to Do

### Option 1: Use Only Confirmed Studies (SAFEST) ⭐

Update `label_transfer_benchmark.py`:
```python
REFERENCE_STUDY = 'van_galen_2019'
TARGET_STUDIES = [
    'velten_2021',    # ✅ Confirmed
    'zhai_2022',      # ✅ Confirmed (if in atlas)
]
```

**Pros:**
- ✅ Scientifically rigorous
- ✅ Guaranteed to work
- ✅ Meaningful metrics

**Cons:**
- Only 2 targets (but that's fine!)

---

### Option 2: Run Inspection Script (RECOMMENDED) 🔍

**I created a script to check which studies actually have compatible labels:**

```bash
python inspect_target_studies.py
```

**This will:**
1. Load your AML scAtlas
2. Check van Galen's cell type labels
3. Check each target study's labels
4. Calculate % overlap
5. Tell you which studies are compatible
6. Give you the exact code to use

**Example output:**
```
velten_2021:
  ✅ HIGHLY COMPATIBLE (85% overlap)
  → Excellent for label transfer validation!

zhang_2023:
  ❌ INCOMPATIBLE (15% overlap)
  → Not suitable for label-based validation

RECOMMENDATION:
TARGET_STUDIES = [
    'velten_2021',   # ✅ 17/20 matching labels
    'zhai_2022',     # ✅ 16/20 matching labels
]
```

---

### Option 3: Use Marker-Based Validation (ALTERNATIVE)

**If no studies have compatible labels:**

Use `validate_marker_expression.py` instead!

**This approach:**
- ✅ Doesn't require matching labels
- ✅ Validates using marker gene expression
- ✅ Works even with "Cluster_1, Cluster_2" labels
- ✅ Still scientifically valid

**How it works:**
1. Define marker genes for each cell type (HSC: CD34, AVP; Monocyte: CD14, LYZ)
2. Compute marker scores in predicted cell types
3. Check if predicted HSCs express HSC markers
4. Validates biological correctness without label matching

---

## Action Plan

### STEP 1: Run Inspection Script ⚡
```bash
python inspect_target_studies.py
```

### STEP 2: Based on Results...

**If 2+ studies are compatible:**
```python
# Update label_transfer_benchmark.py with compatible studies
TARGET_STUDIES = ['velten_2021', 'zhai_2022', ...]
```

**If only 1 study is compatible:**
```python
# Use that one study, or add marker validation
TARGET_STUDIES = ['velten_2021']
```

**If 0 studies are compatible:**
```bash
# Use marker-based validation instead
python validate_marker_expression.py
```

### STEP 3: Run Your Benchmark
```bash
python label_transfer_benchmark.py
```

---

## Why This Matters

### For Your Paper:

**With compatible labels:**
> "We evaluated label transfer using studies that explicitly adopted van Galen's
> cell type framework (Velten et al. 2021, Zhai et al. 2022), achieving ARI of
> 0.85 vs 0.72 for traditional methods."

**With incompatible labels:**
> "Our evaluation achieved ARI of 0.15... wait, what does this mean? 😕"

---

## Quick Reference

| Study | Compatible? | Evidence |
|-------|------------|----------|
| velten_2021 | ✅ YES | Nature 2021, uses van Galen framework |
| zhai_2022 | ✅ YES | Nat Comm 2022, uses van Galen classifier |
| zhang_2023 | ❓ CHECK | Published after van Galen, unclear |
| beneyto-calabuig-2023 | ❓ CHECK | Published after van Galen, unclear |
| jiang_2020 | ❓ CHECK | Contemporary with van Galen, unclear |
| pei_2020 | ❌ NO | Uses own annotation scheme |
| setty_2019 | ❌ NO | Published before van Galen |

---

## Bottom Line

**You were absolutely right to question this!** 🎯

The current target study list is **not validated** and may give meaningless results.

**What to do RIGHT NOW:**

1. ✅ Run `python inspect_target_studies.py`
2. ✅ Update `TARGET_STUDIES` based on results
3. ✅ Or use marker-based validation
4. ✅ Then run your benchmark

---

## Files Created

- **`STUDY_SELECTION_ANALYSIS.md`** - Detailed analysis (10 pages)
- **`inspect_target_studies.py`** - Automated checking script ⭐
- **`STUDY_SELECTION_SUMMARY.md`** - This file (quick reference)

All committed and pushed to your branch!

---

**Next step: Run the inspection script! 🚀**
```bash
python inspect_target_studies.py
```
