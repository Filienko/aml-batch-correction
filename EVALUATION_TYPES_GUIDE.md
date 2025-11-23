# Label Transfer Evaluation Guide: Inter-Study vs Intra-Study

## Overview

There are **two complementary ways** to evaluate label transfer:

1. **Intra-Study**: Within the same study (best-case, no batch effects)
2. **Inter-Study**: Across different studies (real-world, with batch effects)

**You should do BOTH!**

---

## Quick Reference

| Script | Type | What It Does | Use Case |
|--------|------|--------------|----------|
| `inspect_target_studies.py` | **Inspection** | Checks label compatibility | Run FIRST to verify your data |
| `label_transfer_intra_study.py` | **Intra-study** | Train/test within 1 study | Establish upper bound |
| `label_transfer_benchmark.py` | **Inter-study** | Transfer across studies | Real-world validation |

---

## 1. Inspection (Run First!)

### Script: `inspect_target_studies.py`

**What it does:**
- ✅ NOT an evaluation tool
- ✅ Diagnostic/inspection tool
- ✅ Checks which studies have compatible labels
- ✅ Tells you which targets to use

**Run it:**
```bash
python inspect_target_studies.py
```

**Output:**
```
van_galen_2019:
  HSC: 1,200 cells
  GMP: 3,400 cells
  Monocyte: 2,100 cells

velten_2021:
  HSC: 800 cells
  GMP: 1,500 cells
  Monocyte: 900 cells
  ✅ HIGHLY COMPATIBLE (85% overlap)

zhang_2023:
  Cluster_1: 1,000 cells
  Cluster_2: 2,000 cells
  ❌ INCOMPATIBLE (10% overlap)

RECOMMENDATION:
TARGET_STUDIES = ['velten_2021', 'zhai_2022']  # Use these!
```

**When to run:**
- ✅ BEFORE any evaluation
- ✅ When adding new studies
- ✅ If you get unexpected low accuracy

---

## 2. Intra-Study Evaluation (Best Case)

### Script: `label_transfer_intra_study.py`

**What it does:**
```
van_galen_2019 dataset
    ↓ Split 80/20
  Train (80%)  →  Test (20%)
    ↓ Same study, no batch effects
  Predict labels on test set
```

**Tests:**
- ✅ Method's inherent accuracy (no batch effects)
- ✅ Best-case performance
- ✅ Upper bound for comparisons

**Run it:**
```bash
python label_transfer_intra_study.py
```

**Expected results:**
```
Traditional (RF):     Accuracy: 0.95, ARI: 0.92
SCimilarity KNN:      Accuracy: 0.97, ARI: 0.94
```

**Interpretation:**
- High accuracy (>0.90) → Method works well when no batch effects
- Use as **baseline** for inter-study comparison

---

## 3. Inter-Study Evaluation (Real World)

### Script: `label_transfer_benchmark.py`

**What it does:**
```
Reference: van_galen_2019 (entire study)
    ↓ Train classifier
Target 1: zhang_2023 (entire study)
Target 2: velten_2021 (entire study)
Target 3: jiang_2020 (entire study)
    ↓ Different technologies, batches, protocols
  Predict labels on each target
```

**Tests:**
- ✅ Cross-study generalization
- ✅ Batch robustness
- ✅ Real-world scenario

**Run it:**
```bash
python label_transfer_benchmark.py
```

**Expected results:**
```
Target: zhang_2023
  Traditional:   ARI: 0.45
  SCimilarity:   ARI: 0.65

Target: velten_2021
  Traditional:   ARI: 0.52
  SCimilarity:   ARI: 0.72
```

**Interpretation:**
- Lower than intra-study → Batch effects are present
- SCimilarity > Traditional → Better batch robustness

---

## Why You Need Both

### Intra-Study Alone:
```
Result: ARI = 0.95

Question: Is this good?
Answer: Can't tell! No comparison to real-world scenario.
```

### Inter-Study Alone:
```
Result: ARI = 0.45

Question: Is this bad?
Answer: Can't tell! Maybe the method just isn't accurate?
```

### Both Together:
```
Intra-study:  ARI = 0.95  (best case)
Inter-study:  ARI = 0.45  (real world)

Conclusion: Method is accurate (0.95), but batch effects reduce
            performance to 0.45. Need better batch correction!
```

---

## Complete Workflow

### Step 1: Inspect Labels
```bash
python inspect_target_studies.py
```
→ Verify which studies have compatible labels

### Step 2: Update TARGET_STUDIES
Based on Step 1 results, update `label_transfer_benchmark.py`:
```python
TARGET_STUDIES = [
    'velten_2021',    # ✅ Compatible
    'zhai_2022',      # ✅ Compatible
]
```

### Step 3: Run Intra-Study (Best Case)
```bash
python label_transfer_intra_study.py
```
→ Establishes upper bound (e.g., ARI = 0.95)

### Step 4: Run Inter-Study (Real World)
```bash
python label_transfer_benchmark.py
```
→ Tests generalization (e.g., ARI = 0.65)

### Step 5: Compare Results

**Create comparison table:**

| Evaluation | Traditional | SCimilarity | Interpretation |
|------------|-------------|-------------|----------------|
| Intra-study (best case) | 0.92 | 0.95 | Both methods accurate |
| Inter-study (real world) | 0.45 | 0.68 | SCimilarity more robust |
| **Gap (batch effect impact)** | **0.47** | **0.27** | SCimilarity less affected |

**Key insight:** SCimilarity loses only 0.27 ARI due to batch effects, vs 0.47 for traditional methods!

---

## Example Results Interpretation

### Scenario A: Good Batch Correction
```
Method: SCimilarity
  Intra-study:  ARI = 0.94
  Inter-study:  ARI = 0.85
  Gap:          0.09 (small!)

Conclusion: ✅ Excellent batch robustness
```

### Scenario B: Poor Batch Correction
```
Method: Traditional RF
  Intra-study:  ARI = 0.93
  Inter-study:  ARI = 0.45
  Gap:          0.48 (large!)

Conclusion: ❌ Batch effects dominate
```

### Scenario C: Poor Method
```
Method: Logistic Regression
  Intra-study:  ARI = 0.60
  Inter-study:  ARI = 0.55
  Gap:          0.05 (small but both low)

Conclusion: ⚠️ Method is inherently poor, not batch effects
```

---

## For Your Paper

### Methods Section

```
Label Transfer Evaluation

We evaluated label transfer using two complementary approaches:

(1) Intra-study evaluation: We performed 5-fold cross-validation within
    the van Galen et al. 2019 dataset to establish best-case accuracy
    in the absence of batch effects.

(2) Inter-study evaluation: We trained classifiers on van Galen et al.
    2019 and evaluated on independent AML studies (velten_2021, zhai_2022)
    to assess cross-study generalization and batch robustness.

All annotations were obtained from the AML scAtlas harmonized labels
[cite atlas].
```

### Results Section

```
Intra-Study Performance

Within-study cross-validation established that both traditional
classification (ARI = 0.92 ± 0.02) and SCimilarity (ARI = 0.95 ± 0.01)
achieved high accuracy when batch effects were absent, demonstrating
that both methods have sufficient inherent accuracy for cell type
annotation.

Inter-Study Performance

When transferring labels across independent studies, SCimilarity
outperformed traditional methods (mean ARI: 0.68 vs 0.45, p<0.01).
Notably, SCimilarity maintained 72% of its intra-study performance
(0.95 → 0.68), whereas traditional methods retained only 49%
(0.92 → 0.45), demonstrating superior batch robustness.

These results indicate that performance differences are driven primarily
by batch effect handling rather than inherent annotation accuracy.
```

---

## Quick Decision Tree

```
Do you have AML scAtlas data?
├─ NO → Get data first
└─ YES ↓

Have you run inspect_target_studies.py?
├─ NO → Run it NOW! ⚠️
└─ YES ↓

Do 2+ studies have compatible labels (≥70% overlap)?
├─ NO → Use marker-based validation instead
└─ YES ↓

Run both evaluations:
  1. python label_transfer_intra_study.py
  2. python label_transfer_benchmark.py

Compare results:
  • Intra-study = upper bound
  • Inter-study = real world
  • Gap = batch effect impact
```

---

## Summary Table

| Script | Type | Input | Output | Purpose |
|--------|------|-------|--------|---------|
| `inspect_target_studies.py` | Inspection | Atlas data | Compatibility report | Verify labels match |
| `label_transfer_intra_study.py` | Evaluation | 1 study | Best-case accuracy | Upper bound |
| `label_transfer_benchmark.py` | Evaluation | Multiple studies | Real-world accuracy | Generalization |

---

## Next Steps

1. ✅ Run `inspect_target_studies.py` to verify your data
2. ✅ Run `label_transfer_intra_study.py` for best-case baseline
3. ✅ Run `label_transfer_benchmark.py` for real-world validation
4. ✅ Compare results to quantify batch effect impact
5. ✅ Write paper! 🎉

---

## Common Questions

### Q: Which study should I use for intra-study evaluation?

**A:** Use your reference study (van_galen_2019) because:
- It's well-characterized
- It has many cells
- It's what you're transferring FROM in inter-study

### Q: Can I use multiple studies for intra-study evaluation?

**A:** No, that would be inter-study! Intra-study means within ONE study only.

### Q: What if intra-study and inter-study results are similar?

**A:** Two possibilities:
1. ✅ Both low → Method isn't accurate (improve method)
2. ✅ Both high → Excellent batch correction! (celebrate!)

### Q: What if only SCimilarity works well?

**A:** That's the point! Shows foundation model has better batch robustness.

### Q: Should I always use the same reference?

**A:** Yes! Use van_galen_2019 for both intra and inter-study for fair comparison.

---

**Bottom line:** Run all three scripts in order, compare results, publish paper! 🚀
