# Papers That Actually Use Van Galen's AML Framework

## Literature Review

After careful review, only specific papers actually **cite and use** van Galen et al. 2019's AML classification methodology:

### ✅ Confirmed Papers Using Van Galen Framework

1. **Velten et al. 2021** (Nature)
   - "Identification of leukemic and pre-leukemic stem cells by clonal tracking from single-cell transcriptomics"
   - **Explicitly uses** van Galen's AML cell type classification
   - Muta-Seq technology
   - 4,191 cells in our atlas

2. **Zhai et al. 2022** (Nature Communications / Leukemia)
   - "Molecular characterization of AML with t(8;21) by integrating genomic and transcriptomic analysis"
   - **Uses van Galen's classifier** to annotate malignant cells
   - SORT-Seq technology
   - **Need to check if in atlas**

### ❌ Papers That DON'T Actually Use Van Galen's Framework

These papers have the same *labels* (because the atlas standardized them) but didn't originally use van Galen's methodology:

1. **Setty et al. 2019** - Published BEFORE van Galen, couldn't have used it
2. **Pei et al. 2020** - Uses their own annotation, not van Galen's
3. **Oetjen et al. 2018** - Published BEFORE van Galen

## The Issue

The current validation uses studies that just happen to have matching labels due to **atlas harmonization**, not because they actually used van Galen's classification framework.

**This is a weaker test** than validating against papers that explicitly adopted van Galen's methodology.

## Recommended Validation Studies

### Primary Test: Velten 2021 ✅
- Confirmed to use van Galen framework
- 4,191 cells
- Muta-Seq (different technology from van Galen's Seq-Well)
- Has all 6 subtypes

### Secondary Test: Zhai 2022 (if available)
- Also uses van Galen framework
- SORT-Seq technology
- Need to confirm if in atlas

### Alternative: Cross-technology validation
If only Velten is available:
- Still valuable: tests Seq-Well (van Galen) → Muta-Seq (Velten)
- Can also do pairwise: van Galen vs Velten only
- More focused, cleaner experimental design

## Updated Validation Strategy

```python
# Only use papers that actually cited/used van Galen
TEST_STUDIES = [
    'velten_2021',  # ✅ Confirmed to use van Galen framework
    'zhai_2022',    # ✅ If available in atlas
]
```

This is a **stronger validation** because:
- These papers independently adopted van Galen's classification
- Tests if SCimilarity can reproduce their findings
- More aligned with original research question

## Why This Matters

**Original question**: "Use papers that look at the same latent subtypes as van Galen"

- ✅ **Correct interpretation**: Papers that *cite and use* van Galen's framework (Velten, Zhai)
- ❌ **Incorrect interpretation**: Any papers with matching labels (Setty, Pei, Oetjen)

The latter is just testing atlas harmonization, not biological reproducibility.
