# NPM1 AML Data: Raw Counts vs Processed Seurat

## The Paper

**Title**: "Resolving inter- and intra-patient heterogeneity in NPM1-mutated AML at single-cell resolution"

**Published**: Leukemia (Nature), January 2025

**Key Info:**
- ✅ **Cites van Galen et al. 2019** for cell type classification
- ✅ 16 NPM1-mutated AML samples
- ✅ 83,162 cells total
- ✅ Uses van Galen's myeloid cell hierarchy

**Data Availability:**
- **Raw data**: EGA accession **EGAS50000000332** (requires application)
- **Processed data**: Figshare (you have this: `AML.seurat.with.clusters.figshare.qs`)

## Your Situation

You have the Figshare Seurat object, but authors say it's **NOT raw counts**.

**Problem**: SCimilarity needs raw counts to work properly.

## Options

### Option 1: Check if Seurat Has Raw Counts Anyway

Sometimes "processed" Seurat objects still keep raw counts in the `@counts` layer.

**Run this:**
```bash
python check_npm1_seurat.py
```

This will give you R code to check if raw counts exist despite being "processed".

**If you find raw counts** (max value > 100):
- ✅ You can use it!
- Convert Seurat → h5ad
- Add to validation

**If NO raw counts** (max value < 20):
- ✗ Can't use for SCimilarity
- Proceed to Option 2

### Option 2: Get Raw Data from EGA

EGA (European Genome-phenome Archive) has the raw data.

**Accession**: EGAS50000000332

**Steps:**
1. **Register** at https://ega-archive.org/
2. **Apply for access** (requires:
   - Research justification
   - Ethics approval
   - Data Access Agreement
3. **Download raw fastq files**
4. **Process with CellRanger** (alignment, counting)
5. **Create AnnData** with raw counts

**Timeline**: 2-4 weeks for access approval + processing time

**Is it worth it?**
- ✅ YES if you want robust validation (83k cells, 16 samples)
- ✅ YES if you're writing a paper and need strong validation
- ❌ NO if you just want quick results

### Option 3: Use Velten 2021 Only

Your current validation with velten_2021 is already good:

```
Results:
scVI:        93.8%
SCimilarity: 88.6%
Uncorrected: 87.4%
Harmony:     85.9%
```

**Pros:**
- ✅ Already works
- ✅ Velten cites van Galen
- ✅ Cross-technology (Seq-Well → Muta-Seq)

**Cons:**
- ⚠️ Small test set (4,191 cells)
- ⚠️ Only 1 test study

**This is sufficient** for demonstrating the validation approach!

### Option 4: Find Other Studies That Cite Van Galen

Search for other papers that:
- Cite van Galen 2019
- Have publicly available raw counts
- Use van Galen's 6 cell type framework

**Databases to check:**
- GEO (Gene Expression Omnibus)
- SRA (Sequence Read Archive)
- Single Cell Portal

## My Recommendation

### For now: **Option 3** (Use Velten only)

Your current results are already valid:
- ✅ Shows scVI has highest accuracy (but has data leakage)
- ✅ Shows SCimilarity has good accuracy without leakage
- ✅ Shows "Uncorrected" PCA works well (batch effects not severe)

**What's missing**: Per-method marker enrichment comparison

### If you need stronger validation: **Option 2** (Get EGA raw data)

The NPM1 study would be **perfect** because:
- ✅ 83k cells (20x larger than Velten)
- ✅ 16 independent samples (vs 1 study)
- ✅ Explicitly uses van Galen framework
- ✅ Published in high-impact journal (Nature Leukemia)

But it requires time and effort to get raw data.

## Next Steps

### Immediate (5 minutes):
```bash
# Check if Seurat has raw counts
python check_npm1_seurat.py
# Run the R code it outputs
```

### If NO raw counts in Seurat:

**Option A - Quick**: Stick with Velten validation
- Fix marker enrichment comparison (I'll help)
- Document results
- Done!

**Option B - Thorough**: Apply for EGA access
- Register at EGA
- Apply for EGAS50000000332
- Wait 2-4 weeks
- Process raw data
- Run validation

## The Marker Enrichment Issue

Your current marker enrichment shows **overall** enrichment, not **per-method** comparison.

We need to see:
```
Method       HSPC_enrichment  Monocyte_enrichment
SCimilarity  8.2x            7.1x  ⭐ (best biology)
scVI         6.5x            6.8x
Uncorrected  5.2x            5.9x
Harmony      6.0x            6.5x
```

**I need to fix the validation script to compute this!**

Would you like me to:
1. Fix the per-method marker enrichment computation?
2. Help you check the Seurat object for raw counts?
3. Document the current Velten-only validation as complete?

Let me know which direction you want to go!
