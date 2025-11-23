# Quick Start: SCimilarity Annotation Replication

## 🎯 What This Does

Shows that **SCimilarity embeddings alone** can approximate the complex annotation pipeline used in the AML scAtlas:

**Expert Pipeline** (what the atlas did):
```
scVI integration
↓
CellTypist annotation
↓
SingleR annotation
↓
scType annotation
↓
Consensus of all three
↓
Manual marker curation
↓
Custom LSC annotation
= Final expert labels
```

**Your Pipeline** (what you're testing):
```
SCimilarity embeddings
↓
Simple Leiden clustering
= Compare to expert labels!
```

---

## 📊 Studies Used

Your script will use these 5 studies from the atlas:
- `van_galen_2019` - Core AML reference dataset
- `jiang_2020` - 10x Chromium
- `beneyto-calabuig-2023` - 10x Chromium
- `velten_2021` - Muta-Seq
- `zhang_2023` - The atlas paper's own dataset (gold standard)

---

## 🚀 How to Run

### 1. Make sure you have the data

```bash
ls data/AML_scAtlas.h5ad
# Should exist with the full atlas
```

### 2. Run the single script

```bash
python scimilarity_annotation_replication.py
```

That's it! The script does everything:
- ✅ Loads the atlas
- ✅ Subsets to your 5 studies
- ✅ Projects to SCimilarity
- ✅ Clusters
- ✅ Compares to expert annotations
- ✅ Generates all figures and metrics

**Runtime**: ~20-40 minutes depending on data size

---

## 📈 What You'll Get

### Figures

1. **`comparison_expert_vs_scimilarity.pdf`**
   - Side-by-side UMAP
   - Left: Expert annotations (ground truth)
   - Right: SCimilarity clusters

2. **`scimilarity_predicted_celltypes.pdf`**
   - UMAP with SCimilarity clusters labeled by predicted cell types
   - Shows which cluster maps to which cell type

3. **`confusion_matrix.pdf`**
   - Heatmap showing agreement between expert and SCimilarity
   - Diagonal = perfect agreement
   - Off-diagonal = misclassifications

### Metrics

1. **Key Scores**:
   - **ARI** (Adjusted Rand Index): 0-1, higher = better agreement
     - >0.7 = Excellent
     - >0.5 = Good
     - >0.3 = Moderate

   - **NMI** (Normalized Mutual Info): 0-1, higher = better
     - How much information is shared between partitions

2. **`cluster_to_celltype_mapping.csv`**:
   ```
   Cluster  Predicted_CellType    Purity  N_Cells
   0        HSC-like             0.92    1234
   1        GMP-like             0.87    2341
   2        Monocyte             0.95    3456
   ...
   ```
   - Shows which SCimilarity cluster represents which cell type
   - Purity = how "pure" the cluster is (% of dominant cell type)

3. **`marker_validation.csv`**:
   - Shows if correct marker genes are enriched
   - Fold change for known cell type markers

---

## 📊 Interpreting Results

### Excellent Result (ARI > 0.7)

```
RESULTS:
  ARI: 0.85
  NMI: 0.82
  Expert cell types: 15
  SCimilarity clusters: 16

✓✓ SUCCESS! SCimilarity closely approximates expert annotations!
```

**Interpretation**: Foundation model alone is equivalent to the complex multi-tool + manual pipeline!

**For your paper**:
> "SCimilarity achieved an ARI of 0.85, demonstrating that a single foundation model can approximate months of expert curation work combining scVI, CellTypist, SingleR, scType, and manual marker validation."

---

### Good Result (ARI 0.5-0.7)

```
RESULTS:
  ARI: 0.62
  NMI: 0.68

✓ GOOD! SCimilarity captures major cell type structure!
```

**Interpretation**: Foundation model captures the main cell types but may split/merge some subtypes differently than experts.

**For your paper**:
> "SCimilarity demonstrated good agreement (ARI=0.62) with expert annotations, successfully identifying major AML cell types without requiring complex consensus annotation pipelines."

---

### What Each Output Tells You

**1. Cluster-to-CellType Mapping**

High purity (>0.8) for most clusters = SCimilarity clusters are biologically meaningful

```
Cluster 3 → Monocyte (purity 0.95, 3456 cells)
```
→ Cluster 3 is 95% Monocytes according to expert labels = GOOD!

**2. Confusion Matrix**

- **Diagonal bright** = Good agreement
- **Off-diagonal spots** = Where SCimilarity disagrees with experts
  - Could be SCimilarity finding subtypes experts merged
  - Or SCimilarity merging subtypes experts separated

**3. Marker Validation**

```
HSC: Markers AVP, CD34, PROM1
     Fold change: 8.5x
```
→ HSC markers are 8.5x higher in HSC cells than others = Biologically valid!

---

## 🔧 Troubleshooting

### Problem: Wrong study names

If you get "0 cells" for a study:

```python
# Check actual study names in your atlas
import scanpy as sc
adata = sc.read_h5ad('data/AML_scAtlas.h5ad')
print(adata.obs['Study'].unique())
```

Then edit `STUDIES` list in the script to match exact names.

---

### Problem: Can't find cell type column

The script will ask you:
```
Enter the cell type annotation column name:
```

Check your atlas:
```python
print(adata.obs.columns.tolist())
```

Common names: `celltype`, `cell_type`, `annotation`, `CellType`

---

### Problem: Low ARI score (<0.3)

This could mean:
1. **Different granularity**: SCimilarity finds finer subtypes than experts
2. **Different grouping**: SCimilarity groups cell types differently
3. **Check resolution**: Try different Leiden resolution (0.3, 0.5, 0.8, 1.0)

Edit in script:
```python
adata = cluster_scimilarity_embeddings(adata, resolution=0.8)  # Try different values
```

---

### Problem: Out of memory

Reduce batch size:
```python
SCIMILARITY_BATCH_SIZE = 2500  # Default is 5000
```

---

## 💡 Tips for Best Results

1. **Check your data first**:
   ```bash
   # Quick check
   python -c "
   import scanpy as sc
   adata = sc.read_h5ad('data/AML_scAtlas.h5ad')
   print('Total cells:', adata.n_obs)
   print('Studies:', adata.obs['Study'].unique())
   print('Cell types:', adata.obs['celltype'].nunique())  # Adjust column name
   "
   ```

2. **Start with one study** to test quickly:
   ```python
   STUDIES = ['van_galen_2019']  # Just one for testing
   ```

3. **Tune resolution** if needed:
   - Lower (0.3) = Fewer, broader clusters
   - Higher (1.0) = More, finer clusters
   - Match the granularity of expert annotations

4. **Check zhang_2023 separately**:
   Since this is the atlas paper's own dataset, it should have the highest agreement

---

## 📝 For Your Paper

### Methods Section

> "We evaluated whether SCimilarity, a foundation model pre-trained on diverse single-cell datasets, could approximate the complex annotation pipeline used in the AML scAtlas. The original atlas employed scVI for batch correction, followed by consensus annotation using CellTypist, SingleR, and scType, with extensive manual curation based on marker gene expression and custom LSC annotation.
>
> We subset the atlas to five key studies (van Galen et al. 2019, Jiang et al. 2020, Beneyto-Calabuig et al. 2023, Velten et al. 2021, and Zhang et al. 2023) and projected cells into SCimilarity's pre-trained latent space without fine-tuning. We then performed Leiden clustering on SCimilarity embeddings and compared resulting clusters to expert annotations using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)."

### Results Section (if ARI > 0.7)

> "SCimilarity demonstrated strong agreement with expert annotations (ARI=X.XX, NMI=X.XX), successfully identifying all major AML cell types. Cluster purity analysis revealed that XX% of SCimilarity clusters showed >80% agreement with expert cell type labels. Notably, this was achieved using only foundation model embeddings, suggesting that months of manual curation work can be approximated by a single automated step."

### Results Section (if ARI 0.5-0.7)

> "SCimilarity showed good agreement with expert annotations (ARI=X.XX, NMI=X.XX), capturing major AML cell type categories. While some fine-grained subtypes were merged or split differently than in expert annotations, marker gene validation confirmed biological validity of SCimilarity-derived clusters (mean fold-change: X.XX)."

---

## ✅ Success Criteria

You've successfully shown SCimilarity can approximate expert annotation if:

1. **ARI > 0.5** ✓ Good agreement
2. **Most clusters have purity > 0.7** ✓ Clusters are meaningful
3. **Marker genes enriched (FC > 2)** ✓ Biologically valid
4. **Major cell types identified** ✓ All known types present

---

## 🎯 Next Steps After Running

1. **Examine the confusion matrix**:
   - Where does SCimilarity agree with experts?
   - Where does it disagree? (Could be interesting biology!)

2. **Check cluster purity**:
   - High purity = Clean agreement
   - Low purity = Mixed clusters (may need resolution tuning)

3. **Validate unexpected findings**:
   - If SCimilarity splits a cell type experts merged → check markers
   - Could be discovering novel subtypes!

4. **Compare zhang_2023 specifically**:
   - This is the atlas paper's own data
   - Should have highest agreement

5. **Write it up!**:
   - Use the figures and metrics for your paper
   - Emphasize the automation and time savings

---

## 📧 Questions?

If something doesn't work:
1. Check that study names exactly match your atlas
2. Check that cell type column name is correct
3. Try with just one study first
4. Check the error message carefully

Good luck! 🚀
