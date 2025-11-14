# AML Atlas Replication with Foundation Models

## Research Question

**Can SCimilarity, a foundation model, automatically replicate the complex, multi-tool, manually-curated annotation pipeline used in the AML scAtlas - without any manual intervention?**

## Background

The AML scAtlas (159 patients) represents state-of-the-art expert curation:
- **Integration**: scVI batch correction
- **Annotation Pipeline**:
  - CellTypist (automated)
  - SingleR (reference-based)
  - scType (marker-based)
  - **Consensus** of all three tools
  - **Manual curation** with marker genes
  - **Custom LSC annotation** using SingleR with Zeng et al. reference
  - LSC6 and LSC17 score correlation

**Our Hypothesis**: SCimilarity's pre-trained latent space can achieve comparable or better results automatically.

## Project Phases

### Phase 1: Data and Model Setup

#### 1A. Load the "Ground Truth" Atlas
- **File**: `AML_scAtlas.h5ad` (159 patients, fully processed)
- **Key metadata**:
  - `dataset_of_origin`: Batch variable (van Galen, Abbas, Wang, etc.)
  - `cell_type_annotation`: Biology variable (LSPC-Quiescent, GMP-like, etc.)
- **Output**:
  - Figure 1A: "Gold Standard" UMAP showing cells clustered by biology, not batch

#### 1B. Acquire the "Raw" Problem Data
- **Sources**:
  1. van Galen et al. 2019 (GSE116256) - AML cell hierarchy reference
  2. Abbas et al. 2021 (GSE198052) - Additional AML cohort
  3. Wang et al. 2024 (GEO from atlas paper) - Recent AML data
- **Processing**:
  - Load original, unintegrated count matrices
  - Merge WITHOUT any integration or batch correction
- **Output**:
  - Figure 1B: "Problem" UMAP showing massive batch effects

#### 1C. Initialize Foundation Model
- Load SCimilarity with pre-trained weights (model_v1.1)

---

### Phase 2: SCimilarity Experiment (The FM Solution)

#### 2A. Project into Latent Space
- Take raw, unintegrated merged dataset
- Independently project every cell into SCimilarity latent space
- **No training or fine-tuning** - just inference

#### 2B. Visualize FM Solution
- Generate UMAP from SCimilarity embeddings
- Color by:
  - `dataset_of_origin` (batch)
  - `cell_type_annotation` (biology)
- **Output**: Figure 1C - "FM-Corrected Atlas"

---

### Phase 3: Quantitative Batch-Effect Benchmarking

**Goal**: Quantitatively compare Problem vs Gold Standard vs FM Solution

#### 3A. Batch Effect Removal (Mixing Metrics)
- **Metrics**: LISI (Local Inverse Simpson's Index) or kBET
- **Batch variable**: `dataset_of_origin`
- **Hypothesis**:
  - Problem: Score ≈ 0 (no mixing)
  - Gold Standard: High score
  - FM Solution: ≥ Gold Standard score

#### 3B. Biological Purity (Conservation Metrics)
- **Metrics**: ARI (Adjusted Rand Index) or NMI (Normalized Mutual Information)
- **Ground truth**: `cell_type_annotation` from atlas
- **Hypothesis**: FM Solution ≥ Gold Standard
  - **Key insight**: If FM > Gold, this suggests expert-curated integration may have "over-corrected" and merged biologically distinct subtypes
  - FM's generalized latent space may preserve finer biological distinctions

---

### Phase 4: Novel Biological Discovery

**Goal**: Use SCimilarity to approximate the manual annotation hierarchy work

#### 4A. AML scAtlas Annotation Process (What We're Replicating)

The atlas team did:
1. scVI integration
2. UMAP + Leiden clustering
3. CellTypist + SingleR + scType consensus
4. Manual marker gene curation
5. Custom LSC annotation with Zeng et al. reference
6. LSC6/LSC17 score correlation

**Our approach**: Skip all of that - just use SCimilarity embeddings

#### 4B. Re-interrogate the Hierarchy

The AML scAtlas identified:
- **12 recurrent aberrant differentiation patterns**
- **PC1**: Primitive vs. GMP (linked to chemo response)
- **PC2**: Primitive vs. Mature (linked to targeted drug response)

**Our experiment**:
1. Take SCimilarity latent space (FM Solution)
2. Calculate centroid (average embedding) for each van Galen/Zeng cell type
   - HSC-like, GMP-like, Monocyte-like, etc.
3. **Hierarchical clustering** of these centroids
4. **Compare** to expert-curated hierarchy from atlas
5. **Test**: Do we recover the same PC1/PC2 axes without manual work?

#### 4C. Cell Type Annotation Without Manual Curation

**Comparison**:

| AML scAtlas Pipeline | SCimilarity Pipeline |
|---------------------|---------------------|
| scVI integration | ❌ Skip |
| UMAP + Leiden | ❌ Skip |
| CellTypist | ❌ Skip |
| SingleR | ❌ Skip |
| scType | ❌ Skip |
| Consensus voting | ❌ Skip |
| Manual marker curation | ❌ Skip |
| Custom LSC reference | ❌ Skip |
| LSC6/LSC17 correlation | ❌ Skip |
| **TOTAL** | **Just use SCimilarity embeddings** |

**Validation**:
- Compare SCimilarity-based clusters to expert annotations
- Compute agreement (ARI/NMI)
- Check marker gene enrichment in SCimilarity clusters

---

## Expected Outcomes

### Best Case Scenario
1. **Batch mixing**: SCimilarity ≥ scVI integration
2. **Biology conservation**: SCimilarity > manual curation
   - This would suggest the FM preserves subtypes that manual process merged
3. **Hierarchy**: SCimilarity recovers the same PC1/PC2 axes
4. **Annotation**: High agreement with expert labels (ARI > 0.8)

### Impact
- **Demonstrates**: Foundation models can replace months of expert manual work
- **Validates**: Pre-trained models capture disease biology without fine-tuning
- **Practical**: Provides automated pipeline for future AML studies

---

## File Structure

```
aml-batch-correction/
├── ATLAS_REPLICATION_PROJECT.md           # This file
├── DATA_SOURCES.md                        # Data download guide
│
├── phase1_ground_truth.py                 # Load atlas + raw data, visualize
├── phase2_scimilarity_projection.py       # Project raw data to SCimilarity
├── phase3_quantitative_benchmark.py       # LISI, kBET, ARI, NMI
├── phase4_biological_discovery.py         # Hierarchy analysis
│
├── run_full_pipeline.py                   # Orchestrates all phases
│
├── data/                                  # Data directory (gitignored)
│   ├── AML_scAtlas.h5ad                  # Ground truth (159 patients)
│   ├── van_galen_2019_raw.h5ad           # Raw van Galen data
│   ├── abbas_2021_raw.h5ad               # Raw Abbas data
│   └── wang_2024_raw.h5ad                # Raw Wang data
│
└── results_atlas_replication/            # Results
    ├── figures/
    │   ├── fig1a_ground_truth_umap.pdf
    │   ├── fig1b_raw_problem_umap.pdf
    │   ├── fig1c_scimilarity_solution_umap.pdf
    │   ├── fig2_quantitative_comparison.pdf
    │   └── fig3_hierarchy_comparison.pdf
    │
    └── metrics/
        ├── batch_mixing_metrics.csv
        ├── bio_conservation_metrics.csv
        └── hierarchy_agreement.csv
```

---

## Data Requirements

### Primary Dataset
- **AML scAtlas**: 159 patients, ~XXX cells
  - Source: [Publication DOI/GEO]
  - File: `AML_scAtlas.h5ad`

### Raw Datasets (for "Problem")
1. **van Galen et al. 2019**
   - GEO: GSE116256
   - Cells: ~30,000
   - Technology: Seq-Well

2. **Abbas et al. 2021**
   - GEO: GSE198052
   - Cells: ~XX,XXX
   - Technology: 10x Genomics

3. **Wang et al. 2024**
   - GEO: [TBD from atlas paper]
   - Cells: ~XX,XXX
   - Technology: [TBD]

---

## Key Metrics

### Batch Mixing (Higher = Better)
- **LISI**: Local Inverse Simpson's Index
  - Perfect mixing = # of batches
  - No mixing = 1
- **kBET**: k-Nearest Neighbor Batch-Effect Test
  - Rejection rate (lower = better mixing)

### Biological Conservation (Higher = Better)
- **ARI**: Adjusted Rand Index (0-1)
  - Measures cluster agreement with ground truth
- **NMI**: Normalized Mutual Information (0-1)
  - Measures information preserved

### Novel Metric: "Over-Correction" Detection
If `ARI_SCimilarity > ARI_scVI`:
- Suggests scVI may have over-corrected
- FM preserves finer biological distinctions

---

## Timeline

1. **Week 1**: Data acquisition and Phase 1
2. **Week 2**: Phase 2 (SCimilarity projection)
3. **Week 3**: Phase 3 (Quantitative benchmarking)
4. **Week 4**: Phase 4 (Biological discovery)
5. **Week 5**: Manuscript writing

---

## Success Criteria

**Minimum Viable Result**:
- SCimilarity achieves batch mixing comparable to scVI
- Biological conservation ≥ 0.7 ARI

**Strong Result**:
- Batch mixing better than scVI
- Biological conservation > scVI (suggests over-correction detection)
- Recovers known hierarchy without manual work

**Blockbuster Result**:
- All of the above, PLUS
- Identifies novel subtypes not in atlas
- Validated with differential expression analysis

---

## References

1. **AML scAtlas**: [Citation TBD]
   - scVI + CellTypist + SingleR + scType + manual curation

2. **van Galen et al. (2019) Cell**
   - "A single-cell RNA-seq map of AML hierarchies"
   - Established LSC markers and hierarchy

3. **Zeng et al.** (referenced in atlas for LSC annotation)

4. **SCimilarity**: Heimberg et al. (2023) bioRxiv
   - Foundation model for single-cell annotation

5. **scIB Metrics**: Luecken et al. (2022) Nature Methods
   - Benchmarking framework
