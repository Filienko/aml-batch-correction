# Overleaf Setup Instructions

## Files Created

1. **paper_main.tex** - Main document structure
2. **paper_motivation.tex** - Introduction & motivation section
3. **paper_methods.tex** - Methods section
4. **paper_results.tex** - Results section
5. **paper_tables.tex** - All results tables
6. **paper_discussion.tex** - Discussion section with implications, limitations, and future directions
7. **paper_bibliography.bib** - Bibliography entries
8. **paper_figure_pipeline.tex** - TikZ figure showing traditional vs SCimilarity pipeline
9. **paper_figure_transfer.tex** - TikZ figure showing label transfer methodology

## How to Use in Overleaf

### Option 1: Upload All Files
1. Create new Overleaf project
2. Upload all 9 .tex and .bib files (including TikZ figures and discussion)
3. Set **paper_main.tex** as main document
4. Compile

### Option 2: Manual Copy-Paste
1. Create new Overleaf project
2. Open paper_main.tex and copy content into main.tex
3. Create new files for each section and paste content
4. Create bibliography.bib and paste bibliography content

## Tables Overview

The paper includes 7 tables:

1. **Table 1**: Per-dataset annotation replication performance
2. **Table 2**: Optimized single-study performance
3. **Table 3**: Resolution sweep analysis
4. **Table 4**: Intra-study baseline (no batch effects)
5. **Table 5**: Cross-study label transfer comparison (averaged)
6. **Table 6**: Per-target cross-study results
7. **Table 7**: Batch correction benchmarking using scib metrics ← NEW!

**Key Tables**:
- **Table 4**: Intra-study baseline showing performance WITHOUT batch effects. Establishes upper bound and quantifies batch effect degradation.
- **Table 7**: Direct batch correction benchmarking comparing SCimilarity, scVI, Harmony, and uncorrected data using standardized scib metrics. Shows SCimilarity is competitive with scVI (0.692 vs 0.729) without dataset-specific training.

## Citations to Update

**CRITICAL - UPDATE THESE:**

1. **scimilarity2023** - Currently placeholder, update with actual SCimilarity paper
   - Check if it's Heimberg et al. or different authors
   - Update journal, year, volume, pages

2. **amlAtlas2024** - UPDATE with actual AML scAtlas citation
   - Authors, title, journal
   - This is the dataset source

## Figures to Add

### TikZ Figures (Already Created!)

**Figure 1: Pipeline Comparison** - Add after Introduction section:
```latex
\input{paper_figure_pipeline.tex}
```
This shows the traditional (scVI + consensus) vs SCimilarity (direct embedding) pipelines.

**Figure 2: Label Transfer Methodology** - Add in Methods section:
```latex
\input{paper_figure_transfer.tex}
```
This visualizes the label transfer comparison between traditional RF and SCimilarity KNN.

### Optional: Data Figures from Results

If you want to add plots from your experimental results:

1. **Resolution sweep plot**
   - From: results_scimilarity_diagnostics/resolution_sweep.pdf
   - Shows optimal resolution selection

2. **Label transfer comparison**
   - From: results_label_transfer/figures/label_transfer_comparison.pdf
   - Bar charts comparing methods

3. **Heatmap**
   - From: results_label_transfer/figures/label_transfer_heatmap.pdf
   - ARI by target study

Add these with:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{resolution_sweep.pdf}
\caption{Your caption here}
\label{fig:resolution}
\end{figure}
```

## Compilation

Should compile with standard LaTeX:
- pdflatex paper_main.tex
- bibtex paper_main
- pdflatex paper_main.tex
- pdflatex paper_main.tex

Or just use Overleaf's compile button!

## Next Steps

1. Upload to Overleaf
2. Update citations (scimilarity2023, amlAtlas2024)
3. Fill in Table 5 with per-target values (already done, but verify)
4. Add optional data figures if desired (resolution sweep, heatmaps, etc.)
5. Proofread and adjust wording/framing as needed

## Tips

- Use \cite{vangalen2019} for citations
- Tables use booktabs package for professional appearance
- All results are already formatted with proper LaTeX
- Adjust wording/framing as needed for your story

Good luck!
