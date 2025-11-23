# Overleaf Setup Instructions

## Files Created

1. **paper_main.tex** - Main document structure
2. **paper_motivation.tex** - Introduction & motivation section
3. **paper_methods.tex** - Methods section
4. **paper_results.tex** - Results section
5. **paper_tables.tex** - All results tables
6. **paper_bibliography.bib** - Bibliography entries

## How to Use in Overleaf

### Option 1: Upload All Files
1. Create new Overleaf project
2. Upload all 6 .tex and .bib files
3. Set **paper_main.tex** as main document
4. Compile

### Option 2: Manual Copy-Paste
1. Create new Overleaf project
2. Open paper_main.tex and copy content into main.tex
3. Create new files for each section and paste content
4. Create bibliography.bib and paste bibliography content

## Tables That Need Updating

**Table 5 (Per-Target Results)** needs actual values from your results.

Get them from:
```bash
cat results_label_transfer/metrics/label_transfer_results.csv
```

Fill in the --- placeholders in paper_tables.tex with actual ARI and Macro F1 values for each target study.

## Citations to Update

**CRITICAL - UPDATE THESE:**

1. **scimilarity2023** - Currently placeholder, update with actual SCimilarity paper
   - Check if it's Heimberg et al. or different authors
   - Update journal, year, volume, pages

2. **amlAtlas2024** - UPDATE with actual AML scAtlas citation
   - Authors, title, journal
   - This is the dataset source

## Figures to Add

You should create figures from your results and reference them:

1. **Figure 1**: Resolution sweep plot
   - From: results_scimilarity_diagnostics/resolution_sweep.pdf
   - Shows optimal resolution selection

2. **Figure 2**: Label transfer comparison
   - From: results_label_transfer/figures/label_transfer_comparison.pdf
   - Bar charts comparing methods

3. **Figure 3**: Heatmap
   - From: results_label_transfer/figures/label_transfer_heatmap.pdf
   - ARI by target study

Add in Results section with:
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
3. Fill in Table 5 with per-target values
4. Add figures
5. Write Discussion section (I can help!)
6. Proofread and adjust

## Tips

- Use \cite{vangalen2019} for citations
- Tables use booktabs package for professional appearance
- All results are already formatted with proper LaTeX
- Adjust wording/framing as needed for your story

Good luck!
