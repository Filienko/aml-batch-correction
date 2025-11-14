#!/usr/bin/env Rscript
#
# Check NPM1 Seurat object for raw counts and van Galen labels
#
# This script inspects the Seurat object to determine if it can be used
# for validation against van Galen's framework.
#

cat("================================================================================\n")
cat("CHECKING NPM1 SEURAT OBJECT FOR VAN GALEN VALIDATION\n")
cat("================================================================================\n\n")

# Load required libraries
suppressPackageStartupMessages({
  if (!require("qs", quietly = TRUE)) {
    cat("Installing qs package...\n")
    install.packages("qs", repos = "https://cloud.r-project.org")
  }
  library(qs)

  if (!require("Seurat", quietly = TRUE)) {
    cat("Error: Seurat package not installed!\n")
    cat("Install with: install.packages('Seurat')\n")
    quit(status = 1)
  }
  library(Seurat)
})

# File path
seurat_file <- "data/AML.seurat.with.clusters.figshare.qs"

if (!file.exists(seurat_file)) {
  cat("✗ ERROR: File not found:", seurat_file, "\n")
  cat("\nExpected location: data/AML.seurat.with.clusters.figshare.qs\n")
  quit(status = 1)
}

cat("Loading Seurat object from:", seurat_file, "\n")
cat("This may take a minute for 83k cells...\n\n")

# Load Seurat object
seurat_obj <- qread(seurat_file)

cat("✓ Loaded successfully!\n\n")

# Basic info
cat("================================================================================\n")
cat("BASIC INFORMATION\n")
cat("================================================================================\n\n")

cat("Cells:", ncol(seurat_obj), "\n")
cat("Genes:", nrow(seurat_obj), "\n")

# Check assays
cat("\nAssays available:\n")
assays <- names(seurat_obj@assays)
for (assay in assays) {
  cat("  -", assay, "\n")
}

# Focus on RNA assay
if ("RNA" %in% assays) {
  cat("\n================================================================================\n")
  cat("RNA ASSAY INSPECTION\n")
  cat("================================================================================\n\n")

  rna <- seurat_obj@assays$RNA

  # Check slots/layers
  cat("Available slots/layers:\n")
  slot_names <- slotNames(rna)
  for (slot_name in slot_names) {
    if (slot_name %in% c("counts", "data", "scale.data")) {
      cat("  ✓", slot_name, "\n")
    }
  }

  # Check counts
  cat("\n" , "="*80, "\n")
  cat("CRITICAL: CHECKING FOR RAW COUNTS\n")
  cat("================================================================================\n\n")

  has_counts <- FALSE
  counts_max <- 0

  if ("counts" %in% slotNames(rna)) {
    counts_matrix <- slot(rna, "counts")

    if (length(counts_matrix) > 0) {
      has_counts <- TRUE
      counts_max <- max(counts_matrix)
      counts_min <- min(counts_matrix)

      # Sample some values
      nonzero <- counts_matrix[counts_matrix > 0]
      if (length(nonzero) > 1000) {
        sample_vals <- sample(nonzero, 1000)
      } else {
        sample_vals <- nonzero
      }

      cat("Counts slot:\n")
      cat("  Shape:", dim(counts_matrix)[1], "×", dim(counts_matrix)[2], "\n")
      cat("  Max value:", counts_max, "\n")
      cat("  Min value:", counts_min, "\n")
      cat("  Sparsity:", sprintf("%.1f%%", 100 * sum(counts_matrix == 0) / length(counts_matrix)), "\n")

      # Check if values are integers
      are_integers <- all(abs(sample_vals - round(sample_vals)) < 0.01)

      cat("\n")
      if (counts_max > 100 && are_integers) {
        cat("✓✓✓ THIS LOOKS LIKE RAW COUNTS! ✓✓✓\n")
        cat("  - Max value > 100 (", counts_max, ")\n", sep = "")
        cat("  - Values are integers\n")
        cat("  → SCimilarity can use this!\n")
      } else if (counts_max < 20) {
        cat("✗✗✗ WARNING: NOT RAW COUNTS ✗✗✗\n")
        cat("  - Max value < 20 (", counts_max, ")\n", sep = "")
        cat("  - This looks normalized/log-transformed\n")
        cat("  → SCimilarity CANNOT use this\n")
      } else {
        cat("⚠ UNCLEAR\n")
        cat("  - Max value:", counts_max, "\n")
        cat("  - Might be raw counts, might not\n")
      }
    }
  } else {
    cat("✗ No 'counts' slot found in RNA assay\n")
  }

  # Check data slot
  cat("\n")
  if ("data" %in% slotNames(rna)) {
    data_matrix <- slot(rna, "data")
    data_max <- max(data_matrix)

    cat("Data slot (normalized):\n")
    cat("  Max value:", data_max, "\n")

    if (data_max < 20) {
      cat("  → This is log-normalized data\n")
    }
  }
}

# Check metadata
cat("\n================================================================================\n")
cat("METADATA COLUMNS\n")
cat("================================================================================\n\n")

meta <- seurat_obj@meta.data
cat("Total columns:", ncol(meta), "\n\n")

# Show first 20 columns
cat("Available columns:\n")
col_names <- colnames(meta)
for (i in 1:min(20, length(col_names))) {
  col <- col_names[i]

  # Check if it's a cell type column
  if (grepl("cell.*type|cluster|annotation|leiden|louvain", col, ignore.case = TRUE)) {
    n_unique <- length(unique(meta[[col]]))
    cat("  ★", col, "(", n_unique, "unique values) ← Potential cell type labels\n")
  } else {
    cat("  -", col, "\n")
  }
}

if (length(col_names) > 20) {
  cat("  ... and", length(col_names) - 20, "more columns\n")
}

# Look for cell type annotations
cat("\n================================================================================\n")
cat("SEARCHING FOR VAN GALEN CELL TYPE LABELS\n")
cat("================================================================================\n\n")

# Common cell type column names
possible_cols <- c("cell_type", "celltype", "Cell_Type", "CellType",
                   "annotation", "Annotation", "cell.type", "cluster",
                   "seurat_clusters", "predicted.celltype")

found_cell_types <- FALSE

for (col in possible_cols) {
  if (col %in% colnames(meta)) {
    found_cell_types <- TRUE

    cat("Found:", col, "\n")

    # Get unique values
    labels <- table(meta[[col]])
    labels <- sort(labels, decreasing = TRUE)

    cat("  Total unique labels:", length(labels), "\n")
    cat("  Top labels:\n")

    # Show top 15
    for (i in 1:min(15, length(labels))) {
      label_name <- names(labels)[i]
      count <- labels[i]
      pct <- 100 * count / ncol(seurat_obj)
      cat(sprintf("    %s: %d (%.1f%%)\n", label_name, count, pct))
    }

    # Check for van Galen subtypes
    van_galen_subtypes <- c("HSPC", "HSC", "CMP", "GMP", "ProMono", "Promono",
                            "CD14+ Mono", "Monocyte", "cDC", "Dendritic")

    found_vg <- c()
    for (vg in van_galen_subtypes) {
      if (any(grepl(vg, names(labels), ignore.case = TRUE))) {
        found_vg <- c(found_vg, vg)
      }
    }

    if (length(found_vg) > 0) {
      cat("\n  ✓ Found van Galen-like labels:", paste(found_vg, collapse = ", "), "\n")
    } else {
      cat("\n  ✗ No obvious van Galen subtype labels\n")
    }

    cat("\n")
  }
}

if (!found_cell_types) {
  cat("✗ No cell type columns found with standard names\n")
  cat("  Check the column list above manually\n")
}

# Check for sample/patient info
cat("================================================================================\n")
cat("SAMPLE/PATIENT INFORMATION\n")
cat("================================================================================\n\n")

sample_cols <- c("sample", "Sample", "patient", "Patient", "orig.ident", "donor")

for (col in sample_cols) {
  if (col %in% colnames(meta)) {
    n_samples <- length(unique(meta[[col]]))
    cat("Found:", col, "→", n_samples, "unique samples\n")

    if (n_samples <= 20) {
      sample_counts <- table(meta[[col]])
      sample_counts <- sort(sample_counts, decreasing = TRUE)
      cat("  Sample sizes:\n")
      for (i in 1:min(10, length(sample_counts))) {
        cat(sprintf("    %s: %d cells\n", names(sample_counts)[i], sample_counts[i]))
      }
    }
    cat("\n")
  }
}

# Final summary
cat("================================================================================\n")
cat("SUMMARY FOR VAN GALEN VALIDATION\n")
cat("================================================================================\n\n")

can_use <- TRUE
issues <- c()

if (!has_counts) {
  can_use <- FALSE
  issues <- c(issues, "✗ No raw counts found")
} else if (counts_max < 100) {
  can_use <- FALSE
  issues <- c(issues, "✗ Counts appear normalized, not raw")
} else {
  cat("✓ Raw counts available (max =", counts_max, ")\n")
}

if (!found_cell_types) {
  issues <- c(issues, "⚠ Cell type labels unclear - need manual inspection")
  can_use <- FALSE
} else {
  cat("✓ Cell type labels found\n")
}

cat("\n")

if (can_use) {
  cat("="*80, "\n")
  cat("✓✓✓ THIS DATA CAN BE USED FOR VALIDATION! ✓✓✓\n")
  cat("="*80, "\n\n")

  cat("Next steps:\n")
  cat("  1. Convert to h5ad format:\n")
  cat("     source('convert_npm1_to_h5ad.R')\n")
  cat("  2. Add to validation pipeline\n")
  cat("  3. Run: python validate_aml_subtypes.py\n")
} else {
  cat("="*80, "\n")
  cat("✗✗✗ CANNOT USE THIS DATA ✗✗✗\n")
  cat("="*80, "\n\n")

  cat("Issues:\n")
  for (issue in issues) {
    cat(" ", issue, "\n")
  }

  cat("\nOptions:\n")
  cat("  1. Get raw data from EGA (accession: EGAS50000000332)\n")
  cat("  2. Contact authors for raw counts version\n")
  cat("  3. Use current validation with velten_2021 (already valid)\n")
}

cat("\n")
