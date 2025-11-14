#!/usr/bin/env Rscript
#
# Convert NPM1 Seurat object to h5ad format for Python/scanpy
#
# Run this ONLY if inspect_npm1_seurat.R confirmed raw counts exist!
#

cat("================================================================================\n")
cat("CONVERTING NPM1 SEURAT TO H5AD FORMAT\n")
cat("================================================================================\n\n")

# Load required libraries
suppressPackageStartupMessages({
  required_packages <- c("qs", "Seurat", "SeuratDisk")

  for (pkg in required_packages) {
    if (!require(pkg, quietly = TRUE, character.only = TRUE)) {
      cat("Installing", pkg, "...\n")
      if (pkg == "SeuratDisk") {
        if (!require("remotes", quietly = TRUE)) {
          install.packages("remotes")
        }
        remotes::install_github("mojaveazure/seurat-disk")
      } else {
        install.packages(pkg, repos = "https://cloud.r-project.org")
      }
      library(pkg, character.only = TRUE)
    }
  }
})

# File paths
seurat_file <- "data/AML.seurat.with.clusters.figshare.qs"
output_h5seurat <- "data/NPM1_AML.h5Seurat"
output_h5ad <- "data/NPM1_AML.h5ad"

if (!file.exists(seurat_file)) {
  cat("✗ ERROR: File not found:", seurat_file, "\n")
  quit(status = 1)
}

# Load Seurat object
cat("1. Loading Seurat object...\n")
seurat_obj <- qread(seurat_file)
cat("   ✓ Loaded:", ncol(seurat_obj), "cells ×", nrow(seurat_obj), "genes\n\n")

# Verify it has raw counts
if (!"counts" %in% slotNames(seurat_obj@assays$RNA)) {
  cat("✗ ERROR: No raw counts found in Seurat object!\n")
  cat("   This data cannot be used for SCimilarity.\n")
  quit(status = 1)
}

counts_max <- max(slot(seurat_obj@assays$RNA, "counts"))
if (counts_max < 100) {
  cat("✗ WARNING: Counts max =", counts_max, "< 100\n")
  cat("   This may not be raw counts!\n")
  cat("   Continue anyway? (y/n): ")
  response <- readline()
  if (tolower(response) != "y") {
    quit(status = 0)
  }
}

cat("   ✓ Raw counts verified (max =", counts_max, ")\n\n")

# Add study identifier
cat("2. Adding metadata...\n")
seurat_obj$Study <- "npm1_2024"
cat("   ✓ Added Study column\n\n")

# Save as h5Seurat
cat("3. Saving as h5Seurat format...\n")
SaveH5Seurat(seurat_obj, filename = output_h5seurat, overwrite = TRUE)
cat("   ✓ Saved:", output_h5seurat, "\n\n")

# Convert to h5ad
cat("4. Converting to h5ad format...\n")
Convert(output_h5seurat, dest = "h5ad", overwrite = TRUE)
cat("   ✓ Saved:", output_h5ad, "\n\n")

# Verify the output
cat("5. Verifying h5ad file...\n")
if (file.exists(output_h5ad)) {
  file_size_mb <- file.size(output_h5ad) / 1024^2
  cat("   ✓ File exists (", sprintf("%.1f", file_size_mb), " MB)\n", sep = "")
} else {
  cat("   ✗ Conversion failed!\n")
  quit(status = 1)
}

cat("\n================================================================================\n")
cat("✓ CONVERSION COMPLETE!\n")
cat("================================================================================\n\n")

cat("Output file:", output_h5ad, "\n")
cat("\nNext steps:\n")
cat("  1. Verify in Python:\n")
cat("     python check_npm1_h5ad.py\n\n")
cat("  2. Add to validation:\n")
cat("     Edit validate_aml_subtypes.py:\n")
cat("     TEST_STUDIES = ['velten_2021', 'npm1_2024']\n\n")
cat("  3. Run validation:\n")
cat("     python validate_aml_subtypes.py\n\n")
