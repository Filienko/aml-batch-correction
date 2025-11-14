#!/usr/bin/env python3
"""
Check if NPM1 Seurat object has raw counts despite being "processed"
"""

import os

# Check if we need to convert from .qs (R quickserve format)
seurat_file = "data/AML.seurat.with.clusters.figshare.qs"

if os.path.exists(seurat_file):
    print("Found Seurat file:", seurat_file)
    print("\nTo check if it has raw counts, run this in R:")
    print("=" * 80)
    print("""
# Load the .qs file
library(qs)
library(Seurat)

# Load Seurat object
seurat_obj <- qread("data/AML.seurat.with.clusters.figshare.qs")

# Check what's in it
print("=== SEURAT OBJECT SUMMARY ===")
print(seurat_obj)

# Check assays
print("\n=== AVAILABLE ASSAYS ===")
print(names(seurat_obj@assays))

# Check if RNA assay has counts
print("\n=== RNA ASSAY LAYERS ===")
print(names(seurat_obj@assays$RNA))

# Check max values to see if raw counts
if ("counts" %in% names(seurat_obj@assays$RNA)) {
    max_counts <- max(seurat_obj@assays$RNA@counts)
    print(paste("\nMax value in @counts:", max_counts))
    if (max_counts > 100) {
        print("✓ This looks like RAW COUNTS! (values > 100)")
    } else {
        print("✗ This looks normalized (values < 100)")
    }
} else {
    print("\n✗ No 'counts' layer found")
}

# Check data layer
max_data <- max(seurat_obj@assays$RNA@data)
print(paste("\nMax value in @data:", max_data))
if (max_data < 20) {
    print("This is log-normalized data (values < 20)")
}

# Check metadata columns
print("\n=== METADATA COLUMNS ===")
print(colnames(seurat_obj@meta.data))

# Check for cell type annotations
if ("cell_type" %in% colnames(seurat_obj@meta.data)) {
    print("\n=== CELL TYPE LABELS ===")
    print(table(seurat_obj@meta.data$cell_type))
} else if ("seurat_clusters" %in% colnames(seurat_obj@meta.data)) {
    print("\n=== SEURAT CLUSTERS ===")
    print(table(seurat_obj@meta.data$seurat_clusters))
}

# Check sample info
if ("sample" %in% colnames(seurat_obj@meta.data)) {
    print("\n=== SAMPLES ===")
    print(table(seurat_obj@meta.data$sample))
} else if ("orig.ident" %in% colnames(seurat_obj@meta.data)) {
    print("\n=== SAMPLES (orig.ident) ===")
    print(table(seurat_obj@meta.data$orig.ident))
}
""")
    print("=" * 80)
    print("\nIf @counts exists with max > 100, you have raw counts!")
    print("If not, you need to get raw data from EGA.")
else:
    print(f"✗ File not found: {seurat_file}")
    print("\nMake sure the file is in the data/ directory")
