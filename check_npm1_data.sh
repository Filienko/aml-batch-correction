#!/bin/bash
#
# Comprehensive check of NPM1 Seurat data
#
# This script:
# 1. Checks if R and required packages are available
# 2. Runs R script to inspect Seurat object
# 3. Optionally converts to h5ad if data is usable
#

set -e  # Exit on error

echo "================================================================================"
echo "NPM1 DATA CHECKER FOR VAN GALEN VALIDATION"
echo "================================================================================"
echo

# Check if R is available
if ! command -v Rscript &> /dev/null; then
    echo "✗ R not found!"
    echo "  Install R from: https://www.r-project.org/"
    exit 1
fi

echo "✓ R found: $(Rscript --version 2>&1 | head -1)"

# Check if file exists
SEURAT_FILE="data/AML.seurat.with.clusters.figshare.qs"
if [ ! -f "$SEURAT_FILE" ]; then
    echo
    echo "✗ Seurat file not found: $SEURAT_FILE"
    echo
    echo "Expected file from NPM1 AML paper (Leukemia 2025):"
    echo "  'Resolving inter- and intra-patient heterogeneity in NPM1-mutated AML'"
    echo
    echo "Download from paper's Figshare repository and place in data/"
    exit 1
fi

echo "✓ Seurat file found: $SEURAT_FILE"
FILE_SIZE_MB=$(du -m "$SEURAT_FILE" | cut -f1)
echo "  File size: ${FILE_SIZE_MB} MB"

echo
echo "================================================================================"
echo "STEP 1: INSPECTING SEURAT OBJECT"
echo "================================================================================"
echo

# Run inspection
Rscript inspect_npm1_seurat.R

INSPECT_STATUS=$?

if [ $INSPECT_STATUS -ne 0 ]; then
    echo
    echo "✗ Inspection failed or encountered issues"
    exit 1
fi

echo
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo

# Ask if user wants to convert
echo "The inspection is complete. Review the output above."
echo
echo "If raw counts were found and data looks usable:"
echo "  → Run: Rscript convert_npm1_to_h5ad.R"
echo "  → Then: python check_npm1_h5ad.py"
echo
echo "If no raw counts or data is not usable:"
echo "  → Apply for raw data from EGA: EGAS50000000332"
echo "  → Or continue with current validation (velten_2021 only)"
echo
