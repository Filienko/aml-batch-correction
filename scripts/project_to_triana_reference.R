#!/usr/bin/env Rscript
#
# Project single-cell data to the Triana et al. (2021) healthy bone-marrow
# reference and export projected cell-type labels as a CSV.
#
# Reference
# ---------
# Triana S et al. "Single-cell proteo-genomic reference maps of the
# hematopoietic system enable the purification and massive profiling of
# precisely defined cell states."  Nature Immunology (2021).
# https://doi.org/10.1038/s41590-021-01059-0
#
# Reference object (figshare)
# https://doi.org/10.6084/m9.figshare.13397651
#
# Projection method
# -----------------
# Adapted from the Projection_Vignette in
# https://git.embl.org/triana/nrn/-/tree/master/Projection_Vignette
# using scmap nearest-neighbour projection onto the MOFA UMAP embedding.
#
# Usage
# -----
#   Rscript project_to_triana_reference.R \
#       --input  /path/to/query.h5ad \
#       --output /path/to/output_labels.csv \
#       [--reference /path/to/triana_reference.rds] \
#       [--k 5] \
#       [--seed 42]
#
# Output CSV columns
# ------------------
#   barcode            : cell barcode (row name from the query object)
#   projected_celltype : nearest-neighbour majority cell type in the reference
#   correlation_score  : mean scmap correlation score (0–1; higher = better match)
#   pseudotime         : mean pseudotime of reference nearest neighbours
#                        (reference endpoint: "Myelocytes")
#
# The output CSV can be loaded by the Python experiment scripts instead of
# manually crafted harmonisation dictionaries.
#

suppressPackageStartupMessages({
  required <- c("optparse", "Seurat", "SingleCellExperiment", "scmap",
                "zellkonverter", "BiocParallel", "plyr")
  missing  <- required[!sapply(required, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing R packages: ", paste(missing, collapse = ", "), "\n",
      "Install with:\n",
      "  BiocManager::install(c('SingleCellExperiment', 'scmap', 'zellkonverter'))\n",
      "  install.packages(c('optparse', 'Seurat', 'plyr'))\n"
    )
  }
  library(optparse)
  library(Seurat)
  library(SingleCellExperiment)
  library(scmap)
  library(zellkonverter)
  library(BiocParallel)
  library(plyr)
})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
option_list <- list(
  make_option("--input",     type = "character", help = "Path to query .h5ad file [required]"),
  make_option("--output",    type = "character", help = "Path for output CSV [required]"),
  make_option("--reference", type = "character", default = NULL,
              help = "Path to Triana reference .rds Seurat object. If absent the script
              downloads the file from figshare into the same directory as --input."),
  make_option("--k",    type = "integer", default = 5,
              help = "Number of nearest neighbours for scmap projection [default: %default]"),
  make_option("--seed", type = "integer", default = 42,
              help = "Random seed [default: %default]")
)
parser <- OptionParser(option_list = option_list,
                       description = paste(
                         "Project query cells to the Triana et al. healthy",
                         "bone-marrow reference and output projected labels."
                       ))
args <- parse_args(parser)

if (is.null(args$input) || is.null(args$output)) {
  print_help(parser)
  stop("--input and --output are required.")
}

set.seed(args$seed)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Download the Triana reference from figshare if not already present.
#'
#' The dataset at https://doi.org/10.6084/m9.figshare.13397651 contains a
#' Seurat object of annotated healthy bone-marrow cells. We retrieve the
#' direct download link via the figshare API.
download_triana_reference <- function(dest_dir) {
  dest_file <- file.path(dest_dir, "triana_healthy_reference.rds")
  if (file.exists(dest_file)) {
    message("Using cached reference: ", dest_file)
    return(dest_file)
  }

  message("Fetching figshare file list for dataset 13397651 ...")
  api_url <- "https://api.figshare.com/v2/articles/13397651/files"
  resp    <- tryCatch(
    readLines(url(api_url), warn = FALSE),
    error = function(e) stop("Cannot reach figshare API: ", conditionMessage(e))
  )
  close(url(api_url))
  json <- paste(resp, collapse = "")

  # Extract download URLs for .rds files
  rds_urls <- regmatches(json, gregexpr('"download_url":"[^"]+\\.rds"', json))[[1]]
  if (length(rds_urls) == 0) {
    rds_urls <- regmatches(json, gregexpr('"download_url":"[^"]+\\.RDS"', json))[[1]]
  }
  if (length(rds_urls) == 0) {
    stop(
      "No .rds file found in figshare dataset 13397651.\n",
      "Please manually download the reference Seurat object from:\n",
      "  https://doi.org/10.6084/m9.figshare.13397651\n",
      "and pass it via --reference /path/to/reference.rds"
    )
  }
  dl_url <- sub('"download_url":"', "", rds_urls[[1]])
  dl_url <- sub('"$', "", dl_url)

  message("Downloading Triana reference from: ", dl_url)
  message("Destination: ", dest_file)
  download.file(dl_url, dest_file, mode = "wb", quiet = FALSE)
  message("Download complete.")
  dest_file
}


#' Project query cells onto the Triana healthy reference using scmap.
#'
#' Adapted from `project_anyref()` in the Triana Projection_Vignette.
#'
#' @param query_sce  SingleCellExperiment of query cells (log-normalised counts)
#' @param ref_seurat Seurat reference object with a MOFAUMAP dimensionality
#'                   reduction and cell-type metadata column "celltype"
#' @param k          Number of nearest neighbours
#' @param features   Character vector of gene names to use; if NULL, uses the
#'                   reference object's variable features
#' @return data.frame with columns: barcode, projected_celltype,
#'         correlation_score, pseudotime
project_to_reference <- function(query_sce, ref_seurat,
                                 k        = 5,
                                 features = NULL) {

  # Determine projection genes
  if (is.null(features)) {
    features <- VariableFeatures(ref_seurat)
    if (length(features) == 0) {
      stop("Reference Seurat object has no variable features. ",
           "Run FindVariableFeatures() first or supply --features.")
    }
    message("  Using ", length(features), " variable features from reference.")
  }

  # Subset features present in both
  common_genes <- intersect(features, rownames(query_sce))
  common_genes <- intersect(common_genes, rownames(ref_seurat))
  if (length(common_genes) < 50) {
    stop("Fewer than 50 genes in common between query and reference (",
         length(common_genes), " found). Check that gene names match.")
  }
  message("  Projecting using ", length(common_genes), " common genes.")

  query_sce_sub <- query_sce[common_genes, ]

  # Build reference SCE
  ref_counts <- GetAssayData(ref_seurat, slot = "data")  # log-normalised
  ref_sce    <- SingleCellExperiment(
    assays   = list(normcounts = as.matrix(ref_counts[common_genes, ])),
    colData  = ref_seurat@meta.data
  )
  logcounts(ref_sce) <- assay(ref_sce, "normcounts")

  # Determine which column holds cell-type labels
  ct_col <- intersect(c("celltype", "cell_type", "CellType", "Annotation",
                        "annotation", "label", "cluster"),
                      colnames(colData(ref_sce)))
  if (length(ct_col) == 0) {
    stop("Cannot find a cell-type column in the reference metadata.\n",
         "Available columns: ", paste(colnames(colData(ref_sce)), collapse = ", "))
  }
  ct_col <- ct_col[1]
  message("  Using reference cell-type column: '", ct_col, "'")

  # Determine pseudotime column
  pt_col <- intersect(c("Myelocytes", "pseudotime", "Pseudotime", "dpt_pseudotime"),
                      colnames(colData(ref_sce)))
  has_pseudotime <- length(pt_col) > 0
  if (has_pseudotime) pt_col <- pt_col[1]

  # scmap cell index on reference
  rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
  rowData(query_sce_sub)$feature_symbol <- rownames(query_sce_sub)

  ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE)
  ref_sce <- indexCell(ref_sce)

  # Run projection
  message("  Running scmap kNN projection (k=", k, ") ...")
  proj_result <- scmapCell(
    projection = query_sce_sub,
    index_list = list(ref = metadata(ref_sce)$scmap_cell_index),
    w          = k
  )
  nn_idx <- proj_result$ref$cells  # matrix: k × n_query

  # For each query cell: majority-vote cell type and mean scores
  n_query    <- ncol(query_sce_sub)
  proj_types <- character(n_query)
  proj_score <- numeric(n_query)
  proj_pt    <- if (has_pseudotime) numeric(n_query) else rep(NA_real_, n_query)

  ref_labels <- as.character(colData(ref_sce)[[ct_col]])
  if (has_pseudotime) ref_pt <- as.numeric(colData(ref_sce)[[pt_col]])
  ref_scores <- proj_result$ref$similarities  # matrix: k × n_query

  for (i in seq_len(n_query)) {
    nn       <- nn_idx[, i]
    valid    <- !is.na(nn) & nn > 0
    if (sum(valid) == 0) {
      proj_types[i] <- NA_character_
      proj_score[i] <- NA_real_
      if (has_pseudotime) proj_pt[i] <- NA_real_
      next
    }
    nn_v            <- nn[valid]
    proj_types[i]   <- names(sort(table(ref_labels[nn_v]), decreasing = TRUE))[1]
    proj_score[i]   <- mean(ref_scores[valid, i], na.rm = TRUE)
    if (has_pseudotime) proj_pt[i] <- mean(ref_pt[nn_v], na.rm = TRUE)
  }

  data.frame(
    barcode            = colnames(query_sce_sub),
    projected_celltype = proj_types,
    correlation_score  = proj_score,
    pseudotime         = proj_pt,
    stringsAsFactors   = FALSE
  )
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

message("=== Triana Reference Projection ===")
message("Query  : ", args$input)
message("Output : ", args$output)

# 1. Locate / download reference
if (!is.null(args$reference)) {
  ref_path <- args$reference
  if (!file.exists(ref_path)) stop("Reference file not found: ", ref_path)
} else {
  ref_path <- download_triana_reference(dirname(args$input))
}
message("\nLoading reference: ", ref_path)
ref_seurat <- readRDS(ref_path)
message("  Reference cells: ", ncol(ref_seurat))

# 2. Load query h5ad
message("\nLoading query: ", args$input)
query_sce <- readH5AD(args$input, use_hdf5 = FALSE)
message("  Query cells: ", ncol(query_sce))

# Ensure log-normalised counts are in logcounts assay
if (!"logcounts" %in% assayNames(query_sce)) {
  if ("X" %in% assayNames(query_sce)) {
    logcounts(query_sce) <- assay(query_sce, "X")
  } else {
    stop("Cannot find log-normalised expression in query h5ad.\n",
         "Available assays: ", paste(assayNames(query_sce), collapse = ", "))
  }
}

# 3. Run projection
message("\nRunning projection ...")
results <- project_to_reference(
  query_sce  = query_sce,
  ref_seurat = ref_seurat,
  k          = args$k
)

# 4. Write output
dir.create(dirname(args$output), showWarnings = FALSE, recursive = TRUE)
write.csv(results, args$output, row.names = FALSE, quote = TRUE)

message("\nDone. ", nrow(results), " cells written to: ", args$output)
n_na <- sum(is.na(results$projected_celltype))
if (n_na > 0) {
  message("  Warning: ", n_na, " cells could not be projected (NA in output).")
}
message("\nLabel distribution:")
print(sort(table(results$projected_celltype), decreasing = TRUE))
