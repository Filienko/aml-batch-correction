
"""
Configuration file for batch correction evaluation pipeline
"""

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_PATH = "data/AML_scAtlas.h5ad"
SCVI_PATH = "data/AML_scAtlas_X_scVI.h5ad.gz"
SCIMILARITY_MODEL_PATH = "models/model_v1.1"

# ============================================================================
# METADATA KEYS
# ============================================================================
# Batch key options: "sample" or "study"
BATCH_KEY = "Sample"
# Label key for cell types (check your .obs columns)
LABEL_KEY = "Cell Type"

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# Number of highly variable genes for uncorrected PCA
N_HVGS = 2000

# Number of parallel jobs for scIB benchmarking
N_JOBS = 8

# Output directory
OUTPUT_DIR = "batch_correction_results"

# ============================================================================
# METHODS TO EVALUATE
# ============================================================================
EVALUATE_METHODS = {
    'uncorrected': True,
    'scvi': True,
    'scimilarity': True,
}

# ============================================================================
# SCIB METRICS CONFIGURATION
# ============================================================================
# Configure which bio conservation metrics to compute
BIO_CONSERVATION_CONFIG = {
    'isolated_labels': True,
    'nmi_ari_cluster_labels_leiden': True,
    'nmi_ari_cluster_labels_kmeans': True,
}

