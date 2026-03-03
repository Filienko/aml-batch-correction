"""
Cross-Dataset Transfer: Velten (AML blasts) ↔ Beneyto (AML + hematopoiesis)
=============================================================================

Scientific argument
-------------------
CellTypist assigns each cell to a **discrete gene-signature** — a fixed point
in expression space.  This works well for canonical, well-separated cell types
but degrades for cells that live on the *continuum* between states:
intermediate progenitors, cycling cells, or AML blasts that are arrested at
various stages of differentiation.

SCimilarity and similar foundation models learn a **continuous embedding
space** where biological gradients are preserved.  A cell that is
"between" an HSC and an early myeloid progenitor will land between those
clusters in the embedding, and the nearest-neighbour classifier will assign
it to the biologically closest label rather than snapping it to a distant
discrete signature.

Datasets
--------
Beneyto  (101 767 cells, 39 146 genes): large AML cohort with rich
         hematopoietic labels spanning normal progenitors → mature cells.
Velten   (  5 228 cells, 27 059 genes): AML dataset dominated by blast
         subtypes explicitly labelled as intermediate or unclear.

The blast subtypes in Velten are the key test cases:
  - "CD34+ Blasts and HSPCs"       explicitly intermediate between HSC & blast
  - "CD34- Blasts (Intermediate)"  explicitly intermediate / ambiguous
  - "Mitotic HSPCs (G2/M)"         cycling state confounds discrete signatures
  - "CD34+ Blasts / CD34+HBZ+"     progenitor-arrested blasts

Primary analysis: Train on Beneyto → Predict Velten
  A model trained on normal haematopoiesis must assign each AML blast to the
  normal progenitor stage it most resembles.  The continuous embedding should
  do this more accurately for ambiguous blasts than a discrete-signature
  classifier.

Supplementary: Train on Velten → Predict Beneyto (reversed roles).

Outputs (FIGURE_DIR)
---------------------
1. per_celltype_accuracy_<direction>.png
   Main story: per-cell-type accuracy for each original velten/beneyto
   cell type, intermediate types highlighted in orange.
2. overall_metrics_<direction>.png
   Accuracy / ARI / F1-macro for every method (bar chart).
3. continuous_advantage_<direction>.png
   Scatter: SCimilarity-mlp F1 (y) vs CellTypist F1 (x) per cell type —
   points above the diagonal show where continuous embeddings win.
4. intermediate_confusion_<direction>.png
   For each intermediate/blast type, stacked bar of predicted categories
   (SCimilarity vs CellTypist) to show biologically coherent vs incoherent
   errors.
5. umap_novel_<direction>.png
   UMAP of SCimilarity embeddings coloured by fine-grained label; novel /
   absent types drawn in bright colours, all other types in muted grey.
   Even when the classifier assigns the wrong harmonised label, tight clusters
   for novel types indicate the embedding preserves their biological identity.
6. knn_purity_<direction>.png
   kNN purity (fraction of k=30 nearest neighbours sharing the same
   fine-grained label) for each cell type in SCimilarity embedding vs PCA
   baseline.  Novel types are highlighted; high purity in SCimilarity despite
   low accuracy implies the model recognises the type even if it lacks a label.
7. centroid_separation_<direction>.png
   Distance between a novel type's embedding centroid and the nearest
   non-novel-type centroid.  Larger separation in SCimilarity vs PCA shows the
   foundation model places novel types in a distinct region of latent space.
"""

import sys
import warnings
import time
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, adjusted_rand_score, accuracy_score

from sccl import Pipeline
from sccl.models.celltypist import CellTypistModel

plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BENEYTO_PATH = Path("/home/daniilf/aml-batch-correction/data/beneyto.h5ad")
VELTEN_PATH  = Path("/home/daniilf/aml-batch-correction/data/velten.h5ad")
MODEL_PATH   = "/home/daniilf/aml-batch-correction/model_v1.1"

OUTPUT_DIR  = Path(__file__).parent / "results"
FIGURE_DIR  = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Cap reference/query to save memory (set None to use all cells)
MAX_REF_CELLS   = 20_000
MAX_QUERY_CELLS = None   # velten is only 5 228 — keep all

RUN_SCIMILARITY = True
RUN_CELLTYPIST  = True
RUN_SINGLER     = True   # requires the `singler` Python package

# ==============================================================================
# HARMONISED LABEL SPACE
# ==============================================================================
# Both datasets are mapped to an 11-category shared space that preserves
# biologically meaningful intermediate states.  The mapping is intentionally
# fine enough to distinguish key intermediate types while being coarse enough
# that both datasets have cells in most categories.

# Categories present only in Beneyto training data (models must generalise):
#   "Lymphomyeloid Progenitor", "Erythro-myeloid Progenitor",
#   "Monocyte", "Dendritic Cell"
# These are the categories where SCimilarity's continuous space should shine —
# Velten blast cells that resemble these states must be interpolated correctly.

BENEYTO_CT_TO_HARMONISED = {
    # Immature / early progenitors
    'HSCs & MPPs':                          'HSC/Progenitor',
    'NK cell progenitors':                  'HSC/Progenitor',
    # Intermediate bipotent progenitors — KEY types for this story
    'Lymphomyeloid prog':                   'Lymphomyeloid Progenitor',
    'Erythro-myeloid progenitors':          'Erythro-myeloid Progenitor',
    'Eosinophil-basophil-mast cell progenitors': 'Erythro-myeloid Progenitor',
    # Myeloid differentiation axis
    'Early promyelocytes':                  'Early Myeloid',
    'Late promyelocytes':                   'Late Myeloid',
    'Myelocytes':                           'Late Myeloid',
    'Classical Monocytes':                  'Monocyte',
    'Non-classical monocytes':              'Monocyte',
    'Monocyte-like blasts':                 'Monocyte',
    # Dendritic cells
    'Conventional dendritic cell 1':        'Dendritic Cell',
    'Conventional dendritic cell 2':        'Dendritic Cell',
    'Plasmacytoid dendritic cells':         'Dendritic Cell',
    'Plasmacytoid dendritic cell progenitors': 'Dendritic Cell',
    # Lymphoid
    'CD56dimCD16+ NK cells':               'NK Cell',
    'CD56brightCD16- NK cells':            'NK Cell',
    'NK T cells':                          'NK Cell',
    'CD4+ memory T cells':                 'T Cell',
    'CD4+ naive T cells':                  'T Cell',
    'CD8+ effector memory T cells':        'T Cell',
    'CD8+ central memory T cells':         'T Cell',
    'CD8+CD103+ tissue resident memory T cells': 'T Cell',
    'CD8+ naive T cells':                  'T Cell',
    'CD4+ cytotoxic T cells':              'T Cell',
    'GammaDelta T cells':                  'T Cell',
    'CD69+PD-1+ memory CD4+ T cells':     'T Cell',
    # B cell axis
    'Mature naive B cells':                'B Cell',
    'Nonswitched memory B cells':          'B Cell',
    'Class switched memory B cells':       'B Cell',
    'CD11c+ memory B cells':              'B Cell',
    'Plasma cells':                        'B Cell',
    'Pro-B cells':                         'B Progenitor',
    'Pre-B cells':                         'B Progenitor',
    'Immature B cells':                    'B Progenitor',
    'Pre-pro-B cells':                     'B Progenitor',
    'Small pre-B cell':                    'B Progenitor',
    # Erythroid / megakaryocyte axis
    'Early erythroid progenitor':          'Erythroid/MEP',
    'Late erythroid progenitor':           'Erythroid/MEP',
    'Aberrant erythroid':                  'Erythroid/MEP',
    'Megakaryocyte progenitors':           'Erythroid/MEP',
    # Exclude
    'Mesenchymal cells_1':                 None,
}

VELTEN_CT_TO_HARMONISED = {
    # Normal / committed types
    'HSC/MPPs':                            'HSC/Progenitor',
    'Mitotic HSPCs (G2/M)':               'HSC/Progenitor',     # cycling ← ambiguous
    'Neutrophil precursors':               'Late Myeloid',
    'MEP':                                 'Erythroid/MEP',
    'Erythroid precursors':               'Erythroid/MEP',
    'B cells and progenitors':            'B Progenitor',
    'NK cells':                           'NK Cell',
    'NK and T cells':                     'NK Cell',
    'Central memory T-cells':             'T Cell',
    'Effector memory T-cells':            'T Cell',
    'Cytotoxic T-cells':                  'T Cell',
    'Other T/NK cells':                   'T Cell',
    # Blast / intermediate types — the core test cases
    # CD34+ blasts: arrested at HSC/progenitor stage
    'CD34+ Blasts':                       'HSC/Progenitor',
    'CD34+HBZ+ Blasts':                   'HSC/Progenitor',
    # Transitional between HSC and committed progenitor — KEY INTERMEDIATE
    'CD34+ Blasts and HSPCs':             'Lymphomyeloid Progenitor',
    # CD34- blasts: variably differentiated along myeloid axis
    'CD34- Blasts (Calprotectin+AZU1-)':  'Early Myeloid',
    'CD34- Blasts (Intermediate)':         'Early Myeloid',      # KEY INTERMEDIATE
    'CD34- Blasts (Calprotectin+AZU1+)':  'Late Myeloid',
    'CD34- Blasts (Calprotectin-AZU1+)':  'Late Myeloid',
    'CD34- Blasts (Unclear)':             'Late Myeloid',
}

# Velten types classified as "intermediate" for highlighted visualisation
INTERMEDIATE_VELTEN = {
    'CD34+ Blasts and HSPCs',      # explicitly intermediate
    'CD34- Blasts (Intermediate)', # explicitly intermediate
    'Mitotic HSPCs (G2/M)',        # cell-cycle state confounds signature
}
# Velten blast types (arrested progenitors — not intermediate but still hard)
BLAST_VELTEN = {
    'CD34+ Blasts',
    'CD34+HBZ+ Blasts',
    'CD34- Blasts (Calprotectin+AZU1-)',
    'CD34- Blasts (Calprotectin+AZU1+)',
    'CD34- Blasts (Calprotectin-AZU1+)',
    'CD34- Blasts (Unclear)',
}

# Intermediate types in Beneyto (for velten→beneyto direction)
INTERMEDIATE_BENEYTO = {
    'Lymphomyeloid prog',
    'Erythro-myeloid progenitors',
    'Eosinophil-basophil-mast cell progenitors',
}

# Novel types: present in the query but absent (as fine-grained labels) from
# the reference dataset.  Even if the classifier assigns an incorrect
# harmonised label, tight distinct clusters in the embedding indicate the
# foundation model has captured the biology without supervision.
NOVEL_TYPES_VELTEN  = INTERMEDIATE_VELTEN | BLAST_VELTEN
NOVEL_TYPES_BENEYTO = INTERMEDIATE_BENEYTO


def _velten_type_category(t):
    """Return display category for a velten cell_type."""
    if t in INTERMEDIATE_VELTEN:
        return 'intermediate'
    if t in BLAST_VELTEN:
        return 'blast'
    return 'normal'


def _beneyto_type_category(t):
    if t in INTERMEDIATE_BENEYTO:
        return 'intermediate'
    return 'normal'


# ==============================================================================
# DATA LOADING
# ==============================================================================
def plot_trajectory_interpolation(emb_ref, emb_query, labels_ref, labels_query, 
                                  ct_prob_matrix, type_category_fn, direction, 
                                  start_state='HSC/Progenitor', end_state='Monocyte',
                                  output_dir=None):
    """
    Measures if intermediate blasts are correctly positioned between 
    HSCs and mature states, comparing SCimilarity distances vs CellTypist probabilities.
    """
    # 1. SCimilarity Continuous Trajectory
    # Find reference centroids
    valid_ref = pd.notna(labels_ref)
    emb_ref_v = emb_ref[valid_ref]
    labels_ref_v = labels_ref[valid_ref]
    
    if start_state not in labels_ref_v or end_state not in labels_ref_v:
        print(f"  Skipping interpolation plot: anchors {start_state} or {end_state} missing.")
        return
        
    c_start = emb_ref_v[labels_ref_v == start_state].mean(axis=0).reshape(1, -1)
    c_end = emb_ref_v[labels_ref_v == end_state].mean(axis=0).reshape(1, -1)
    
    # Calculate relative projection for query cells
    dist_to_start = cdist(emb_query, c_start, metric='euclidean').flatten()
    dist_to_end = cdist(emb_query, c_end, metric='euclidean').flatten()
    scim_progression = dist_to_start / (dist_to_start + dist_to_end)
    
    # 2. CellTypist Discrete Probability Trajectory
    # Assumes ct_prob_matrix is a DataFrame with column names matching reference states
    if start_state in ct_prob_matrix.columns and end_state in ct_prob_matrix.columns:
        p_start = ct_prob_matrix[start_state].values
        p_end = ct_prob_matrix[end_state].values
        # Add epsilon to avoid div-by-zero
        ct_progression = p_end / (p_start + p_end + 1e-9)
    else:
        ct_progression = np.full(len(emb_query), np.nan)

    # 3. Build DataFrame
    df = pd.DataFrame({
        'cell_type': labels_query,
        'category': [type_category_fn(t) for t in labels_query],
        'SCimilarity_Progression': scim_progression,
        'CellTypist_Progression': ct_progression
    })
    
    # Filter to intermediate and blast types
    df_focus = df[df['category'].isin(['intermediate', 'blast'])]
    if df_focus.empty: return
    summary_df = df_focus.groupby('cell_type')[['SCimilarity_Progression', 'CellTypist_Progression']].agg(['mean', 'std'])
    print(f"\n  [Trajectory Interpolation Summary - {direction}]")
    print(summary_df.to_string(float_format='{:.3f}'.format))
    
    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        summary_df.to_csv(output_dir / f"trajectory_summary_{safe}.csv")
        df_focus.to_csv(output_dir / f"trajectory_raw_{safe}.csv", index=False)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    sns.kdeplot(data=df_focus, x='SCimilarity_Progression', hue='cell_type', 
                fill=True, ax=axes[0], palette='YlOrRd', alpha=0.5)
    axes[0].set_title('SCimilarity: Continuous Spatial Projection', fontweight='bold')
    axes[0].set_xlabel(f'Relative Distance: {start_state} -> {end_state}')
    axes[0].set_xlim(0, 1)
    
    sns.kdeplot(data=df_focus, x='CellTypist_Progression', hue='cell_type', 
                fill=True, ax=axes[1], palette='YlOrRd', alpha=0.5)
    axes[1].set_title('CellTypist: Relative Probability', fontweight='bold')
    axes[1].set_xlabel(f'Relative Probability: {start_state} -> {end_state}')
    axes[1].set_xlim(0, 1)

    fig.suptitle(f'Trajectory Interpolation of Ambiguous Blasts\n{direction}', fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        plt.savefig(output_dir / f"trajectory_interpolation_{safe}.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cross_dataset_mixing(emb_ref, emb_query, labels_query, type_category_fn, 
                              direction, output_dir=None, k=30):
    """
    Evaluates foundation model batch-integration by measuring the fraction of 
    reference cells in the local neighborhood of query cells.
    """
    joint_emb = np.vstack([emb_ref, emb_query])
    is_ref = np.array([True] * len(emb_ref) + [False] * len(emb_query))
    
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(joint_emb)
    # Get neighbors for query cells only
    distances, indices = nn.kneighbors(emb_query)
    
    # Fraction of neighbors that belong to the reference dataset
    ref_fractions = is_ref[indices].mean(axis=1)
    
    df = pd.DataFrame({
        'cell_type': labels_query,
        'category': [type_category_fn(t) for t in labels_query],
        'ref_mixing_fraction': ref_fractions
    })
    
    # Sort for plotting: normal -> intermediate -> blast
    cat_order = ['normal', 'intermediate', 'blast']
    df['cat_ord'] = pd.Categorical(df['category'], categories=cat_order, ordered=True)
    df = df.sort_values(['cat_ord', 'cell_type'])
    summary_df = df.groupby(['category', 'cell_type'])['ref_mixing_fraction'].agg(['mean', 'median', 'std']).dropna()
    print(f"\n  [Cross-Dataset Mixing Summary (k={k}) - {direction}]")
    print(summary_df.to_string(float_format='{:.3f}'.format))
    
    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        summary_df.to_csv(output_dir / f"dataset_mixing_summary_{safe}.csv")
        df.to_csv(output_dir / f"dataset_mixing_raw_{safe}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='cell_type', y='ref_mixing_fraction', hue='category', 
                palette=_TYPE_COLORS, ax=ax, fliersize=1)
    
    ax.axhline(0.5, ls='--', c='black', alpha=0.3, label='Perfect Mixing')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Fraction of Reference Neighbors', fontweight='bold')
    ax.set_title(f'Cross-Dataset Neighborhood Integration (k={k})\n{direction}', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        plt.savefig(output_dir / f"dataset_mixing_{safe}.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_entropy(ct_prob_matrix, labels_query, type_category_fn, 
                            direction, output_dir=None):
    """
    Calculates Shannon entropy of the discrete classifier's probability distribution.
    High entropy implies the classifier is forcing a discrete label onto an ambiguous cell.
    """
    # Calculate Shannon entropy row-wise
    entropies = scipy.stats.entropy(ct_prob_matrix.values, axis=1)
    
    df = pd.DataFrame({
        'cell_type': labels_query,
        'category': [type_category_fn(t) for t in labels_query],
        'entropy': entropies
    })
    summary_df = df.groupby(['category', 'cell_type'])['entropy'].agg(['mean', 'median', 'std']).dropna()
    print(f"\n  [Prediction Entropy Summary - {direction}]")
    print(summary_df.to_string(float_format='{:.3f}'.format))
    
    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        summary_df.to_csv(output_dir / f"prediction_entropy_summary_{safe}.csv")
        df.to_csv(output_dir / f"prediction_entropy_raw_{safe}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x='category', y='entropy', palette=_TYPE_COLORS, 
                   inner="quartile", ax=ax)
    
    ax.set_ylabel('Shannon Entropy (bits)', fontweight='bold')
    ax.set_xlabel('Biological State', fontweight='bold')
    ax.set_title(f'CellTypist Prediction Ambiguity (Entropy)\n{direction}', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        plt.savefig(output_dir / f"prediction_entropy_{safe}.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_and_prep(path: Path, label_col: str, harmonised_map: dict,
                  max_cells: int = None, seed: int = 42) -> pd.DataFrame:
    """Load h5ad, apply harmonised label map, optionally subsample.

    Returns the AnnData object and a Series of harmonised labels aligned to
    adata.obs_names.
    """
    adata = sc.read_h5ad(path)

    # Use logcounts layer if .X is not already log-normalised
    if 'logcounts' in adata.layers:
        adata.X = adata.layers['logcounts']
        adata.uns['log1p'] = {}     # Scanpy expects a dictionary here

    # Map labels
    raw_labels = adata.obs[label_col].astype(str)
    harmonised = raw_labels.map(harmonised_map)  # NaN for unmapped
    adata.obs['_harmonised'] = harmonised
    adata.obs['_raw_label']  = raw_labels

    # Drop cells with no harmonised label (e.g. "Mesenchymal cells_1")
    keep = harmonised.notna()
    adata = adata[keep].copy()
    print(f"  {path.name}: {adata.n_obs:,} cells after removing unmapped types "
          f"({keep.sum():,} / {len(keep):,})")

    # Subsample if requested
    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  Subsampled to {adata.n_obs:,} cells")

    return adata


def align_genes(adata_ref, adata_query):
    """Subset both objects to their shared gene set (sorted for consistency)."""
    common = np.intersect1d(adata_ref.var_names, adata_query.var_names)
    print(f"  Gene intersection: {len(common):,} genes "
          f"(ref {adata_ref.n_vars:,}, query {adata_query.n_vars:,})")
    return adata_ref[:, common].copy(), adata_query[:, common].copy()


# ==============================================================================
# SCIMILARITY HELPERS  (reused from exp_ensemble_embeddings)
# ==============================================================================

def get_scimilarity_embeddings(adata, model_path):
    pipeline = Pipeline(model='scimilarity', model_params={'model_path': model_path})
    return pipeline.model.get_embedding(adata)


def _refine(embeddings, predictions, k=30):
    """kNN majority-vote smoothing on query's own embedding graph."""
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(embeddings)
    neighbours = nn.kneighbors(embeddings, return_distance=False)[:, 1:]  # drop self
    return np.array([Counter(predictions[idx]).most_common(1)[0][0]
                     for idx in neighbours])


# ==============================================================================
# SINGLER HELPER
# ==============================================================================

def log_normalize(X, target_sum=1e4):
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return np.log1p(X / row_sums * target_sum)


def run_singler(adata_ref, adata_query, label_col='_harmonised'):
    try:
        import singler
    except ImportError:
        print("    singler not installed; skipping.")
        return None

    ref_genes   = adata_ref.var_names.values
    query_genes = adata_query.var_names.values

    X_ref   = adata_ref.X
    X_query = adata_query.X

    # Normalise on FULL gene set before subsetting (library-size preservation)
    X_ref_norm   = log_normalize(X_ref)
    X_query_norm = log_normalize(X_query)

    common = np.intersect1d(ref_genes, query_genes)
    if len(common) < 100:
        print(f"    SingleR: only {len(common)} common genes — skipping.")
        return None

    ref_idx   = [np.where(ref_genes == g)[0][0]   for g in common]
    query_idx = [np.where(query_genes == g)[0][0] for g in common]

    X_ref_sub   = X_ref_norm[:, ref_idx]
    X_query_sub = X_query_norm[:, query_idx]

    ref_labels = adata_ref.obs[label_col].values
    valid = pd.notna(ref_labels)
    X_ref_sub   = X_ref_sub[valid]
    ref_labels  = ref_labels[valid]

    results = singler.annotate_single(
        test_data=X_query_sub.T,
        test_features=common,
        ref_data=X_ref_sub.T,
        ref_labels=ref_labels,
        ref_features=common,
        num_threads=8,
    )
    return np.asarray(results.column("best"))


# ==============================================================================
# METRICS
# ==============================================================================

def per_type_accuracy(y_true_fine, y_pred_harmonised, y_true_harmonised):
    """For each fine-grained true type, compute % cells correctly harmonised.

    Returns a DataFrame with columns [fine_type, accuracy, n_cells].
    """
    rows = []
    for ftype in np.unique(y_true_fine):
        mask  = y_true_fine == ftype
        n     = mask.sum()
        expected = y_true_harmonised[mask][0]  # all same by construction
        correct  = (y_pred_harmonised[mask] == expected).sum()
        rows.append({'cell_type': ftype, 'accuracy': correct / n, 'n_cells': n,
                     'expected_harmonised': expected})
    return pd.DataFrame(rows).sort_values('accuracy')


def macro_f1(y_true, y_pred):
    valid = pd.notna(y_true)
    return f1_score(np.asarray(y_true)[valid], np.asarray(y_pred)[valid],
                    average='macro', zero_division=0)


def ari(y_true, y_pred):
    valid = pd.notna(y_true)
    return adjusted_rand_score(np.asarray(y_true)[valid],
                               np.asarray(y_pred)[valid])


def accuracy(y_true, y_pred):
    valid = pd.notna(y_true)
    return accuracy_score(np.asarray(y_true)[valid],
                          np.asarray(y_pred)[valid])


# ==============================================================================
# VISUALISATIONS
# ==============================================================================

# Colour palette shared across figures
_PALETTE = {
    'SCimilarity-knn':  '#2196F3',
    'SCimilarity-logreg': '#1976D2',
    'SCimilarity-mlp':  '#0D47A1',
    'CellTypist':        '#FF7043',
    'SingleR':           '#4CAF50',
}
_TYPE_COLORS = {
    'intermediate': '#E65100',  # dark orange — the highlight
    'blast':        '#B71C1C',  # dark red
    'normal':       '#455A64',  # blue-grey
}


def plot_per_celltype_accuracy(results_dict, fine_labels, harmonised_true,
                               type_category_fn, direction, output_dir=None):
    """Horizontal bar chart: per-fine-cell-type accuracy for each method.

    Intermediate types are drawn with a warm highlight band.

    Parameters
    ----------
    results_dict : dict[method_name -> np.ndarray of harmonised predictions]
    fine_labels  : np.ndarray of original (fine) cell-type labels for query
    harmonised_true : np.ndarray of harmonised ground-truth labels for query
    type_category_fn : callable(fine_type) -> 'normal'|'intermediate'|'blast'
    direction : str  e.g. "Beneyto → Velten"
    output_dir : Path, optional
    """
    # Build per-type accuracy table for each method
    all_tables = {}
    for method, preds in results_dict.items():
        df = per_type_accuracy(fine_labels, preds, harmonised_true)
        df['category'] = df['cell_type'].map(type_category_fn)
        all_tables[method] = df

    # Align on cell types present in all methods
    common_types = list(all_tables[next(iter(all_tables))]['cell_type'])

    n_types   = len(common_types)
    n_methods = len(results_dict)
    bar_h     = 0.8 / n_methods
    fig_h     = max(6, n_types * 0.55)
    fig, ax   = plt.subplots(figsize=(12, fig_h))

    method_names = list(results_dict.keys())
    y_base = np.arange(n_types)

    # Draw highlight bands for intermediate types
    for i, ct in enumerate(common_types):
        cat = type_category_fn(ct)
        if cat == 'intermediate':
            ax.axhspan(i - 0.45, i + 0.45, color='#FFE0B2', alpha=0.4, zorder=0)
        elif cat == 'blast':
            ax.axhspan(i - 0.45, i + 0.45, color='#FFCDD2', alpha=0.25, zorder=0)

    # Draw bars
    offsets = np.linspace(-(n_methods - 1) * bar_h / 2,
                          (n_methods - 1) * bar_h / 2, n_methods)
    for j, method in enumerate(method_names):
        df = all_tables[method].set_index('cell_type').reindex(common_types)
        color = _PALETTE.get(method, f'C{j}')
        ax.barh(y_base + offsets[j], df['accuracy'], height=bar_h * 0.9,
                color=color, alpha=0.88, edgecolor='white', linewidth=0.4,
                label=method, zorder=2)

    # Axis formatting
    ax.set_yticks(y_base)
    ax.set_yticklabels(common_types, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Accuracy (fraction correctly harmonised)', fontsize=11,
                  fontweight='bold')
    ax.set_title(f'Per-Cell-Type Accuracy — {direction}\n'
                 f'(orange band = intermediate, red band = blast)',
                 fontsize=13, fontweight='bold', pad=10)

    # Colour the y-tick labels by category
    for label, ct in zip(ax.get_yticklabels(), common_types):
        cat = type_category_fn(ct)
        label.set_color(_TYPE_COLORS[cat])
        if cat in ('intermediate', 'blast'):
            label.set_fontweight('bold')

    # Legend: methods + category explanation
    method_patches = [mpatches.Patch(fc=_PALETTE.get(m, f'C{i}'), label=m, alpha=0.88)
                      for i, m in enumerate(method_names)]
    cat_patches = [
        mpatches.Patch(fc='#E65100', label='Intermediate type', alpha=0.7),
        mpatches.Patch(fc='#B71C1C', label='Blast type',        alpha=0.5),
        mpatches.Patch(fc='#455A64', label='Normal type',       alpha=0.7),
    ]
    ax.legend(handles=method_patches + cat_patches,
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
              frameon=True, title='Method / Category', title_fontsize=10)

    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn = output_dir / f"per_celltype_accuracy_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


def plot_continuous_advantage(results_dict, fine_labels, harmonised_true,
                              type_category_fn, direction,
                              ref_method='CellTypist',
                              cmp_method='SCimilarity-mlp',
                              output_dir=None):
    """Scatter: CellTypist accuracy (x) vs SCimilarity accuracy (y) per type.

    Points above the diagonal indicate types where the continuous embedding
    space outperforms discrete signatures.
    """
    if ref_method not in results_dict or cmp_method not in results_dict:
        print(f"    plot_continuous_advantage: {ref_method} or {cmp_method} missing.")
        return

    df_ref = per_type_accuracy(fine_labels, results_dict[ref_method], harmonised_true)
    df_cmp = per_type_accuracy(fine_labels, results_dict[cmp_method], harmonised_true)

    merged = df_ref.rename(columns={'accuracy': 'acc_ref'})\
                   .merge(df_cmp.rename(columns={'accuracy': 'acc_cmp'}),
                          on='cell_type')
    merged['category'] = merged['cell_type'].map(type_category_fn)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, zorder=0)
    ax.fill_between([0, 1], [0, 1], [1, 1], color='#E3F2FD', alpha=0.3,
                    label=f'{cmp_method} wins')
    ax.fill_between([0, 1], [0, 0], [0, 1], color='#FFEBEE', alpha=0.3,
                    label=f'{ref_method} wins')

    for cat, grp in merged.groupby('category'):
        color  = _TYPE_COLORS[cat]
        marker = {'intermediate': '*', 'blast': '^', 'normal': 'o'}[cat]
        sizes  = {'intermediate': 200,  'blast': 100,  'normal': 60}[cat]
        ax.scatter(grp['acc_ref'], grp['acc_cmp'],
                   c=color, marker=marker, s=sizes, alpha=0.85,
                   edgecolors='white', linewidths=0.5, label=cat, zorder=3)

    # Label intermediate and blast types
    for _, row in merged[merged['category'].isin(('intermediate', 'blast'))].iterrows():
        ax.annotate(row['cell_type'], (row['acc_ref'], row['acc_cmp']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7, color=_TYPE_COLORS[row['category']], zorder=4)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(f'{ref_method} accuracy', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{cmp_method} accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Continuous Embedding Advantage — {direction}\n'
                 f'Points above diagonal: foundation model wins',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, frameon=True)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn = output_dir / f"continuous_advantage_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


def plot_overall_metrics(metrics_list, direction, output_dir=None):
    """Grouped bar chart: accuracy, ARI, F1-macro for each method."""
    df = pd.DataFrame(metrics_list)
    metric_cols = ['accuracy', 'ari', 'f1_macro']
    label_map   = {'accuracy': 'Accuracy', 'ari': 'ARI', 'f1_macro': 'F1 Macro'}
    methods = df['method'].tolist()

    x     = np.arange(len(metric_cols))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 5))

    for j, method in enumerate(methods):
        row    = df[df['method'] == method].iloc[0]
        vals   = [row[m] for m in metric_cols]
        offset = (j - (len(methods) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=method, color=_PALETTE.get(method, f'C{j}'),
                        alpha=0.88, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in metric_cols], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Overall Metrics — {direction}',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn = output_dir / f"overall_metrics_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


def plot_intermediate_confusion(results_dict, fine_labels, harmonised_categories,
                                intermediate_types, direction, output_dir=None):
    """Stacked bar: for each intermediate cell type, distribution of predicted
    harmonised categories across methods.

    Shows whether model errors are biologically coherent (similar progenitors)
    or random (e.g. T-cell assigned to blast-like cell).
    """
    focus_types = [t for t in intermediate_types if t in np.unique(fine_labels)]
    if not focus_types:
        return

    all_cats = sorted(harmonised_categories)
    cmap     = plt.cm.get_cmap('tab20', len(all_cats))
    cat_color = {c: cmap(i) for i, c in enumerate(all_cats)}

    n_rows = len(focus_types)
    n_cols = len(results_dict)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             squeeze=False)

    for row_i, ct in enumerate(focus_types):
        mask = fine_labels == ct
        for col_j, (method, preds) in enumerate(results_dict.items()):
            ax = axes[row_i][col_j]
            pred_subset = preds[mask]
            counts      = Counter(pred_subset)
            total       = mask.sum()
            fracs       = [(c, counts.get(c, 0) / total) for c in all_cats]
            fracs       = [(c, f) for c, f in fracs if f > 0]

            left = 0.0
            for cat, frac in fracs:
                ax.barh(0, frac, left=left, height=0.5,
                        color=cat_color[cat], label=cat if row_i == 0 else None)
                if frac > 0.08:
                    ax.text(left + frac / 2, 0,
                            f'{frac:.0%}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')
                left += frac

            ax.set_xlim(0, 1)
            ax.set_ylim(-0.4, 0.4)
            ax.set_yticks([])
            ax.set_xlabel('Fraction of cells', fontsize=8)
            if col_j == 0:
                ax.set_ylabel(ct, fontsize=9, fontweight='bold',
                              color=_TYPE_COLORS.get('intermediate', '#E65100'))
            if row_i == 0:
                ax.set_title(method, fontsize=10, fontweight='bold')

    # Shared legend
    handles = [mpatches.Patch(color=cat_color[c], label=c) for c in all_cats]
    fig.legend(handles=handles, title='Predicted Category',
               bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=8,
               title_fontsize=9, frameon=True)

    fig.suptitle(f'Predicted Label Distribution for Intermediate Types\n{direction}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn = output_dir / f"intermediate_confusion_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


# ==============================================================================
# NOVEL TYPE SEPARATION ANALYSIS
# ==============================================================================

def compute_knn_purity(embeddings: np.ndarray, fine_labels, k: int = 30) -> dict:
    """Per-type mean kNN purity in an embedding space.

    For each cell, compute the fraction of its k nearest neighbours (excluding
    itself) that share the same fine-grained label.  Return the mean purity
    over all cells of each type.

    A high purity value for a novel type — even when the classifier assigns
    it the wrong harmonised label — shows the embedding has clustered those
    cells together, indicating latent recognition of the biology.

    Parameters
    ----------
    embeddings : ndarray, shape (n_cells, n_dims)
    fine_labels : array-like, shape (n_cells,)
    k : int
        Number of nearest neighbours (self excluded).

    Returns
    -------
    purity : dict  {cell_type -> mean_purity}
    """
    fine_labels = np.asarray(fine_labels)
    k_eff = min(k, len(fine_labels) - 1)
    nn   = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1).fit(embeddings)
    nbrs = nn.kneighbors(embeddings, return_distance=False)[:, 1:]  # drop self

    purity = {}
    for ct in np.unique(fine_labels):
        mask = fine_labels == ct
        if mask.sum() < 2:
            purity[ct] = 1.0
            continue
        nbr_labels  = fine_labels[nbrs[mask]]          # (n_ct, k_eff)
        cell_purity = (nbr_labels == ct).mean(axis=1)
        purity[ct]  = float(cell_purity.mean())
    return purity


def compute_centroid_separation(embeddings: np.ndarray, fine_labels,
                                novel_types: set) -> dict:
    """Relative centroid separation for each novel type.

    For each novel type, compute the distance from its centroid to the nearest
    non-novel centroid, then divide by the mean pairwise centroid distance
    across **all** types in this embedding.

    Normalising by the mean pairwise distance makes the metric dimensionless
    and comparable across embedding spaces (SCimilarity latent space vs PCA
    components): a value of 1.0 means "as far as types are on average", 2.0
    means "twice as far from the nearest normal centroid as types typically are
    from each other".  Without this normalisation, raw Euclidean distances in
    PCA space (large-variance gene expression) are always ~20–100× larger than
    those in the compact SCimilarity latent space, making comparison meaningless.

    Parameters
    ----------
    embeddings : ndarray, shape (n_cells, n_dims)
    fine_labels : array-like
    novel_types : set of str

    Returns
    -------
    separation : dict  {novel_type -> relative_separation}
        Values are dimensionless fractions of the mean inter-centroid distance.
    """
    from scipy.spatial.distance import cdist

    fine_labels  = np.asarray(fine_labels)
    all_types    = np.unique(fine_labels)
    normal_types = [t for t in all_types if t not in novel_types]

    if not normal_types:
        return {}

    centroids = {ct: embeddings[fine_labels == ct].mean(axis=0)
                 for ct in all_types if (fine_labels == ct).any()}

    # Normalisation factor: mean pairwise centroid distance in this space
    all_ctrs = np.stack(list(centroids.values()))
    pw       = cdist(all_ctrs, all_ctrs, metric='euclidean')
    n_ctrs   = len(all_ctrs)
    triu_idx = np.triu_indices(n_ctrs, k=1)
    mean_pw  = float(pw[triu_idx].mean()) if len(triu_idx[0]) > 0 else 1.0
    mean_pw  = max(mean_pw, 1e-10)   # guard against degenerate embeddings

    normal_centroids = np.stack([centroids[t] for t in normal_types
                                 if t in centroids])

    separation = {}
    for ct in novel_types:
        if ct not in centroids:
            continue
        dists          = cdist(centroids[ct].reshape(1, -1),
                               normal_centroids, metric='euclidean')[0]
        separation[ct] = float(dists.min()) / mean_pw
    return separation


def plot_umap_novel_types(emb_2d: np.ndarray, fine_labels, novel_types: set,
                          direction: str, output_dir=None):
    """UMAP scatter: novel/absent types in bright colours, all others muted grey.

    Parameters
    ----------
    emb_2d : ndarray, shape (n_cells, 2)   pre-computed 2-D UMAP coordinates
    fine_labels : array-like
    novel_types : set of str   types to highlight
    direction : str
    output_dir : Path, optional
    """
    fine_labels   = np.asarray(fine_labels)
    present_novel = sorted(t for t in novel_types if t in fine_labels)

    cmap_bright  = plt.cm.get_cmap('tab10', max(len(present_novel), 1))
    novel_colors = {t: cmap_bright(i) for i, t in enumerate(present_novel)}

    fig, ax = plt.subplots(figsize=(9, 7))

    # Background: non-novel cells in light grey
    bg_mask = ~np.isin(fine_labels, present_novel)
    ax.scatter(emb_2d[bg_mask, 0], emb_2d[bg_mask, 1],
               c='#BDBDBD', s=3, alpha=0.25, rasterized=True, label='Other types')

    # Foreground: novel types in bright, distinct colours
    for ct in present_novel:
        mask = fine_labels == ct
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[novel_colors[ct]], s=18, alpha=0.80,
                   label=ct, rasterized=True, zorder=3)

    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(f'SCimilarity Embedding — Novel / Absent Types\n{direction}',
                 fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
              markerscale=3, frameon=True,
              title='Novel types (bright)\nOther types (grey)',
              title_fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn   = output_dir / f"umap_novel_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


def plot_knn_purity_comparison(purity_scim: dict, purity_pca: dict,
                                fine_labels, novel_types: set,
                                direction: str, output_dir=None):
    """Grouped horizontal bar: kNN purity per cell type, SCimilarity vs PCA.

    Novel/absent types are highlighted with an orange background band.
    High purity in SCimilarity despite low classification accuracy is the
    key signal: the model recognises the type even without a training label.

    Parameters
    ----------
    purity_scim, purity_pca : dict  {cell_type -> mean_purity}
    fine_labels : array-like
    novel_types : set of str
    direction : str
    output_dir : Path, optional
    """
    fine_labels = np.asarray(fine_labels)
    all_types   = sorted((t for t in purity_scim if t in purity_pca),
                         key=lambda t: purity_scim[t])

    n  = len(all_types)
    bh = 0.35
    y  = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.45)))

    # Highlight novel types with a warm band
    for i, ct in enumerate(all_types):
        if ct in novel_types:
            ax.axhspan(i - 0.48, i + 0.48, color='#FFE0B2', alpha=0.45, zorder=0)

    scim_vals = [purity_scim[t] for t in all_types]
    pca_vals  = [purity_pca[t]  for t in all_types]

    ax.barh(y + bh / 2, scim_vals, bh, label='SCimilarity',
            color='#1976D2', alpha=0.85, edgecolor='white', linewidth=0.4, zorder=2)
    ax.barh(y - bh / 2, pca_vals,  bh, label='PCA baseline',
            color='#78909C', alpha=0.75, edgecolor='white', linewidth=0.4, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(all_types, fontsize=8)
    for label, ct in zip(ax.get_yticklabels(), all_types):
        if ct in novel_types:
            label.set_color(_TYPE_COLORS.get('intermediate', '#E65100'))
            label.set_fontweight('bold')

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Mean kNN purity (k=30)', fontsize=11, fontweight='bold')
    ax.set_title(f'kNN Purity — SCimilarity vs PCA Baseline\n{direction}\n'
                 f'(orange band = novel/absent types; '
                 f'high purity → tighter cluster in embedding)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=10, frameon=True, loc='lower right')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn   = output_dir / f"knn_purity_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


def plot_centroid_separation(sep_scim: dict, sep_pca: dict,
                              novel_types: set, direction: str,
                              output_dir=None):
    """Grouped bar: relative centroid separation, SCimilarity vs PCA baseline.

    Each bar shows the distance from the novel type's centroid to its nearest
    non-novel centroid, divided by the mean pairwise centroid distance in that
    embedding space.  This normalisation makes SCimilarity and PCA values
    directly comparable despite their different coordinate scales.

    Values > 1.0 mean the novel type sits farther from normal types than the
    average inter-type centroid distance — i.e. it forms a genuinely distinct
    cluster.

    Parameters
    ----------
    sep_scim, sep_pca : dict  {cell_type -> relative_separation}
    novel_types : set of str
    direction : str
    output_dir : Path, optional
    """
    types = sorted(t for t in sep_scim if t in sep_pca)
    if not types:
        return

    x = np.arange(len(types))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(types) * 1.4), 5))

    scim_vals = [sep_scim[t] for t in types]
    pca_vals  = [sep_pca[t]  for t in types]

    ax.bar(x - w / 2, scim_vals, w, label='SCimilarity',
           color='#1976D2', alpha=0.85, edgecolor='white')
    ax.bar(x + w / 2, pca_vals,  w, label='PCA baseline',
           color='#78909C', alpha=0.75, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha='right', fontsize=9)
    for label in ax.get_xticklabels():
        if label.get_text() in novel_types:
            label.set_color(_TYPE_COLORS.get('intermediate', '#E65100'))
            label.set_fontweight('bold')

    ax.set_ylabel('Relative separation\n(× mean pairwise centroid distance)',
                  fontsize=11, fontweight='bold')
    ax.set_title(f'Relative Centroid Separation — {direction}\n'
                 f'(normalised to mean inter-centroid distance; '
                 f'>1.0 → more distinct than average)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10, frameon=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        safe = direction.replace(' ', '_').replace('→', 'to').replace('/', '_')
        fn   = output_dir / f"centroid_separation_{safe}.png"
        plt.savefig(fn, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fn.name}")
    else:
        plt.show()


def analyze_novel_type_separation(adata_query, emb_scimilarity: np.ndarray,
                                   novel_types: set, direction: str,
                                   output_dir=None):
    """Test whether SCimilarity embeddings cluster novel/absent types distinctly.

    Even when a classifier assigns an incorrect harmonised label (because the
    type was absent from training), tight distinct clusters in the embedding
    space show that the foundation model has captured the underlying biology.

    Produces three figures:
      * UMAP coloured by novel types (if umap-learn is available)
      * kNN purity comparison (SCimilarity vs PCA baseline)
      * Centroid separation comparison (SCimilarity vs PCA baseline)

    Parameters
    ----------
    adata_query : AnnData
        Query dataset with obs['_raw_label'] (original fine-grained labels)
        and .X containing the expression matrix used for the PCA baseline.
    emb_scimilarity : np.ndarray, shape (n_cells, n_dims)
        SCimilarity embeddings of the query cells.
    novel_types : set of str
        Fine-grained types absent from the reference (training) dataset.
    direction : str
        Direction label used for plot titles and file names.
    output_dir : Path, optional
    """
    import scipy.sparse
    from sklearn.decomposition import PCA

    fine_labels   = adata_query.obs['_raw_label'].values
    present_novel = sorted(t for t in novel_types if t in fine_labels)

    if not present_novel:
        print(f"  No novel types found in query for '{direction}'. Skipping.")
        return

    print(f"\n  Novel type separation analysis ({direction}):")
    print(f"  Types analysed: {present_novel}")

    # ---- PCA baseline on raw query expression ----
    X_raw = adata_query.X
    if scipy.sparse.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw   = np.asarray(X_raw, dtype=np.float32)
    n_comps = min(50, X_raw.shape[1] - 1, X_raw.shape[0] - 1)
    emb_pca = PCA(n_components=n_comps, random_state=42).fit_transform(X_raw)
    print(f"  PCA baseline: {emb_pca.shape}")

    # ---- kNN purity ----
    print("  Computing kNN purity (k=30)...")
    purity_scim = compute_knn_purity(emb_scimilarity, fine_labels, k=30)
    purity_pca  = compute_knn_purity(emb_pca,          fine_labels, k=30)
    for ct in present_novel:
        print(f"    {ct:50s}  "
              f"SCimilarity={purity_scim.get(ct, float('nan')):.3f}  "
              f"PCA={purity_pca.get(ct, float('nan')):.3f}")

    # ---- Centroid separation ----
    print("  Computing centroid separation...")
    sep_scim = compute_centroid_separation(emb_scimilarity, fine_labels,
                                           set(present_novel))
    sep_pca  = compute_centroid_separation(emb_pca, fine_labels,
                                           set(present_novel))
    for ct in present_novel:
        print(f"    {ct:50s}  "
              f"SCimilarity rel={sep_scim.get(ct, float('nan')):.3f}  "
              f"PCA rel={sep_pca.get(ct, float('nan')):.3f}  "
              f"(× mean pairwise centroid dist)")

    # ---- UMAP (optional) ----
    try:
        import umap as umap_mod
        print("  Computing UMAP...")
        reducer = umap_mod.UMAP(n_components=2, random_state=42,
                                n_jobs=1, verbose=False)
        emb_2d  = reducer.fit_transform(emb_scimilarity)
        plot_umap_novel_types(emb_2d, fine_labels, set(present_novel),
                              direction, output_dir)
    except ImportError:
        print("  umap-learn not installed; skipping UMAP plot.")

    # ---- Bar chart figures ----
    plot_knn_purity_comparison(purity_scim, purity_pca, fine_labels,
                               set(present_novel), direction, output_dir)
    plot_centroid_separation(sep_scim, sep_pca, set(present_novel),
                             direction, output_dir)


# ==============================================================================
# SINGLE EXPERIMENT RUN
# ==============================================================================

def run_transfer(adata_ref, adata_query, direction, type_category_fn):
    """Run one direction of cross-dataset transfer.

    Parameters
    ----------
    adata_ref   : AnnData  with obs['_harmonised'] and obs['_raw_label']
    adata_query : AnnData  with obs['_harmonised'] and obs['_raw_label']
    direction   : str  label for plots
    type_category_fn : callable

    Returns
    -------
    predictions  : dict[method -> np.ndarray of harmonised predictions]
    overall_metrics : list[dict]
    """
    y_true_fine       = adata_query.obs['_raw_label'].values
    y_true_harmonised = adata_query.obs['_harmonised'].values
    labels_ref        = adata_ref.obs['_harmonised'].values

    predictions    = {}
    overall_metrics = []
    emb_query_out  = None   # set in SCimilarity block if run successfully

    def _metrics_row(method, preds):
        return {
            'method':   method,
            'accuracy': accuracy(y_true_harmonised, preds),
            'ari':      ari(y_true_harmonised, preds),
            'f1_macro': macro_f1(y_true_harmonised, preds),
        }

    # ------------------------------------------------------------------
    # SCimilarity
    # ------------------------------------------------------------------
    emb_ref = None
    if RUN_SCIMILARITY:
        try:
            print("  [SCimilarity] computing embeddings...")
            t0 = time.time()
            emb_ref   = get_scimilarity_embeddings(adata_ref,   MODEL_PATH)
            emb_query = get_scimilarity_embeddings(adata_query, MODEL_PATH)
            emb_query_out = emb_query
            print(f"    Embeddings: ref {emb_ref.shape}, query {emb_query.shape}  "
                  f"({time.time()-t0:.1f}s)")

            # Drop NaN labels from reference
            valid = pd.notna(labels_ref)
            emb_ref_v   = emb_ref[valid]
            labels_ref_v = labels_ref[valid]

            classifiers = {
                'SCimilarity-knn':    KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
                'SCimilarity-logreg': LogisticRegression(max_iter=1000, n_jobs=-1,
                                                         random_state=42),
                'SCimilarity-mlp':    MLPClassifier(hidden_layer_sizes=(128, 64),
                                                    max_iter=300, alpha=0.001,
                                                    random_state=42),
            }
            for name, clf in classifiers.items():
                t1 = time.time()
                clf.fit(emb_ref_v, labels_ref_v)
                raw_pred = clf.predict(emb_query)
                pred     = _refine(emb_query, raw_pred, k=30)
                predictions[name] = pred
                row = _metrics_row(name, pred)
                overall_metrics.append(row)
                print(f"    [{name}] Acc {row['accuracy']:.3f}  ARI {row['ari']:.3f}  "
                      f"F1 {row['f1_macro']:.3f}  ({time.time()-t1:.1f}s)")

        except Exception as exc:
            print(f"  [SCimilarity] ERROR: {exc}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # CellTypist
    # ------------------------------------------------------------------
    ct_prob_matrix = None
    if RUN_CELLTYPIST:
        try:
            print("  [CellTypist] training...")
            t0  = time.time()
            ct  = CellTypistModel(majority_voting=True)
            ct.fit(adata_ref, target_column='_harmonised')
            print(f"    Trained in {time.time()-t0:.1f}s")
            
            t1  = time.time()
            pred = ct.predict(adata_query)
            print(f"    Predicted in {time.time()-t1:.1f}s")

            # probability_matrix is stored on the wrapper by predict()
            ct_prob_matrix = getattr(ct, 'probability_matrix', None)
            if ct_prob_matrix is None:
                print("    [CellTypist] Warning: no probability_matrix — entropy/trajectory plots skipped.")
                
            predictions['CellTypist'] = pred
            row = _metrics_row('CellTypist', pred)
            overall_metrics.append(row)
            print(f"    [CellTypist] Acc {row['accuracy']:.3f}  ARI {row['ari']:.3f}  "
                  f"F1 {row['f1_macro']:.3f}")
                  
        except Exception as exc:
            print(f"  [CellTypist] ERROR: {exc}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # SingleR
    # ------------------------------------------------------------------
    if RUN_SINGLER:
        print("  [SingleR] running...")
        t0   = time.time()
        pred = run_singler(adata_ref, adata_query)
        if pred is not None:
            predictions['SingleR'] = pred
            row = _metrics_row('SingleR', pred)
            overall_metrics.append(row)
            print(f"    [SingleR] Acc {row['accuracy']:.3f}  ARI {row['ari']:.3f}  "
                  f"F1 {row['f1_macro']:.3f}  ({time.time()-t0:.1f}s)")

    return predictions, overall_metrics, emb_ref, emb_query_out, ct_prob_matrix


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("Cross-Dataset Transfer: Velten ↔ Beneyto")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Check data availability
    # -------------------------------------------------------------------------
    for p in [BENEYTO_PATH, VELTEN_PATH]:
        if not p.exists():
            print(f"\nERROR: data file not found: {p}")
            print("Please update BENEYTO_PATH and VELTEN_PATH at the top of this script.")
            return

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print("\nLoading Beneyto...")
    adata_beneyto = load_and_prep(
        BENEYTO_PATH, label_col='ct',
        harmonised_map=BENEYTO_CT_TO_HARMONISED,
        max_cells=MAX_REF_CELLS,
    )
    print("\nLoading Velten...")
    adata_velten = load_and_prep(
        VELTEN_PATH, label_col='cell_type',
        harmonised_map=VELTEN_CT_TO_HARMONISED,
        max_cells=MAX_QUERY_CELLS,
    )

    print(f"\nBeneyto harmonised label distribution:")
    print(adata_beneyto.obs['_harmonised'].value_counts().to_string())
    print(f"\nVelten harmonised label distribution:")
    print(adata_velten.obs['_harmonised'].value_counts().to_string())

    # Gene alignment is handled internally by SCimilarity (its own vocab) and
    # by CellTypist (trained-gene list).  For SingleR we align inside
    # run_singler().  For SCimilarity the alignment to its internal gene
    # vocabulary is done inside get_embedding(); the two datasets can keep
    # their full gene sets.
    #
    # NOTE: sklearn-based methods that work on raw .X (not used here) would
    # need explicit gene alignment.  CellTypistModel and SCimilarityModel
    # each handle their own gene space internally.

    # -------------------------------------------------------------------------
    # Direction A: Beneyto → Velten
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Direction A: Train on Beneyto → Predict Velten")
    print("=" * 60)

    preds_a, metrics_a, emb_ref_a, emb_query_a, ct_probs_a = run_transfer(
        adata_ref=adata_beneyto,
        adata_query=adata_velten,
        direction='Beneyto → Velten',
        type_category_fn=_velten_type_category,
    )


    if preds_a:
        fine_v = adata_velten.obs['_raw_label'].values
        harm_v = adata_velten.obs['_harmonised'].values

        print("\n  Generating figures for Beneyto → Velten...")
        plot_per_celltype_accuracy(
            preds_a, fine_v, harm_v,
            _velten_type_category, 'Beneyto → Velten', FIGURE_DIR)

        plot_continuous_advantage(
            preds_a, fine_v, harm_v,
            _velten_type_category, 'Beneyto → Velten',
            ref_method='CellTypist', cmp_method='SCimilarity-mlp',
            output_dir=FIGURE_DIR)

        plot_overall_metrics(metrics_a, 'Beneyto → Velten', FIGURE_DIR)

        plot_intermediate_confusion(
            preds_a, fine_v,
            harmonised_categories=sorted((set(VELTEN_CT_TO_HARMONISED.values()) |
                                          set(BENEYTO_CT_TO_HARMONISED.values())) - {None}),
            intermediate_types=INTERMEDIATE_VELTEN,
            direction='Beneyto → Velten',
            output_dir=FIGURE_DIR,
        )

        if emb_ref_a is not None and emb_query_a is not None:
            # 1. Dataset mixing
            plot_cross_dataset_mixing(
                emb_ref=emb_ref_a, 
                emb_query=emb_query_a, 
                labels_query=fine_v, 
                type_category_fn=_velten_type_category, 
                direction='Beneyto → Velten',
                output_dir=FIGURE_DIR
            )
            
            # 2. Trajectory Interpolation 
            if ct_probs_a is not None:
                plot_trajectory_interpolation(
                    emb_ref=emb_ref_a, 
                    emb_query=emb_query_a, 
                    labels_ref=adata_beneyto.obs['_harmonised'].values, 
                    labels_query=fine_v, 
                    ct_prob_matrix=ct_probs_a, 
                    type_category_fn=_velten_type_category, 
                    direction='Beneyto → Velten',
                    output_dir=FIGURE_DIR
                )
                
        # 3. Prediction Entropy
        if ct_probs_a is not None:
            plot_prediction_entropy(
                ct_prob_matrix=ct_probs_a, 
                labels_query=fine_v, 
                type_category_fn=_velten_type_category, 
                direction='Beneyto → Velten',
                output_dir=FIGURE_DIR
            )

        # Save per-type accuracy tables
        for method, pred in preds_a.items():
            df = per_type_accuracy(fine_v, pred, harm_v)
            df['category'] = df['cell_type'].map(_velten_type_category)
            df.to_csv(OUTPUT_DIR / f"per_celltype_accuracy_B2V_{method}.csv",
                      index=False)

    # Novel type separation: do Velten blast/intermediate types cluster
    # distinctly in SCimilarity space even when labelled incorrectly?
    if emb_query_a is not None:
        analyze_novel_type_separation(
            adata_velten, emb_query_a, NOVEL_TYPES_VELTEN,
            'Beneyto → Velten', FIGURE_DIR,
        )

    # -------------------------------------------------------------------------
    # Direction B: Velten → Beneyto  (supplementary)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Direction B: Train on Velten → Predict Beneyto")
    print("=" * 60)

    preds_b, metrics_b, emb_ref_b, emb_query_b, ct_probs_b = run_transfer(
        adata_ref=adata_velten,
        adata_query=adata_beneyto,
        direction='Velten → Beneyto',
        type_category_fn=_beneyto_type_category,
    )

    if preds_b:
        fine_b = adata_beneyto.obs['_raw_label'].values
        harm_b = adata_beneyto.obs['_harmonised'].values

        print("\n  Generating figures for Velten → Beneyto...")
        plot_per_celltype_accuracy(
            preds_b, fine_b, harm_b,
            _beneyto_type_category, 'Velten → Beneyto', FIGURE_DIR)

        plot_continuous_advantage(
            preds_b, fine_b, harm_b,
            _beneyto_type_category, 'Velten → Beneyto',
            ref_method='CellTypist', cmp_method='SCimilarity-mlp',
            output_dir=FIGURE_DIR)

        plot_overall_metrics(metrics_b, 'Velten → Beneyto', FIGURE_DIR)

        plot_intermediate_confusion(
            preds_b, fine_b,
            harmonised_categories=sorted((set(VELTEN_CT_TO_HARMONISED.values()) |
                                          set(BENEYTO_CT_TO_HARMONISED.values())) - {None}),
            intermediate_types=INTERMEDIATE_BENEYTO,
            direction='Velten → Beneyto',
            output_dir=FIGURE_DIR,
        )

        if emb_ref_b is not None and emb_query_b is not None:
            # 1. Dataset mixing
            plot_cross_dataset_mixing(
                emb_ref=emb_ref_b,
                emb_query=emb_query_b,
                labels_query=fine_b,
                type_category_fn=_beneyto_type_category,
                direction='Velten → Beneyto',
                output_dir=FIGURE_DIR
            )

            # 2. Trajectory Interpolation
            if ct_probs_b is not None:
                plot_trajectory_interpolation(
                    emb_ref=emb_ref_b,
                    emb_query=emb_query_b,
                    labels_ref=adata_velten.obs['_harmonised'].values,
                    labels_query=fine_b,
                    ct_prob_matrix=ct_probs_b,
                    type_category_fn=_beneyto_type_category,
                    direction='Velten → Beneyto',
                    output_dir=FIGURE_DIR
                )

        # 3. Prediction Entropy
        if ct_probs_b is not None:
            plot_prediction_entropy(
                ct_prob_matrix=ct_probs_b,
                labels_query=fine_b,
                type_category_fn=_beneyto_type_category,
                direction='Velten → Beneyto',
                output_dir=FIGURE_DIR
            )

        for method, pred in preds_b.items():
            df = per_type_accuracy(fine_b, pred, harm_b)
            df['category'] = df['cell_type'].map(_beneyto_type_category)
            df.to_csv(OUTPUT_DIR / f"per_celltype_accuracy_V2B_{method}.csv",
                      index=False)

    # Novel type separation: do Beneyto intermediate progenitors cluster
    # distinctly in SCimilarity space even when trained only on Velten blasts?
    if emb_query_b is not None:
        analyze_novel_type_separation(
            adata_beneyto, emb_query_b, NOVEL_TYPES_BENEYTO,
            'Velten → Beneyto', FIGURE_DIR,
        )

    # -------------------------------------------------------------------------
    # Summary printout
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY — Beneyto → Velten")
    print("=" * 60)
    if metrics_a:
        df_sum = pd.DataFrame(metrics_a).set_index('method')
        print(df_sum.to_string(float_format='{:.4f}'.format))
        df_sum.to_csv(OUTPUT_DIR / "summary_B2V.csv")

    print("\n" + "=" * 60)
    print("SUMMARY — Velten → Beneyto")
    print("=" * 60)
    if metrics_b:
        df_sum = pd.DataFrame(metrics_b).set_index('method')
        print(df_sum.to_string(float_format='{:.4f}'.format))
        df_sum.to_csv(OUTPUT_DIR / "summary_V2B.csv")

    print(f"\nAll figures saved to: {FIGURE_DIR}")


if __name__ == '__main__':
    main()
