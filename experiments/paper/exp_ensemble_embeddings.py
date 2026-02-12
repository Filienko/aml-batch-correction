"""
Experiment: Comprehensive Cell Type Annotation Benchmark
=========================================================

Compares multiple cell type annotation methods:
1. CellTypist (reference-based, trained on reference)
2. SCimilarity + Ensemble (embedding-based classifiers)
3. SingleR (reference-based, correlation-based)
4. scTab (zero-shot foundation model)

All methods are evaluated on the same reference/query pairs for fair comparison.

Outputs:
- UMAP visualizations (ground truth vs predictions)
- Box-whisker plots for F1 scores across methods/datasets
- Box-whisker plots for runtimes
- Per-cell-type F1 breakdown for SCimilarity-MLP
"""
import os
import sys
import warnings
import gc
import time
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, classification_report

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.evaluation import compute_metrics
from sccl.models.celltypist import CellTypistModel

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Benchmark data directory (for Zheng dataset)
BENCHMARK_DATA_DIR = Path(__file__).parent / "benchmark_data"

# scTab model paths (for zero-shot baseline)
SCTAB_CHECKPOINT = Path("scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt")
MERLIN_DIR = Path("merlin_cxg_2023_05_15_sf-log1p_minimal")

# Set to None to use full data
MAX_CELLS_PER_STUDY = 15000

SCENARIOS_SHORT = [
    {
        'name': 'Cross-Platform: van_galen (Seq-Well) -> Zhai (SORT-seq)',
        'reference': 'van_galen_2019',
        'query': 'zhai_2022',
    },
]

# AML Atlas scenarios (subset from single file)
AML_SCENARIOS = [
    {
        'name': 'Same-Platform: beneyto (10X Genomics) → Zhang (10X Genomics)',
        'reference': 'beneyto-calabuig-2023',
        'query': 'zhang_2023',
    },
    {
        'name': 'Cross-Platform: Zhai (SORT-seq) → Zhang (10X Genomics)',
        'reference': 'zhai_2022',
        'query': 'zhang_2023',
    },
    {
        'name': 'Cross-Platform: van_galen (Seq-Well) → velten (Muta-Seq)',
        'reference': 'van_galen_2019',
        'query': 'velten_2021',
    },
    {
        'name': 'Cross-Platform: van_galen (Seq-Well) → beneyto (10X Genomics)',
        'reference': 'van_galen_2019',
        'query': 'beneyto-calabuig-2023',
    },
    {
        'name': 'Same-Platform: van_galen (Seq-Well) -> Zhai (SORT-seq)',
        'reference': 'van_galen_2019',
        'query': 'zhai_2022',
    },
]

# Zheng dataset scenarios (separate files for reference/query)
ZHENG_SCENARIOS = [
    {
        'name': 'Zheng PBMC: Train → Test',
        'reference_file': BENCHMARK_DATA_DIR / 'zheng_train.h5ad',
        'query_file': BENCHMARK_DATA_DIR / 'zheng_test.h5ad',
        'type': 'separate_files',
    },
]

# Combine all scenarios (AML uses 'reference'/'query' study names, Zheng uses file paths)
SCENARIOS = AML_SCENARIOS + ZHENG_SCENARIOS

# Which methods to run (set to False to skip)
RUN_CELLTYPIST = True
RUN_SCIMILARITY = True
RUN_SINGLER = True
RUN_SCTAB = True

# Number of runs for statistical analysis (box-whisker plots)
N_RUNS = 5

# Figure output directory
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# ==============================================================================
# SCIMILARITY METHODS
# ==============================================================================

def get_scimilarity_embeddings(adata, model_path):
    """Get SCimilarity embeddings for data."""
    pipeline = Pipeline(
        model='scimilarity',
        model_params={'model_path': model_path}
    )
    embeddings = pipeline.model.get_embedding(adata)
    return embeddings


def create_ensemble_classifiers():
    """Create individual classifiers for ensemble."""
    classifiers = {
        'knn': KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        'logreg': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        'rf': RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.001, random_state=42),
    }
    return classifiers


def refine_predictions(embeddings, raw_predictions, k=50):
    """Smooths predictions using the query dataset's own internal structure."""
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(embeddings)
    neighbors = nn.kneighbors(embeddings, return_distance=False)

    refined_preds = []
    from collections import Counter

    for i, neighbor_indices in enumerate(neighbors):
        neighbor_labels = raw_predictions[neighbor_indices]
        vote = Counter(neighbor_labels).most_common(1)[0][0]
        refined_preds.append(vote)

    return np.array(refined_preds)


def train_ensemble(embeddings_ref, labels_ref, ensemble_type='voting'):
    """Train ensemble of classifiers on embeddings."""
    classifiers = create_ensemble_classifiers()

    if ensemble_type == 'voting_hard':
        estimators = [(name, clf) for name, clf in classifiers.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
        ensemble.fit(embeddings_ref, labels_ref)
        return ensemble

    elif ensemble_type == 'voting_soft':
        estimators = [(name, clf) for name, clf in classifiers.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(embeddings_ref, labels_ref)
        return ensemble

    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


def predict_ensemble(ensemble, embeddings_query, ensemble_type='voting'):
    """Make predictions with ensemble."""
    return ensemble.predict(embeddings_query)


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_umap_comparison(adata_query, embeddings_query, y_true, y_pred,
                         method_name, scenario_name, output_dir):
    """
    Create UMAP visualization comparing ground truth vs predicted cell types.

    Parameters
    ----------
    adata_query : AnnData
        Query data
    embeddings_query : np.ndarray
        SCimilarity embeddings for query cells
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    method_name : str
        Name of the method (e.g., 'SCimilarity-mlp')
    scenario_name : str
        Name of the scenario
    output_dir : Path
        Output directory for figures
    """
    from sklearn.manifold import TSNE
    import umap

    # Filter out NaN labels for visualization
    valid_mask = pd.notna(y_true)
    embeddings_valid = embeddings_query[valid_mask]
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Compute UMAP on embeddings
    print(f"    Computing UMAP for {method_name}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(embeddings_valid)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get unique labels for consistent coloring
    all_labels = np.unique(np.concatenate([y_true_valid, y_pred_valid]))
    n_colors = len(all_labels)
    colors = sns.color_palette("husl", n_colors)
    label_to_color = {label: colors[i] for i, label in enumerate(all_labels)}

    # Generate descriptive title for scenario
    if 'Zheng' in scenario_name:
        scenario_title = 'Zheng PBMC (Train → Test, Same Dataset)'
    else:
        scenario_title = scenario_name

    # Plot ground truth
    ax1 = axes[0]
    for label in all_labels:
        mask = y_true_valid == label
        if mask.sum() > 0:
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[label_to_color[label]], label=label, s=5, alpha=0.6)
    ax1.set_title(f'Ground Truth\n{scenario_title}', fontsize=12)
    ax1.set_xlabel('UMAP1')
    ax1.set_ylabel('UMAP2')

    # Plot predictions
    ax2 = axes[1]
    for label in all_labels:
        mask = y_pred_valid == label
        if mask.sum() > 0:
            ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[label_to_color[label]], label=label, s=5, alpha=0.6)
    ax2.set_title(f'{method_name} Predictions\n{scenario_title}', fontsize=12)
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')

    # Add legend (shared)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5),
               fontsize=8, markerscale=2)

    plt.tight_layout()

    # Save figure
    safe_scenario = scenario_name.replace(':', '').replace(' ', '_').replace('→', 'to')[:50]
    safe_method = method_name.replace('-', '_')
    filename = output_dir / f"umap_{safe_method}_{safe_scenario}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved UMAP to {filename}")

    return umap_coords


def plot_method_comparison_boxplot(df_results, metric='f1_macro', output_dir=None):
    """
    Create box-whisker plot comparing methods across datasets.
    Style: horizontal boxes, one color per method, one plot per dataset.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with columns: scenario, method, run, f1_macro, time_sec
    metric : str
        Metric to plot ('f1_macro' or 'time_sec')
    output_dir : Path
        Output directory for figures
    """
    from matplotlib.patches import Patch

    # Filter to main methods only
    main_methods = ['CellTypist', 'SingleR', 'scTab', 'SCimilarity-mlp']
    df_plot = df_results[df_results['method'].isin(main_methods)].copy()

    if len(df_plot) == 0:
        print(f"    No data for main methods, using all methods")
        df_plot = df_results.copy()
        main_methods = df_results['method'].unique().tolist()

    scenarios = df_plot['scenario'].unique()

    # Create one plot per scenario
    for scenario in scenarios:
        df_scenario = df_plot[df_plot['scenario'] == scenario].copy()

        # Get methods present in this scenario, sorted by median metric (ascending)
        method_medians = df_scenario.groupby('method')[metric].median().sort_values(ascending=True)
        method_order = method_medians.index.tolist()

        # Create figure
        n_methods = len(method_order)
        fig_height = max(4, n_methods * 0.8)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        # Color palette - distinct color per method
        colors = sns.color_palette("husl", n_methods)
        color_dict = {m: colors[i] for i, m in enumerate(method_order)}

        # Prepare data for boxplot
        box_data = [df_scenario[df_scenario['method'] == m][metric].values
                    for m in method_order]

        # Create horizontal box plot
        bp = ax.boxplot(
            box_data,
            vert=False,
            patch_artist=True,
            labels=method_order,
            widths=0.6,
        )

        # Color each box
        for patch, method in zip(bp['boxes'], method_order):
            patch.set_facecolor(color_dict[method])
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Style whiskers, caps, medians
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1)
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)

        # Formatting
        if metric == 'f1_macro':
            ax.set_xlabel('F1 Score (Macro)', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.05)
        else:
            ax.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('')

        # Add gridlines
        ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.set_axisbelow(True)

        # Generate descriptive title
        if 'Zheng' in scenario:
            # For Zheng dataset, make it clear it's same-dataset train/test
            title = 'Zheng PBMC\n(Train → Test, Same Dataset)'
        else:
            # For cross-study scenarios, show the transfer direction
            short_name = scenario.split(':')[-1].strip() if ':' in scenario else scenario
            title = short_name.replace('→', '→\n').replace('->', '→\n')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Create legend
        legend_patches = [Patch(facecolor=color_dict[m], label=m, alpha=0.85,
                                edgecolor='black', linewidth=0.5)
                         for m in reversed(method_order)]
        ax.legend(handles=legend_patches, title='Method',
                 bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                 title_fontsize=11, frameon=True)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_dir:
            safe_scenario = scenario.replace(':', '').replace(' ', '_').replace('→', 'to').replace('->', 'to')[:40]
            filename = output_dir / f"methods_{metric}_{safe_scenario}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved method comparison: {filename.name}")
        else:
            plt.show()


def plot_per_celltype_f1(df_per_celltype, method_name, output_dir=None):
    """
    Create box-whisker plot showing F1/Accuracy per cell type for a specific method.
    Style: horizontal boxes, one color per cell type, one plot per dataset.

    Parameters
    ----------
    df_per_celltype : pd.DataFrame
        DataFrame with columns: scenario, cell_type, f1, run
    method_name : str
        Name of the method
    output_dir : Path
        Output directory for figures
    """
    from matplotlib.patches import Patch

    # Create one plot per scenario (dataset)
    scenarios = df_per_celltype['scenario'].unique()

    for scenario in scenarios:
        df_scenario = df_per_celltype[df_per_celltype['scenario'] == scenario].copy()

        # Get unique cell types and sort by median F1 (ascending so highest is at top)
        cell_type_medians = df_scenario.groupby('cell_type')['f1'].median().sort_values(ascending=True)
        cell_type_order = cell_type_medians.index.tolist()

        # Create figure - height based on number of cell types
        n_celltypes = len(cell_type_order)
        fig_height = max(5, n_celltypes * 0.45)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        # Color palette - distinct color per cell type
        colors = sns.color_palette("husl", n_celltypes)
        color_dict = {ct: colors[i] for i, ct in enumerate(cell_type_order)}

        # Prepare data for boxplot
        box_data = [df_scenario[df_scenario['cell_type'] == ct]['f1'].values
                    for ct in cell_type_order]

        # Create horizontal box plot
        bp = ax.boxplot(
            box_data,
            vert=False,
            patch_artist=True,
            labels=cell_type_order,
            widths=0.6,
        )

        # Color each box with its cell type color
        for patch, ct in zip(bp['boxes'], cell_type_order):
            patch.set_facecolor(color_dict[ct])
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Style whiskers, caps, medians
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1)
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)

        # Formatting
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylabel('')

        # Add gridlines
        ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.set_axisbelow(True)

        # Generate descriptive title
        if 'Zheng' in scenario:
            title = 'Zheng PBMC\n(Train → Test, Same Dataset)'
        else:
            short_name = scenario.split(':')[-1].strip() if ':' in scenario else scenario
            title = short_name.replace('→', '→\n').replace('->', '→\n')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Create legend with cell type colors (reversed to match plot order)
        legend_patches = [Patch(facecolor=color_dict[ct], label=ct, alpha=0.85,
                                edgecolor='black', linewidth=0.5)
                         for ct in reversed(cell_type_order)]
        ax.legend(handles=legend_patches, title='cell type',
                 bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
                 title_fontsize=10, frameon=True)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_dir:
            safe_scenario = scenario.replace(':', '').replace(' ', '_').replace('→', 'to').replace('->', 'to')[:40]
            safe_method = method_name.replace('-', '_')
            filename = output_dir / f"percelltype_{safe_method}_{safe_scenario}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved per-celltype plot: {filename.name}")
        else:
            plt.show()


def plot_timing_breakdown(df_results, output_dir=None):
    """
    Create stacked bar plot showing training vs inference time breakdown.
    Each method gets a horizontal bar split into training (teal) and inference (coral).

    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with columns: scenario, method, train_time_sec, inference_time_sec
    output_dir : Path
        Output directory for figures
    """
    from matplotlib.patches import Patch

    # Check if timing columns exist
    if 'train_time_sec' not in df_results.columns or 'inference_time_sec' not in df_results.columns:
        print("    Timing breakdown columns not found, skipping...")
        return

    # Filter to main methods
    main_methods = ['CellTypist', 'SingleR', 'scTab', 'SCimilarity-mlp']
    df_plot = df_results[df_results['method'].isin(main_methods)].copy()

    if len(df_plot) == 0:
        df_plot = df_results.copy()
        main_methods = df_results['method'].unique().tolist()

    scenarios = df_plot['scenario'].unique()

    # Colors for train/inference
    train_color = '#4ECDC4'  # Teal
    infer_color = '#FF6B6B'  # Coral

    for scenario in scenarios:
        df_scenario = df_plot[df_plot['scenario'] == scenario].copy()

        # Aggregate by method (mean times)
        timing = df_scenario.groupby('method').agg({
            'train_time_sec': 'mean',
            'inference_time_sec': 'mean',
        }).reset_index()

        # Sort by total time
        timing['total'] = timing['train_time_sec'] + timing['inference_time_sec']
        timing = timing.sort_values('total', ascending=True)

        # Create figure
        n_methods = len(timing)
        fig_height = max(4, n_methods * 0.8)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        y_pos = np.arange(n_methods)

        # Create stacked horizontal bar
        bars_train = ax.barh(y_pos, timing['train_time_sec'], color=train_color,
                             label='Training', alpha=0.85, edgecolor='black', linewidth=0.5)
        bars_infer = ax.barh(y_pos, timing['inference_time_sec'], left=timing['train_time_sec'],
                             color=infer_color, label='Inference', alpha=0.85,
                             edgecolor='black', linewidth=0.5)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(timing['method'])
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')

        # Add value labels on bars
        for i, (train, infer) in enumerate(zip(timing['train_time_sec'], timing['inference_time_sec'])):
            total = train + infer
            if train > total * 0.15:  # Only show if bar is wide enough
                ax.text(train/2, i, f'{train:.1f}s', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
            if infer > total * 0.15:
                ax.text(train + infer/2, i, f'{infer:.1f}s', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

        # Generate descriptive title
        if 'Zheng' in scenario:
            title = 'Zheng PBMC (Train → Test, Same Dataset)\nTiming Breakdown'
        else:
            short_name = scenario.split(':')[-1].strip() if ':' in scenario else scenario
            title = f'{short_name}\nTiming Breakdown'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Add legend
        legend_patches = [
            Patch(facecolor=train_color, label='Training', alpha=0.85, edgecolor='black', linewidth=0.5),
            Patch(facecolor=infer_color, label='Inference', alpha=0.85, edgecolor='black', linewidth=0.5),
        ]
        ax.legend(handles=legend_patches, title='Phase', loc='lower right', fontsize=10,
                 title_fontsize=11, frameon=True)

        # Grid and spines
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_dir:
            safe_scenario = scenario.replace(':', '').replace(' ', '_').replace('→', 'to').replace('->', 'to')[:40]
            # Use same filename pattern as timing figure
            filename = output_dir / f"methods_time_sec_{safe_scenario}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved timing breakdown: {filename.name}")
        else:
            plt.show()


def compute_per_celltype_f1(y_true, y_pred):
    """
    Compute F1 score for each cell type.

    Returns
    -------
    dict : {cell_type: f1_score}
    """
    # Filter NaN
    valid_mask = pd.notna(y_true)
    y_true_valid = np.asarray(y_true)[valid_mask]
    y_pred_valid = np.asarray(y_pred)[valid_mask]

    # Get classification report as dict
    report = classification_report(y_true_valid, y_pred_valid,
                                   output_dict=True, zero_division=0)

    # Extract per-class F1
    per_class_f1 = {}
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            per_class_f1[label] = metrics['f1-score']

    return per_class_f1


# ==============================================================================
# SINGLER METHODS
# ==============================================================================

def preprocess_zheng_data(adata):
    """
    Preprocess Zheng dataset to standardize format.

    - Sets var.index to gene symbols (from var['gene_symbols'])
    - Converts integer cell type labels to string names (using uns['cell_type_names'])

    Parameters
    ----------
    adata : AnnData
        Zheng dataset

    Returns
    -------
    adata : AnnData
        Preprocessed dataset with gene symbols as var.index and string labels
    """
    adata = adata.copy()

    # Fix gene names: set var.index to gene symbols
    if 'gene_symbols' in adata.var.columns:
        print(f"    Setting var.index to gene symbols...")
        adata.var.index = adata.var['gene_symbols'].values
        adata.var_names_make_unique()

    # Fix cell type labels: convert integers to string names
    if 'cell_type_names' in adata.uns:
        cell_type_names = adata.uns['cell_type_names']

        # Find the cell type column
        for col in ['cell_type_label', 'cell_type', 'celltype', 'label', 'labels']:
            if col in adata.obs.columns:
                labels = adata.obs[col].values
                # Check if labels are integers
                if np.issubdtype(labels.dtype, np.integer):
                    print(f"    Converting integer labels to cell type names...")
                    # Map integer indices to string names
                    string_labels = [cell_type_names[i] if i < len(cell_type_names) else f'Unknown_{i}'
                                    for i in labels]
                    adata.obs[col] = string_labels
                break

    return adata


def log_normalize(X, target_sum=1e4):
    """Log-normalize count matrix (like scanpy.pp.normalize_total + log1p)."""
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_norm = X / row_sums * target_sum

    return np.log1p(X_norm)


def run_singler_prediction(adata_ref, adata_query, cell_type_col):
    """
    Run SingleR reference-based annotation.

    SingleR compares query cells to reference cells using correlation-based
    scoring to transfer labels.
    """
    try:
        import singler
    except ImportError:
        print("    singler package not found. Install with: pip install singler")
        return None

    # Get raw counts and gene names
    if adata_ref.raw is not None:
        X_ref = adata_ref.raw.X
        ref_genes = adata_ref.raw.var_names.values
    else:
        X_ref = adata_ref.X
        ref_genes = adata_ref.var_names.values

    if adata_query.raw is not None:
        X_query = adata_query.raw.X
        query_genes = adata_query.raw.var_names.values
    else:
        X_query = adata_query.X
        query_genes = adata_query.var_names.values

    ref_labels = adata_ref.obs[cell_type_col].values

    # Find common genes
    common_genes = np.intersect1d(ref_genes, query_genes)
    print(f"    Common genes: {len(common_genes)}")

    if len(common_genes) < 100:
        print("    ERROR: Too few common genes!")
        return None

    # Subset to common genes
    ref_gene_idx = [np.where(ref_genes == g)[0][0] for g in common_genes]
    query_gene_idx = [np.where(query_genes == g)[0][0] for g in common_genes]

    X_ref_common = X_ref[:, ref_gene_idx]
    X_query_common = X_query[:, query_gene_idx]

    # Log-normalize
    X_ref_norm = log_normalize(X_ref_common)
    X_query_norm = log_normalize(X_query_common)

    # Remove NaN labels from reference
    valid_ref = pd.notna(ref_labels)
    X_ref_norm = X_ref_norm[valid_ref]
    ref_labels_valid = ref_labels[valid_ref]

    # Run SingleR (expects genes x cells)
    results = singler.annotate_single(
        test_data=X_query_norm.T,
        test_features=common_genes,
        ref_data=X_ref_norm.T,
        ref_labels=ref_labels_valid,
        ref_features=common_genes,
        num_threads=8,
    )

    predictions = np.asarray(results.column("best"))
    return predictions


# ==============================================================================
# SCTAB METHODS
# ==============================================================================

# Label mapping from scTab Cell Ontology to abbreviated labels
SCTAB_LABEL_MAP = {
    'B cell': 'B', 'naive B cell': 'B', 'memory B cell': 'B',
    'plasma cell': 'Plasma', 'plasmablast': 'Plasma',
    'CD14-positive monocyte': 'CD14+ Mono',
    'CD14-positive, CD16-negative classical monocyte': 'CD14+ Mono',
    'CD14-low, CD16-positive monocyte': 'CD16+ Mono',
    'CD16-positive, CD14-low monocyte': 'CD16+ Mono',
    'classical monocyte': 'CD14+ Mono',
    'non-classical monocyte': 'CD16+ Mono',
    'intermediate monocyte': 'CD14+ Mono',
    'natural killer cell': 'NK',
    'CD16-negative, CD56-bright natural killer cell, human': 'NK',
    'CD16-positive, CD56-dim natural killer cell, human': 'NK',
    'T cell': 'T',
    'CD4-positive, alpha-beta T cell': 'CD4+ T',
    'CD8-positive, alpha-beta T cell': 'CD8+ T',
    'CD4-positive helper T cell': 'CD4+ T',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ T',
    'CD8-positive, alpha-beta memory T cell': 'CD8+ T',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'CD4+ T',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'CD8+ T',
    'regulatory T cell': 'Treg',
    'gamma-delta T cell': 'gdT',
    'dendritic cell': 'DC',
    'conventional dendritic cell': 'cDC',
    'plasmacytoid dendritic cell': 'pDC',
    'CD1c-positive myeloid dendritic cell': 'cDC',
    'myeloid dendritic cell': 'cDC',
    'hematopoietic stem cell': 'HSPC',
    'hematopoietic multipotent progenitor cell': 'HSPC',
    'common myeloid progenitor': 'CMP',
    'granulocyte monocyte progenitor cell': 'GMP',
    'megakaryocyte-erythroid progenitor cell': 'MEP',
    'common lymphoid progenitor': 'CLP',
    'megakaryocyte': 'MEP',
    'erythroid lineage cell': 'Erythroid',
    'erythrocyte': 'Erythroid',
    'proerythroblast': 'Erythroid',
    'erythroblast': 'Erythroid',
    'neutrophil': 'Neutrophil',
    'basophil': 'Basophil',
    'eosinophil': 'Eosinophil',
    'mast cell': 'Mast',
}


def harmonize_sctab_labels(predictions, ground_truth_labels=None):
    """Harmonize scTab Cell Ontology labels to match ground truth vocabulary."""
    predictions = np.asarray(predictions)
    harmonized = predictions.copy()

    for sctab_label, target_label in SCTAB_LABEL_MAP.items():
        mask = predictions == sctab_label
        harmonized[mask] = target_label

    # Fuzzy matching for unmapped labels
    if ground_truth_labels is not None:
        # Convert ground truth to strings for comparison
        gt_valid = ground_truth_labels[pd.notna(ground_truth_labels)]
        gt_labels_set = set([str(x) for x in np.unique(gt_valid)])
        unmapped = set(np.unique(harmonized)) - gt_labels_set

        for pred_label in unmapped:
            # Ensure pred_label is string
            if not isinstance(pred_label, str):
                continue
            pred_lower = pred_label.lower()
            for gt_label in gt_labels_set:
                # Ensure gt_label is string
                if not isinstance(gt_label, str):
                    continue
                gt_lower = gt_label.lower()
                if gt_lower in pred_lower or pred_lower in gt_lower:
                    mask = harmonized == pred_label
                    harmonized[mask] = gt_label
                    break

    return harmonized


def sf_log1p_norm(x):
    """Normalize each cell to 10000 counts and apply log1p transform (for scTab)."""
    import torch
    counts = torch.sum(x, dim=1, keepdim=True)
    counts += counts == 0.
    scaling_factor = 10000. / counts
    return torch.log1p(scaling_factor * x)


def run_sctab_prediction(adata_query, cell_type_col):
    """
    Run scTab zero-shot cell type prediction.

    scTab is a pre-trained foundation model that predicts cell types
    without needing reference data (zero-shot).
    """
    try:
        import torch
        import yaml
        from tqdm import tqdm
    except ImportError as e:
        print(f"    Missing dependency for scTab: {e}")
        return None

    # Check if model files exist
    if not SCTAB_CHECKPOINT.exists():
        print(f"    scTab checkpoint not found: {SCTAB_CHECKPOINT}")
        return None
    if not MERLIN_DIR.exists():
        print(f"    Merlin directory not found: {MERLIN_DIR}")
        return None

    try:
        # Load model metadata
        var_path = MERLIN_DIR / "var.parquet"
        genes_from_model = pd.read_parquet(var_path)
        model_genes = genes_from_model['feature_name'].values

        cell_type_path = MERLIN_DIR / "categorical_lookup" / "cell_type.parquet"
        cell_type_mapping = pd.read_parquet(cell_type_path)

        hparams_path = SCTAB_CHECKPOINT.parent / "hparams.yaml"
        with open(hparams_path) as f:
            model_params = yaml.full_load(f.read())

        # Get raw counts
        if adata_query.raw is not None:
            X = adata_query.raw.X
            gene_names = adata_query.raw.var_names.values
        else:
            X = adata_query.X
            gene_names = adata_query.var_names.values

        # Match genes
        gene_mask = np.isin(gene_names, model_genes)
        print(f"    Gene overlap: {gene_mask.sum()}/{len(model_genes)}")

        if gene_mask.sum() < 1000:
            print("    WARNING: Low gene overlap for scTab")

        # Subset and align genes
        X_subset = csc_matrix(X)[:, gene_mask]
        gene_names_subset = gene_names[gene_mask]

        # Create aligned matrix
        gene_to_idx = {g: i for i, g in enumerate(gene_names_subset)}
        n_cells = X_subset.shape[0]
        n_model_genes = len(model_genes)
        aligned = np.zeros((n_cells, n_model_genes), dtype=np.float32)

        if hasattr(X_subset, 'toarray'):
            X_dense = X_subset.toarray()
        else:
            X_dense = np.asarray(X_subset)

        for i, gene in enumerate(model_genes):
            if gene in gene_to_idx:
                aligned[:, i] = X_dense[:, gene_to_idx[gene]]

        # Load model
        if torch.cuda.is_available():
            ckpt = torch.load(SCTAB_CHECKPOINT, weights_only=False)
            device = 'cuda'
        else:
            ckpt = torch.load(SCTAB_CHECKPOINT, map_location=torch.device('cpu'), weights_only=False)
            device = 'cpu'

        tabnet_weights = OrderedDict()
        for name, weight in ckpt['state_dict'].items():
            if 'classifier.' in name:
                tabnet_weights[name.replace('classifier.', '')] = weight

        from cellnet.tabnet.tab_network import TabNet

        model = TabNet(
            input_dim=model_params['gene_dim'],
            output_dim=model_params['type_dim'],
            n_d=model_params['n_d'],
            n_a=model_params['n_a'],
            n_steps=model_params['n_steps'],
            gamma=model_params['gamma'],
            n_independent=model_params['n_independent'],
            n_shared=model_params['n_shared'],
            epsilon=model_params['epsilon'],
            virtual_batch_size=model_params['virtual_batch_size'],
            momentum=model_params['momentum'],
            mask_type=model_params['mask_type'],
        )
        model.load_state_dict(tabnet_weights)
        model.eval()
        if device == 'cuda':
            model = model.cuda()

        # Run inference
        batch_size = 2048
        preds = []

        with torch.no_grad():
            for i in tqdm(range(0, n_cells, batch_size), desc="    scTab inference"):
                batch_x = aligned[i:i + batch_size]
                x_tensor = torch.from_numpy(batch_x).float()
                if device == 'cuda':
                    x_tensor = x_tensor.cuda()

                x_input = sf_log1p_norm(x_tensor)
                logits, _ = model(x_input)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(batch_preds)

        preds = np.hstack(preds)
        predictions = cell_type_mapping.loc[preds]['label'].values

        # Harmonize labels
        y_true = adata_query.obs[cell_type_col].values
        predictions = harmonize_sctab_labels(predictions, ground_truth_labels=y_true)

        return predictions

    except Exception as e:
        print(f"    scTab error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# MAIN
# ==============================================================================

def run_single_experiment(scenario, adata, study_col, cell_type_col, run_id=0,
                          generate_umap=False):
    """
    Run a single experiment for one scenario.

    Returns
    -------
    results : list of dict
        Results for all methods
    per_celltype_results : list of dict
        Per-cell-type F1 for SCimilarity-mlp
    embeddings_query : np.ndarray or None
        SCimilarity embeddings (for UMAP)
    mlp_predictions : np.ndarray or None
        MLP predictions (for UMAP)
    """
    results = []
    per_celltype_results = []
    embeddings_query = None
    mlp_predictions = None

    # Set random seed for this run
    np.random.seed(42 + run_id)

    # 1. Prepare Data - handle both scenario types
    if scenario.get('type') == 'separate_files':
        # Zheng-style: separate files for reference and query
        ref_file = scenario['reference_file']
        query_file = scenario['query_file']

        if not Path(ref_file).exists():
            if run_id == 0:
                print(f"  SKIP: Reference file not found: {ref_file}")
            return [], [], None, None, None
        if not Path(query_file).exists():
            if run_id == 0:
                print(f"  SKIP: Query file not found: {query_file}")
            return [], [], None, None, None

        adata_ref = sc.read_h5ad(ref_file)
        adata_query = sc.read_h5ad(query_file)

        # Preprocess Zheng data (fix gene names and convert integer labels)
        if run_id == 0:
            print("  Preprocessing Zheng data...")
        adata_ref = preprocess_zheng_data(adata_ref)
        adata_query = preprocess_zheng_data(adata_query)

        # Auto-detect cell type column for Zheng data
        cell_type_col_local = cell_type_col
        for col in ['cell_type_label', 'cell_type', 'celltype', 'Cell Type', 'label', 'labels', 'cell_label']:
            if col in adata_ref.obs.columns:
                cell_type_col_local = col
                break
    else:
        # AML-style: subset from single file using study names
        adata_ref = subset_data(adata, studies=[scenario['reference']]).to_memory()
        adata_query = subset_data(adata, studies=[scenario['query']]).to_memory()
        cell_type_col_local = cell_type_col

    # Subsample if needed
    if MAX_CELLS_PER_STUDY:
        if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_ref = adata_ref[indices].copy()
        if adata_query.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_query = adata_query[indices].copy()

    if run_id == 0:
        print(f"  Reference: {adata_ref.n_obs:,} cells")
        print(f"  Query:     {adata_query.n_obs:,} cells")

    y_true = adata_query.obs[cell_type_col_local].values

    # Helper to compute F1
    def compute_f1(y_t, y_p):
        valid_mask = pd.notna(y_t)
        return f1_score(np.asarray(y_t)[valid_mask], np.asarray(y_p)[valid_mask],
                       average='macro', zero_division=0)

    # =========================================================================
    # Method 1: CellTypist (Reference-based)
    # =========================================================================
    if RUN_CELLTYPIST:
        try:
            # Training
            train_start = time.time()
            ct_model = CellTypistModel(majority_voting=True)
            ct_model.fit(adata_ref, target_column=cell_type_col_local)
            train_time = time.time() - train_start

            # Inference
            infer_start = time.time()
            ct_pred = ct_model.predict(adata_query)
            infer_time = time.time() - infer_start

            metrics = compute_metrics(y_true=y_true, y_pred=ct_pred, metrics=['accuracy', 'ari'])
            results.append({
                'scenario': scenario['name'],
                'method': 'CellTypist',
                'type': 'reference-based',
                'accuracy': metrics['accuracy'],
                'ari': metrics['ari'],
                'f1_macro': compute_f1(y_true, ct_pred),
                'train_time_sec': train_time,
                'inference_time_sec': infer_time,
                'time_sec': train_time + infer_time,
                'run': run_id,
            })
            if run_id == 0:
                print(f"    [CellTypist] Acc: {metrics['accuracy']:.3f}, F1: {results[-1]['f1_macro']:.3f}, Train: {train_time:.1f}s, Infer: {infer_time:.1f}s")
        except Exception as e:
            if run_id == 0:
                print(f"    [CellTypist] Error: {e}")
        finally:
            gc.collect()

    # =========================================================================
    # Method 2: SingleR (Reference-based, correlation-based - no explicit training)
    # Note: SingleR is correlation-based, so "training" = building reference profile
    #       and "inference" = computing correlations for query cells. The annotate_single
    #       function does both internally, so we report train_time=0 (no learnable params).
    # =========================================================================
    if RUN_SINGLER:
        try:
            # SingleR has no explicit training step - it's correlation-based
            train_time = 0.0

            # Inference (includes building reference profile + computing correlations)
            infer_start = time.time()
            singler_pred = run_singler_prediction(adata_ref, adata_query, cell_type_col_local)
            infer_time = time.time() - infer_start

            if singler_pred is not None:
                metrics = compute_metrics(y_true=y_true, y_pred=singler_pred, metrics=['accuracy', 'ari'])
                results.append({
                    'scenario': scenario['name'],
                    'method': 'SingleR',
                    'type': 'reference-based',
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'f1_macro': compute_f1(y_true, singler_pred),
                    'train_time_sec': train_time,
                    'inference_time_sec': infer_time,
                    'time_sec': train_time + infer_time,
                    'run': run_id,
                })
                if run_id == 0:
                    print(f"    [SingleR] Acc: {metrics['accuracy']:.3f}, F1: {results[-1]['f1_macro']:.3f}, Train: {train_time:.1f}s, Infer: {infer_time:.1f}s")
        except Exception as e:
            if run_id == 0:
                print(f"    [SingleR] Error: {e}")
        finally:
            gc.collect()

    # =========================================================================
    # Method 3: scTab (Zero-shot Foundation Model - no training, pre-trained)
    # =========================================================================
    if RUN_SCTAB:
        try:
            # scTab is zero-shot - no training step, model is pre-trained
            train_time = 0.0

            # Inference only
            infer_start = time.time()
            sctab_pred = run_sctab_prediction(adata_query, cell_type_col_local)
            infer_time = time.time() - infer_start

            if sctab_pred is not None:
                metrics = compute_metrics(y_true=y_true, y_pred=sctab_pred, metrics=['accuracy', 'ari'])
                results.append({
                    'scenario': scenario['name'],
                    'method': 'scTab',
                    'type': 'zero-shot',
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'f1_macro': compute_f1(y_true, sctab_pred),
                    'train_time_sec': train_time,
                    'inference_time_sec': infer_time,
                    'time_sec': train_time + infer_time,
                    'run': run_id,
                })
                if run_id == 0:
                    print(f"    [scTab] Acc: {metrics['accuracy']:.3f}, F1: {results[-1]['f1_macro']:.3f}, Train: {train_time:.1f}s, Infer: {infer_time:.1f}s")
        except Exception as e:
            if run_id == 0:
                print(f"    [scTab] Error: {e}")
        finally:
            gc.collect()

    # =========================================================================
    # Method 4: SCimilarity + Classifiers
    # Training time = reference preprocessing + reference embedding + classifier fit
    # Inference time = query preprocessing + query embedding + classifier predict + refinement
    # =========================================================================
    if RUN_SCIMILARITY:
        try:
            # Training phase: preprocess and embed reference data
            train_start = time.time()
            adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
            emb_ref = get_scimilarity_embeddings(adata_ref_prep, MODEL_PATH)
            labels_ref = adata_ref.obs[cell_type_col_local].values
            ref_embed_time = time.time() - train_start  # Time to prepare reference embeddings

            # Inference phase: preprocess and embed query data
            infer_start = time.time()
            adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)
            emb_query = get_scimilarity_embeddings(adata_query_prep, MODEL_PATH)
            query_embed_time = time.time() - infer_start  # Time to prepare query embeddings

            # Store for UMAP (only first run)
            if generate_umap and run_id == 0:
                embeddings_query = emb_query.copy()

            del adata_ref_prep, adata_query_prep
            gc.collect()

            # Individual classifiers
            classifiers = create_ensemble_classifiers()

            for clf_name, clf in classifiers.items():
                try:
                    # Training: classifier fit (add to reference embedding time)
                    clf_train_start = time.time()
                    clf.fit(emb_ref, labels_ref)
                    clf_train_time = time.time() - clf_train_start
                    total_train_time = ref_embed_time + clf_train_time

                    # Inference: predict + refinement (add to query embedding time)
                    clf_infer_start = time.time()
                    pred_raw = clf.predict(emb_query)
                    pred = refine_predictions(emb_query, pred_raw, k=50)
                    clf_infer_time = time.time() - clf_infer_start
                    total_infer_time = query_embed_time + clf_infer_time

                    metrics = compute_metrics(y_true=y_true, y_pred=pred, metrics=['accuracy', 'ari'])
                    f1_val = compute_f1(y_true, pred)

                    results.append({
                        'scenario': scenario['name'],
                        'method': f'SCimilarity-{clf_name}',
                        'type': 'embedding-based',
                        'accuracy': metrics['accuracy'],
                        'ari': metrics['ari'],
                        'f1_macro': f1_val,
                        'train_time_sec': total_train_time,
                        'inference_time_sec': total_infer_time,
                        'time_sec': total_train_time + total_infer_time,
                        'run': run_id,
                    })

                    if run_id == 0:
                        print(f"    [SCimilarity-{clf_name}] Acc: {metrics['accuracy']:.3f}, F1: {f1_val:.3f}, Train: {total_train_time:.1f}s, Infer: {total_infer_time:.1f}s")

                    # Store MLP predictions for UMAP and per-celltype analysis
                    if clf_name == 'mlp':
                        if generate_umap and run_id == 0:
                            mlp_predictions = pred.copy()

                        # Compute per-cell-type F1
                        per_ct_f1 = compute_per_celltype_f1(y_true, pred)
                        for ct, f1_val in per_ct_f1.items():
                            per_celltype_results.append({
                                'scenario': scenario['name'],
                                'cell_type': ct,
                                'f1': f1_val,
                                'run': run_id,
                            })

                except Exception as e:
                    if run_id == 0:
                        print(f"    [SCimilarity-{clf_name}] Error: {e}")
                finally:
                    gc.collect()

            # Ensemble Soft Voting
            try:
                # Training: ensemble fit
                ens_train_start = time.time()
                ensemble = train_ensemble(emb_ref, labels_ref, 'voting_soft')
                ens_train_time = time.time() - ens_train_start
                total_train_time = ref_embed_time + ens_train_time

                # Inference: ensemble predict + refinement
                ens_infer_start = time.time()
                pred_raw = predict_ensemble(ensemble, emb_query, 'voting_soft')
                pred = refine_predictions(emb_query, pred_raw, k=50)
                ens_infer_time = time.time() - ens_infer_start
                total_infer_time = query_embed_time + ens_infer_time

                metrics = compute_metrics(y_true=y_true, y_pred=pred, metrics=['accuracy', 'ari'])
                results.append({
                    'scenario': scenario['name'],
                    'method': 'SCimilarity-Ensemble',
                    'type': 'embedding-based',
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'f1_macro': compute_f1(y_true, pred),
                    'train_time_sec': total_train_time,
                    'inference_time_sec': total_infer_time,
                    'time_sec': total_train_time + total_infer_time,
                    'run': run_id,
                })
                if run_id == 0:
                    print(f"    [SCimilarity-Ensemble] Acc: {metrics['accuracy']:.3f}, F1: {results[-1]['f1_macro']:.3f}, Train: {total_train_time:.1f}s, Infer: {total_infer_time:.1f}s")
            except Exception as e:
                if run_id == 0:
                    print(f"    [SCimilarity-Ensemble] Error: {e}")
            finally:
                del emb_ref, emb_query
                gc.collect()

        except Exception as e:
            if run_id == 0:
                print(f"    SCimilarity error: {e}")

    # Cleanup
    del adata_ref, adata_query
    gc.collect()

    return results, per_celltype_results, embeddings_query, mlp_predictions, y_true


def main():
    print("=" * 80)
    print("Comprehensive Cell Type Annotation Benchmark")
    print("=" * 80)
    print("\nMethods:")
    print(f"  - CellTypist (reference-based): {RUN_CELLTYPIST}")
    print(f"  - SCimilarity + Ensemble:       {RUN_SCIMILARITY}")
    print(f"  - SingleR (reference-based):    {RUN_SINGLER}")
    print(f"  - scTab (zero-shot):            {RUN_SCTAB}")
    print(f"\nNumber of runs per scenario: {N_RUNS}")

    all_results = []
    all_per_celltype = []

    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH, backed='r')
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    for scenario in SCENARIOS:
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {scenario['name']}")
        print('=' * 80)

        for run_id in range(N_RUNS):
            print(f"\n  --- Run {run_id + 1}/{N_RUNS} ---")

            # Generate UMAP only on first run
            generate_umap = (run_id == 0)

            results, per_ct, emb_query, mlp_pred, y_true = run_single_experiment(
                scenario, adata, study_col, cell_type_col,
                run_id=run_id, generate_umap=generate_umap
            )

            all_results.extend(results)
            all_per_celltype.extend(per_ct)

            # Generate UMAP visualization (first run only)
            if generate_umap and emb_query is not None and mlp_pred is not None:
                try:
                    plot_umap_comparison(
                        adata_query=None,
                        embeddings_query=emb_query,
                        y_true=y_true,
                        y_pred=mlp_pred,
                        method_name='SCimilarity-mlp',
                        scenario_name=scenario['name'],
                        output_dir=FIGURE_DIR
                    )
                except Exception as e:
                    print(f"    UMAP error: {e}")

    # Convert to DataFrames
    df_results = pd.DataFrame(all_results)
    df_per_celltype = pd.DataFrame(all_per_celltype)

    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Box-whisker plot: F1 comparison across methods
    if len(df_results) > 0:
        print("\n  Creating F1 comparison boxplot...")
        plot_method_comparison_boxplot(df_results, metric='f1_macro', output_dir=FIGURE_DIR)

        print("\n  Creating runtime comparison boxplot (train vs inference breakdown)...")
        plot_timing_breakdown(df_results, output_dir=FIGURE_DIR)

    # Per-cell-type F1 for SCimilarity-mlp
    if len(df_per_celltype) > 0:
        print("\n  Creating per-cell-type F1 boxplot...")
        plot_per_celltype_f1(df_per_celltype, 'SCimilarity-mlp', output_dir=FIGURE_DIR)

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS (aggregated over {} runs)".format(N_RUNS))
    print("=" * 80)

    if len(df_results) == 0:
        print("\nNo results collected! Check if data files exist and methods ran successfully.")
        return

    # Aggregate results by scenario and method
    agg_dict = {
        'accuracy': ['mean', 'std'],
        'f1_macro': ['mean', 'std'],
        'ari': ['mean', 'std'],
        'time_sec': ['mean', 'std'],
    }
    # Add timing columns if they exist
    if 'train_time_sec' in df_results.columns:
        agg_dict['train_time_sec'] = ['mean', 'std']
    if 'inference_time_sec' in df_results.columns:
        agg_dict['inference_time_sec'] = ['mean', 'std']

    summary = df_results.groupby(['scenario', 'method']).agg(agg_dict).round(4)

    print("\n" + summary.to_string())

    # Save detailed results
    output_file = OUTPUT_DIR / "comprehensive_benchmark_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save per-cell-type results
    if len(df_per_celltype) > 0:
        percelltype_file = OUTPUT_DIR / "percelltype_f1_results.csv"
        df_per_celltype.to_csv(percelltype_file, index=False)
        print(f"Per-cell-type F1 saved to: {percelltype_file}")

    # Save summary
    summary_file = OUTPUT_DIR / "benchmark_summary.csv"
    summary.to_csv(summary_file)
    print(f"Summary saved to: {summary_file}")

    print(f"\nFigures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
