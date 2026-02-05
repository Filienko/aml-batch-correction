"""
Experiment: Comprehensive Cell Type Annotation Benchmark
=========================================================

Compares multiple cell type annotation methods:
1. CellTypist (reference-based, trained on reference)
2. SCimilarity + Ensemble (embedding-based classifiers)
3. SingleR (reference-based, correlation-based)
4. scTab (zero-shot foundation model)

All methods are evaluated on the same reference/query pairs for fair comparison.
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
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.evaluation import compute_metrics
from sccl.models.celltypist import CellTypistModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

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

SCENARIOS = [
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
# Which methods to run (set to False to skip)
RUN_CELLTYPIST = True
RUN_SCIMILARITY = True
RUN_SINGLER = True
RUN_SCTAB = True


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
# SINGLER METHODS
# ==============================================================================

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
        gt_labels_set = set(np.unique(ground_truth_labels[pd.notna(ground_truth_labels)]))
        unmapped = set(np.unique(harmonized)) - gt_labels_set

        for pred_label in unmapped:
            pred_lower = pred_label.lower()
            for gt_label in gt_labels_set:
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

def main():
    print("=" * 80)
    print("Comprehensive Cell Type Annotation Benchmark")
    print("=" * 80)
    print("\nMethods:")
    print(f"  - CellTypist (reference-based): {RUN_CELLTYPIST}")
    print(f"  - SCimilarity + Ensemble:       {RUN_SCIMILARITY}")
    print(f"  - SingleR (reference-based):    {RUN_SINGLER}")
    print(f"  - scTab (zero-shot):            {RUN_SCTAB}")

    results = []

    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH, backed='r')
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    for scenario in SCENARIOS:
        print(f"\n{'=' * 80}")
        print(f"{scenario['name']}")
        print('=' * 80)

        # 1. Prepare Data
        adata_ref = subset_data(adata, studies=[scenario['reference']]).to_memory()
        adata_query = subset_data(adata, studies=[scenario['query']]).to_memory()

        if MAX_CELLS_PER_STUDY:
            if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_ref = adata_ref[indices].copy()
            if adata_query.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_query = adata_query[indices].copy()

        print(f"  Reference: {adata_ref.n_obs:,} cells")
        print(f"  Query:     {adata_query.n_obs:,} cells")

        y_true = adata_query.obs[cell_type_col].values

        # =====================================================================
        # Method 1: CellTypist (Reference-based)
        # =====================================================================
        if RUN_CELLTYPIST:
            print("\n  [CellTypist]...", end=' ')
            try:
                start = time.time()
                ct_model = CellTypistModel(majority_voting=True)
                ct_model.fit(adata_ref, target_column=cell_type_col)
                ct_pred = ct_model.predict(adata_query)
                elapsed = time.time() - start

                metrics = compute_metrics(y_true=y_true, y_pred=ct_pred, metrics=['accuracy', 'ari'])
                results.append({
                    'scenario': scenario['name'],
                    'method': 'CellTypist',
                    'type': 'reference-based',
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                    'time_sec': elapsed,
                })
                print(f"Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}, Time: {elapsed:.1f}s")

            except Exception as e:
                print(f"Error: {e}")
            finally:
                gc.collect()

        # =====================================================================
        # Method 2: SingleR (Reference-based)
        # =====================================================================
        if RUN_SINGLER:
            print("\n  [SingleR]...", end=' ')
            try:
                start = time.time()
                singler_pred = run_singler_prediction(adata_ref, adata_query, cell_type_col)
                elapsed = time.time() - start

                if singler_pred is not None:
                    metrics = compute_metrics(y_true=y_true, y_pred=singler_pred, metrics=['accuracy', 'ari'])
                    results.append({
                        'scenario': scenario['name'],
                        'method': 'SingleR',
                        'type': 'reference-based',
                        'accuracy': metrics['accuracy'],
                        'ari': metrics['ari'],
                        'time_sec': elapsed,
                    })
                    print(f"Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}, Time: {elapsed:.1f}s")
                else:
                    print("Failed")

            except Exception as e:
                print(f"Error: {e}")
            finally:
                gc.collect()

        # =====================================================================
        # Method 3: scTab (Zero-shot Foundation Model)
        # =====================================================================
        if RUN_SCTAB:
            print("\n  [scTab]...", end=' ')
            try:
                start = time.time()
                sctab_pred = run_sctab_prediction(adata_query, cell_type_col)
                elapsed = time.time() - start

                if sctab_pred is not None:
                    metrics = compute_metrics(y_true=y_true, y_pred=sctab_pred, metrics=['accuracy', 'ari'])
                    results.append({
                        'scenario': scenario['name'],
                        'method': 'scTab',
                        'type': 'zero-shot',
                        'accuracy': metrics['accuracy'],
                        'ari': metrics['ari'],
                        'time_sec': elapsed,
                    })
                    print(f"Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}, Time: {elapsed:.1f}s")
                else:
                    print("Failed (model files not found or error)")

            except Exception as e:
                print(f"Error: {e}")
            finally:
                gc.collect()

        # =====================================================================
        # Method 4: SCimilarity + Ensemble
        # =====================================================================
        if RUN_SCIMILARITY:
            print("\n  Computing SCimilarity embeddings...")
            try:
                adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
                adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

                embeddings_ref = get_scimilarity_embeddings(adata_ref_prep, MODEL_PATH)
                embeddings_query = get_scimilarity_embeddings(adata_query_prep, MODEL_PATH)
                labels_ref = adata_ref.obs[cell_type_col].values

                del adata_ref_prep, adata_query_prep
                gc.collect()

                # Individual classifiers
                print("\n  Individual Classifiers:")
                classifiers = create_ensemble_classifiers()

                for clf_name, clf in classifiers.items():
                    print(f"    [{clf_name}]...", end=' ')
                    try:
                        start = time.time()
                        clf.fit(embeddings_ref, labels_ref)
                        pred_raw = clf.predict(embeddings_query)
                        pred = refine_predictions(embeddings_query, pred_raw, k=50)
                        elapsed = time.time() - start

                        metrics = compute_metrics(y_true=y_true, y_pred=pred, metrics=['accuracy', 'ari'])
                        results.append({
                            'scenario': scenario['name'],
                            'method': f'SCimilarity-{clf_name}',
                            'type': 'embedding-based',
                            'accuracy': metrics['accuracy'],
                            'ari': metrics['ari'],
                            'time_sec': elapsed,
                        })
                        print(f"Acc: {metrics['accuracy']:.3f}")

                    except Exception as e:
                        print(f"Error: {e}")
                    finally:
                        gc.collect()

                # Ensemble Soft Voting
                print(f"  [Ensemble-SoftVoting]...", end=' ')
                try:
                    start = time.time()
                    ensemble = train_ensemble(embeddings_ref, labels_ref, 'voting_soft')
                    pred_raw = predict_ensemble(ensemble, embeddings_query, 'voting_soft')
                    pred = refine_predictions(embeddings_query, pred_raw, k=50)
                    elapsed = time.time() - start

                    metrics = compute_metrics(y_true=y_true, y_pred=pred, metrics=['accuracy', 'ari'])
                    results.append({
                        'scenario': scenario['name'],
                        'method': 'SCimilarity-Ensemble',
                        'type': 'embedding-based',
                        'accuracy': metrics['accuracy'],
                        'ari': metrics['ari'],
                        'time_sec': elapsed,
                    })
                    print(f"Acc: {metrics['accuracy']:.3f}")

                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    del embeddings_ref, embeddings_query
                    gc.collect()

            except Exception as e:
                print(f"  SCimilarity error: {e}")

        # Cleanup for next scenario
        del adata_ref, adata_query
        gc.collect()

    # Results Table
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    for scenario_name in df_results['scenario'].unique():
        print(f"\n{scenario_name}:")
        scenario_df = df_results[df_results['scenario'] == scenario_name].copy()
        scenario_df = scenario_df.sort_values('accuracy', ascending=False)
        print(scenario_df[['method', 'type', 'accuracy', 'ari', 'time_sec']].to_string(index=False))

    output_file = OUTPUT_DIR / "comprehensive_benchmark_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
