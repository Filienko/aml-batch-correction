"""
Experiment: Ensemble Methods on SCimilarity Embeddings
=======================================================

Instead of using a single classifier on embeddings, combine multiple:
- KNN (local patterns)
- Logistic Regression (linear patterns)
- Random Forest (non-linear patterns)
- MLP (neural patterns)

Ensemble strategies:
1. Hard Voting: Majority vote across classifiers
2. Soft Voting: Average probabilities, then predict

Now includes LIVE CellTypist training for a fair baseline comparison.
"""
import os
import sys
import warnings
import gc
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.evaluation import compute_metrics
# IMPORT CELLTYPIST MODEL
from sccl.models.celltypist import CellTypistModel 

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set to None to use full data (Recommended for best performance)
MAX_CELLS_PER_STUDY = 1000

SCENARIOS = [
    {
        'name': 'Same-Platform: beneyto → jiang',
        'reference': 'beneyto-calabuig-2023',
        'query': 'jiang_2020',
    },
    {
        'name': 'Cross-Platform: van_galen → jiang',
        'reference': 'van_galen_2019',
        'query': 'jiang_2020',
    },
]


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
        'rf': RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.001, random_state=42),
    }
    return classifiers


def refine_predictions(embeddings, raw_predictions, k=50):
    """
    Smooths predictions using the query dataset's own internal structure.
    Increased k=50 for better stability on large datasets.
    """
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


def main():
    print("="*80)
    print("Ensemble Methods vs Trained CellTypist")
    print("="*80)
    
    results = []

    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"{scenario['name']}")
        print('='*80)

        # 1. Prepare Data
        adata_ref = subset_data(adata, studies=[scenario['reference']])
        adata_query = subset_data(adata, studies=[scenario['query']])

        if MAX_CELLS_PER_STUDY:
            if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_ref = adata_ref[indices].copy()
            if adata_query.n_obs > MAX_CELLS_PER_STUDY:
                indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
                adata_query = adata_query[indices].copy()

        print(f"  Reference: {adata_ref.n_obs:,} cells")
        print(f"  Query:     {adata_query.n_obs:,} cells")

        # 2. Train & Test CellTypist (The Real Baseline)
        print("\n  [Baseline-CellTypist]...", end=' ')
        try:
            ct_model = CellTypistModel(model=None)
            # Create a lightweight copy for CellTypist
            ct_model.fit(adata_ref, target_column=cell_type_col)

            metrics = compute_metrics(
                y_true=adata_query.obs[cell_type_col].values,
                y_pred=ct_pred,
                metrics=['accuracy', 'ari']
            )

            results.append({
                'scenario': scenario['name'],
                'method': 'Baseline-CellTypist',
                'ensemble_type': 'baseline',
                'accuracy': metrics['accuracy'],
                'ari': metrics['ari'],
            })
            print(f"✓ Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        # 3. SCimilarity Embeddings
        print("\n  Computing SCimilarity embeddings...")
        adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
        adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)
        
        embeddings_ref = get_scimilarity_embeddings(adata_ref_prep, MODEL_PATH)
        embeddings_query = get_scimilarity_embeddings(adata_query_prep, MODEL_PATH)
        labels_ref = adata_ref.obs[cell_type_col].values

        # 4. Individual Classifiers (With Refinement!)
        print("\n  Individual Classifiers:")
        classifiers = create_ensemble_classifiers()

        for clf_name, clf in classifiers.items():
            print(f"    [{clf_name}]...", end=' ')
            try:
                clf.fit(embeddings_ref, labels_ref)
                
                # Get RAW prediction
                pred_raw = clf.predict(embeddings_query)
                
                # Apply Refinement (Smooth with neighbors)
                pred = refine_predictions(embeddings_query, pred_raw, k=50)

                metrics = compute_metrics(
                    y_true=adata_query.obs[cell_type_col].values,
                    y_pred=pred,
                    metrics=['accuracy', 'ari']
                )

                results.append({
                    'scenario': scenario['name'],
                    'method': f'Individual-{clf_name}',
                    'ensemble_type': 'individual',
                    'accuracy': metrics['accuracy'],
                    'ari': metrics['ari'],
                })
                print(f"✓ Acc: {metrics['accuracy']:.3f}")

            except Exception as e:
                print(f"✗ Error: {e}")
            finally:
                gc.collect()

        # 5. Ensemble Soft Voting (With Refinement!)
        print(f"  [Ensemble-SoftVoting]...", end=' ')
        try:
            ensemble = train_ensemble(embeddings_ref, labels_ref, 'voting_soft')
            pred_raw = predict_ensemble(ensemble, embeddings_query, 'voting_soft')
            pred = refine_predictions(embeddings_query, pred_raw, k=50)

            metrics = compute_metrics(
                y_true=adata_query.obs[cell_type_col].values,
                y_pred=pred,
                metrics=['accuracy', 'ari']
            )

            results.append({
                'scenario': scenario['name'],
                'method': 'Ensemble-SoftVoting',
                'ensemble_type': 'voting_soft',
                'accuracy': metrics['accuracy'],
                'ari': metrics['ari'],
            })
            print(f"✓ Acc: {metrics['accuracy']:.3f}")

        except Exception as e:
            print(f"✗ Error: {e}")
        finally:
            del ensemble, pred
            gc.collect()

        # Cleanup for next scenario
        del adata_ref, adata_query
        del embeddings_ref, embeddings_query
        gc.collect()

    # Results Table
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    df_results = pd.DataFrame(results)

    for scenario_name in df_results['scenario'].unique():
        print(f"\n{scenario_name}:")
        scenario_df = df_results[df_results['scenario'] == scenario_name].copy()
        scenario_df = scenario_df.sort_values('accuracy', ascending=False)
        print(scenario_df[['method', 'accuracy', 'ari']].to_string(index=False))

    output_file = OUTPUT_DIR / "ensemble_results_with_celltypist.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
