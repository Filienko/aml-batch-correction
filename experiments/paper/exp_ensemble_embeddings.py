#!/usr/bin/env python3
"""
Experiment: Ensemble Methods on SCimilarity Embeddings
=======================================================

Instead of using a single classifier on embeddings, combine multiple:
- KNN (local patterns)
- Logistic Regression (linear patterns)
- Random Forest (non-linear patterns)
- SVM (margin-based patterns)

Ensemble strategies:
1. Hard Voting: Majority vote across classifiers
2. Soft Voting: Average probabilities, then predict
3. Weighted Voting: Weight by validation accuracy

This should improve beyond single classifiers (current best: 51.2%)
"""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sccl import Pipeline
from sccl.data import subset_data, preprocess_data, get_study_column, get_cell_type_column
from sccl.evaluation import compute_metrics

# Configuration
DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
MODEL_PATH = "/home/daniilf/aml-batch-correction/model_v1.1"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CELLS_PER_STUDY = 10000

# Test scenarios
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
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
    }
    return classifiers


def train_ensemble(embeddings_ref, labels_ref, ensemble_type='voting'):
    """Train ensemble of classifiers on embeddings.

    Parameters
    ----------
    embeddings_ref : np.ndarray
        Reference embeddings
    labels_ref : np.ndarray
        Reference labels
    ensemble_type : str
        'voting_hard', 'voting_soft', or 'individual'

    Returns
    -------
    ensemble : fitted ensemble or dict of fitted classifiers
    """
    classifiers = create_ensemble_classifiers()

    if ensemble_type == 'voting_hard':
        # Hard voting: majority vote
        estimators = [(name, clf) for name, clf in classifiers.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
        ensemble.fit(embeddings_ref, labels_ref)
        return ensemble

    elif ensemble_type == 'voting_soft':
        # Soft voting: average probabilities
        estimators = [(name, clf) for name, clf in classifiers.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(embeddings_ref, labels_ref)
        return ensemble

    elif ensemble_type == 'individual':
        # Train each classifier individually (for later weighted combination)
        fitted_classifiers = {}
        for name, clf in classifiers.items():
            clf.fit(embeddings_ref, labels_ref)
            fitted_classifiers[name] = clf
        return fitted_classifiers

    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


def predict_ensemble(ensemble, embeddings_query, ensemble_type='voting'):
    """Make predictions with ensemble.

    Parameters
    ----------
    ensemble : fitted ensemble or dict of fitted classifiers
    embeddings_query : np.ndarray
        Query embeddings
    ensemble_type : str
        'voting_hard', 'voting_soft', or 'individual'

    Returns
    -------
    predictions : np.ndarray
        Predicted labels
    """
    if ensemble_type in ['voting_hard', 'voting_soft']:
        return ensemble.predict(embeddings_query)

    elif ensemble_type == 'individual':
        # Get predictions from each classifier
        all_predictions = []
        for name, clf in ensemble.items():
            preds = clf.predict(embeddings_query)
            all_predictions.append(preds)

        # Majority vote
        all_predictions = np.array(all_predictions)
        from collections import Counter

        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)

        return np.array(final_predictions)


def main():
    """Run ensemble experiment."""
    print("="*80)
    print("Ensemble Methods on SCimilarity Embeddings")
    print("="*80)
    print("\nStrategies:")
    print("  1. Individual classifiers (KNN, LogReg, RF, SVM)")
    print("  2. Hard Voting: Majority vote across classifiers")
    print("  3. Soft Voting: Average probabilities, then predict")
    print("\nGoal: Improve beyond best single classifier (51.2%)")
    print("="*80)

    results = []

    # Detect columns
    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH)
    study_col = get_study_column(adata)
    cell_type_col = get_cell_type_column(adata)

    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"{scenario['name']}")
        print(f"  Reference: {scenario['reference']}")
        print(f"  Query:     {scenario['query']}")
        print('='*80)

        # Get data
        adata_ref = subset_data(adata, studies=[scenario['reference']])
        adata_query = subset_data(adata, studies=[scenario['query']])

        # Subsample
        if adata_ref.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_ref.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_ref = adata_ref[indices].copy()

        if adata_query.n_obs > MAX_CELLS_PER_STUDY:
            indices = np.random.choice(adata_query.n_obs, MAX_CELLS_PER_STUDY, replace=False)
            adata_query = adata_query[indices].copy()

        print(f"  Reference: {adata_ref.n_obs:,} cells")
        print(f"  Query:     {adata_query.n_obs:,} cells")

        # Preprocess
        adata_ref_prep = preprocess_data(adata_ref.copy(), batch_key=None)
        adata_query_prep = preprocess_data(adata_query.copy(), batch_key=None)

        # Get embeddings (once for all classifiers)
        print("\n  Computing SCimilarity embeddings...")
        embeddings_ref = get_scimilarity_embeddings(adata_ref_prep, MODEL_PATH)
        embeddings_query = get_scimilarity_embeddings(adata_query_prep, MODEL_PATH)
        labels_ref = adata_ref.obs[cell_type_col].values

        print(f"  Embeddings: {embeddings_ref.shape[1]} dimensions")

        # Test individual classifiers first
        print("\n  Individual Classifiers:")
        classifiers = create_ensemble_classifiers()

        for clf_name, clf in classifiers.items():
            print(f"    [{clf_name}]...", end=' ')

            try:
                clf.fit(embeddings_ref, labels_ref)
                pred = clf.predict(embeddings_query)

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

                print(f"✓ Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")

            except Exception as e:
                print(f"✗ Error: {e}")

            finally:
                del pred
                gc.collect()

        # Test Hard Voting ensemble
        print(f"\n  [Ensemble-HardVoting]...", end=' ')
        try:
            ensemble = train_ensemble(embeddings_ref, labels_ref, 'voting_hard')
            pred = predict_ensemble(ensemble, embeddings_query, 'voting_hard')

            metrics = compute_metrics(
                y_true=adata_query.obs[cell_type_col].values,
                y_pred=pred,
                metrics=['accuracy', 'ari']
            )

            results.append({
                'scenario': scenario['name'],
                'method': 'Ensemble-HardVoting',
                'ensemble_type': 'voting_hard',
                'accuracy': metrics['accuracy'],
                'ari': metrics['ari'],
            })

            print(f"✓ Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            del ensemble, pred
            gc.collect()

        # Test Soft Voting ensemble
        print(f"  [Ensemble-SoftVoting]...", end=' ')
        try:
            ensemble = train_ensemble(embeddings_ref, labels_ref, 'voting_soft')
            pred = predict_ensemble(ensemble, embeddings_query, 'voting_soft')

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

            print(f"✓ Acc: {metrics['accuracy']:.3f}, ARI: {metrics['ari']:.3f}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            del ensemble, pred
            gc.collect()

        # Cleanup
        del adata_ref, adata_query, adata_ref_prep, adata_query_prep
        del embeddings_ref, embeddings_query
        gc.collect()

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    df_results = pd.DataFrame(results)

    # Group by scenario
    for scenario_name in df_results['scenario'].unique():
        print(f"\n{scenario_name}:")
        scenario_df = df_results[df_results['scenario'] == scenario_name].copy()
        scenario_df = scenario_df.sort_values('accuracy', ascending=False)
        print(scenario_df[['method', 'accuracy', 'ari']].to_string(index=False))

    # Average performance
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE")
    print("="*80)

    avg_results = df_results.groupby('method').agg({
        'accuracy': 'mean',
        'ari': 'mean'
    }).reset_index()
    avg_results = avg_results.sort_values('accuracy', ascending=False)

    print("\n" + avg_results.to_string(index=False))

    # Save results
    output_file = OUTPUT_DIR / "ensemble_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Compare to baselines
    print("\n" + "="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)

    best_ensemble = avg_results.iloc[0]
    print(f"\nBest Ensemble: {best_ensemble['method']}")
    print(f"  Accuracy: {best_ensemble['accuracy']:.3f}")
    print(f"  ARI: {best_ensemble['ari']:.3f}")

    print("\nBaselines:")
    print(f"  SCimilarity+KNN (pure):  51.2% (single classifier)")
    print(f"  CellTypist (pure):       59.9% (target)")

    gap = 0.599 - best_ensemble['accuracy']
    print(f"\nGap to CellTypist: {gap:.3f} ({gap/0.599*100:.1f}%)")

    if best_ensemble['accuracy'] > 0.512:
        improvement = best_ensemble['accuracy'] - 0.512
        print(f"Improvement over single KNN: {improvement:+.3f} ({improvement/0.512*100:+.1f}%)")


if __name__ == "__main__":
    main()
