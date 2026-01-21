#!/usr/bin/env python3
"""
Experiment 1: Annotation Replication
=====================================
Can SCimilarity approximate expert consensus annotations?

Expected: ARI > 0.70 indicates good replication

Usage:
    python exp1_annotation_replication.py
    python exp1_annotation_replication.py --data /path/to/data.h5ad
    python exp1_annotation_replication.py --study-col batch --cell-type-col annotation
"""

import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# Add SCCL to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sccl import Pipeline
from sccl.data import subset_data, list_columns
from sccl.evaluation import compute_metrics, compute_per_class_metrics, plot_confusion_matrix

# Default configuration
DEFAULT_DATA_PATH = "/home/daniilf/full_aml_tasks/batch_correction/data/AML_scAtlas.h5ad"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Studies with Van Galen-style consensus annotations
VAN_GALEN_STUDIES = [
    'van_galen_2019',
    'zhang_2023',
#    'beneyto-calabuig-2023',
#    'jiang_2020',
#    'velten_2021',
#    'zhai_2022',
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 1: Annotation Replication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (auto-detect column names)
  python exp1_annotation_replication.py

  # Custom data path
  python exp1_annotation_replication.py --data /path/to/data.h5ad

  # Custom column names
  python exp1_annotation_replication.py --study-col batch_id --cell-type-col annotation

  # List available columns first
  python exp1_annotation_replication.py --list-columns
        """
    )

    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                       help='Path to h5ad data file')
    parser.add_argument('--study-col', type=str, default=None,
                       help='Study/batch column name (auto-detected if not specified)')
    parser.add_argument('--cell-type-col', type=str, default=None,
                       help='Cell type column name (auto-detected if not specified)')
    parser.add_argument('--list-columns', action='store_true',
                       help='List available columns and exit')
    parser.add_argument('--species', type=str, default='human',
                       choices=['human', 'mouse'],
                       help='Species for SCimilarity gene alignment')

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    print("="*80)
    print("EXPERIMENT 1: Annotation Replication")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    adata = sc.read_h5ad(args.data)
    print(f"   Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    # List columns if requested
    if args.list_columns:
        columns = list_columns(adata, verbose=True)
        return

    # Detect or use specified columns
    from sccl.data import get_study_column, get_cell_type_column

    study_col = get_study_column(adata, args.study_col)
    cell_type_col = get_cell_type_column(adata, args.cell_type_col)

    print(f"\n   Using columns:")
    print(f"     Study: {study_col}")
    print(f"     Cell Type: {cell_type_col}")

    # Subset to Van Galen studies
    print("\n2. Subsetting to Van Galen studies...")
    available_studies = adata.obs[study_col].unique()
    valid_studies = [s for s in VAN_GALEN_STUDIES if s in available_studies]

    if not valid_studies:
        print("   ERROR: No Van Galen studies found!")
        print(f"   Available studies: {list(available_studies)[:5]}...")
        return

    print(f"   Using {len(valid_studies)} studies:")
    for study in valid_studies:
        n_cells = (adata.obs[study_col] == study).sum()
        print(f"     • {study}: {n_cells:,} cells")

    adata = subset_data(adata, studies=valid_studies, study_col=study_col)
    print(f"   Subset: {adata.n_obs:,} cells")

    # Run SCimilarity
    print("\n3. Running SCimilarity predictions...")
    pipeline = Pipeline(
        model="scimilarity",
        batch_key=study_col,
        model_params={'species': args.species}
    )
    predictions = pipeline.predict(adata.copy(), target_column=cell_type_col)

    # Compute overall metrics
    print("\n4. Computing metrics...")
    metrics = compute_metrics(
        y_true=adata.obs[cell_type_col].values,
        y_pred=predictions,
        adata=adata,
        metrics=['accuracy', 'ari', 'nmi', 'f1']
    )

    # Per-class metrics
    per_class = compute_per_class_metrics(
        y_true=adata.obs[cell_type_col].values,
        y_pred=predictions
    )
    per_class_df = pd.DataFrame(per_class).T.sort_values('support')

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():20s}: {value:.4f}")

    print("\nPer-Class Performance (sorted by rarity):")
    print(per_class_df.to_string())

    # Identify rare types
    threshold = adata.n_obs * 0.01
    rare_types = per_class_df[per_class_df['support'] < threshold]

    if len(rare_types) > 0:
        print(f"\nRare Cell Types (< 1% frequency):")
        print(rare_types[['f1', 'precision', 'recall', 'support']].to_string())
        print(f"\n  Average F1 on rare types: {rare_types['f1'].mean():.3f}")

    # Save results
    print(f"\n5. Saving results to {OUTPUT_DIR}/")

    # Overall metrics
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "exp1_overall_metrics.csv", index=False)

    # Per-class
    per_class_df.to_csv(OUTPUT_DIR / "exp1_perclass_performance.csv")

    # Confusion matrix plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plot_confusion_matrix(
        y_true=adata.obs[cell_type_col].values,
        y_pred=predictions,
        normalize=True,
        figsize=(14, 12),
        save=str(OUTPUT_DIR / "exp1_confusion_matrix.pdf")
    )
    plt.close()

    print("   ✓ exp1_overall_metrics.csv")
    print("   ✓ exp1_perclass_performance.csv")
    print("   ✓ exp1_confusion_matrix.pdf")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    ari = metrics['ari']
    if ari > 0.80:
        print("✅ EXCELLENT: SCimilarity closely approximates expert consensus")
    elif ari > 0.70:
        print("✅ GOOD: SCimilarity approximates expert consensus well")
    elif ari > 0.60:
        print("⚠️ MODERATE: Some agreement but room for improvement")
    else:
        print("❌ LOW: Significant discrepancy from expert annotations")

    print(f"\nKey finding: ARI = {ari:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
