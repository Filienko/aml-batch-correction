#!/usr/bin/env python3
"""
Analyze Cross-Platform Results
================================

This script analyzes the results from exp_cross_platform.py and creates
publication-ready tables and statistics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "exp_cross_platform.csv"


def main():
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found: {RESULTS_FILE}")
        print("Please run exp_cross_platform.py first.")
        return

    print("="*80)
    print("Cross-Platform Transfer Analysis")
    print("="*80)

    # Load results
    df = pd.read_csv(RESULTS_FILE)

    print(f"\nLoaded {len(df)} results from {len(df['query_study'].unique())} studies")
    print(f"Models tested: {df['model'].unique().tolist()}")

    # 1. Performance by study and model
    print("\n" + "="*80)
    print("Table 1: Performance by Query Study and Model")
    print("="*80)

    pivot = df.pivot_table(
        values='ari',
        index='query_study',
        columns='model',
        aggfunc='mean'
    )

    # Add platform information
    platform_col = df.groupby('query_study')['query_platform'].first()
    pivot.insert(0, 'Platform', platform_col)

    # Sort by best SCimilarity model
    scim_cols = [c for c in pivot.columns if 'SCimilarity' in c]
    if scim_cols:
        pivot = pivot.sort_values(scim_cols[0], ascending=False)

    print("\nARI scores by study:")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # 2. Average performance by model
    print("\n" + "="*80)
    print("Table 2: Average Performance Across All Cross-Platform Transfers")
    print("="*80)

    avg_performance = df.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
    avg_performance = avg_performance.sort_values('ari', ascending=False)

    print(avg_performance.to_string(float_format=lambda x: f"{x:.4f}"))

    # 3. SCimilarity vs Traditional comparison
    print("\n" + "="*80)
    print("Table 3: SCimilarity vs Traditional ML")
    print("="*80)

    scim_models = df[df['model'].str.contains('SCimilarity', na=False)]
    trad_models = df[~df['model'].str.contains('SCimilarity', na=False)]

    scim_avg = scim_models.groupby('model')[['accuracy', 'ari', 'nmi']].mean()
    trad_avg = trad_models.groupby('model')[['accuracy', 'ari', 'nmi']].mean()

    print("\nSCimilarity (embeddings + classifier):")
    print(scim_avg.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nTraditional ML (raw gene expression):")
    print(trad_avg.to_string(float_format=lambda x: f"{x:.4f}"))

    # 4. Statistical summary
    print("\n" + "="*80)
    print("Statistical Summary")
    print("="*80)

    best_scim = scim_avg['ari'].idxmax()
    best_scim_ari = scim_avg.loc[best_scim, 'ari']

    best_trad = trad_avg['ari'].idxmax()
    best_trad_ari = trad_avg.loc[best_trad, 'ari']

    improvement = ((best_scim_ari - best_trad_ari) / best_trad_ari) * 100
    absolute_diff = best_scim_ari - best_trad_ari

    print(f"\nBest SCimilarity model:  {best_scim}")
    print(f"  Average ARI: {best_scim_ari:.4f}")
    print(f"\nBest Traditional model:  {best_trad}")
    print(f"  Average ARI: {best_trad_ari:.4f}")
    print(f"\nImprovement:")
    print(f"  Absolute: +{absolute_diff:.4f} ARI")
    print(f"  Relative: +{improvement:.1f}%")

    # 5. Per-study breakdown
    print("\n" + "="*80)
    print("Table 4: Best Model Performance Per Study")
    print("="*80)

    study_best = []
    for study in df['query_study'].unique():
        study_df = df[df['query_study'] == study]
        best_idx = study_df['ari'].idxmax()
        best_row = study_df.loc[best_idx]

        study_best.append({
            'Study': study,
            'Platform': best_row['query_platform'],
            'Best Model': best_row['model'],
            'ARI': best_row['ari'],
            'Accuracy': best_row['accuracy'],
            'Cells': int(best_row['n_cells'])
        })

    study_best_df = pd.DataFrame(study_best)
    print(study_best_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 6. Create publication table
    print("\n" + "="*80)
    print("Publication-Ready Table (LaTeX format)")
    print("="*80)

    # Create a nice table for publication
    pub_table = df.pivot_table(
        values=['ari', 'accuracy'],
        index='query_study',
        columns='model',
        aggfunc='mean'
    )

    print("\nNote: Copy this to your paper:")
    print("\\begin{table}[h]")
    print("\\caption{Cross-platform label transfer performance (Seq-Well → 10x Genomics)}")
    print("\\begin{tabular}{l" + "c" * len(df['model'].unique()) + "}")
    print("\\toprule")

    # Header
    models = sorted(df['model'].unique())
    print("Study & " + " & ".join(models) + " \\\\")
    print("\\midrule")

    # Data rows
    for study in sorted(df['query_study'].unique()):
        row_data = [study.replace('_', '\\_')]
        for model in models:
            study_model = df[(df['query_study'] == study) & (df['model'] == model)]
            if len(study_model) > 0:
                ari = study_model['ari'].values[0]
                row_data.append(f"{ari:.3f}")
            else:
                row_data.append("--")
        print(" & ".join(row_data) + " \\\\")

    print("\\midrule")

    # Average row
    row_data = ["Average"]
    for model in models:
        model_df = df[df['model'] == model]
        avg_ari = model_df['ari'].mean()
        row_data.append(f"\\textbf{{{avg_ari:.3f}}}")
    print(" & ".join(row_data) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # 7. Save summary
    summary_file = RESULTS_DIR / "exp_cross_platform_analysis.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Cross-Platform Transfer Analysis Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Reference: van_galen_2019 (Seq-Well)\n")
        f.write(f"Query: Multiple studies (10x Genomics)\n\n")
        f.write(f"Best SCimilarity: {best_scim} (ARI = {best_scim_ari:.4f})\n")
        f.write(f"Best Traditional: {best_trad} (ARI = {best_trad_ari:.4f})\n")
        f.write(f"Improvement: +{improvement:.1f}% ({absolute_diff:+.4f} ARI)\n\n")
        f.write("="*80 + "\n")
        f.write("Average Performance\n")
        f.write("="*80 + "\n")
        f.write(avg_performance.to_string())

    print(f"\n✓ Analysis saved to {summary_file}")

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
