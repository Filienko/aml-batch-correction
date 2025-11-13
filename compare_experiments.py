#!/usr/bin/env python
"""
Compare Results: Cross-Mechanism vs Within-Mechanism Batch Correction

This script compares the performance of batch correction methods across
two experimental setups:
1. Cross-mechanism: Different technologies (microwell, well-based, droplet)
2. Within-mechanism: Same technology (all droplet-based)

Generates comparison tables and visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CROSS_MECHANISM_FILE = "results_cross_mechanism/cross_mechanism_results.csv"
WITHIN_MECHANISM_FILE = "results_within_mechanism/within_mechanism_results.csv"
OUTPUT_DIR = "results_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    """
    Load results from both experiments.

    Returns:
        cross_df, within_df: DataFrames with results
    """
    print("="*80)
    print("LOADING EXPERIMENT RESULTS")
    print("="*80)

    # Load cross-mechanism results
    if os.path.exists(CROSS_MECHANISM_FILE):
        cross_df = pd.read_csv(CROSS_MECHANISM_FILE, index_col=0)
        print(f"\n✓ Loaded cross-mechanism results:")
        print(f"  File: {CROSS_MECHANISM_FILE}")
        print(f"  Methods: {', '.join(cross_df.index.tolist())}")
    else:
        print(f"\n✗ Cross-mechanism results not found: {CROSS_MECHANISM_FILE}")
        print(f"  Run experiment_cross_mechanism.py first")
        cross_df = None

    # Load within-mechanism results
    if os.path.exists(WITHIN_MECHANISM_FILE):
        within_df = pd.read_csv(WITHIN_MECHANISM_FILE, index_col=0)
        print(f"\n✓ Loaded within-mechanism results:")
        print(f"  File: {WITHIN_MECHANISM_FILE}")
        print(f"  Methods: {', '.join(within_df.index.tolist())}")
    else:
        print(f"\n✗ Within-mechanism results not found: {WITHIN_MECHANISM_FILE}")
        print(f"  Run experiment_within_mechanism.py first")
        within_df = None

    if cross_df is None and within_df is None:
        print("\n✗ No results available for comparison")
        return None, None

    return cross_df, within_df


def create_comparison_table(cross_df, within_df):
    """
    Create a comparison table showing metrics for both experiments.

    Returns:
        DataFrame with comparison
    """
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLE")
    print("="*80)

    # Key metrics to compare
    metrics = ['Total', 'Batch correction', 'Bio conservation']

    # Get common methods
    if cross_df is not None and within_df is not None:
        common_methods = list(set(cross_df.index) & set(within_df.index))
    elif cross_df is not None:
        common_methods = cross_df.index.tolist()
    else:
        common_methods = within_df.index.tolist()

    print(f"\nMethods to compare: {', '.join(common_methods)}")

    # Create comparison table
    comparison_data = []

    for method in common_methods:
        for metric in metrics:
            row = {
                'Method': method,
                'Metric': metric,
            }

            # Add cross-mechanism values
            if cross_df is not None and method in cross_df.index:
                row['Cross-Mechanism'] = cross_df.loc[method, metric]
            else:
                row['Cross-Mechanism'] = np.nan

            # Add within-mechanism values
            if within_df is not None and method in within_df.index:
                row['Within-Mechanism'] = within_df.loc[method, metric]
            else:
                row['Within-Mechanism'] = np.nan

            # Calculate difference
            if not np.isnan(row.get('Cross-Mechanism', np.nan)) and \
               not np.isnan(row.get('Within-Mechanism', np.nan)):
                row['Difference'] = row['Within-Mechanism'] - row['Cross-Mechanism']
            else:
                row['Difference'] = np.nan

            comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save
    output_file = os.path.join(OUTPUT_DIR, "experiment_comparison.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison table: {output_file}")

    return comparison_df


def print_comparison_summary(comparison_df):
    """
    Print a formatted summary of the comparison.
    """
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    methods = comparison_df['Method'].unique()

    for method in methods:
        print(f"\n{method}:")
        print("-" * 60)

        method_data = comparison_df[comparison_df['Method'] == method]

        print(f"{'Metric':<25s} {'Cross':>10s} {'Within':>10s} {'Diff':>10s}")
        print("-" * 60)

        for _, row in method_data.iterrows():
            metric = row['Metric']
            cross = row['Cross-Mechanism']
            within = row['Within-Mechanism']
            diff = row['Difference']

            # Format values
            cross_str = f"{cross:.4f}" if not np.isnan(cross) else "N/A"
            within_str = f"{within:.4f}" if not np.isnan(within) else "N/A"
            diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"

            # Add indicator for improvement
            if not np.isnan(diff):
                if metric == 'Batch correction':
                    indicator = "✓" if diff > 0 else "✗"
                elif metric == 'Bio conservation':
                    indicator = "✓" if diff > 0 else "✗"
                elif metric == 'Total':
                    indicator = "✓" if diff > 0 else "✗"
                else:
                    indicator = ""
            else:
                indicator = ""

            print(f"{metric:<25s} {cross_str:>10s} {within_str:>10s} {diff_str:>10s} {indicator}")


def create_visualizations(comparison_df, cross_df, within_df):
    """
    Create comparison visualizations.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. Grouped bar chart: Total scores
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = comparison_df['Method'].unique()
    x = np.arange(len(methods))
    width = 0.35

    cross_total = []
    within_total = []

    for method in methods:
        method_data = comparison_df[
            (comparison_df['Method'] == method) &
            (comparison_df['Metric'] == 'Total')
        ]

        cross_val = method_data['Cross-Mechanism'].values[0]
        within_val = method_data['Within-Mechanism'].values[0]

        cross_total.append(cross_val if not np.isnan(cross_val) else 0)
        within_total.append(within_val if not np.isnan(within_val) else 0)

    ax.bar(x - width/2, cross_total, width, label='Cross-Mechanism', alpha=0.8)
    ax.bar(x + width/2, within_total, width, label='Within-Mechanism', alpha=0.8)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Score', fontsize=12, fontweight='bold')
    ax.set_title('Batch Correction Performance: Cross vs Within Mechanism',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "total_score_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # 2. Heatmap: All metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['Total', 'Batch correction', 'Bio conservation']

    # Cross-mechanism heatmap
    if cross_df is not None:
        cross_data = cross_df[metrics].T
        sns.heatmap(cross_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Score'})
        axes[0].set_title('Cross-Mechanism', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Metric', fontsize=11)
        axes[0].set_xlabel('Method', fontsize=11)

    # Within-mechanism heatmap
    if within_df is not None:
        within_data = within_df[metrics].T
        sns.heatmap(within_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Score'})
        axes[1].set_title('Within-Mechanism', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Metric', fontsize=11)
        axes[1].set_xlabel('Method', fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "metrics_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # 3. Difference plot
    fig, ax = plt.subplots(figsize=(10, 6))

    diff_data = comparison_df.pivot(index='Metric', columns='Method', values='Difference')

    sns.heatmap(diff_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
               ax=ax, cbar_kws={'label': 'Difference (Within - Cross)'})

    ax.set_title('Performance Difference: Within-Mechanism vs Cross-Mechanism\n(Positive = Within performs better)',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=11)
    ax.set_xlabel('Method', fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "performance_difference.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def main():
    """
    Main comparison pipeline.
    """
    print("="*80)
    print("BATCH CORRECTION EXPERIMENT COMPARISON")
    print("="*80)

    # Load results
    cross_df, within_df = load_results()

    if cross_df is None and within_df is None:
        print("\n✗ No results to compare. Please run experiments first:")
        print("  1. python experiment_cross_mechanism.py")
        print("  2. python experiment_within_mechanism.py")
        return

    # Create comparison table
    comparison_df = create_comparison_table(cross_df, within_df)

    # Print summary
    print_comparison_summary(comparison_df)

    # Create visualizations
    create_visualizations(comparison_df, cross_df, within_df)

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    if cross_df is not None and within_df is not None:
        # Calculate average differences
        avg_diff = comparison_df.groupby('Metric')['Difference'].mean()

        print("\nAverage performance difference (Within - Cross):")
        for metric, diff in avg_diff.items():
            direction = "better" if diff > 0 else "worse"
            print(f"  {metric}: {diff:+.4f} ({direction} within-mechanism)")

        print("\nConclusions:")
        if avg_diff['Batch correction'] > 0:
            print("  ✓ Batch correction is easier within same technology")
        else:
            print("  ⚠ Cross-mechanism correction performs surprisingly well")

        if avg_diff['Bio conservation'] > 0:
            print("  ✓ Biological signal better preserved within same technology")
        else:
            print("  ⚠ Cross-mechanism preserves biology as well as within-mechanism")

    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    print("  - experiment_comparison.csv")
    print("  - total_score_comparison.png")
    print("  - metrics_heatmap.png")
    print("  - performance_difference.png")


if __name__ == "__main__":
    main()
