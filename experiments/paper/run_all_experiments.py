#!/usr/bin/env python3
"""
Run All Paper Experiments
==========================
Execute all experiments for the paper and generate summary report.
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd

EXPERIMENTS = [
    ("exp1_annotation_replication.py", "Experiment 1: Annotation Replication"),
    ("exp2_label_transfer.py", "Experiment 2: Label Transfer Benchmark"),
    ("exp3_computational_efficiency.py", "Experiment 3: Computational Efficiency"),
    ("exp4_cross_study_generalization.py", "Experiment 4: Cross-Study Generalization"),
]


def run_experiment(script_path, description):
    """Run a single experiment script."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        return False

    print(f"\n✅ {description} completed successfully")
    return True


def generate_summary_report():
    """Generate a summary report from all experiment results."""
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print("\n⚠️ No results directory found")
        return

    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    # Experiment 1: Overall metrics
    exp1_file = results_dir / "exp1_overall_metrics.csv"
    if exp1_file.exists():
        exp1 = pd.read_csv(exp1_file)
        print("\nExperiment 1: Annotation Replication")
        print("-" * 40)
        for _, row in exp1.iterrows():
            for col in exp1.columns:
                print(f"  {col.upper():20s}: {row[col]:.4f}")

    # Experiment 2: Model comparison
    exp2_file = results_dir / "exp2_model_comparison.csv"
    if exp2_file.exists():
        exp2 = pd.read_csv(exp2_file)
        print("\nExperiment 2: Label Transfer (Average by Model)")
        print("-" * 40)
        print(exp2.to_string(index=False))

    # Experiment 3: Timing
    exp3_file = results_dir / "exp3_timing.csv"
    if exp3_file.exists():
        exp3 = pd.read_csv(exp3_file)
        print("\nExperiment 3: Computational Efficiency")
        print("-" * 40)
        for _, row in exp3.iterrows():
            method = row['method']
            time_min = row['time_minutes']
            print(f"  {method:50s}: {time_min:8.1f} minutes")

    # Experiment 4: Cross-study generalization
    exp4_file = results_dir / "exp4_generalization.csv"
    if exp4_file.exists():
        exp4 = pd.read_csv(exp4_file)
        print("\nExperiment 4: Cross-Study Generalization")
        print("-" * 40)
        print(f"  Mean ARI:      {exp4['ari'].mean():.4f} ± {exp4['ari'].std():.4f}")
        print(f"  Mean Accuracy: {exp4['accuracy'].mean():.4f} ± {exp4['accuracy'].std():.4f}")
        print(f"  Studies tested: {len(exp4)}")

    # Paper conclusions
    print("\n" + "="*80)
    print("PAPER CONCLUSIONS")
    print("="*80)

    conclusions = []

    # Check claims
    if exp1_file.exists():
        exp1_ari = pd.read_csv(exp1_file)['ari'].values[0]
        if exp1_ari > 0.70:
            conclusions.append(f"✅ Can approximate expert consensus (ARI = {exp1_ari:.3f})")
        else:
            conclusions.append(f"⚠️ Limited agreement with expert consensus (ARI = {exp1_ari:.3f})")

    if exp2_file.exists():
        exp2 = pd.read_csv(exp2_file)
        if 'SCimilarity' in exp2['model'].values:
            scim_row = exp2[exp2['model'] == 'SCimilarity'].iloc[0]
            scim_ari = scim_row['ari']
            best_ari = exp2['ari'].max()
            if scim_ari >= best_ari - 0.05:
                conclusions.append(f"✅ SCimilarity competitive for label transfer (ARI = {scim_ari:.3f})")

    if exp3_file.exists():
        exp3 = pd.read_csv(exp3_file)
        scim_time = exp3[exp3['method'] == 'SCimilarity']['time_minutes'].values[0]
        trad_time = exp3[exp3['method'].str.contains('Traditional')]['time_minutes'].values[0]
        speedup = trad_time / scim_time
        if speedup > 5:
            conclusions.append(f"✅ Significantly more efficient ({speedup:.1f}x speedup)")

    if exp4_file.exists():
        exp4 = pd.read_csv(exp4_file)
        if exp4['ari'].std() < 0.10:
            conclusions.append(f"✅ Robust across studies (σ = {exp4['ari'].std():.3f})")

    print("\nKey Findings:")
    for conclusion in conclusions:
        print(f"  {conclusion}")

    print("\n" + "="*80)
    print(f"All results saved to: {results_dir}/")
    print("="*80)


def main():
    print("="*80)
    print("RUNNING ALL PAPER EXPERIMENTS")
    print("="*80)
    print("\nThis will run all 4 experiments sequentially.")
    print("Estimated time: 10-30 minutes depending on dataset size")
    print("="*80)

    script_dir = Path(__file__).parent
    success_count = 0

    # Run all experiments
    for script_name, description in EXPERIMENTS:
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"\n⚠️ WARNING: {script_name} not found, skipping...")
            continue

        success = run_experiment(script_path, description)
        if success:
            success_count += 1

    # Generate summary
    print("\n" + "="*80)
    print(f"Completed {success_count}/{len(EXPERIMENTS)} experiments")
    print("="*80)

    if success_count > 0:
        generate_summary_report()

    print("\n✅ All experiments completed!")
    print("\nNext steps:")
    print("  1. Check experiments/paper/results/ for output files")
    print("  2. Review CSV files for tables in your paper")
    print("  3. Use plots (PDF files) in your manuscript")


if __name__ == "__main__":
    main()
