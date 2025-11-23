#!/usr/bin/env python
"""
Complete Atlas Replication Pipeline

This script orchestrates all 4 phases of the atlas replication project:
1. Phase 1: Load ground truth + create raw problem
2. Phase 2: Project to SCimilarity latent space
3. Phase 3: Quantitative benchmarking
4. Phase 4: Biological discovery

Usage:
    python run_full_pipeline.py              # Run all phases
    python run_full_pipeline.py --phase 2    # Run specific phase only
    python run_full_pipeline.py --skip-phase1 # Skip phase 1
"""

import argparse
import sys
import time
from pathlib import Path

# Import phase modules
import phase1_ground_truth as phase1
import phase2_scimilarity_projection as phase2
import phase3_quantitative_benchmark as phase3
import phase4_biological_discovery as phase4


def print_banner(text):
    """Print a nice banner"""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def check_data_availability():
    """Check if required data files exist"""
    print("Checking data availability...")

    required_files = {
        'AML_scAtlas.h5ad': Path("data/AML_scAtlas.h5ad"),
    }

    all_exist = True
    for name, path in required_files.items():
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} not found at {path}")
            all_exist = False

    if not all_exist:
        print("\n⚠ Missing required data files!")
        print("  Please see DATA_SOURCES.md for download instructions")
        print("  Or run: python phase1_ground_truth.py to start anyway")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    print()


def run_phase_with_timing(phase_num, phase_func, phase_name):
    """Run a phase and track execution time"""
    print_banner(f"PHASE {phase_num}: {phase_name}")

    start_time = time.time()

    try:
        result = phase_func()
        elapsed = time.time() - start_time

        print(f"\n✓ Phase {phase_num} completed in {elapsed//60:.0f} min {elapsed%60:.0f} sec")

        return result, True

    except Exception as e:
        elapsed = time.time() - start_time

        print(f"\n✗ Phase {phase_num} failed after {elapsed//60:.0f} min {elapsed%60:.0f} sec")
        print(f"Error: {e}")

        import traceback
        traceback.print_exc()

        return None, False


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='Run the complete AML Atlas Replication pipeline'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run only a specific phase (1-4)'
    )
    parser.add_argument(
        '--skip-phase1',
        action='store_true',
        help='Skip Phase 1 (useful if data already prepared)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip initial data availability checks'
    )

    args = parser.parse_args()

    # Print header
    print_banner("AML ATLAS REPLICATION WITH FOUNDATION MODELS")
    print("Research Question:")
    print("  Can SCimilarity automatically replicate the complex, manually-curated")
    print("  annotation pipeline used in the AML scAtlas?\n")

    # Check data
    if not args.skip_checks:
        check_data_availability()

    # Track overall timing
    pipeline_start = time.time()
    results = {}
    all_success = True

    # Determine which phases to run
    if args.phase:
        phases_to_run = [args.phase]
    else:
        phases_to_run = [1, 2, 3, 4]
        if args.skip_phase1:
            phases_to_run = [2, 3, 4]

    # Run phases
    for phase_num in phases_to_run:
        if phase_num == 1:
            result, success = run_phase_with_timing(
                1, phase1.main, "Data and Model Setup"
            )
            results['phase1'] = result

        elif phase_num == 2:
            result, success = run_phase_with_timing(
                2, phase2.main, "SCimilarity Projection"
            )
            results['phase2'] = result

        elif phase_num == 3:
            result, success = run_phase_with_timing(
                3, phase3.main, "Quantitative Benchmarking"
            )
            results['phase3'] = result

        elif phase_num == 4:
            result, success = run_phase_with_timing(
                4, phase4.main, "Biological Discovery"
            )
            results['phase4'] = result

        if not success:
            all_success = False
            print(f"\n⚠ Phase {phase_num} failed, stopping pipeline")
            break

        # Small pause between phases
        if phase_num < max(phases_to_run):
            print("\n" + "-" * 80)
            time.sleep(2)

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    print_banner("PIPELINE COMPLETE")

    print("Execution Summary:")
    print(f"  Total time: {pipeline_elapsed//60:.0f} min {pipeline_elapsed%60:.0f} sec")
    print(f"  Phases run: {len(phases_to_run)}")
    print(f"  Status: {'✓ SUCCESS' if all_success else '✗ FAILED'}\n")

    if all_success:
        print("Results Location:")
        print("  results_atlas_replication/")
        print("    ├── figures/")
        print("    │   ├── fig1a_ground_truth_atlas.pdf")
        print("    │   ├── fig1b_raw_problem.pdf")
        print("    │   ├── fig1c_scimilarity_solution.pdf")
        print("    │   ├── fig2_quantitative_comparison.pdf")
        print("    │   └── fig2_metrics_heatmap.pdf")
        print("    ├── metrics/")
        print("    │   ├── quantitative_comparison.csv")
        print("    │   └── marker_gene_validation.csv")
        print("    ├── hierarchy/")
        print("    │   ├── fig3_hierarchy_dendrogram.pdf")
        print("    │   └── fig3_principal_axes.pdf")
        print("    └── data/")
        print("        ├── merged_raw_problem.h5ad")
        print("        └── scimilarity_solution.h5ad")

        print("\nNext Steps:")
        print("  1. Review figures in results_atlas_replication/figures/")
        print("  2. Check metrics in results_atlas_replication/metrics/")
        print("  3. Write up results for publication")

        print("\nKey Questions to Answer:")
        print("  • Does SCimilarity achieve batch mixing ≥ scVI?")
        print("  • Does SCimilarity preserve biology better than manual curation?")
        print("  • Does SCimilarity recover the known cell type hierarchies?")
        print("  • Are marker genes correctly enriched in SCimilarity clusters?")

    else:
        print("\n⚠ Pipeline did not complete successfully")
        print("  Check error messages above for details")
        print("  You can re-run individual phases with --phase <N>")

    return results, all_success


if __name__ == "__main__":
    results, success = main()
    sys.exit(0 if success else 1)
