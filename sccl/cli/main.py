"""Main CLI entry point."""

import argparse
import logging
import sys
from pathlib import Path
import yaml

from ..pipeline import Pipeline, run_pipeline_from_config
from ..data import generate_synthetic_data, load_data
from ..models import AVAILABLE_MODELS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_predict(args):
    """Run prediction command."""
    logger.info("Running prediction...")

    # Load data
    adata = load_data(args.data)

    # Create pipeline
    pipeline = Pipeline(
        model=args.model,
        batch_key=args.batch_key,
        preprocess=not args.no_preprocess,
    )

    # Subset if requested
    subset_params = None
    if args.subset_studies:
        subset_params = {'studies': args.subset_studies.split(',')}

    # Predict
    predictions = pipeline.predict(
        adata=adata,
        target_column=args.target if args.train_split else None,
        subset_params=subset_params,
    )

    # Save predictions
    if args.output:
        import pandas as pd
        output = Path(args.output)
        pd.DataFrame({
            'cell_id': adata.obs_names,
            'prediction': predictions,
        }).to_csv(output, index=False)
        logger.info(f"Saved predictions to {output}")
    else:
        logger.info(f"Predictions: {predictions[:10]}...")

    logger.info("Prediction completed!")


def cmd_evaluate(args):
    """Run evaluation command."""
    logger.info("Running evaluation...")

    # Load data
    adata = load_data(args.data)

    # Create pipeline
    pipeline = Pipeline(
        model=args.model,
        batch_key=args.batch_key,
        preprocess=not args.no_preprocess,
    )

    # Evaluate
    metrics = pipeline.evaluate(
        adata=adata,
        target_column=args.target,
        test_size=args.test_size,
    )

    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    for metric, value in metrics.items():
        logger.info(f"{metric:20s}: {value:.4f}")
    logger.info("="*50)

    # Save results
    if args.output:
        import json
        output = Path(args.output)
        with open(output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved results to {output}")


def cmd_compare(args):
    """Run model comparison command."""
    logger.info("Running model comparison...")

    # Load data
    adata = load_data(args.data)

    # Parse models
    models = args.models.split(',')

    # Create pipeline
    pipeline = Pipeline(
        model=models[0],  # Dummy, will be replaced
        batch_key=args.batch_key,
        preprocess=not args.no_preprocess,
    )

    # Compare
    comparison = pipeline.compare_models(
        adata=adata,
        target_column=args.target,
        models=models,
        test_size=args.test_size,
    )

    # Save results
    if args.output:
        output = Path(args.output)
        comparison.to_csv(output)
        logger.info(f"Saved comparison to {output}")

    logger.info("Comparison completed!")


def cmd_generate(args):
    """Generate synthetic data command."""
    logger.info("Generating synthetic data...")

    adata = generate_synthetic_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_cell_types=args.n_cell_types,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)

    logger.info(f"Saved synthetic data to {output}")
    logger.info(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    logger.info(f"  {args.n_cell_types} cell types, {args.n_batches} batches")


def cmd_run(args):
    """Run pipeline from config file."""
    logger.info(f"Running pipeline from config: {args.config}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Run
    results = run_pipeline_from_config(config)

    logger.info("Pipeline completed!")

    # Save results
    if args.output:
        output = Path(args.output)
        results['comparison'].to_csv(output)
        logger.info(f"Saved results to {output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SCCL: Single Cell Classification Library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict cell types')
    predict_parser.add_argument('--data', required=True, help='Path to h5ad file')
    predict_parser.add_argument('--model', required=True, choices=list(AVAILABLE_MODELS.keys()),
                               help='Model to use')
    predict_parser.add_argument('--target', help='Target column (for supervised models)')
    predict_parser.add_argument('--batch-key', help='Batch column name')
    predict_parser.add_argument('--subset-studies', help='Comma-separated list of studies to include')
    predict_parser.add_argument('--train-split', type=float, help='Fraction for training (if supervised)')
    predict_parser.add_argument('--no-preprocess', action='store_true',
                               help='Skip preprocessing')
    predict_parser.add_argument('--output', help='Output file for predictions')
    predict_parser.set_defaults(func=cmd_predict)

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--data', required=True, help='Path to h5ad file')
    eval_parser.add_argument('--model', required=True, choices=list(AVAILABLE_MODELS.keys()),
                            help='Model to use')
    eval_parser.add_argument('--target', required=True, help='Target column')
    eval_parser.add_argument('--batch-key', help='Batch column name')
    eval_parser.add_argument('--test-size', type=float, default=0.2,
                            help='Test set fraction (default: 0.2)')
    eval_parser.add_argument('--no-preprocess', action='store_true',
                            help='Skip preprocessing')
    eval_parser.add_argument('--output', help='Output file for results (JSON)')
    eval_parser.set_defaults(func=cmd_evaluate)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--data', required=True, help='Path to h5ad file')
    compare_parser.add_argument('--models', required=True,
                               help='Comma-separated list of models')
    compare_parser.add_argument('--target', required=True, help='Target column')
    compare_parser.add_argument('--batch-key', help='Batch column name')
    compare_parser.add_argument('--test-size', type=float, default=0.2,
                               help='Test set fraction (default: 0.2)')
    compare_parser.add_argument('--no-preprocess', action='store_true',
                               help='Skip preprocessing')
    compare_parser.add_argument('--output', help='Output CSV file for comparison')
    compare_parser.set_defaults(func=cmd_compare)

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--output', required=True, help='Output h5ad file')
    gen_parser.add_argument('--n-cells', type=int, default=1000,
                           help='Number of cells (default: 1000)')
    gen_parser.add_argument('--n-genes', type=int, default=2000,
                           help='Number of genes (default: 2000)')
    gen_parser.add_argument('--n-cell-types', type=int, default=5,
                           help='Number of cell types (default: 5)')
    gen_parser.add_argument('--n-batches', type=int, default=3,
                           help='Number of batches (default: 3)')
    gen_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    gen_parser.set_defaults(func=cmd_generate)

    # Run command (from config)
    run_parser = subparsers.add_parser('run', help='Run pipeline from config file')
    run_parser.add_argument('--config', required=True, help='Path to YAML config file')
    run_parser.add_argument('--output', help='Output CSV file for results')
    run_parser.set_defaults(func=cmd_run)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
