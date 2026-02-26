"""Main CLI entry point for SCCL.

Provides subcommands:
    sccl predict      Label transfer (reference -> query) or train/test prediction
    sccl evaluate     Evaluate a model with train/test split
    sccl compare      Compare multiple models side-by-side
    sccl generate     Generate synthetic data for testing
    sccl info         Inspect an .h5ad file
    sccl list-models  Show available models
    sccl run          Run pipeline from a YAML config file
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..pipeline import Pipeline, run_pipeline_from_config
from ..data import (
    generate_synthetic_data,
    load_data,
    detect_batch_key,
    detect_label_key,
)
from ..data.preprocessing import preprocess_data
from ..evaluation.metrics import compute_metrics
from ..models import AVAILABLE_MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Set up logging based on --verbose flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )


def _resolve_target(adata, explicit_target):
    """Return the target column, auto-detecting if not provided."""
    if explicit_target:
        if explicit_target not in adata.obs.columns:
            sys.exit(
                f"Error: target column '{explicit_target}' not found in data. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return explicit_target
    detected = detect_label_key(adata)
    if detected is None:
        sys.exit(
            "Error: could not auto-detect a cell-type column. "
            "Pass --target explicitly."
        )
    return detected


def _resolve_batch_key(adata, explicit_key):
    """Return the batch key, auto-detecting if not provided."""
    if explicit_key:
        return explicit_key
    return detect_batch_key(adata)


def _parse_model_params(raw):
    """Parse 'key=val,key=val' into a dict, casting numbers automatically."""
    if not raw:
        return {}
    params = {}
    for pair in raw.split(","):
        if "=" not in pair:
            sys.exit(f"Error: bad --model-params entry '{pair}' (expected key=value)")
        key, val = pair.split("=", 1)
        # Try int, then float, then leave as string
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        params[key.strip()] = val
    return params


def _print_metrics(metrics):
    """Pretty-print a metrics dict to stdout."""
    print()
    print("=" * 40)
    print("RESULTS")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name:20s}  {value:.4f}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_predict(args):
    """Label transfer: train on reference, predict on query.

    Three modes:
      1. --reference + --data   : train on reference, predict on data
      2. --data + --test-size   : split data, train on train, predict on test
      3. --data only            : predict with a pre-trained model
    """
    adata = load_data(args.data)
    batch_key = _resolve_batch_key(adata, args.batch_key)
    model_params = _parse_model_params(args.model_params)
    do_preprocess = not args.no_preprocess

    pipeline = Pipeline(
        model=args.model,
        batch_key=batch_key,
        preprocess=False,  # we handle preprocessing ourselves for ref/query
        model_params=model_params,
    )

    # ----- Mode 1: separate reference file -----------------------------------
    if args.reference:
        adata_ref = load_data(args.reference)
        target = _resolve_target(adata_ref, args.target)

        if do_preprocess:
            adata_ref = preprocess_data(adata_ref.copy(), batch_key=batch_key)
            adata = preprocess_data(adata.copy(), batch_key=batch_key)

        print(f"Training {args.model} on reference ({adata_ref.n_obs} cells)...")
        pipeline.model.fit(adata_ref, target_column=target, batch_key=batch_key)

        print(f"Predicting on query ({adata.n_obs} cells)...")
        predictions = pipeline.model.predict(adata, batch_key=batch_key)

        # Evaluate if query also has labels
        if args.target and args.target in adata.obs.columns:
            y_true = adata.obs[args.target].values
            metrics = compute_metrics(y_true=y_true, y_pred=predictions)
            _print_metrics(metrics)

    # ----- Mode 2: single file with train/test split -------------------------
    elif args.test_size:
        target = _resolve_target(adata, args.target)
        from sklearn.model_selection import train_test_split

        indices = np.arange(adata.n_obs)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=42,
            stratify=adata.obs[target].values,
        )
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        if do_preprocess:
            adata_train = preprocess_data(adata_train, batch_key=batch_key)
            adata_test = preprocess_data(adata_test, batch_key=batch_key)

        print(f"Training {args.model} on {adata_train.n_obs} cells...")
        pipeline.model.fit(adata_train, target_column=target, batch_key=batch_key)

        print(f"Predicting on {adata_test.n_obs} cells...")
        predictions = pipeline.model.predict(adata_test, batch_key=batch_key)

        # Evaluate
        y_true = adata_test.obs[target].values
        metrics = compute_metrics(y_true=y_true, y_pred=predictions)
        _print_metrics(metrics)

        # For output: only test cells
        adata = adata_test

    # ----- Mode 3: no training (pre-trained model) ---------------------------
    else:
        if do_preprocess:
            adata = preprocess_data(adata.copy(), batch_key=batch_key)

        print(f"Predicting on {adata.n_obs} cells with {args.model}...")
        predictions = pipeline.model.predict(adata, batch_key=batch_key)

    # ----- Save / print predictions ------------------------------------------
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cell_id": adata.obs_names,
            "prediction": predictions,
        }).to_csv(out, index=False)
        print(f"Saved predictions to {out}")
    else:
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\nPredicted {len(predictions)} cells into {len(unique)} types:")
        for ct, n in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  {ct:30s}  {n:>6d}")


def cmd_evaluate(args):
    """Evaluate a model with a train/test split and print metrics."""
    adata = load_data(args.data)
    target = _resolve_target(adata, args.target)
    batch_key = _resolve_batch_key(adata, args.batch_key)
    model_params = _parse_model_params(args.model_params)

    pipeline = Pipeline(
        model=args.model,
        batch_key=batch_key,
        preprocess=not args.no_preprocess,
        model_params=model_params,
    )

    metrics = pipeline.evaluate(
        adata=adata,
        target_column=target,
        test_size=args.test_size,
    )

    _print_metrics(metrics)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved results to {out}")


def cmd_compare(args):
    """Compare multiple models side-by-side."""
    adata = load_data(args.data)
    target = _resolve_target(adata, args.target)
    batch_key = _resolve_batch_key(adata, args.batch_key)

    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in AVAILABLE_MODELS:
            sys.exit(
                f"Error: unknown model '{m}'. "
                f"Run 'sccl list-models' to see options."
            )

    pipeline = Pipeline(
        model=models[0],
        batch_key=batch_key,
        preprocess=not args.no_preprocess,
    )

    comparison = pipeline.compare_models(
        adata=adata,
        target_column=target,
        models=models,
        test_size=args.test_size,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string())
    print("=" * 60)

    best = comparison["accuracy"].idxmax()
    print(f"\nBest model by accuracy: {best} ({comparison.loc[best, 'accuracy']:.4f})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out)
        print(f"\nSaved comparison to {out}")


def cmd_generate(args):
    """Generate synthetic data for testing."""
    adata = generate_synthetic_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_cell_types=args.n_cell_types,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)

    print(f"Saved synthetic data to {output}")
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  {args.n_cell_types} cell types, {args.n_batches} batches")
    print(f"\nCell types: {sorted(adata.obs['cell_type'].unique().tolist())}")


def cmd_info(args):
    """Inspect an .h5ad dataset."""
    adata = load_data(args.data)

    print(f"File:       {args.data}")
    print(f"Cells:      {adata.n_obs:,}")
    print(f"Genes:      {adata.n_vars:,}")
    print(f"Columns:    {list(adata.obs.columns)}")

    # Auto-detect key columns
    label_col = detect_label_key(adata)
    batch_col = detect_batch_key(adata)

    if label_col:
        n_types = adata.obs[label_col].nunique()
        print(f"\nCell-type column (auto-detected): '{label_col}' ({n_types} types)")
        top = adata.obs[label_col].value_counts().head(10)
        for ct, n in top.items():
            print(f"  {str(ct):30s}  {n:>6d}")
        if n_types > 10:
            print(f"  ... and {n_types - 10} more")
    else:
        print("\nNo cell-type column detected.")

    if batch_col:
        n_batches = adata.obs[batch_col].nunique()
        print(f"\nBatch column (auto-detected): '{batch_col}' ({n_batches} batches)")
        for b, n in adata.obs[batch_col].value_counts().items():
            print(f"  {str(b):30s}  {n:>6d}")
    else:
        print("\nNo batch column detected.")


def cmd_list_models(_args):
    """List available models."""
    print("Available models:\n")
    for name in sorted(AVAILABLE_MODELS):
        cls = AVAILABLE_MODELS[name]
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"  {name:25s}  {doc}")
    print()
    print("Optional models not installed are hidden. Install extras with:")
    print("  pip install -e '.[scimilarity]'")
    print("  pip install -e '.[scvi]'")
    print("  pip install celltypist")


def cmd_run(args):
    """Run pipeline from a YAML config file."""
    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = run_pipeline_from_config(config)

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(results["comparison"].to_string())
    print("=" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        results["comparison"].to_csv(out)
        print(f"\nSaved results to {out}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common_args(parser):
    """Add flags shared across several subcommands."""
    parser.add_argument(
        "--batch-key",
        help="Batch/study column (auto-detected if omitted)",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip preprocessing (normalise, HVG, PCA)",
    )
    parser.add_argument(
        "--model-params",
        help="Extra model params as key=value,key=value "
             "(e.g. n_estimators=200,max_depth=10)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show debug-level logging",
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sccl",
        description="SCCL: Single Cell Classification Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  sccl info --data my_data.h5ad
  sccl list-models
  sccl generate -o synthetic.h5ad
  sccl evaluate --data synthetic.h5ad --model random_forest
  sccl predict --reference ref.h5ad --data query.h5ad --model random_forest --target cell_type -o preds.csv
  sccl predict --data data.h5ad --model random_forest --test-size 0.2
  sccl compare --data data.h5ad --models random_forest,svm,knn
  sccl run --config example_config.yaml
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show debug-level logging",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- predict -------------------------------------------------------------
    p = subparsers.add_parser(
        "predict",
        help="Label transfer: train on reference, predict on query",
        description=(
            "Three modes:\n"
            "  1) --reference + --data : train on reference, predict on data\n"
            "  2) --data + --test-size : split data, train on train, predict on test\n"
            "  3) --data alone         : predict with a pre-trained model"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", required=True,
                   help="Path to .h5ad file (query data or single dataset)")
    p.add_argument("--reference",
                   help="Path to labeled reference .h5ad (for label transfer)")
    p.add_argument("--model", required=True, choices=sorted(AVAILABLE_MODELS),
                   help="Model to use")
    p.add_argument("--target",
                   help="Cell-type column name (auto-detected if omitted)")
    p.add_argument("--test-size", type=float,
                   help="If no --reference, split data into train/test (e.g. 0.2)")
    p.add_argument("-o", "--output", help="Save predictions to CSV")
    _add_common_args(p)
    p.set_defaults(func=cmd_predict)

    # --- evaluate ------------------------------------------------------------
    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate a model with train/test split",
    )
    p.add_argument("--data", required=True, help="Path to .h5ad file")
    p.add_argument("--model", required=True, choices=sorted(AVAILABLE_MODELS),
                   help="Model to use")
    p.add_argument("--target",
                   help="Cell-type column (auto-detected if omitted)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test set fraction (default: 0.2)")
    p.add_argument("-o", "--output", help="Save results to JSON")
    _add_common_args(p)
    p.set_defaults(func=cmd_evaluate)

    # --- compare -------------------------------------------------------------
    p = subparsers.add_parser(
        "compare",
        help="Compare multiple models side-by-side",
    )
    p.add_argument("--data", required=True, help="Path to .h5ad file")
    p.add_argument("--models", required=True,
                   help="Comma-separated model names (e.g. random_forest,svm,knn)")
    p.add_argument("--target",
                   help="Cell-type column (auto-detected if omitted)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test set fraction (default: 0.2)")
    p.add_argument("-o", "--output", help="Save comparison to CSV")
    _add_common_args(p)
    p.set_defaults(func=cmd_compare)

    # --- generate ------------------------------------------------------------
    p = subparsers.add_parser(
        "generate",
        help="Generate synthetic data for testing",
    )
    p.add_argument("-o", "--output", required=True, help="Output .h5ad file")
    p.add_argument("--n-cells", type=int, default=1000,
                   help="Number of cells (default: 1000)")
    p.add_argument("--n-genes", type=int, default=2000,
                   help="Number of genes (default: 2000)")
    p.add_argument("--n-cell-types", type=int, default=5,
                   help="Cell types (default: 5)")
    p.add_argument("--n-batches", type=int, default=3,
                   help="Batches (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.set_defaults(func=cmd_generate)

    # --- info ----------------------------------------------------------------
    p = subparsers.add_parser(
        "info",
        help="Inspect an .h5ad dataset",
    )
    p.add_argument("--data", required=True, help="Path to .h5ad file")
    p.set_defaults(func=cmd_info)

    # --- list-models ---------------------------------------------------------
    p = subparsers.add_parser(
        "list-models",
        help="List available models",
    )
    p.set_defaults(func=cmd_list_models)

    # --- run (config) --------------------------------------------------------
    p = subparsers.add_parser(
        "run",
        help="Run pipeline from YAML config file",
    )
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("-o", "--output", help="Save results to CSV")
    p.set_defaults(func=cmd_run)

    # --- parse and dispatch --------------------------------------------------
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _configure_logging(args.verbose if hasattr(args, "verbose") else False)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(args, "verbose") and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
