#!/usr/bin/env python3
"""Evaluation script for hierarchical demand reconciliation model."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.data.loader import (
    M5DataLoader,
    HierarchicalTimeSeriesDataset,
)
from hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.data.preprocessing import (
    HierarchicalPreprocessor,
)
from hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.models.model import (
    HierarchicalReconciliationTransformer,
)
from hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.evaluation.metrics import (
    HierarchicalMetrics,
    WRMSSEMetric,
    CoherenceErrorMetric,
    AnomalyMetrics,
)
from hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.utils.config import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical demand reconciliation model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to M5 dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )

    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use synthetic data instead of M5 dataset",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file",
    )

    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create visualization plots",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def create_synthetic_data() -> pd.DataFrame:
    """Create synthetic data for evaluation (same as in train.py)."""
    np.random.seed(42)

    n_items = 50  # Smaller for evaluation
    n_days = 100
    n_stores = 5
    n_depts = 3
    n_cats = 2
    n_states = 2

    # Create hierarchy structure
    items = [f"FOODS_{i//25:01d}_{(i%25)//5:03d}_{i%5:03d}_evaluation" for i in range(n_items)]
    stores = [f"{['CA', 'TX'][i%n_states]}_{(i%n_stores):01d}" for i in range(n_items)]
    depts = [f"FOODS_{i//17:01d}" for i in range(n_items)]
    cats = [f"FOODS" for _ in range(n_items)]
    states = [store.split('_')[0] for store in stores]

    # Generate time series data
    data = []

    for i, (item_id, store_id, dept_id, cat_id, state_id) in enumerate(
        zip(items, stores, depts, cats, states)
    ):
        # Base demand pattern
        base_level = 8 + np.random.uniform(-2, 2)
        trend = np.random.uniform(-0.005, 0.005)

        # Weekly seasonality
        weekly_pattern = 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 7 + np.random.uniform(0, 2*np.pi))

        # Random events/anomalies
        anomaly_days = np.random.choice(n_days, size=np.random.randint(2, 8), replace=False)
        anomaly_effect = np.zeros(n_days)
        anomaly_effect[anomaly_days] = np.random.uniform(3, 10, len(anomaly_days))

        # Combine effects
        sales_pattern = (
            base_level
            + trend * np.arange(n_days)
            + weekly_pattern
            + anomaly_effect
            + np.random.normal(0, 0.5, n_days)
        )

        # Ensure non-negative
        sales_pattern = np.maximum(0, sales_pattern)

        # Create daily records
        for day in range(n_days):
            data.append({
                "item_id": item_id,
                "dept_id": dept_id,
                "cat_id": cat_id,
                "store_id": store_id,
                "state_id": state_id,
                "d": f"d_{day + 1}",
                "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=day),
                "sales": sales_pattern[day],
            })

    return pd.DataFrame(data)


def load_model_and_data(
    checkpoint_path: str,
    config_path: str,
    data_path: str,
    use_synthetic: bool = False,
) -> Tuple[HierarchicalReconciliationTransformer, HierarchicalTimeSeriesDataset, HierarchicalPreprocessor]:
    """Load model and prepare evaluation data."""
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(config_path)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = HierarchicalReconciliationTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Prepare data
    if use_synthetic:
        logger.info("Using synthetic data for evaluation")
        sales_data = create_synthetic_data()

        # Create simple hierarchy
        hierarchy_data = {
            "item_id": sales_data.copy(),
            "total": sales_data.groupby("date")["sales"].sum().reset_index(),
        }
        hierarchy_data["total"]["item_id"] = "TOTAL"

        prepared_data = sales_data

    else:
        logger.info("Loading M5 data for evaluation")
        try:
            loader = M5DataLoader(data_path=data_path, download=False)
            sales_df, calendar_df, prices_df = loader.load_raw_data()
            hierarchy_data = loader.create_hierarchical_structure(sales_df)
            prepared_data = loader.prepare_features(sales_df, calendar_df, prices_df)
        except FileNotFoundError:
            logger.warning("M5 data not found, using synthetic data")
            return load_model_and_data(checkpoint_path, config_path, data_path, use_synthetic=True)

    # Initialize preprocessor
    preprocessor = HierarchicalPreprocessor(
        sequence_length=config.data.sequence_length,
        prediction_length=config.data.prediction_length,
        anomaly_window=config.model.anomaly_detector.window_size,
        anomaly_threshold=config.model.anomaly_detector.contamination,
    )

    # Use only recent data for evaluation
    recent_data = prepared_data.tail(1000) if len(prepared_data) > 1000 else prepared_data

    # Fit preprocessor and transform data
    preprocessor.fit(recent_data)
    transformed_data = preprocessor.transform(recent_data)

    # Create dataset
    hierarchy_mapping = {
        level: idx for idx, level in enumerate(config.data.hierarchy_levels)
    }

    eval_dataset = HierarchicalTimeSeriesDataset(
        data=transformed_data,
        sequence_length=config.data.sequence_length,
        prediction_length=config.data.prediction_length,
        hierarchy_mapping=hierarchy_mapping,
        features=["sales"],
    )

    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    return model, eval_dataset, preprocessor


def evaluate_model(
    model: HierarchicalReconciliationTransformer,
    dataset: HierarchicalTimeSeriesDataset,
    preprocessor: HierarchicalPreprocessor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict:
    """Evaluate model on dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Running model evaluation...")

    if device is None:
        device = get_device()

    model = model.to(device)

    # Create data loader
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    all_hierarchy_levels = []
    all_anomaly_scores = []
    all_base_predictions = []
    all_reconciled_predictions = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            hierarchy_levels = batch["hierarchy_level"].to(device)

            # Forward pass
            outputs = model(inputs, hierarchy_levels)

            # Collect results
            all_predictions.append(outputs["reconciled_forecasts"].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_hierarchy_levels.append(hierarchy_levels.cpu().numpy())
            all_anomaly_scores.append(outputs["anomaly_scores"].cpu().numpy())
            all_base_predictions.append(outputs["base_forecasts"].cpu().numpy())
            all_reconciled_predictions.append(outputs["reconciled_forecasts"].cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    hierarchy_levels = np.concatenate(all_hierarchy_levels, axis=0)
    anomaly_scores = np.concatenate(all_anomaly_scores, axis=0)
    base_predictions = np.concatenate(all_base_predictions, axis=0)
    reconciled_predictions = np.concatenate(all_reconciled_predictions, axis=0)

    logger.info(f"Collected {len(predictions)} predictions")

    # Flatten for metric computation
    predictions_flat = predictions.reshape(-1, predictions.shape[-1]).mean(axis=1)
    targets_flat = targets.reshape(-1, targets.shape[-1]).mean(axis=1)
    base_predictions_flat = base_predictions.reshape(-1, base_predictions.shape[-1]).mean(axis=1)
    reconciled_predictions_flat = reconciled_predictions.reshape(-1, reconciled_predictions.shape[-1]).mean(axis=1)

    # Expand hierarchy levels to match flattened shape
    hierarchy_levels_flat = np.repeat(hierarchy_levels, predictions.shape[1])

    # Initialize metrics
    hierarchical_metrics = HierarchicalMetrics()

    # Compute comprehensive metrics
    metrics = hierarchical_metrics.compute_all_metrics(
        predictions=reconciled_predictions_flat,
        targets=targets_flat,
        hierarchy_levels=hierarchy_levels_flat,
        anomaly_scores=anomaly_scores.flatten(),
    )

    # Compute improvement from reconciliation
    improvement_metrics = hierarchical_metrics.evaluate_forecast_improvement(
        base_predictions=base_predictions_flat,
        reconciled_predictions=reconciled_predictions_flat,
        targets=targets_flat,
        hierarchy_levels=hierarchy_levels_flat,
    )

    metrics.update(improvement_metrics)

    # Compute additional metrics
    additional_metrics = {
        "n_samples": len(predictions),
        "n_hierarchy_levels": len(np.unique(hierarchy_levels)),
        "anomaly_detection_rate": float(np.mean(anomaly_scores > 0.5)),
        "mean_prediction": float(np.mean(predictions_flat)),
        "mean_target": float(np.mean(targets_flat)),
        "prediction_std": float(np.std(predictions_flat)),
        "target_std": float(np.std(targets_flat)),
    }

    metrics.update(additional_metrics)

    logger.info("Evaluation completed")

    return {
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
        "hierarchy_levels": hierarchy_levels,
        "anomaly_scores": anomaly_scores,
        "base_predictions": base_predictions,
        "reconciled_predictions": reconciled_predictions,
    }


def save_results(
    results: Dict,
    output_dir: Path,
    save_predictions: bool = False,
) -> None:
    """Save evaluation results."""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)

    logger.info(f"Metrics saved to {metrics_file}")

    # Save predictions if requested
    if save_predictions:
        predictions_file = output_dir / "predictions.npz"
        np.savez(
            predictions_file,
            predictions=results["predictions"],
            targets=results["targets"],
            hierarchy_levels=results["hierarchy_levels"],
            anomaly_scores=results["anomaly_scores"],
            base_predictions=results["base_predictions"],
            reconciled_predictions=results["reconciled_predictions"],
        )

        logger.info(f"Predictions saved to {predictions_file}")


def create_visualizations(
    results: Dict,
    output_dir: Path,
) -> None:
    """Create visualization plots."""
    logger = logging.getLogger(__name__)
    logger.info("Creating visualization plots...")

    plt.style.use('default')
    fig_size = (12, 8)

    # 1. Prediction vs Target scatter plot
    plt.figure(figsize=fig_size)
    predictions_flat = results["predictions"].reshape(-1)
    targets_flat = results["targets"].reshape(-1)

    plt.scatter(targets_flat, predictions_flat, alpha=0.5, s=1)
    plt.plot([0, max(targets_flat.max(), predictions_flat.max())],
             [0, max(targets_flat.max(), predictions_flat.max())], 'r--', label='Perfect prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.savefig(output_dir / "predictions_vs_targets.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Anomaly score distribution
    plt.figure(figsize=fig_size)
    anomaly_scores = results["anomaly_scores"].flatten()
    plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.savefig(output_dir / "anomaly_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Hierarchy level performance
    hierarchy_levels = results["hierarchy_levels"]
    unique_levels = np.unique(hierarchy_levels)

    level_errors = []
    level_names = []

    for level in unique_levels:
        level_mask = hierarchy_levels == level
        if np.any(level_mask):
            level_preds = results["reconciled_predictions"][level_mask].reshape(-1)
            level_targets = results["targets"][level_mask].reshape(-1)
            level_error = np.mean(np.abs(level_preds - level_targets))
            level_errors.append(level_error)
            level_names.append(f"Level {level}")

    plt.figure(figsize=fig_size)
    plt.bar(level_names, level_errors)
    plt.xlabel('Hierarchy Level')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance by Hierarchy Level')
    plt.xticks(rotation=45)
    plt.savefig(output_dir / "hierarchy_level_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Base vs Reconciled predictions
    base_preds_flat = results["base_predictions"].reshape(-1)
    reconciled_preds_flat = results["reconciled_predictions"].reshape(-1)

    plt.figure(figsize=fig_size)
    plt.scatter(base_preds_flat, reconciled_preds_flat, alpha=0.5, s=1)
    plt.plot([0, max(base_preds_flat.max(), reconciled_preds_flat.max())],
             [0, max(base_preds_flat.max(), reconciled_preds_flat.max())], 'r--', label='No change')
    plt.xlabel('Base Predictions')
    plt.ylabel('Reconciled Predictions')
    plt.title('Base vs Reconciled Predictions')
    plt.legend()
    plt.savefig(output_dir / "base_vs_reconciled.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization plots saved to {output_dir}")


def print_metrics_summary(metrics: Dict) -> None:
    """Print a summary of key metrics."""
    logger = logging.getLogger(__name__)

    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)

    # Key forecasting metrics
    if "wrmsse" in metrics:
        print(f"WRMSSE: {metrics['wrmsse']:.4f}")
    if "rmse" in metrics:
        print(f"RMSE: {metrics['rmse']:.4f}")
    if "mae" in metrics:
        print(f"MAE: {metrics['mae']:.4f}")
    if "smape" in metrics:
        print(f"sMAPE: {metrics['smape']:.2f}%")

    # Reconciliation metrics
    if "coherence_error" in metrics:
        print(f"Coherence Error: {metrics['coherence_error']:.4f}")
    if "wrmsse_improvement_pct" in metrics:
        print(f"WRMSSE Improvement: {metrics['wrmsse_improvement_pct']:.2f}%")

    # Anomaly detection metrics
    if "precision_at_k" in metrics:
        print(f"Anomaly Precision@K: {metrics['precision_at_k']:.4f}")
    if "anomaly_detection_rate" in metrics:
        print(f"Anomaly Detection Rate: {metrics['anomaly_detection_rate']:.2f}")

    # Coverage metrics
    for coverage_key in ["coverage_95", "coverage_67", "coverage_50"]:
        if coverage_key in metrics:
            level = coverage_key.split("_")[1]
            print(f"Coverage {level}%: {metrics[coverage_key]:.3f}")

    print(f"Total Samples: {metrics.get('n_samples', 'N/A')}")
    print(f"Hierarchy Levels: {metrics.get('n_hierarchy_levels', 'N/A')}")

    print("="*60)


def main() -> None:
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "evaluation.log"
    setup_logging(log_level=args.log_level, log_file=log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting hierarchical demand reconciliation evaluation")

    try:
        # Check checkpoint exists
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model and data
        model, eval_dataset, preprocessor = load_model_and_data(
            str(checkpoint_path),
            args.config,
            args.data_path,
            use_synthetic=args.synthetic_data,
        )

        # Evaluate model
        device = get_device()
        results = evaluate_model(
            model,
            eval_dataset,
            preprocessor,
            batch_size=args.batch_size,
            device=device,
        )

        # Print metrics summary
        print_metrics_summary(results["metrics"])

        # Save results
        save_results(
            results,
            output_dir,
            save_predictions=args.save_predictions,
        )

        # Create visualizations
        if args.create_plots:
            create_visualizations(results, output_dir)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()