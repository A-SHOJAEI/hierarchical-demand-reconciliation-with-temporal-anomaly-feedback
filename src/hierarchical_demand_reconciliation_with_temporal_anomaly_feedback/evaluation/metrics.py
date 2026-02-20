"""Comprehensive evaluation metrics for hierarchical demand reconciliation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class WRMSSEMetric:
    """Weighted Root Mean Squared Scaled Error (WRMSSE) metric."""

    def __init__(self, hierarchy_weights: Optional[np.ndarray] = None) -> None:
        """Initialize WRMSSE metric.

        Args:
            hierarchy_weights: Weights for different hierarchy levels
        """
        self.hierarchy_weights = hierarchy_weights
        self.naive_forecasts_cache = {}

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        hierarchy_levels: np.ndarray,
        train_targets: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute WRMSSE metric.

        Args:
            predictions: Predicted values (n_samples, prediction_horizon)
            targets: True values (n_samples, prediction_horizon)
            hierarchy_levels: Hierarchy level indicators (n_samples,)
            train_targets: Training targets for scaling (optional)

        Returns:
            Dictionary with WRMSSE and component metrics
        """
        # Compute RMSSE for each series
        rmsse_values = self._compute_rmsse(predictions, targets, train_targets)

        # Compute weights if not provided
        if self.hierarchy_weights is None:
            weights = self._compute_default_weights(hierarchy_levels)
        else:
            weights = self.hierarchy_weights

        # Compute weighted RMSSE
        wrmsse = np.average(rmsse_values, weights=weights)

        # Compute level-wise RMSSE
        level_rmsse = {}
        for level in np.unique(hierarchy_levels):
            level_mask = hierarchy_levels == level
            if np.any(level_mask):
                level_rmsse[f"level_{level}_rmsse"] = np.mean(rmsse_values[level_mask])

        return {
            "wrmsse": wrmsse,
            "mean_rmsse": np.mean(rmsse_values),
            **level_rmsse,
        }

    def _compute_rmsse(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        train_targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute RMSSE for each time series."""
        mse = np.mean((predictions - targets) ** 2, axis=1)
        rmse = np.sqrt(mse)

        # Compute scaling factors (naive forecast errors)
        if train_targets is not None:
            scaling_factors = self._compute_scaling_factors(train_targets)
        else:
            # Use simple scaling based on target variance
            scaling_factors = np.std(targets, axis=1)

        # Avoid division by zero
        scaling_factors = np.maximum(scaling_factors, 1e-8)

        rmsse = rmse / scaling_factors

        return rmsse

    def _compute_scaling_factors(self, train_targets: np.ndarray) -> np.ndarray:
        """Compute scaling factors based on naive forecasts."""
        # Compute seasonal naive forecasts (weekly seasonality)
        seasonal_period = 7
        naive_errors = []

        for i, series in enumerate(train_targets):
            if len(series) <= seasonal_period:
                # Use simple naive forecast if series too short
                naive_forecast = np.roll(series, 1)
                naive_forecast[0] = series[0]
            else:
                # Seasonal naive forecast
                naive_forecast = np.roll(series, seasonal_period)
                naive_forecast[:seasonal_period] = series[:seasonal_period]

            naive_error = np.mean((series - naive_forecast) ** 2)
            naive_errors.append(np.sqrt(naive_error))

        return np.array(naive_errors)

    def _compute_default_weights(self, hierarchy_levels: np.ndarray) -> np.ndarray:
        """Compute default weights based on hierarchy structure."""
        weights = np.ones(len(hierarchy_levels))

        # Higher weight for more aggregated levels
        for level in np.unique(hierarchy_levels):
            level_mask = hierarchy_levels == level
            # Higher levels get more weight
            level_weight = 1.0 + 0.1 * level
            weights[level_mask] = level_weight

        return weights / np.sum(weights)


class CoherenceErrorMetric:
    """Metric to evaluate hierarchical coherence."""

    def __init__(self, reconciliation_matrix: np.ndarray) -> None:
        """Initialize coherence error metric.

        Args:
            reconciliation_matrix: Summing matrix S for hierarchy
        """
        self.S = reconciliation_matrix

    def compute(
        self,
        predictions: np.ndarray,
        hierarchy_levels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute hierarchical coherence error.

        Args:
            predictions: Predicted values (n_samples, prediction_horizon)
            hierarchy_levels: Hierarchy level indicators

        Returns:
            Dictionary with coherence metrics
        """
        # Aggregate predictions by hierarchy level
        aggregated_predictions = self._aggregate_by_level(predictions, hierarchy_levels)

        # Compute coherence violations
        coherence_error = self._compute_coherence_violations(aggregated_predictions)

        # Compute relative coherence error
        total_magnitude = np.mean(np.abs(predictions))
        relative_coherence_error = coherence_error / (total_magnitude + 1e-8)

        return {
            "coherence_error": coherence_error,
            "relative_coherence_error": relative_coherence_error,
            "coherence_violations": (coherence_error > 0.01).astype(float),
        }

    def _aggregate_by_level(
        self,
        predictions: np.ndarray,
        hierarchy_levels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Aggregate predictions by hierarchy level."""
        aggregated = {}

        for level in np.unique(hierarchy_levels):
            level_mask = hierarchy_levels == level
            level_predictions = predictions[level_mask]
            aggregated[level] = level_predictions

        return aggregated

    def _compute_coherence_violations(self, aggregated_predictions: Dict[int, np.ndarray]) -> float:
        """Compute coherence violations based on summing constraints."""
        violations = []

        # Compare adjacent levels (simplified)
        levels = sorted(aggregated_predictions.keys())

        for i in range(len(levels) - 1):
            lower_level = levels[i]
            upper_level = levels[i + 1]

            lower_preds = aggregated_predictions[lower_level]
            upper_preds = aggregated_predictions[upper_level]

            if len(lower_preds) > 0 and len(upper_preds) > 0:
                # Sum lower level predictions
                summed_lower = np.sum(lower_preds, axis=0, keepdims=True)

                # Compare with upper level predictions
                if summed_lower.shape == upper_preds.shape:
                    violation = np.mean(np.abs(summed_lower - upper_preds))
                    violations.append(violation)

        return np.mean(violations) if violations else 0.0


class AnomalyMetrics:
    """Metrics for evaluating anomaly detection performance."""

    def __init__(self, k: int = 20) -> None:
        """Initialize anomaly metrics.

        Args:
            k: Number of top predictions for precision@k
        """
        self.k = k

    def compute(
        self,
        anomaly_scores: np.ndarray,
        true_anomalies: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute anomaly detection metrics.

        Args:
            anomaly_scores: Predicted anomaly scores (0-1)
            true_anomalies: True anomaly labels (binary)
            threshold: Threshold for binary classification

        Returns:
            Dictionary with anomaly detection metrics
        """
        metrics = {}

        # Basic statistics
        metrics["anomaly_score_mean"] = np.mean(anomaly_scores)
        metrics["anomaly_score_std"] = np.std(anomaly_scores)
        metrics["anomaly_score_max"] = np.max(anomaly_scores)

        # Precision@K
        top_k_indices = np.argsort(anomaly_scores)[-self.k:]
        metrics["precision_at_k"] = self._compute_precision_at_k(
            anomaly_scores, true_anomalies, self.k
        )

        if true_anomalies is not None:
            # Convert scores to binary predictions
            predicted_anomalies = (anomaly_scores > threshold).astype(int)

            # Standard classification metrics
            if len(np.unique(true_anomalies)) > 1:
                metrics.update({
                    "precision": precision_score(true_anomalies, predicted_anomalies, zero_division=0),
                    "recall": recall_score(true_anomalies, predicted_anomalies, zero_division=0),
                    "f1_score": f1_score(true_anomalies, predicted_anomalies, zero_division=0),
                })

                # ROC-AUC if we have both classes
                try:
                    metrics["roc_auc"] = roc_auc_score(true_anomalies, anomaly_scores)
                except ValueError:
                    metrics["roc_auc"] = 0.0

        return metrics

    def _compute_precision_at_k(
        self,
        scores: np.ndarray,
        true_labels: Optional[np.ndarray],
        k: int,
    ) -> float:
        """Compute precision at top-k predictions."""
        if true_labels is None:
            return 0.0

        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:]

        # Compute precision
        if len(top_k_indices) == 0:
            return 0.0

        true_positives = np.sum(true_labels[top_k_indices])
        precision_at_k = true_positives / k

        return precision_at_k


class CoverageMetric:
    """Metric for evaluating prediction interval coverage."""

    def __init__(self, confidence_levels: List[float] = [0.5, 0.67, 0.95]) -> None:
        """Initialize coverage metric.

        Args:
            confidence_levels: List of confidence levels to evaluate
        """
        self.confidence_levels = confidence_levels

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prediction_intervals: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, float]:
        """Compute prediction interval coverage.

        Args:
            predictions: Point predictions
            targets: True values
            prediction_intervals: Dict mapping confidence level to (lower, upper) bounds

        Returns:
            Dictionary with coverage metrics
        """
        metrics = {}

        if prediction_intervals is None:
            # Estimate intervals using prediction residuals
            residuals = targets - predictions
            prediction_intervals = self._estimate_intervals(predictions, residuals)

        for confidence_level in self.confidence_levels:
            if confidence_level in prediction_intervals:
                lower_bounds, upper_bounds = prediction_intervals[confidence_level]

                # Check coverage
                in_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
                coverage_rate = np.mean(in_interval)

                # Interval width
                interval_width = np.mean(upper_bounds - lower_bounds)

                metrics.update({
                    f"coverage_{int(confidence_level * 100)}": coverage_rate,
                    f"interval_width_{int(confidence_level * 100)}": interval_width,
                })

        return metrics

    def _estimate_intervals(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Estimate prediction intervals from residuals."""
        intervals = {}

        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2

            # Compute quantiles of residuals
            lower_error = np.quantile(residuals, lower_quantile)
            upper_error = np.quantile(residuals, upper_quantile)

            # Create intervals
            lower_bounds = predictions + lower_error
            upper_bounds = predictions + upper_error

            intervals[confidence_level] = (lower_bounds, upper_bounds)

        return intervals


class HierarchicalMetrics:
    """Comprehensive metrics suite for hierarchical forecasting."""

    def __init__(
        self,
        reconciliation_matrix: Optional[np.ndarray] = None,
        hierarchy_weights: Optional[np.ndarray] = None,
        anomaly_k: int = 20,
        confidence_levels: List[float] = [0.5, 0.67, 0.95],
    ) -> None:
        """Initialize hierarchical metrics suite.

        Args:
            reconciliation_matrix: Matrix for coherence evaluation
            hierarchy_weights: Weights for WRMSSE computation
            anomaly_k: K for precision@k in anomaly detection
            confidence_levels: Confidence levels for coverage evaluation
        """
        self.wrmsse_metric = WRMSSEMetric(hierarchy_weights)

        if reconciliation_matrix is not None:
            self.coherence_metric = CoherenceErrorMetric(reconciliation_matrix)
        else:
            self.coherence_metric = None

        self.anomaly_metric = AnomalyMetrics(k=anomaly_k)
        self.coverage_metric = CoverageMetric(confidence_levels)

    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        hierarchy_levels: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None,
        true_anomalies: Optional[np.ndarray] = None,
        prediction_intervals: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None,
        train_targets: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all hierarchical forecasting metrics.

        Args:
            predictions: Predicted values
            targets: True values
            hierarchy_levels: Hierarchy level indicators
            anomaly_scores: Anomaly scores (optional)
            true_anomalies: True anomaly labels (optional)
            prediction_intervals: Prediction intervals (optional)
            train_targets: Training targets for scaling (optional)

        Returns:
            Dictionary with all computed metrics
        """
        logger.info("Computing comprehensive hierarchical metrics...")

        all_metrics = {}

        # WRMSSE and related metrics
        try:
            wrmsse_metrics = self.wrmsse_metric.compute(
                predictions, targets, hierarchy_levels, train_targets
            )
            all_metrics.update(wrmsse_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute WRMSSE metrics: {e}")

        # Coherence metrics
        if self.coherence_metric is not None:
            try:
                coherence_metrics = self.coherence_metric.compute(predictions, hierarchy_levels)
                all_metrics.update(coherence_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute coherence metrics: {e}")

        # Anomaly detection metrics
        if anomaly_scores is not None:
            try:
                anomaly_metrics = self.anomaly_metric.compute(
                    anomaly_scores, true_anomalies
                )
                all_metrics.update(anomaly_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute anomaly metrics: {e}")

        # Coverage metrics
        try:
            coverage_metrics = self.coverage_metric.compute(
                predictions, targets, prediction_intervals
            )
            all_metrics.update(coverage_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute coverage metrics: {e}")

        # Basic forecasting metrics
        try:
            basic_metrics = self._compute_basic_metrics(predictions, targets)
            all_metrics.update(basic_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute basic metrics: {e}")

        logger.info(f"Computed {len(all_metrics)} metrics")

        return all_metrics

    def _compute_basic_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """Compute basic forecasting metrics."""
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

        # sMAPE
        smape = np.mean(2 * np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions) + 1e-8)) * 100

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape,
        }

    def evaluate_forecast_improvement(
        self,
        base_predictions: np.ndarray,
        reconciled_predictions: np.ndarray,
        targets: np.ndarray,
        hierarchy_levels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate improvement from reconciliation.

        Args:
            base_predictions: Base model predictions
            reconciled_predictions: Reconciled predictions
            targets: True values
            hierarchy_levels: Hierarchy levels

        Returns:
            Dictionary with improvement metrics
        """
        # Compute metrics for base predictions
        base_metrics = self.wrmsse_metric.compute(
            base_predictions, targets, hierarchy_levels
        )

        # Compute metrics for reconciled predictions
        reconciled_metrics = self.wrmsse_metric.compute(
            reconciled_predictions, targets, hierarchy_levels
        )

        # Compute improvement percentages
        improvement_metrics = {}
        for metric_name in ["wrmsse", "mean_rmsse"]:
            if metric_name in base_metrics and metric_name in reconciled_metrics:
                base_value = base_metrics[metric_name]
                reconciled_value = reconciled_metrics[metric_name]

                if base_value > 0:
                    improvement_pct = ((base_value - reconciled_value) / base_value) * 100
                    improvement_metrics[f"{metric_name}_improvement_pct"] = improvement_pct

        return improvement_metrics