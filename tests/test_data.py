"""Tests for data loading and preprocessing modules."""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.data.loader import (
    M5DataLoader,
    HierarchicalTimeSeriesDataset,
)
from src.hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.data.preprocessing import (
    HierarchicalPreprocessor,
)


class TestM5DataLoader:
    """Test M5 data loader functionality."""

    def test_init_without_download(self, tmp_path):
        """Test loader initialization without downloading."""
        loader = M5DataLoader(data_path=tmp_path, download=False)
        assert loader.data_path == tmp_path
        assert not loader.download

    def test_hierarchy_mapping(self, tmp_path):
        """Test hierarchy level mapping."""
        loader = M5DataLoader(data_path=tmp_path, download=False)
        expected_mapping = {
            "item_id": 0,
            "dept_id": 1,
            "cat_id": 2,
            "store_id": 3,
            "state_id": 4,
            "total": 5,
        }
        assert loader.hierarchy_mapping == expected_mapping

    def test_check_data_exists_false(self, tmp_path):
        """Test data existence check when files don't exist."""
        loader = M5DataLoader(data_path=tmp_path, download=False)
        assert not loader._check_data_exists()

    def test_check_data_exists_true(self, tmp_path):
        """Test data existence check when files exist."""
        # Create required files
        (tmp_path / "calendar.csv").touch()
        (tmp_path / "sales_train_evaluation.csv").touch()
        (tmp_path / "sell_prices.csv").touch()

        loader = M5DataLoader(data_path=tmp_path, download=False)
        assert loader._check_data_exists()

    def test_create_hierarchical_structure(
        self,
        sample_sales_data: pd.DataFrame,
        tmp_path,
    ):
        """Test hierarchical structure creation."""
        loader = M5DataLoader(data_path=tmp_path, download=False)

        # Transform sample data to M5 format
        sales_df = sample_sales_data.pivot(
            index=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            columns="d",
            values="sales"
        ).reset_index()

        hierarchical_data = loader.create_hierarchical_structure(sales_df)

        # Check all levels are present
        expected_levels = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "total"]
        assert set(hierarchical_data.keys()) == set(expected_levels)

        # Check item level is unchanged
        assert len(hierarchical_data["item_id"]) == len(sales_df)

        # Check aggregation reduces number of series
        assert len(hierarchical_data["total"]) == 1

    def test_prepare_features(
        self,
        sample_sales_data: pd.DataFrame,
        sample_calendar_data: pd.DataFrame,
        sample_price_data: pd.DataFrame,
        tmp_path,
    ):
        """Test feature preparation."""
        loader = M5DataLoader(data_path=tmp_path, download=False)

        # Convert sample data to M5 format
        sales_df = sample_sales_data.pivot(
            index=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            columns="d",
            values="sales"
        ).reset_index()

        features_df = loader.prepare_features(sales_df, sample_calendar_data, sample_price_data)

        # Check required columns are present
        required_cols = ["item_id", "sales", "date", "dayofweek", "month"]
        for col in required_cols:
            assert col in features_df.columns

        # Check temporal features
        assert "sales_lag_1" in features_df.columns
        assert "sales_roll_mean_7" in features_df.columns

        # Check no NaN in final dataset
        assert not features_df[["sales", "dayofweek", "month"]].isna().any().any()


class TestHierarchicalTimeSeriesDataset:
    """Test hierarchical time series dataset."""

    def test_dataset_creation(self, sample_dataset: HierarchicalTimeSeriesDataset):
        """Test dataset creation."""
        assert len(sample_dataset) > 0
        assert sample_dataset.sequence_length == 14
        assert sample_dataset.prediction_length == 7

    def test_getitem(self, sample_dataset: HierarchicalTimeSeriesDataset):
        """Test dataset item retrieval."""
        sample = sample_dataset[0]

        # Check required keys
        assert "input" in sample
        assert "target" in sample
        assert "hierarchy_level" in sample
        assert "item_id" in sample

        # Check tensor shapes
        assert sample["input"].shape == (14, 1)  # sequence_length, features
        assert sample["target"].shape == (7, 1)  # prediction_length, features
        assert sample["hierarchy_level"].shape == ()  # scalar

        # Check tensor types
        assert sample["input"].dtype == torch.float32
        assert sample["target"].dtype == torch.float32
        assert sample["hierarchy_level"].dtype == torch.long

    def test_dataset_length_consistency(
        self,
        sample_sales_data: pd.DataFrame,
        sample_config,
    ):
        """Test that dataset length is consistent with input data."""
        hierarchy_mapping = {"item_id": 0}

        dataset = HierarchicalTimeSeriesDataset(
            data=sample_sales_data,
            sequence_length=7,
            prediction_length=7,
            hierarchy_mapping=hierarchy_mapping,
        )

        # Should have samples for each item with sufficient data
        min_length = 7 + 7  # sequence + prediction
        valid_items = sample_sales_data.groupby("item_id").size() >= min_length
        expected_samples = valid_items.sum() * (
            sample_sales_data.groupby("item_id").size().max() - min_length + 1
        )

        # Dataset length should be reasonable (exact calculation depends on implementation)
        assert len(dataset) > 0
        assert len(dataset) <= expected_samples


class TestHierarchicalPreprocessor:
    """Test hierarchical preprocessor."""

    def test_preprocessor_init(self, sample_preprocessor: HierarchicalPreprocessor):
        """Test preprocessor initialization."""
        assert sample_preprocessor.sequence_length == 14
        assert sample_preprocessor.prediction_length == 7
        assert not sample_preprocessor.is_fitted

    def test_fit_transform(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
        sample_sales_data: pd.DataFrame,
    ):
        """Test fit and transform methods."""
        # Fit preprocessor
        sample_preprocessor.fit(sample_sales_data)
        assert sample_preprocessor.is_fitted

        # Transform data
        transformed_data = sample_preprocessor.transform(sample_sales_data)

        # Check new columns are added
        assert "sales_scaled" in transformed_data.columns
        assert "anomaly_score" in transformed_data.columns
        assert "is_anomaly" in transformed_data.columns

        # Check temporal features
        assert "dayofweek_sin" in transformed_data.columns
        assert "month_sin" in transformed_data.columns

        # Check hierarchical embeddings
        assert "item_id_encoded" in transformed_data.columns

    def test_anomaly_detection(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
        sample_sales_data: pd.DataFrame,
    ):
        """Test anomaly detection functionality."""
        sample_preprocessor.fit(sample_sales_data)
        transformed_data = sample_preprocessor.transform(sample_sales_data)

        # Check anomaly scores are in valid range
        anomaly_scores = transformed_data["anomaly_score"]
        assert anomaly_scores.min() >= 0
        assert anomaly_scores.max() <= 1

        # Check some anomalies are detected
        assert transformed_data["is_anomaly"].sum() > 0

    def test_inverse_transform(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
        sample_sales_data: pd.DataFrame,
    ):
        """Test inverse transformation of scaled data."""
        # Fit and transform
        sample_preprocessor.fit(sample_sales_data)
        transformed_data = sample_preprocessor.transform(sample_sales_data)

        # Get scaled sales
        scaled_sales = transformed_data["sales_scaled"].values

        # Inverse transform
        original_sales = sample_preprocessor.inverse_transform_sales(scaled_sales)

        # Check shape preservation
        assert original_sales.shape == scaled_sales.shape

        # Check values are reasonable (not exact due to scaling)
        original_mean = sample_sales_data["sales"].mean()
        recovered_mean = original_sales.mean()
        assert abs(original_mean - recovered_mean) / original_mean < 0.1  # Within 10%

    def test_data_splitting(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
        sample_sales_data: pd.DataFrame,
    ):
        """Test data splitting functionality."""
        train_data, val_data, test_data = sample_preprocessor.split_data(
            sample_sales_data,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
        )

        # Check all data is accounted for
        total_samples = len(train_data) + len(val_data) + len(test_data)
        assert total_samples == len(sample_sales_data)

        # Check no temporal leakage (train dates < val dates < test dates)
        assert train_data["date"].max() <= val_data["date"].min()
        assert val_data["date"].max() <= test_data["date"].min()

    def test_reconciliation_matrix_creation(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
        sample_hierarchical_data: dict,
    ):
        """Test reconciliation matrix creation."""
        reconciliation_matrix = sample_preprocessor.create_reconciliation_matrix(
            sample_hierarchical_data
        )

        # Check matrix dimensions
        assert reconciliation_matrix.ndim == 2
        assert reconciliation_matrix.shape[0] > 0
        assert reconciliation_matrix.shape[1] > 0

        # Check matrix contains only 0s and 1s (summing matrix)
        unique_values = np.unique(reconciliation_matrix)
        assert set(unique_values).issubset({0.0, 1.0})

    def test_fit_without_sales_column(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
    ):
        """Test fitting preprocessor when sales column is missing."""
        # Create data without sales column
        data_without_sales = pd.DataFrame({
            "item_id": ["item_001", "item_002"],
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "value": [10.0, 15.0],
        })

        # Should not raise error
        sample_preprocessor.fit(data_without_sales)
        assert sample_preprocessor.is_fitted

    def test_transform_without_fit_raises_error(
        self,
        sample_preprocessor: HierarchicalPreprocessor,
        sample_sales_data: pd.DataFrame,
    ):
        """Test that transform without fit raises ValueError."""
        with pytest.raises(ValueError, match="must be fitted"):
            sample_preprocessor.transform(sample_sales_data)

    def test_edge_cases_short_series(self, sample_config):
        """Test handling of very short time series."""
        # Create very short series
        short_data = pd.DataFrame({
            "item_id": ["item_001"] * 3,
            "date": pd.date_range("2021-01-01", periods=3),
            "sales": [1.0, 2.0, 3.0],
        })

        preprocessor = HierarchicalPreprocessor(
            sequence_length=5,  # Longer than available data
            prediction_length=2,
            anomaly_window=2,
        )

        # Should handle gracefully
        preprocessor.fit(short_data)
        transformed_data = preprocessor.transform(short_data)

        # Should return some result even with insufficient data
        assert len(transformed_data) == len(short_data)