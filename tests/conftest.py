"""Test configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig

from src.hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.data.loader import (
    HierarchicalTimeSeriesDataset,
    M5DataLoader,
)
from src.hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.data.preprocessing import (
    HierarchicalPreprocessor,
)
from src.hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.utils.config import (
    set_seed,
)


@pytest.fixture(scope="session")
def random_seed() -> int:
    """Random seed for reproducible tests."""
    seed = 42
    set_seed(seed)
    return seed


@pytest.fixture
def sample_config() -> DictConfig:
    """Sample configuration for testing."""
    config = DictConfig({
        "data": {
            "sequence_length": 14,
            "prediction_length": 7,
            "validation_split": 0.2,
            "test_split": 0.1,
            "hierarchy_levels": ["item_id", "dept_id", "cat_id", "store_id", "state_id", "total"],
        },
        "model": {
            "forecasting_model": {
                "hidden_size": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
            },
            "anomaly_detector": {
                "window_size": 7,
                "threshold_method": "isolation_forest",
                "contamination": 0.1,
            },
            "reconciliation": {
                "method": "weighted_least_squares",
                "coherence_weight": 1.0,
                "anomaly_weight": 0.5,
            },
            "loss_weights": {
                "forecast_loss": 1.0,
                "reconciliation_loss": 0.3,
                "anomaly_loss": 0.2,
                "coherence_loss": 0.1,
            },
        },
        "training": {
            "batch_size": 16,
            "max_epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 3,
            "monitor": "val_wrmsse",
            "mode": "min",
            "save_top_k": 1,
        },
        "system": {
            "seed": 42,
            "num_workers": 0,
            "pin_memory": False,
            "log_every_n_steps": 10,
            "deterministic": True,
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "checkpoints",
            "results_dir": "results",
            "logs_dir": "logs",
        },
    })
    return config


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Generate sample sales data for testing."""
    np.random.seed(42)

    # Create sample hierarchical data
    n_items = 50
    n_days = 100

    # Item hierarchy
    items = [f"item_{i:03d}" for i in range(n_items)]
    depts = [f"dept_{i//10:02d}" for i in range(n_items)]
    cats = [f"cat_{i//25:02d}" for i in range(n_items)]
    stores = [f"store_{i%5:02d}" for i in range(n_items)]
    states = [f"state_{i%2:02d}" for i in range(n_items)]

    # Generate time series data
    data = []
    for i in range(n_items):
        # Base demand with trend and seasonality
        trend = 0.01 * i
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        noise = np.random.normal(0, 1, n_days)
        base_demand = 10 + trend * np.arange(n_days) + seasonal + noise

        # Ensure non-negative values
        sales_values = np.maximum(0, base_demand)

        for day in range(n_days):
            data.append({
                "item_id": items[i],
                "dept_id": depts[i],
                "cat_id": cats[i],
                "store_id": stores[i],
                "state_id": states[i],
                "d": f"d_{day+1}",
                "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=day),
                "sales": sales_values[day],
            })

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_calendar_data() -> pd.DataFrame:
    """Generate sample calendar data for testing."""
    n_days = 100
    calendar_data = []

    for day in range(n_days):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
        calendar_data.append({
            "d": f"d_{day+1}",
            "date": date,
            "wm_yr_wk": date.isocalendar()[1] + 100,  # Simple week mapping
            "weekday": date.weekday(),
            "wday": date.weekday() + 1,
            "month": date.month,
            "year": date.year,
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
        })

    return pd.DataFrame(calendar_data)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)

    n_items = 50
    n_weeks = 15  # Approximate weeks for 100 days

    price_data = []
    items = [f"item_{i:03d}" for i in range(n_items)]
    stores = [f"store_{i%5:02d}" for i in range(n_items)]

    for item_id in items:
        for store_id in stores:
            base_price = 10 + np.random.uniform(-2, 2)

            for week in range(n_weeks):
                # Price variations
                price = base_price + np.random.uniform(-1, 1)
                price_data.append({
                    "store_id": store_id,
                    "item_id": item_id,
                    "wm_yr_wk": week + 100,
                    "sell_price": max(0.1, price),  # Ensure positive prices
                })

    return pd.DataFrame(price_data)


@pytest.fixture
def sample_hierarchical_data(sample_sales_data: pd.DataFrame) -> dict:
    """Generate sample hierarchical aggregation structure."""
    # This would normally be created by M5DataLoader.create_hierarchical_structure
    # For testing, we'll create a simplified version

    meta_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # Item level (finest granularity)
    item_level = sample_sales_data.copy()

    # Create simplified higher levels for testing
    hierarchical_data = {
        "item_id": item_level,
        "dept_id": item_level.groupby(["dept_id", "store_id"]).first().reset_index(),
        "cat_id": item_level.groupby(["cat_id", "state_id"]).first().reset_index(),
        "store_id": item_level.groupby(["store_id"]).first().reset_index(),
        "state_id": item_level.groupby(["state_id"]).first().reset_index(),
        "total": item_level.head(1),  # Single total series
    }

    return hierarchical_data


@pytest.fixture
def sample_preprocessor(sample_config: DictConfig) -> HierarchicalPreprocessor:
    """Create sample preprocessor for testing."""
    return HierarchicalPreprocessor(
        sequence_length=sample_config.data.sequence_length,
        prediction_length=sample_config.data.prediction_length,
        anomaly_window=sample_config.model.anomaly_detector.window_size,
        anomaly_threshold=sample_config.model.anomaly_detector.contamination,
    )


@pytest.fixture
def sample_dataset(
    sample_sales_data: pd.DataFrame,
    sample_config: DictConfig,
) -> HierarchicalTimeSeriesDataset:
    """Create sample dataset for testing."""
    hierarchy_mapping = {
        "item_id": 0,
        "dept_id": 1,
        "cat_id": 2,
        "store_id": 3,
        "state_id": 4,
        "total": 5,
    }

    return HierarchicalTimeSeriesDataset(
        data=sample_sales_data,
        sequence_length=sample_config.data.sequence_length,
        prediction_length=sample_config.data.prediction_length,
        hierarchy_mapping=hierarchy_mapping,
        features=["sales"],
    )


@pytest.fixture
def sample_reconciliation_matrix() -> torch.Tensor:
    """Create sample reconciliation matrix for testing."""
    # Simple 6x6 matrix for 6 hierarchy levels
    # In practice, this would be computed based on actual hierarchy structure
    matrix = torch.eye(6, dtype=torch.float32)
    return matrix


@pytest.fixture
def sample_batch() -> dict:
    """Create sample batch for model testing."""
    batch_size = 8
    sequence_length = 14
    prediction_length = 7

    return {
        "input": torch.randn(batch_size, sequence_length, 1),
        "target": torch.randn(batch_size, prediction_length, 1),
        "hierarchy_level": torch.randint(0, 6, (batch_size,)),
        "item_id": [f"item_{i}" for i in range(batch_size)],
    }


@pytest.fixture
def device() -> torch.device:
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")