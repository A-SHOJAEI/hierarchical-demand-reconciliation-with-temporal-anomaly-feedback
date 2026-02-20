# Hierarchical Demand Reconciliation with Temporal Anomaly Feedback

A closed-loop hierarchical forecasting system that uses anomaly detection at fine-grained product levels to automatically trigger forecast reconciliation adjustments at higher aggregation levels. The system fuses forecasting and anomaly detection through a custom coherence-constrained loss function, learning which anomaly patterns are reconciliation-worthy versus noise.

## Quick Start

### Installation

```bash
pip install -e .
```

### Usage

```python
from hierarchical_demand_reconciliation_with_temporal_anomaly_feedback import (
    M5DataLoader, HierarchicalPreprocessor, HierarchicalReconciliationTransformer
)

# Load and preprocess data
loader = M5DataLoader("data/m5", download=True)
sales_df, calendar_df, prices_df = loader.load_raw_data()
hierarchy_data = loader.create_hierarchical_structure(sales_df)
features_df = loader.prepare_features(sales_df, calendar_df, prices_df)

# Initialize preprocessor
preprocessor = HierarchicalPreprocessor(sequence_length=28, prediction_length=28)
preprocessor.fit(features_df)
processed_data = preprocessor.transform(features_df)

# Train model
python scripts/train.py --config configs/default.yaml
```

### Training

```bash
# Train with M5 data
python scripts/train.py --data-path data/m5

# Train with synthetic data for testing
python scripts/train.py --synthetic-data

# Resume training from checkpoint
python scripts/train.py --resume-from checkpoints/model.ckpt
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/final_model.ckpt --create-plots

# Save predictions for analysis
python scripts/evaluate.py --checkpoint checkpoints/final_model.ckpt --save-predictions
```

## Training Results

The model was trained on synthetic M5-style hierarchical demand data (100 items, 365 days, 36,500 records) on an NVIDIA RTX 4090 GPU for 100 epochs.

### Key Metrics

| Metric | Value |
|--------|-------|
| Best Validation Loss | 15.3086 (epoch 2) |
| Final Train Loss | 12.9133 |
| Final Validation Loss | 15.3904 |
| Epochs Trained | 100 |
| Model Parameters | 3.2M |

### Training Progression

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 18.0524 | 17.1897 |
| **2** | **13.2066** | **15.3086** |
| 5 | 13.0915 | 15.4574 |
| 10 | 13.0584 | 16.5016 |
| 25 | 12.9831 | 15.5303 |
| 50 | 12.9635 | 15.4489 |
| 75 | 12.9517 | 15.4133 |
| 100 | 12.9133 | 15.3904 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Data | Synthetic (100 items, 365 days) |
| Train / Val / Test | 25,500 / 7,300 / 3,700 samples |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Hidden Size | 256 |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Sequence Length | 28 |
| Prediction Length | 28 |
| Hardware | NVIDIA RTX 4090 |

### Analysis

The model shows rapid convergence, with train loss dropping from 18.05 to 13.21 after just 2 epochs. The best validation loss of 15.31 was achieved at epoch 2, indicating the model quickly learns the fundamental temporal patterns in the hierarchical demand data. Subsequent training continued to reduce train loss (from 13.21 to 12.91) but validation loss remained relatively stable around 15.3-15.5, suggesting the model reached near-optimal generalization early in training. The gap between train and validation loss indicates mild overfitting on the synthetic data, which is expected given the controlled generation process.

## Architecture

The system combines three key components:

1. **Hierarchical Transformer**: Multi-level forecasting with temporal attention
2. **Anomaly Detection Module**: Isolation forest-based detection with temporal context
3. **Reconciliation Module**: Coherence-constrained adjustments based on anomaly feedback

## Project Structure

```
hierarchical-demand-reconciliation-with-temporal-anomaly-feedback/
├── src/hierarchical_demand_reconciliation_with_temporal_anomaly_feedback/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model implementations
│   ├── training/      # Training pipeline
│   ├── evaluation/    # Metrics and evaluation
│   └── utils/         # Utilities and configuration
├── tests/             # Unit tests
├── scripts/           # Training and evaluation scripts
├── configs/           # Configuration files
└── notebooks/         # Example notebooks
```

## Features

- Hierarchical forecasting with 12 aggregation levels
- Temporal anomaly detection with attention mechanism
- Coherence-constrained reconciliation
- MLflow experiment tracking
- Comprehensive evaluation metrics
- Production-ready PyTorch Lightning implementation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning
- PyTorch Forecasting
- Darts
- MLflow
- M5 Competition dataset

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.