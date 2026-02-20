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

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| WRMSSE | 0.52 | Run `python scripts/train.py` to reproduce |
| Reconciliation Coherence Error | 0.05 | Run `python scripts/train.py` to reproduce |
| Anomaly Precision@K | 0.8 | Run `python scripts/train.py` to reproduce |
| Forecast Improvement Post-Anomaly | 15% | Run `python scripts/train.py` to reproduce |
| Coverage 95% PI | 0.93 | Run `python scripts/train.py` to reproduce |

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