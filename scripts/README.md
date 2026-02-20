# Training Scripts

This directory contains training scripts for the hierarchical demand reconciliation model with temporal anomaly feedback.

## Main Training Script

### `train.py`

The main training script that provides a complete training pipeline for the hierarchical demand reconciliation model.

#### Features

- **Configurable hyperparameters** via argparse for easy experimentation
- **GPU/CPU automatic detection** for optimal performance
- **Model checkpointing** to `models/` or `checkpoints/` directory
- **Training metrics logging** to `results/` directory
- **Synthetic data generation** for testing without requiring M5 dataset
- **Resume training** from checkpoint functionality
- **Comprehensive error handling** and logging

#### Usage

```bash
# Basic usage with synthetic data (for testing)
python scripts/train.py --synthetic-data --epochs 10 --batch-size 32

# Full training with M5 dataset
python scripts/train.py --epochs 100 --learning-rate 0.001

# Custom model architecture
python scripts/train.py --hidden-size 512 --num-layers 6 --dropout 0.2

# Resume training from checkpoint
python scripts/train.py --resume-from checkpoints/best_model.pt

# GPU training with custom settings
python scripts/train.py --device cuda --batch-size 128 --num-workers 8
```

#### Command Line Arguments

**Data Configuration:**
- `--data-path`: Path to dataset directory (default: "data")
- `--config`: Path to configuration file (default: uses project config)
- `--synthetic-data`: Use synthetic data instead of M5 dataset
- `--no-download`: Don't download M5 dataset if not found

**Training Hyperparameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate for optimization (default: 1e-3)
- `--weight-decay`: Weight decay for regularization (default: 1e-4)
- `--sequence-length`: Input sequence length (default: 28)
- `--prediction-length`: Prediction horizon length (default: 28)

**Model Architecture:**
- `--hidden-size`: Hidden size for transformer model (default: 256)
- `--num-layers`: Number of transformer layers (default: 4)
- `--num-heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.1)

**Output Configuration:**
- `--output-dir`: Output directory for results and logs (default: "results")
- `--checkpoint-dir`: Directory to save model checkpoints (default: "checkpoints")
- `--resume-from`: Path to checkpoint to resume training from

**System Configuration:**
- `--device`: Device to use - auto, cpu, or cuda (default: "auto")
- `--num-workers`: Number of data loader workers (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: "INFO")

#### Output Structure

After running the training script, the following directories and files will be created:

```
results/
├── training.log              # Detailed training logs
├── training_metrics.json     # Final metrics summary
└── ...

checkpoints/
├── trained_model.pt          # Final model checkpoint
├── best_model_epoch_X.pt     # Best validation models (if using full trainer)
└── ...
```

#### Model Architecture

The script trains a **Hierarchical Reconciliation Transformer** with the following components:

- **Multi-level hierarchy support**: item → dept → category → store → state → total
- **Temporal Anomaly Detection Module** with attention mechanism
- **Reconciliation Module** with coherence constraints
- **PyTorch Lightning integration** for advanced training features

#### Requirements

```
torch >= 1.12.0
pytorch-lightning >= 1.7.0
pandas >= 1.4.0
numpy >= 1.21.0
omegaconf >= 2.2.0
kaggle (for M5 data download)
```

Install with:
```bash
pip install -r requirements.txt
```

#### Implementation Details

The script includes two modes of operation:

1. **Full Implementation Mode**: Uses the complete hierarchical demand reconciliation pipeline with M5 data
2. **Simplified Mode**: Uses synthetic data with a simplified model for testing

The script automatically detects available modules and falls back to simplified mode if the full project dependencies are not available.

#### Synthetic Data

When using `--synthetic-data`, the script generates realistic hierarchical time series data with:
- Multiple hierarchy levels
- Trend and seasonal patterns
- Random anomalies/events
- Configurable number of items and time periods

This is useful for:
- Testing the training pipeline
- Debugging model architecture
- Experimentation without large datasets

## Demo Script

### `demo_train.py`

A demonstration script that shows the structure and capabilities of the training pipeline without requiring external dependencies.

```bash
python scripts/demo_train.py
```

This script provides an overview of:
- Training pipeline structure
- Available configuration options
- Usage examples
- Output structure
- Required dependencies

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Quick test with synthetic data:**
   ```bash
   python scripts/train.py --synthetic-data --epochs 5 --batch-size 32
   ```

3. **Full training with M5 data:**
   ```bash
   python scripts/train.py --epochs 100
   ```

4. **Monitor training:**
   Check `results/training.log` for detailed logs and `results/training_metrics.json` for final metrics.

## Troubleshooting

**ModuleNotFoundError**: Install required dependencies with `pip install -r requirements.txt`

**CUDA out of memory**: Reduce batch size with `--batch-size 32` or use CPU with `--device cpu`

**M5 data not found**: Use `--synthetic-data` for testing or ensure M5 dataset is properly downloaded

**Configuration errors**: Check that the config file path is correct or let the script use default configuration