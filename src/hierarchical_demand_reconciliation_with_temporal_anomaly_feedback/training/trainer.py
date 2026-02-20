"""Training pipeline for hierarchical demand reconciliation model."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from ..data.loader import HierarchicalTimeSeriesDataset
from ..models.model import HierarchicalReconciliationTransformer
from ..utils.config import get_device, set_seed

logger = logging.getLogger(__name__)


class HierarchicalTrainer:
    """Trainer for hierarchical demand reconciliation model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Set random seed
        set_seed(config.system.seed)

        # Get device
        self.device = get_device()

        # Initialize MLflow logger
        self.mlflow_logger = None
        if hasattr(config.system, 'experiment_name'):
            self._setup_mlflow_logger()

        # Initialize callbacks
        self.callbacks = self._setup_callbacks()

        # Initialize trainer
        self.trainer = None

    def _setup_mlflow_logger(self) -> None:
        """Setup MLflow logger for experiment tracking."""
        try:
            self.mlflow_logger = MLFlowLogger(
                experiment_name=self.config.system.experiment_name,
                tracking_uri=self.config.system.get('tracking_uri', 'file:./mlruns'),
                log_model=True,
            )
            logger.info("MLflow logger initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow logger: {e}")
            self.mlflow_logger = None

    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.paths.models_dir,
            filename='hierarchical-reconciliation-{epoch:02d}-{val_wrmsse:.4f}',
            monitor=self.config.training.monitor,
            mode=self.config.training.mode,
            save_top_k=self.config.training.save_top_k,
            save_weights_only=False,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping callback
        if hasattr(self.config.training, 'patience'):
            early_stopping = EarlyStopping(
                monitor=self.config.training.monitor,
                patience=self.config.training.patience,
                mode=self.config.training.mode,
                verbose=True,
            )
            callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        return callbacks

    def prepare_model(self, reconciliation_matrix: Optional[torch.Tensor] = None) -> HierarchicalReconciliationTransformer:
        """Prepare model for training.

        Args:
            reconciliation_matrix: Optional reconciliation matrix

        Returns:
            Initialized model
        """
        logger.info("Preparing model...")

        # Model configuration
        model_config = {
            "input_dim": 1,  # Sales feature
            "hidden_size": self.config.model.forecasting_model.hidden_size,
            "num_layers": self.config.model.forecasting_model.num_layers,
            "num_heads": self.config.model.forecasting_model.num_heads,
            "dropout": self.config.model.forecasting_model.dropout,
            "prediction_length": self.config.data.prediction_length,
            "num_hierarchy_levels": len(self.config.data.hierarchy_levels),
            "reconciliation_matrix": reconciliation_matrix,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "loss_weights": dict(self.config.model.loss_weights),
        }

        model = HierarchicalReconciliationTransformer(**model_config)

        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

        return model

    def prepare_trainer(self) -> pl.Trainer:
        """Prepare PyTorch Lightning trainer.

        Returns:
            Configured trainer
        """
        logger.info("Preparing trainer...")

        trainer_config = {
            "max_epochs": self.config.training.max_epochs,
            "callbacks": self.callbacks,
            "logger": self.mlflow_logger,
            "log_every_n_steps": self.config.system.log_every_n_steps,
            "deterministic": self.config.system.deterministic,
            "gradient_clip_val": self.config.training.gradient_clip_val,
            "gradient_clip_algorithm": "norm",
        }

        # GPU configuration
        if torch.cuda.is_available():
            trainer_config.update({
                "accelerator": "gpu",
                "devices": 1,
            })
        else:
            trainer_config.update({
                "accelerator": "cpu",
            })

        self.trainer = pl.Trainer(**trainer_config)

        return self.trainer

    def train(
        self,
        model: HierarchicalReconciliationTransformer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> HierarchicalReconciliationTransformer:
        """Train the model.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader

        Returns:
            Trained model
        """
        logger.info("Starting training...")

        # Prepare trainer if not already done
        if self.trainer is None:
            self.prepare_trainer()

        # Log hyperparameters
        if self.mlflow_logger is not None:
            try:
                self.mlflow_logger.log_hyperparams(dict(self.config))
            except Exception as e:
                logger.warning(f"Failed to log hyperparameters: {e}")

        # Train model
        self.trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Load best checkpoint
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            model = HierarchicalReconciliationTransformer.load_from_checkpoint(best_model_path)

        logger.info("Training completed")

        return model

    def validate(
        self,
        model: HierarchicalReconciliationTransformer,
        val_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            model: Model to validate
            val_dataloader: Validation data loader

        Returns:
            Validation metrics
        """
        logger.info("Running validation...")

        if self.trainer is None:
            self.prepare_trainer()

        # Run validation
        val_results = self.trainer.validate(model, val_dataloader)

        # Extract metrics
        metrics = {}
        if val_results:
            metrics = val_results[0]  # First validation result

        logger.info(f"Validation metrics: {metrics}")

        return metrics

    def test(
        self,
        model: HierarchicalReconciliationTransformer,
        test_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Test the model.

        Args:
            model: Model to test
            test_dataloader: Test data loader

        Returns:
            Test metrics
        """
        logger.info("Running testing...")

        if self.trainer is None:
            self.prepare_trainer()

        # Run testing
        test_results = self.trainer.test(model, test_dataloader)

        # Extract metrics
        metrics = {}
        if test_results:
            metrics = test_results[0]  # First test result

        logger.info(f"Test metrics: {metrics}")

        return metrics

    def save_model(
        self,
        model: HierarchicalReconciliationTransformer,
        output_path: str,
    ) -> None:
        """Save trained model.

        Args:
            model: Model to save
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save using PyTorch Lightning
        self.trainer.save_checkpoint(str(output_path))

        logger.info(f"Model saved to {output_path}")

    def load_model(
        self,
        checkpoint_path: str,
        **kwargs
    ) -> HierarchicalReconciliationTransformer:
        """Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            **kwargs: Additional arguments for model loading

        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {checkpoint_path}")

        model = HierarchicalReconciliationTransformer.load_from_checkpoint(
            checkpoint_path,
            **kwargs
        )

        logger.info("Model loaded successfully")

        return model

    def create_data_loaders(
        self,
        train_dataset: HierarchicalTimeSeriesDataset,
        val_dataset: HierarchicalTimeSeriesDataset,
        test_dataset: Optional[HierarchicalTimeSeriesDataset] = None,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create data loaders for training, validation, and testing.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Creating data loaders...")

        # Training data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory,
        )

        # Validation data loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory,
        )

        # Test data loader
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.system.num_workers,
                pin_memory=self.config.system.pin_memory,
            )

        logger.info(f"Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")
        if test_loader:
            logger.info(f"Test: {len(test_loader)}")

        return train_loader, val_loader, test_loader