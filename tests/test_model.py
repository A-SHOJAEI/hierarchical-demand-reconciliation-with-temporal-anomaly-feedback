"""Tests for model components."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.hierarchical_demand_reconciliation_with_temporal_anomaly_feedback.models.model import (
    HierarchicalReconciliationTransformer,
    AnomalyDetectionModule,
    ReconciliationModule,
    PositionalEncoding,
)


class TestPositionalEncoding:
    """Test positional encoding module."""

    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        d_model = 64
        max_len = 100
        pos_encoder = PositionalEncoding(d_model, max_len)

        # Test input
        seq_len, batch_size = 20, 8
        x = torch.randn(seq_len, batch_size, d_model)

        # Forward pass
        output = pos_encoder(x)

        # Check output shape
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model = 32
        pos_encoder = PositionalEncoding(d_model)

        x = torch.randn(10, 4, d_model)

        output1 = pos_encoder(x)
        output2 = pos_encoder(x)

        # Should be identical
        assert torch.allclose(output1, output2)

    def test_positional_encoding_different_lengths(self):
        """Test positional encoding with different sequence lengths."""
        d_model = 32
        pos_encoder = PositionalEncoding(d_model, max_len=50)

        # Test different lengths
        for seq_len in [5, 15, 30]:
            x = torch.randn(seq_len, 2, d_model)
            output = pos_encoder(x)
            assert output.shape == x.shape


class TestAnomalyDetectionModule:
    """Test anomaly detection module."""

    def test_anomaly_detection_init(self):
        """Test anomaly detection module initialization."""
        input_dim = 1
        hidden_dim = 32
        window_size = 7

        module = AnomalyDetectionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_size=window_size,
        )

        assert module.window_size == window_size

    def test_anomaly_detection_forward(self):
        """Test anomaly detection forward pass."""
        input_dim = 1
        hidden_dim = 32
        window_size = 5

        module = AnomalyDetectionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_size=window_size,
        )

        # Test input
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, input_dim)

        # Forward pass
        anomaly_scores, attention_weights = module(x)

        # Check output shapes
        assert anomaly_scores.shape == (batch_size, 1)
        assert attention_weights.shape[0] == batch_size

        # Check score range
        assert torch.all(anomaly_scores >= 0)
        assert torch.all(anomaly_scores <= 1)

    def test_anomaly_detection_short_sequence(self):
        """Test anomaly detection with short sequences."""
        module = AnomalyDetectionModule(
            input_dim=1,
            hidden_dim=16,
            window_size=10,  # Longer than sequence
        )

        # Short sequence
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, 1)

        # Should handle gracefully
        anomaly_scores, attention_weights = module(x)

        assert anomaly_scores.shape == (batch_size, 1)

    def test_detect_anomalies(self):
        """Test anomaly detection method."""
        module = AnomalyDetectionModule(input_dim=1, threshold=0.5)

        # Test scores
        scores = torch.tensor([[0.3], [0.7], [0.2], [0.8]])
        detected = module.detect_anomalies(scores)

        expected = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        assert torch.allclose(detected, expected)


class TestReconciliationModule:
    """Test reconciliation module."""

    def test_reconciliation_init(self, sample_reconciliation_matrix: torch.Tensor):
        """Test reconciliation module initialization."""
        num_levels = 6
        module = ReconciliationModule(
            num_levels=num_levels,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        assert module.num_levels == num_levels
        assert torch.allclose(module.S, sample_reconciliation_matrix)

    def test_reconciliation_forward(self, sample_reconciliation_matrix: torch.Tensor):
        """Test reconciliation forward pass."""
        num_levels = 6
        module = ReconciliationModule(
            num_levels=num_levels,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Test inputs
        batch_size, seq_len = 8, 10
        forecasts = torch.randn(batch_size, seq_len, 1)
        anomaly_scores = torch.rand(batch_size, 1)  # 0-1 range
        hierarchy_levels = torch.randint(0, num_levels, (batch_size,))

        # Forward pass
        reconciled_forecasts, coherence_loss = module(
            forecasts, anomaly_scores, hierarchy_levels
        )

        # Check output shapes
        assert reconciled_forecasts.shape == forecasts.shape
        assert coherence_loss.dim() == 0  # Scalar

        # Check loss is non-negative
        assert coherence_loss >= 0

    def test_reconciliation_different_levels(self, sample_reconciliation_matrix: torch.Tensor):
        """Test reconciliation with different hierarchy levels."""
        num_levels = 6
        module = ReconciliationModule(
            num_levels=num_levels,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Test with all levels represented
        batch_size = 12  # 2 samples per level
        forecasts = torch.randn(batch_size, 5, 1)
        anomaly_scores = torch.rand(batch_size, 1)
        hierarchy_levels = torch.repeat_interleave(torch.arange(num_levels), 2)

        reconciled_forecasts, coherence_loss = module(
            forecasts, anomaly_scores, hierarchy_levels
        )

        assert reconciled_forecasts.shape == forecasts.shape


class TestHierarchicalReconciliationTransformer:
    """Test main hierarchical reconciliation model."""

    def test_model_init(self, sample_reconciliation_matrix: torch.Tensor):
        """Test model initialization."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Check model components
        assert model.input_dim == 1
        assert model.hidden_size == 64
        assert model.prediction_length == 7

    def test_model_forward(self, sample_batch: dict, sample_reconciliation_matrix: torch.Tensor):
        """Test model forward pass."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Forward pass
        x = sample_batch["input"]
        hierarchy_levels = sample_batch["hierarchy_level"]

        outputs = model(x, hierarchy_levels)

        # Check required outputs
        assert "base_forecasts" in outputs
        assert "reconciled_forecasts" in outputs
        assert "anomaly_scores" in outputs
        assert "anomaly_attention" in outputs
        assert "coherence_loss" in outputs

        # Check output shapes
        batch_size, seq_len, features = x.shape
        assert outputs["base_forecasts"].shape == (batch_size, 7, 1)
        assert outputs["reconciled_forecasts"].shape == (batch_size, 7, 1)
        assert outputs["anomaly_scores"].shape == (batch_size, 1)

    def test_training_step(self, sample_batch: dict, sample_reconciliation_matrix: torch.Tensor):
        """Test model training step."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Training step
        loss = model.training_step(sample_batch, batch_idx=0)

        # Check loss is scalar and non-negative
        assert loss.dim() == 0
        assert loss >= 0

    def test_validation_step(self, sample_batch: dict, sample_reconciliation_matrix: torch.Tensor):
        """Test model validation step."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Validation step (should not raise error)
        model.validation_step(sample_batch, batch_idx=0)

    def test_model_different_input_sizes(self, sample_reconciliation_matrix: torch.Tensor):
        """Test model with different input sizes."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Test different batch sizes and sequence lengths
        test_cases = [
            (4, 10),
            (8, 28),
            (1, 14),
            (16, 7),
        ]

        for batch_size, seq_len in test_cases:
            x = torch.randn(batch_size, seq_len, 1)
            hierarchy_levels = torch.randint(0, 6, (batch_size,))

            outputs = model(x, hierarchy_levels)

            assert outputs["base_forecasts"].shape == (batch_size, 7, 1)
            assert outputs["reconciled_forecasts"].shape == (batch_size, 7, 1)

    def test_model_gradient_flow(self, sample_batch: dict, sample_reconciliation_matrix: torch.Tensor):
        """Test that gradients flow through the model."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Enable gradient computation
        model.train()

        # Forward and backward pass
        loss = model.training_step(sample_batch, batch_idx=0)
        loss.backward()

        # Check that gradients are computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in model parameters"

    def test_model_no_reconciliation_matrix(self):
        """Test model initialization without reconciliation matrix."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=1,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=None,  # Should use default
        )

        # Check that default matrix is used
        assert model.reconciliation_module.S is not None

    def test_configure_optimizers(self, sample_reconciliation_matrix: torch.Tensor):
        """Test optimizer configuration."""
        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=32,
            num_layers=1,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )

        optimizer_config = model.configure_optimizers()

        # Check optimizer and scheduler
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config

        optimizer = optimizer_config["optimizer"]
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    @pytest.mark.parametrize("device_type", ["cpu"])  # Skip CUDA if not available
    def test_model_device_compatibility(
        self,
        device_type: str,
        sample_batch: dict,
        sample_reconciliation_matrix: torch.Tensor,
    ):
        """Test model compatibility with different devices."""
        device = torch.device(device_type)

        # Skip if device not available
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = HierarchicalReconciliationTransformer(
            input_dim=1,
            hidden_size=16,
            num_layers=1,
            prediction_length=7,
            num_hierarchy_levels=6,
            reconciliation_matrix=sample_reconciliation_matrix,
        )

        # Move to device
        model = model.to(device)

        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }

        # Forward pass
        outputs = model(batch["input"], batch["hierarchy_level"])

        # Check outputs are on correct device
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                assert tensor.device == device