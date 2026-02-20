"""Hierarchical forecasting model with temporal anomaly feedback."""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.pe[:x.size(0), :]


class AnomalyDetectionModule(nn.Module):
    """Anomaly detection module with attention mechanism."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        window_size: int = 7,
        threshold: float = 0.1,
    ) -> None:
        """Initialize anomaly detection module.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for features
            window_size: Window size for temporal context
            threshold: Anomaly threshold
        """
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Anomaly score prediction
        self.anomaly_predictor = nn.Linear(hidden_dim // 2, 1)

        # Temporal attention for anomaly context
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for anomaly detection.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Tuple of (anomaly_scores, attention_weights)
        """
        batch_size, seq_len, features = x.shape

        # Create sliding windows
        if seq_len < self.window_size:
            # Pad if sequence is too short
            padding = self.window_size - seq_len
            x = F.pad(x, (0, 0, padding, 0))
            seq_len = self.window_size

        # Extract windows
        windows = []
        for i in range(seq_len - self.window_size + 1):
            window = x[:, i:i + self.window_size, :]
            windows.append(window.reshape(batch_size, -1))

        if not windows:
            # Return zeros if no windows can be created
            return torch.zeros(batch_size, 1), torch.zeros(batch_size, 1, 1)

        windows = torch.stack(windows, dim=1)  # (batch, num_windows, window_features)

        # Extract features
        features = self.feature_extractor(windows)

        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(
            features, features, features
        )

        # Global average pooling over windows
        pooled_features = attended_features.mean(dim=1)

        # Predict anomaly scores
        anomaly_scores = torch.sigmoid(self.anomaly_predictor(pooled_features))

        return anomaly_scores, attention_weights

    def detect_anomalies(self, scores: torch.Tensor) -> torch.Tensor:
        """Detect anomalies based on scores."""
        return (scores > self.threshold).float()


class ReconciliationModule(nn.Module):
    """Hierarchical reconciliation module with coherence constraints."""

    def __init__(
        self,
        num_levels: int,
        reconciliation_matrix: torch.Tensor,
        coherence_weight: float = 1.0,
        anomaly_weight: float = 0.5,
        attention_dim: int = 128,
    ) -> None:
        """Initialize reconciliation module.

        Args:
            num_levels: Number of hierarchy levels
            reconciliation_matrix: Matrix S for reconciliation
            coherence_weight: Weight for coherence loss
            anomaly_weight: Weight for anomaly-driven adjustments
            attention_dim: Dimension for attention mechanism
        """
        super().__init__()
        self.num_levels = num_levels
        self.coherence_weight = coherence_weight
        self.anomaly_weight = anomaly_weight

        # Register reconciliation matrix as buffer
        self.register_buffer('S', reconciliation_matrix)

        # Attention mechanism for anomaly-driven reconciliation
        self.anomaly_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=8,
            batch_first=True
        )

        # Level-specific adjustment networks
        self.level_adjustments = nn.ModuleList([
            nn.Linear(attention_dim, 1) for _ in range(num_levels)
        ])

        # Projection to attention dimension
        self.projection = nn.Linear(1, attention_dim)

    def forward(
        self,
        forecasts: torch.Tensor,
        anomaly_scores: torch.Tensor,
        hierarchy_levels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for reconciliation.

        Args:
            forecasts: Base forecasts (batch, seq_len, 1)
            anomaly_scores: Anomaly scores (batch, 1)
            hierarchy_levels: Hierarchy level indices (batch,)

        Returns:
            Tuple of (reconciled_forecasts, coherence_loss)
        """
        batch_size, seq_len, _ = forecasts.shape

        # Project forecasts to attention dimension
        projected_forecasts = self.projection(forecasts)  # (batch, seq_len, attention_dim)

        # Apply anomaly-driven attention
        anomaly_context = anomaly_scores.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, 1)
        anomaly_context = self.projection(anomaly_context)  # (batch, seq_len, attention_dim)

        attended_forecasts, _ = self.anomaly_attention(
            projected_forecasts, anomaly_context, anomaly_context
        )

        # Apply level-specific adjustments
        reconciled_forecasts = forecasts.clone()

        for level in range(self.num_levels):
            level_mask = (hierarchy_levels == level)
            if level_mask.any():
                level_adjustments = self.level_adjustments[level](
                    attended_forecasts[level_mask]
                )
                adjustment_factor = torch.sigmoid(level_adjustments)

                # Apply anomaly-weighted adjustments
                anomaly_weight = anomaly_scores[level_mask].unsqueeze(-1)
                final_adjustment = (1 + self.anomaly_weight * anomaly_weight *
                                  (adjustment_factor - 1))

                reconciled_forecasts[level_mask] = forecasts[level_mask] * final_adjustment

        # Compute coherence loss
        coherence_loss = self._compute_coherence_loss(reconciled_forecasts, hierarchy_levels)

        return reconciled_forecasts, coherence_loss

    def _compute_coherence_loss(
        self,
        forecasts: torch.Tensor,
        hierarchy_levels: torch.Tensor
    ) -> torch.Tensor:
        """Compute hierarchical coherence loss."""
        # Simplified coherence loss based on level relationships
        total_loss = torch.tensor(0.0, device=forecasts.device)

        for level in range(self.num_levels - 1):
            current_mask = (hierarchy_levels == level)
            parent_mask = (hierarchy_levels == level + 1)

            if current_mask.any() and parent_mask.any():
                current_forecasts = forecasts[current_mask]
                parent_forecasts = forecasts[parent_mask]

                # Aggregate current level to parent level (simplified)
                if len(current_forecasts) > 0 and len(parent_forecasts) > 0:
                    aggregated = current_forecasts.mean(dim=0, keepdim=True)
                    coherence_error = F.mse_loss(aggregated, parent_forecasts.mean(dim=0, keepdim=True))
                    total_loss = total_loss + coherence_error

        return total_loss * self.coherence_weight


class HierarchicalReconciliationTransformer(pl.LightningModule):
    """Main hierarchical forecasting model with temporal anomaly feedback."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        prediction_length: int = 28,
        num_hierarchy_levels: int = 6,
        reconciliation_matrix: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """Initialize hierarchical reconciliation transformer.

        Args:
            input_dim: Input feature dimension
            hidden_size: Hidden size for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            prediction_length: Length of predictions
            num_hierarchy_levels: Number of hierarchy levels
            reconciliation_matrix: Matrix for reconciliation
            learning_rate: Learning rate
            weight_decay: Weight decay for optimization
            loss_weights: Weights for different loss components
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.prediction_length = prediction_length
        self.num_hierarchy_levels = num_hierarchy_levels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss weights
        self.loss_weights = loss_weights or {
            "forecast_loss": 1.0,
            "reconciliation_loss": 0.3,
            "anomaly_loss": 0.2,
            "coherence_loss": 0.1,
        }

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Hierarchy level embedding
        self.hierarchy_embedding = nn.Embedding(num_hierarchy_levels, hidden_size)

        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_length * input_dim),
        )

        # Anomaly detection module
        self.anomaly_detector = AnomalyDetectionModule(
            input_dim=input_dim,
            hidden_dim=64,
            window_size=7,
            threshold=0.1,
        )

        # Reconciliation module
        if reconciliation_matrix is None:
            reconciliation_matrix = torch.eye(num_hierarchy_levels)

        self.reconciliation_module = ReconciliationModule(
            num_levels=num_hierarchy_levels,
            reconciliation_matrix=reconciliation_matrix,
            coherence_weight=1.0,
            anomaly_weight=0.5,
            attention_dim=128,
        )

    def forward(
        self,
        x: torch.Tensor,
        hierarchy_levels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input sequences (batch, seq_len, features)
            hierarchy_levels: Hierarchy level indices (batch,)

        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len, _ = x.shape

        # Detect anomalies
        anomaly_scores, anomaly_attention = self.anomaly_detector(x)

        # Project input to hidden dimension
        x_proj = self.input_projection(x)

        # Add hierarchy level embeddings
        hierarchy_emb = self.hierarchy_embedding(hierarchy_levels)
        hierarchy_emb = hierarchy_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x_proj = x_proj + hierarchy_emb

        # Add positional encoding
        x_pos = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)

        # Transformer encoding
        encoded = self.transformer(x_pos)

        # Global average pooling
        pooled = encoded.mean(dim=1)

        # Generate base forecasts
        base_forecasts = self.forecast_head(pooled)
        base_forecasts = base_forecasts.view(batch_size, self.prediction_length, self.input_dim)

        # Apply reconciliation
        reconciled_forecasts, coherence_loss = self.reconciliation_module(
            base_forecasts, anomaly_scores, hierarchy_levels
        )

        return {
            "base_forecasts": base_forecasts,
            "reconciled_forecasts": reconciled_forecasts,
            "anomaly_scores": anomaly_scores,
            "anomaly_attention": anomaly_attention,
            "coherence_loss": coherence_loss,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["input"]
        y = batch["target"]
        hierarchy_levels = batch["hierarchy_level"]

        # Forward pass
        outputs = self(x, hierarchy_levels)

        # Compute losses
        forecast_loss = F.mse_loss(outputs["base_forecasts"], y)
        reconciliation_loss = F.mse_loss(outputs["reconciled_forecasts"], y)
        coherence_loss = outputs["coherence_loss"]

        # Anomaly loss (encourage detection of actual anomalies)
        anomaly_loss = F.binary_cross_entropy(
            outputs["anomaly_scores"].squeeze(),
            torch.zeros_like(outputs["anomaly_scores"].squeeze())
        )

        # Combined loss
        total_loss = (
            self.loss_weights["forecast_loss"] * forecast_loss +
            self.loss_weights["reconciliation_loss"] * reconciliation_loss +
            self.loss_weights["coherence_loss"] * coherence_loss +
            self.loss_weights["anomaly_loss"] * anomaly_loss
        )

        # Logging
        self.log("train_forecast_loss", forecast_loss, prog_bar=True)
        self.log("train_reconciliation_loss", reconciliation_loss, prog_bar=True)
        self.log("train_coherence_loss", coherence_loss, prog_bar=True)
        self.log("train_anomaly_loss", anomaly_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        x = batch["input"]
        y = batch["target"]
        hierarchy_levels = batch["hierarchy_level"]

        # Forward pass
        outputs = self(x, hierarchy_levels)

        # Compute losses
        forecast_loss = F.mse_loss(outputs["base_forecasts"], y)
        reconciliation_loss = F.mse_loss(outputs["reconciled_forecasts"], y)
        coherence_loss = outputs["coherence_loss"]

        # Log metrics
        self.log("val_forecast_loss", forecast_loss, sync_dist=True)
        self.log("val_reconciliation_loss", reconciliation_loss, sync_dist=True)
        self.log("val_coherence_loss", coherence_loss, sync_dist=True)

        # Compute WRMSSE (simplified version)
        wrmsse = self._compute_wrmsse(outputs["reconciled_forecasts"], y)
        self.log("val_wrmsse", wrmsse, sync_dist=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        x = batch["input"]
        y = batch["target"]
        hierarchy_levels = batch["hierarchy_level"]

        # Forward pass
        outputs = self(x, hierarchy_levels)

        # Compute losses
        forecast_loss = F.mse_loss(outputs["base_forecasts"], y)
        reconciliation_loss = F.mse_loss(outputs["reconciled_forecasts"], y)

        # Log metrics
        self.log("test_forecast_loss", forecast_loss, sync_dist=True)
        self.log("test_reconciliation_loss", reconciliation_loss, sync_dist=True)

        # Compute WRMSSE
        wrmsse = self._compute_wrmsse(outputs["reconciled_forecasts"], y)
        self.log("test_wrmsse", wrmsse, sync_dist=True)

        # Compute RMSE
        rmse = torch.sqrt(reconciliation_loss)
        self.log("test_rmse", rmse, sync_dist=True)
        self.log("test_loss", reconciliation_loss, sync_dist=True)

    def _compute_wrmsse(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute simplified WRMSSE metric."""
        # Simplified implementation - in practice, this would use proper scaling
        mse = F.mse_loss(predictions, targets)
        rmsse = torch.sqrt(mse) / (torch.std(targets) + 1e-8)
        return rmsse  # Weighted version would require proper weights

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Should be set from trainer config
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_wrmsse",
            },
        }