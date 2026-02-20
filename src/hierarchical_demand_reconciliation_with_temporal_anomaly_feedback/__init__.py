"""Hierarchical Demand Reconciliation with Temporal Anomaly Feedback.

A closed-loop hierarchical forecasting system that uses anomaly detection
at fine-grained product levels to automatically trigger forecast reconciliation
adjustments at higher aggregation levels.
"""

__version__ = "1.0.0"
__author__ = "Alireza Shojaei"
__email__ = "alireza.shojaei@example.com"

from .data.loader import M5DataLoader
from .data.preprocessing import HierarchicalPreprocessor
from .models.model import HierarchicalReconciliationTransformer
from .training.trainer import HierarchicalTrainer
from .evaluation.metrics import HierarchicalMetrics
from .utils.config import load_config

__all__ = [
    "M5DataLoader",
    "HierarchicalPreprocessor",
    "HierarchicalReconciliationTransformer",
    "HierarchicalTrainer",
    "HierarchicalMetrics",
    "load_config",
]