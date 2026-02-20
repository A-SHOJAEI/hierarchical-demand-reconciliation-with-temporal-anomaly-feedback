"""Configuration utilities for the hierarchical demand reconciliation system."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration as DictConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = OmegaConf.create(config_dict)
        logger.info(f"Configuration loaded from {config_path}")

        return config

    except yaml.YAMLError as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)

    logger.info(f"Configuration saved to {output_path}")


def get_device() -> torch.device:
    """Get the best available device for PyTorch.

    Returns:
        PyTorch device (CPU or CUDA)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def setup_logging(log_level: str = "INFO", log_file: Union[str, Path, None] = None) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )

    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    logger.info("Logging configured")


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ["data", "model", "training", "evaluation"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate data configuration
    if "sequence_length" not in config.data or config.data.sequence_length <= 0:
        raise ValueError("data.sequence_length must be positive")

    if "prediction_length" not in config.data or config.data.prediction_length <= 0:
        raise ValueError("data.prediction_length must be positive")

    # Validate training configuration
    if "batch_size" not in config.training or config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    if "learning_rate" not in config.training or config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")

    if "max_epochs" not in config.training or config.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive")

    # Validate model configuration
    if "hidden_size" not in config.model.forecasting_model or config.model.forecasting_model.hidden_size <= 0:
        raise ValueError("model.forecasting_model.hidden_size must be positive")

    logger.info("Configuration validation passed")


def update_paths(config: DictConfig, base_path: Union[str, Path]) -> DictConfig:
    """Update relative paths in configuration to be relative to base path.

    Args:
        config: Configuration with potentially relative paths
        base_path: Base path to resolve relative paths from

    Returns:
        Updated configuration with resolved paths
    """
    base_path = Path(base_path)

    # Update data paths
    if hasattr(config, 'data'):
        for path_key in ['dataset_path', 'calendar_path', 'prices_path', 'sales_path']:
            if hasattr(config.data, path_key):
                path_value = getattr(config.data, path_key)
                if not Path(path_value).is_absolute():
                    setattr(config.data, path_key, str(base_path / path_value))

    # Update system paths
    if hasattr(config, 'paths'):
        for path_key in ['data_dir', 'models_dir', 'results_dir', 'logs_dir']:
            if hasattr(config.paths, path_key):
                path_value = getattr(config.paths, path_key)
                if not Path(path_value).is_absolute():
                    setattr(config.paths, path_key, str(base_path / path_value))

    return config


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations with later configs taking precedence.

    Args:
        *configs: Configuration objects to merge

    Returns:
        Merged configuration
    """
    if not configs:
        return DictConfig({})

    merged = configs[0].copy()

    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)

    return merged


class ConfigManager:
    """Configuration manager for handling complex configuration scenarios."""

    def __init__(self, base_config_path: Union[str, Path]) -> None:
        """Initialize configuration manager.

        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = Path(base_config_path)
        self.config = load_config(self.base_config_path)
        self.overrides = DictConfig({})

    def add_override(self, key: str, value: Any) -> None:
        """Add configuration override.

        Args:
            key: Dot-separated configuration key (e.g., 'model.hidden_size')
            value: Value to override
        """
        OmegaConf.set(self.overrides, key, value)
        logger.info(f"Added override: {key} = {value}")

    def add_overrides(self, overrides: Dict[str, Any]) -> None:
        """Add multiple configuration overrides.

        Args:
            overrides: Dictionary of key-value pairs to override
        """
        for key, value in overrides.items():
            self.add_override(key, value)

    def get_config(self) -> DictConfig:
        """Get final configuration with all overrides applied.

        Returns:
            Final configuration
        """
        final_config = merge_configs(self.config, self.overrides)
        validate_config(final_config)
        return final_config

    def reset_overrides(self) -> None:
        """Reset all configuration overrides."""
        self.overrides = DictConfig({})
        logger.info("Configuration overrides reset")