"""
Configuration module for NASCAR Driver Career Analysis project.

This module provides utilities for loading and managing project configuration
from YAML files, ensuring consistent settings across all components.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file. 
                                   Defaults to config/config.yaml
    
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    if config_path is None:
        # Get the directory where this __init__.py file is located
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def get_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract and validate data paths from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, str]: Dictionary of data paths
    """
    paths = config.get('paths', {})
    
    # Ensure all required paths exist
    required_paths = ['raw_data', 'processed_data', 'models', 'outputs']
    for path_key in required_paths:
        if path_key not in paths:
            raise KeyError(f"Required path '{path_key}' not found in configuration")
            
        # Create directories if they don't exist
        path = Path(paths[path_key])
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def get_model_params(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Get model parameters for specific model type.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        model_type (str): Type of model ('lstm', 'clustering', etc.)
        
    Returns:
        Dict[str, Any]: Model parameters
        
    Raises:
        KeyError: If model type not found in configuration
    """
    models_config = config.get('models', {})
    if model_type not in models_config:
        raise KeyError(f"Model type '{model_type}' not found in configuration")
    
    return models_config[model_type]


# Default configuration instance
_config = None

def get_config() -> Dict[str, Any]:
    """
    Get the global configuration instance (singleton pattern).
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


# Make key functions available at package level
__all__ = [
    'load_config',
    'get_config', 
    'get_data_paths',
    'get_model_params'
]