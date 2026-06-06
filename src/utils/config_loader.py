"""
Configuration loading utility for Parkinson's disease progression prediction project.
Handles resolving absolute paths relative to the project root and parsing the config.yaml.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

class ConfigError(Exception):
    """Custom exception raised for errors in project configuration."""
    pass

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    Assumes this script runs from src/utils/ or somewhere in src/.
    
    Returns:
        Path: Project root directory path.
    """
    # src/utils/config_loader.py -> parent is utils, parent.parent is src, parent.parent.parent is project root
    return Path(__file__).resolve().parents[2]

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the config.yaml file from the project root.
    
    Args:
        config_path (str): Relative or absolute path to the configuration file.
        
    Returns:
        Dict[str, Any]: The configuration settings dictionary.
        
    Raises:
        ConfigError: If config file is missing, invalid, or unparseable.
    """
    root_dir = get_project_root()
    full_path = root_dir / config_path
    
    if not full_path.exists():
        raise ConfigError(
            f"Configuration file not found at expected path: {full_path.as_posix()}. "
            "Please ensure you run the pipeline from the project root or provide a valid path."
        )
        
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if config is None:
            raise ConfigError(f"Configuration file at {full_path} is empty.")
            
        return config
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ConfigError(f"Unexpected error loading configuration: {e}")

def resolve_path(relative_path: str) -> Path:
    """
    Resolves a relative path defined in configuration to an absolute path based on the project root.
    
    Args:
        relative_path (str): The relative path.
        
    Returns:
        Path: Resolved absolute path.
    """
    root_dir = get_project_root()
    return (root_dir / relative_path).resolve()
