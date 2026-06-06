"""
Logging setup utility for Parkinson's disease progression prediction project.
Configures logging levels, output formatters, handlers, and creates log directories.
"""

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict
from src.utils.config_loader import get_project_root

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up the logging system based on the configuration dictionary.
    Ensures that the directory for any file handler exists.
    
    Args:
        config (Dict[str, Any]): Loaded project configuration containing the 'logging' section.
    """
    logging_cfg = config.get("logging")
    if not logging_cfg:
        # Fallback logging configuration if none is found
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.warning("No 'logging' section found in config. Using fallback basicConfig.")
        return

    # Deep copy/modify the log file paths to make them absolute based on project root
    # Look for file handlers and resolve their directory paths
    handlers = logging_cfg.get("handlers", {})
    project_root = get_project_root()
    
    for name, handler_cfg in handlers.items():
        if "filename" in handler_cfg:
            rel_path = handler_cfg["filename"]
            abs_path = (project_root / rel_path).resolve()
            
            # Ensure the directory of the log file exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update the configuration dictionary with the absolute path string
            handler_cfg["filename"] = str(abs_path)

    try:
        logging.config.dictConfig(logging_cfg)
        logging.info("Logging configured successfully from configuration settings.")
    except Exception as e:
        # Fallback in case of invalid dictConfig settings
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to configure logging via dictConfig: {e}. Reverted to basicConfig.")
