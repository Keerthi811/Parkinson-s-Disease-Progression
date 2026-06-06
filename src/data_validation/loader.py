"""
Data loader module for Parkinson's Disease voice biomarker datasets.
Reads datasets from disk using pathlib, verifies existence, and handles errors.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoaderError(Exception):
    """Custom exception raised when data loading fails."""
    pass

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads a dataset from the specified file path using pandas.
    Verifies that the file exists and is a valid CSV/text file.
    
    Args:
        file_path (Path): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        DataLoaderError: If the file does not exist, is not readable, 
                         or fails to be parsed by pandas.
    """
    logger.info(f"Attempting to load dataset from: {file_path.as_posix()}")
    
    # Verify file path is not null/empty
    if not file_path:
        raise DataLoaderError("Provided file path is empty or invalid.")
        
    # Check if the file exists on disk
    if not file_path.exists():
        msg = (
            f"Dataset file not found at: '{file_path.resolve().as_posix()}'.\n"
            "Please ensure you download the original 'parkinsons_updrs.data' file "
            "from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring),\n"
            "rename it to 'parkinsons_updrs.csv' (or update config.yaml), and place it in the 'data/raw/' folder."
        )
        logger.error(msg)
        raise DataLoaderError(msg)
        
    # Verify that the path is a file, not a directory
    if not file_path.is_file():
        raise DataLoaderError(f"Target path exists but is not a file: '{file_path.as_posix()}'.")
        
    # Attempt to load the data
    try:
        # The UCI dataset parkinsons_updrs.data is a comma-separated values file.
        # It has a header on the first row. We load it using pandas.
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except pd.errors.EmptyDataError:
        msg = f"The dataset file at '{file_path.as_posix()}' is empty."
        logger.error(msg)
        raise DataLoaderError(msg)
    except pd.errors.ParserError as e:
        msg = f"Failed to parse the CSV file at '{file_path.as_posix()}'. It may be malformed. Error: {e}"
        logger.error(msg)
        raise DataLoaderError(msg)
    except Exception as e:
        msg = f"Unexpected error occurred while reading the dataset file: {e}"
        logger.error(msg)
        raise DataLoaderError(msg)
