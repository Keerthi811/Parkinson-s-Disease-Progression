"""
Data validation module for Parkinson's Disease voice biomarker datasets.
Checks that all required files and columns exist and match expected schemas.
Provides synthetic data generation to test the pipeline out-of-the-box.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised for dataset validation failures."""
    pass

def validate_dataset(df: pd.DataFrame, schema_config: Dict[str, Any]) -> bool:
    """
    Validates the input DataFrame against the configuration schema.
    
    Args:
        df (pd.DataFrame): The loaded dataset to validate.
        schema_config (Dict[str, Any]): Schema settings from config.yaml.
        
    Returns:
        bool: True if validation succeeds.
        
    Raises:
        ValidationError: If schema check, column existence, or null checks fail.
    """
    logger.info("Starting dataset schema validation...")
    
    # 1. Identify expected columns
    subject_col = schema_config.get("subject_id_col")
    test_time_col = schema_config.get("test_time_col")
    motor_target = schema_config.get("motor_updrs_target")
    total_target = schema_config.get("total_updrs_target")
    biomarkers = schema_config.get("voice_biomarkers", [])
    
    required_cols = [subject_col, test_time_col, motor_target, total_target] + biomarkers
    
    # Check if all required keys were defined in the config
    if any(col is None for col in [subject_col, test_time_col, motor_target, total_target]):
        raise ValidationError("Required column mapping is missing in configuration schema.")

    # 2. Check column existence in DataFrame
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in dataset: {missing_cols}")
        
    # 3. Check for empty dataframe
    if len(df) == 0:
        raise ValidationError("Dataset is empty. Validation failed.")
        
    # 4. Check for high proportion of missing values in biomarkers
    null_counts = df[required_cols].isnull().sum()
    high_null_cols = null_counts[null_counts > 0.5 * len(df)].index.tolist()
    if high_null_cols:
        logger.warning(f"Columns with >50% missing values detected: {high_null_cols}")
        
    # 5. Check data types (e.g. subject_id should be numeric)
    non_numeric_cols = []
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
            
    if non_numeric_cols:
        raise ValidationError(f"Columns must be numeric but contain non-numeric types: {non_numeric_cols}")

    logger.info("Dataset validation passed successfully. Data schema matches config expectations.")
    return True

def generate_synthetic_data(file_path: Path, config: Dict[str, Any]) -> None:
    """
    Generates a realistic synthetic longitudinal Parkinson's dataset.
    This allows the pipeline to run out-of-the-box before the user places raw files.
    
    Args:
        file_path (Path): Destination path for the CSV file.
        config (Dict[str, Any]): Project configuration dictionary.
    """
    logger.info(f"Generating synthetic dataset at {file_path.as_posix()} to enable immediate pipeline execution.")
    
    schema_config = config["data_validation"]["schema"]
    subject_col = schema_config["subject_id_col"]
    age_col = schema_config.get("age_col", "age")
    sex_col = schema_config.get("sex_col", "sex")
    test_time_col = schema_config["test_time_col"]
    motor_target = schema_config["motor_updrs_target"]
    total_target = schema_config["total_updrs_target"]
    biomarkers = schema_config["voice_biomarkers"]
    
    # Configure random seeds for synthetic data generation
    seed = config.get("reproducibility", {}).get("seed", 42)
    rng = np.random.default_rng(seed)
    
    # 42 subjects, 6 longitudinal visits each
    n_subjects = 42
    visits_per_subject = 6
    records = []
    
    for subject_id in range(1, n_subjects + 1):
        age = rng.integers(50, 85)
        sex = rng.choice([0, 1])
        base_motor_updrs = rng.uniform(10.0, 30.0)
        base_total_updrs = base_motor_updrs * rng.uniform(1.2, 1.5)
        
        # Longitudinal visits over 6 months
        for visit in range(visits_per_subject):
            test_time = visit * 30.0 + rng.uniform(-3, 3) # Roughly every 30 days
            
            # Simulate progression: UPDRS increases slightly over time
            progression_factor = 1.0 + (test_time / 365.0) * rng.uniform(0.05, 0.2)
            motor_updrs = base_motor_updrs * progression_factor
            total_updrs = base_total_updrs * progression_factor
            
            # Simulate biomarkers correlating weakly with progression
            biomarker_values = {}
            for idx, biomarker in enumerate(biomarkers):
                # Add some correlation with motor UPDRS
                base_val = rng.uniform(0.1, 0.5) if "Jitter" in biomarker or "Shimmer" in biomarker else rng.uniform(0.5, 0.9)
                if "HNR" in biomarker:
                    base_val = rng.uniform(15.0, 25.0) - 0.1 * motor_updrs
                else:
                    base_val += 0.005 * motor_updrs
                
                # Add random variance
                biomarker_values[biomarker] = max(0.001, base_val + rng.normal(0, 0.05 * base_val))
                
            record = {
                subject_col: subject_id,
                age_col: age,
                sex_col: sex,
                test_time_col: test_time,
                motor_target: motor_updrs,
                total_target: total_updrs,
                **biomarker_values
            }
            records.append(record)
            
    df = pd.DataFrame(records)
    
    # Ensure raw directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Generated synthetic dataset with {len(df)} records for {n_subjects} subjects.")

def run_validation_stage(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Orchestrates the validation stage. Ensures raw directories and files exist, 
    validates the file format and schema, and returns the DataFrame.
    
    Args:
        config (Dict[str, Any]): The loaded configuration dictionary.
        
    Returns:
        pd.DataFrame: Validated raw DataFrame.
        
    Raises:
        ValidationError: If raw files are missing and synthetic data generation fails.
    """
    raw_dir = resolve_path(config["paths"]["raw_data_dir"])
    required_files = config["data_validation"]["required_files"]
    schema_config = config["data_validation"]["schema"]
    
    if not required_files:
        raise ValidationError("No required files specified in data_validation configuration.")
        
    primary_file = required_files[0]
    primary_file_path = raw_dir / primary_file
    
    # Handle case where raw dataset is missing
    if not primary_file_path.exists():
        logger.warning(f"Primary raw file not found at: {primary_file_path.as_posix()}")
        try:
            generate_synthetic_data(primary_file_path, config)
        except Exception as e:
            raise ValidationError(f"Failed to generate synthetic data for missing file: {e}")
            
    # Load raw dataset
    try:
        logger.info(f"Loading raw dataset from {primary_file_path.as_posix()}...")
        df = pd.read_csv(primary_file_path)
    except Exception as e:
        raise ValidationError(f"Failed to read raw CSV file: {e}")
        
    # Run schema validation
    validate_dataset(df, schema_config)
    
    return df
