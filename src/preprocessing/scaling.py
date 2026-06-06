"""
Scaling and splitting utilities for longitudinal Parkinson's Disease progression.
Provides modular helper functions for target/feature mapping, patient-grouped 
train/test splitting, patient-grouped cross-validation, and StandardScaler operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

def get_targets() -> List[str]:
    """
    Returns the target variable columns.
    
    Returns:
        List[str]: List containing motor_UPDRS and total_UPDRS.
    """
    return ["motor_UPDRS", "total_UPDRS"]

def get_features(df: pd.DataFrame) -> List[str]:
    """
    Identifies and returns predictor feature columns, excluding patient grouping 
    and target columns.
    
    Args:
        df (pd.DataFrame): Input dataset.
        
    Returns:
        List[str]: Predictor feature column names.
    """
    exclude = ["subject#"] + get_targets()
    features = [col for col in df.columns if col not in exclude]
    return features

def split_train_test(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and hold-out test sets based strictly on patient ID 
    (subject#) to guarantee zero patient overlap.
    
    Args:
        df (pd.DataFrame): Input longitudinal DataFrame.
        test_size (float): Proportion of unique subjects to allocate to the test set.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train DataFrame and Hold-out Test DataFrame.
    """
    logger.info(f"Performing patient-grouped train/test split (test_size={test_size})...")
    
    subject_col = "subject#"
    if subject_col not in df.columns:
        raise ValueError(f"Required subject column '{subject_col}' not found in DataFrame.")
        
    # Extract unique patients
    unique_subjects = df[subject_col].unique()
    n_subjects = len(unique_subjects)
    n_test_subjects = int(np.round(test_size * n_subjects))
    
    # Shuffle subjects reproducibly
    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(unique_subjects)
    
    test_subjects = shuffled_subjects[:n_test_subjects]
    train_subjects = shuffled_subjects[n_test_subjects:]
    
    train_df = df[df[subject_col].isin(train_subjects)].copy()
    test_df = df[df[subject_col].isin(test_subjects)].copy()
    
    logger.info(
        f"Train/Test split complete: "
        f"Train Set: {len(train_subjects)} patients, {len(train_df)} observations. "
        f"Test Set: {len(test_subjects)} patients, {len(test_df)} observations."
    )
    
    return train_df, test_df

def create_groupkfold(
    df: pd.DataFrame, 
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates indices for patient-grouped cross-validation splits using GroupKFold.
    
    Args:
        df (pd.DataFrame): Input training dataset.
        n_splits (int): Number of folds.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (train_idx, val_idx) arrays.
    """
    logger.info(f"Generating {n_splits}-Fold GroupKFold splits grouped by 'subject#'...")
    
    subject_col = "subject#"
    if subject_col not in df.columns:
        raise ValueError(f"Required subject column '{subject_col}' not found in DataFrame.")
        
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[subject_col].values
    
    # Extract X (features) to satisfy scikit-learn API (though not strictly needed for group split mapping)
    features = get_features(df)
    X = df[features].values
    
    splits = list(gkf.split(X, groups=groups))
    return splits

def scale_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits StandardScaler on train features and scales both train and validation DataFrames.
    
    Args:
        train_df (pd.DataFrame): Training set DataFrame.
        val_df (pd.DataFrame): Validation set DataFrame.
        features (List[str]): Columns to scale.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: Scaled train, scaled validation DataFrames 
                                                           (with original non-scaled targets/IDs), 
                                                           and the fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    
    # Fit on training features and transform
    train_scaled[features] = scaler.fit_transform(train_df[features])
    
    # Transform validation features
    if not val_df.empty:
        val_scaled[features] = scaler.transform(val_df[features])
        
    return train_scaled, val_scaled, scaler

def save_scaler(scaler: StandardScaler, path: Path) -> None:
    """
    Saves standard scaler instance to disk.
    
    Args:
        scaler (StandardScaler): The fitted StandardScaler instance.
        path (Path): Path file location to write.
    """
    logger.info(f"Saving fitted scaler to: {path.as_posix()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)
    except Exception as e:
        raise IOError(f"Failed to persist standard scaler object to disk: {e}")
