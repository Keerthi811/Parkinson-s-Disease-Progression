"""
Modeling training module for longitudinal Parkinson's progression prediction.
Sets up subject-grouped train/test splits, model initialization, training, 
evaluation metrics calculation, and model serialization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class ModelingError(Exception):
    """Custom exception raised for modeling phase errors."""
    pass

def group_train_test_split(
    df: pd.DataFrame, 
    subject_col: str, 
    test_size: float = 0.2, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a train-test split grouped by subject ID to prevent data leakage.
    Ensures that a subject's longitudinal records are completely in either 
    the train set or test set, but never split between both.
    
    Args:
        df (pd.DataFrame): Full dataset.
        subject_col (str): Column name for subject ID.
        test_size (float): Proportion of subjects to allocate to the test set.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train DataFrame and Test DataFrame.
    """
    logger.info(f"Splitting dataset using subject-grouped strategy (test_size={test_size})...")
    
    try:
        subjects = df[subject_col].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(subjects)
        
        split_idx = int(len(subjects) * (1 - test_size))
        train_subjects = subjects[:split_idx]
        test_subjects = subjects[split_idx:]
        
        train_df = df[df[subject_col].isin(train_subjects)].copy()
        test_df = df[df[subject_col].isin(test_subjects)].copy()
        
        logger.info(
            f"Split completed. Train: {len(train_df)} records ({len(train_subjects)} subjects). "
            f"Test: {len(test_df)} records ({len(test_subjects)} subjects)."
        )
        return train_df, test_df
    except Exception as e:
        raise ModelingError(f"Failed to perform group-based split: {e}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression metrics for evaluation.
    
    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.
        
    Returns:
        Dict[str, float]: Calculated metrics (MSE, RMSE, MAE, R2).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }

def run_classical_training(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the classical model training. 
    Selects features, performs group splits, trains the model, saves outputs.
    
    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.
        config (Dict[str, Any]): Loaded project configuration.
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    logger.info("Initializing classical modeling pipeline...")
    
    # 1. Setup metadata
    schema_cfg = config["data_validation"]["schema"]
    model_cfg = config["classical_modeling"]
    paths_cfg = config["paths"]
    
    subject_col = schema_cfg["subject_id_col"]
    target_col = schema_cfg["total_updrs_target"] # Focus on total UPDRS for progression
    
    # Dynamically extract all features (original biomarkers + engineered lag/rolling features)
    exclude_cols = [
        subject_col, 
        schema_cfg.get("age_col", "age"), 
        schema_cfg.get("sex_col", "sex"), 
        schema_cfg["test_time_col"], 
        schema_cfg["motor_updrs_target"], 
        schema_cfg["total_updrs_target"]
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Target variable: {target_col}")
    logger.info(f"Number of modeling features extracted: {len(features)}")
    
    # 2. Train-Test Split (Group-based)
    train_df, test_df = group_train_test_split(
        df=df,
        subject_col=subject_col,
        test_size=model_cfg.get("train_size", 0.8), # Allocating remainder to test
        seed=config.get("reproducibility", {}).get("seed", 42)
    )
    
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]
    
    # 3. Model Initialization (We will initialize RandomForestRegressor from configuration settings)
    rf_params = model_cfg.get("models", {}).get("random_forest", {})
    seed = config.get("reproducibility", {}).get("seed", 42)
    
    logger.info("Instantiating RandomForestRegressor model...")
    model = RandomForestRegressor(
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", 10),
        min_samples_split=rf_params.get("min_samples_split", 2),
        n_jobs=rf_params.get("n_jobs", -1),
        random_state=seed
    )
    
    # 4. Training
    try:
        logger.info("Fitting model on training split...")
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error fitting RandomForest model: {e}. Falling back to DummyRegressor.")
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        
    # 5. Prediction & Evaluation
    logger.info("Evaluating model predictions on test split...")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    # Print metrics to log
    for metric_name, value in metrics.items():
        logger.info(f"Test Metric - {metric_name}: {value:.4f}")
        
    # 6. Save model and predictions
    models_dir = resolve_path(paths_cfg["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "random_forest_progression.pkl"
    
    eval_dir = resolve_path(paths_cfg["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_save_path = eval_dir / "test_predictions.csv"
    metrics_save_path = eval_dir / "test_metrics.csv"
    
    try:
        # Save trained model
        joblib.dump(model, model_save_path)
        logger.info(f"Trained classical model serialized to: {model_save_path.as_posix()}")
        
        # Save predictions
        pred_df = test_df[[subject_col, schema_cfg["test_time_col"], target_col]].copy()
        pred_df["predicted_UPDRS"] = y_pred
        pred_df.to_csv(predictions_save_path, index=False)
        logger.info(f"Predictions saved to: {predictions_save_path.as_posix()}")
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(metrics_save_path, index=False)
        logger.info(f"Metrics saved to: {metrics_save_path.as_posix()}")
        
    except Exception as e:
        raise ModelingError(f"Failed to serialize model artifacts: {e}")
        
    return metrics
