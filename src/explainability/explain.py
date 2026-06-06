"""
Explainability and interpretability module for Parkinson's progression model.
Calculates global feature importances and SHAP (SHapley Additive exPlanations) 
values to identify predictive voice biomarkers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
import numpy as np
from src.utils.config_loader import resolve_path
from src.modeling.train import group_train_test_split

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExplainabilityError(Exception):
    """Custom exception raised for errors in the explainability pipeline."""
    pass

class ModelExplainer:
    """
    ModelExplainer handles calculating feature attributions and shapley value 
    summaries for trained classical models.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the explainer with project configs.
        
        Args:
            config (Dict[str, Any]): Project configuration dictionary.
        """
        self.config = config
        self.exp_config = config.get("explainability", {})
        self.schema_config = config.get("data_validation", {}).get("schema", {})
        self.paths_config = config.get("paths", {})
        
        self.subject_col = self.schema_config.get("subject_id_col", "subject#")
        self.target_col = self.schema_config.get("total_updrs_target", "total_UPDRS")

    def get_feature_importances(self, model: Any, features: List[str]) -> pd.DataFrame:
        """
        Extracts built-in feature importances from a trained tree-based model.
        
        Args:
            model (Any): Trained scikit-learn model.
            features (List[str]): List of feature names matching model inputs.
            
        Returns:
            pd.DataFrame: DataFrame containing features sorted by importance.
        """
        logger.info("Computing tree-based feature importances...")
        
        if not hasattr(model, "feature_importances_"):
            logger.warning("Loaded model does not support built-in feature importances.")
            return pd.DataFrame(columns=["feature", "importance"])
            
        try:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": features,
                "importance": importances
            }).sort_values(by="importance", ascending=False)
            
            return importance_df
        except Exception as e:
            raise ExplainabilityError(f"Failed to calculate feature importances: {e}")

    def get_shap_values(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Computes SHAP values for the test dataset using the training dataset as background.
        
        Args:
            model (Any): Trained classical model.
            X_train (pd.DataFrame): Training feature data.
            X_test (pd.DataFrame): Testing feature data.
            
        Returns:
            pd.DataFrame: DataFrame containing mean absolute SHAP values for each feature.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library is not available. Skipping SHAP computation.")
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])
            
        logger.info("Computing SHAP values using TreeExplainer...")
        try:
            bg_samples = self.exp_config.get("shap_background_samples", 100)
            
            # Select background data from training split
            if len(X_train) > bg_samples:
                # Select evenly spaced rows or a sample
                background = X_train.sample(n=bg_samples, random_state=42)
            else:
                background = X_train
                
            explainer = shap.TreeExplainer(model, data=background)
            shap_values = explainer.shap_values(X_test)
            
            # Check for multiple output shapes (e.g. multi-class classification vs regression)
            if isinstance(shap_values, list):
                # Take the first target for regression
                shap_matrix = shap_values[0]
            else:
                shap_matrix = shap_values
                
            # Compute mean absolute SHAP value per feature
            mean_abs_shaps = np.abs(shap_matrix).mean(axis=0)
            
            shap_df = pd.DataFrame({
                "feature": X_test.columns,
                "mean_abs_shap": mean_abs_shaps
            }).sort_values(by="mean_abs_shap", ascending=False)
            
            return shap_df
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}. Returning empty table.")
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])

def run_explainability_stage(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Orchestrates the explainability pipeline. Loads trained models, splits features,
    generates feature importances and SHAP values, and saves tables to reports/tables.
    
    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.
        config (Dict[str, Any]): Loaded project configurations.
    """
    logger.info("Starting explainability pipeline...")
    
    # 1. Load trained model
    paths_cfg = config["paths"]
    models_dir = resolve_path(paths_cfg["models_dir"])
    model_file = models_dir / "random_forest_progression.pkl"
    
    if not model_file.exists():
        logger.warning(f"Trained model not found at {model_file.as_posix()}. Explainability skipped.")
        return
        
    try:
        model = joblib.load(model_file)
    except Exception as e:
        raise ExplainabilityError(f"Failed to load model for explainability analysis: {e}")
        
    # 2. Extract features
    schema_cfg = config["data_validation"]["schema"]
    subject_col = schema_cfg["subject_id_col"]
    target_col = schema_cfg["total_updrs_target"]
    
    exclude_cols = [
        subject_col, 
        schema_cfg.get("age_col", "age"), 
        schema_cfg.get("sex_col", "sex"), 
        schema_cfg["test_time_col"], 
        schema_cfg["motor_updrs_target"], 
        schema_cfg["total_updrs_target"]
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    
    # 3. Create split matching training setup
    train_df, test_df = group_train_test_split(
        df=df,
        subject_col=subject_col,
        test_size=config["classical_modeling"].get("train_size", 0.8),
        seed=config.get("reproducibility", {}).get("seed", 42)
    )
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    # 4. Perform analysis
    explainer = ModelExplainer(config)
    importance_df = explainer.get_feature_importances(model, features)
    shap_df = explainer.get_shap_values(model, X_train, X_test)
    
    # 5. Save tables to reports/tables
    tables_dir = resolve_path(paths_cfg["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    importance_path = tables_dir / "feature_importances.csv"
    shap_path = tables_dir / "shap_values.csv"
    
    try:
        if not importance_df.empty:
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importances saved to: {importance_path.as_posix()}")
            
        if not shap_df.empty:
            shap_df.to_csv(shap_path, index=False)
            logger.info(f"SHAP values saved to: {shap_path.as_posix()}")
            
    except Exception as e:
        raise ExplainabilityError(f"Failed to save explainability tables: {e}")
