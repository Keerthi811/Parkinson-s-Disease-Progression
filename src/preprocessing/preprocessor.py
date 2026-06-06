"""
Preprocessing module for longitudinal Parkinson's voice biomarkers.
Handles subject-level missing value interpolation, outlier detection/removal,
and feature scaling (Standard, MinMax, or Robust).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Custom exception raised for preprocessing step failures."""
    pass

class BiomarkerPreprocessor:
    """
    Preprocessor class to handle cleaning, imputation, outlier detection, 
    and scaling of longitudinal voice biomarkers.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the preprocessor with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Full pipeline configuration.
        """
        self.config = config
        self.prep_config = config.get("preprocessing", {})
        self.schema_config = config.get("data_validation", {}).get("schema", {})
        
        # Extracted config variables
        self.subject_col = self.schema_config.get("subject_id_col", "subject#")
        self.test_time_col = self.schema_config.get("test_time_col", "test_time")
        self.biomarkers = self.schema_config.get("voice_biomarkers", [])
        
        self.scaler_type = self.prep_config.get("scaling_method", "standard")
        self.scaler = None
        
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values. Since the dataset is longitudinal, we perform
        subject-level interpolation or forward/backward filling where appropriate.
        
        Args:
            df (pd.DataFrame): Dataframe containing missing values.
            
        Returns:
            pd.DataFrame: Imputed Dataframe.
        """
        strategy = self.prep_config.get("missing_value_strategy", "interpolate")
        method = self.prep_config.get("interpolation_method", "linear")
        
        logger.info(f"Imputing missing values using subject-level strategy: {strategy}")
        df_imputed = df.copy()
        
        # Group by subject to avoid bleeding data between different patients
        def _impute_group(group: pd.DataFrame) -> pd.DataFrame:
            # Sort group chronologically
            group = group.sort_values(by=self.test_time_col)
            
            if strategy == "interpolate":
                # Interpolate numeric columns, then forward/backward fill remaining NaNs
                group[self.biomarkers] = group[self.biomarkers].interpolate(method=method, limit_direction="both")
            elif strategy == "forward_fill":
                group[self.biomarkers] = group[self.biomarkers].ffill().bfill()
            elif strategy == "median":
                # Fill missing with median of this subject's biomarkers
                for col in self.biomarkers:
                    median_val = group[col].median()
                    if pd.isnull(median_val):
                        # Global median fallback if subject has no valid records for this biomarker
                        median_val = df[col].median()
                    group[col] = group[col].fillna(median_val)
            elif strategy == "drop":
                group = group.dropna(subset=self.biomarkers)
                
            return group

        try:
            # Apply group-wise imputation
            df_imputed = df_imputed.groupby(self.subject_col, group_keys=False).apply(_impute_group)
            
            # Final global fill check to make sure absolutely no NaNs remain
            if df_imputed[self.biomarkers].isnull().any().any():
                logger.warning("NaNs remain after subject-wise imputation. Performing global median fill fallback.")
                for col in self.biomarkers:
                    global_median = df[col].median()
                    df_imputed[col] = df_imputed[col].fillna(global_median)
                    
            return df_imputed
        except Exception as e:
            raise PreprocessingError(f"Error during missing value imputation: {e}")

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects and handles outliers in the voice biomarkers.
        Replaces outliers with upper/lower bounds rather than dropping to preserve longitudinal history.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            
        Returns:
            pd.DataFrame: Outlier-treated DataFrame.
        """
        outlier_removal = self.prep_config.get("outlier_removal", True)
        if not outlier_removal:
            logger.info("Outlier removal is disabled in config.")
            return df
            
        method = self.prep_config.get("outlier_method", "iqr")
        threshold = self.prep_config.get("iqr_threshold", 1.5)
        
        logger.info(f"Handling outliers using method: {method} (threshold={threshold})")
        df_clean = df.copy()
        
        try:
            for col in self.biomarkers:
                if method == "iqr":
                    q1 = df_clean[col].quantile(0.25)
                    q3 = df_clean[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                elif method == "zscore":
                    mean_val = df_clean[col].mean()
                    std_val = df_clean[col].std()
                    # 3 standard deviations
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                else:
                    raise PreprocessingError(f"Unsupported outlier method: {method}")
                
                # Cap outliers instead of dropping to keep time-series sequence length intact
                capped_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if capped_count > 0:
                    logger.debug(f"Capping {capped_count} outliers in biomarker '{col}'")
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    
            return df_clean
        except Exception as e:
            raise PreprocessingError(f"Error during outlier treatment: {e}")

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scales biomarker columns using the configured scaling method.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            fit (bool): If True, fits the scaler on the data before transforming.
            
        Returns:
            pd.DataFrame: Feature-scaled DataFrame.
        """
        logger.info(f"Scaling voice biomarkers using {self.scaler_type} scaler")
        df_scaled = df.copy()
        
        if self.scaler_type == "standard":
            scaler_class = StandardScaler
        elif self.scaler_type == "minmax":
            scaler_class = MinMaxScaler
        elif self.scaler_type == "robust":
            scaler_class = RobustScaler
        else:
            raise PreprocessingError(f"Unsupported scaling method: {self.scaler_type}")
            
        try:
            if fit or self.scaler is None:
                self.scaler = scaler_class()
                df_scaled[self.biomarkers] = self.scaler.fit_transform(df_scaled[self.biomarkers])
            else:
                df_scaled[self.biomarkers] = self.scaler.transform(df_scaled[self.biomarkers])
                
            return df_scaled
        except Exception as e:
            raise PreprocessingError(f"Error during feature scaling: {e}")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the entire cleaning, outlier treatment, and scaling pipeline.
        
        Args:
            df (pd.DataFrame): Raw DataFrame.
            
        Returns:
            pd.DataFrame: Cleaned and scaled DataFrame.
        """
        df_imputed = self.impute_missing_values(df)
        df_outliers = self.handle_outliers(df_imputed)
        df_scaled = self.scale_features(df_outliers, fit=True)
        return df_scaled

def run_preprocessing_stage(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Orchestrates the preprocessing stage. Runs imputation, outlier treatment, 
    scaling, and saves the resulting DataFrame to the processed data directory.
    
    Args:
        df (pd.DataFrame): Raw DataFrame from validation stage.
        config (Dict[str, Any]): Loaded project configuration.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    logger.info("Initializing preprocessing pipeline...")
    
    preprocessor = BiomarkerPreprocessor(config)
    df_processed = preprocessor.fit_transform(df)
    
    # Define destination path
    processed_dir = resolve_path(config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    destination_file = processed_dir / "parkinsons_processed.csv"
    try:
        df_processed.to_csv(destination_file, index=False)
        logger.info(f"Preprocessed dataset successfully saved to: {destination_file.as_posix()}")
    except Exception as e:
        raise PreprocessingError(f"Failed to save preprocessed file to disk: {e}")
        
    return df_processed
