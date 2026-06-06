"""
Feature engineering module for longitudinal Parkinson's voice biomarkers.
Extracts temporal features, historical lags, rolling window aggregations,
and clinical trends to capture disease progression.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class FeatureEngineeringError(Exception):
    """Custom exception raised for feature engineering pipeline failures."""
    pass

class FeatureEngineer:
    """
    FeatureEngineer class to construct historical, rolling, and trend-based 
    longitudinal features from voice biomarkers.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the feature engineer.
        
        Args:
            config (Dict[str, Any]): Project configuration dictionary.
        """
        self.config = config
        self.fe_config = config.get("feature_engineering", {})
        self.schema_config = config.get("data_validation", {}).get("schema", {})
        
        # Extracted configuration details
        self.subject_col = self.schema_config.get("subject_id_col", "subject#")
        self.test_time_col = self.schema_config.get("test_time_col", "test_time")
        self.biomarkers = self.schema_config.get("voice_biomarkers", [])
        
        self.lags = self.fe_config.get("temporal_lags", [1, 2])
        self.windows = self.fe_config.get("rolling_windows", [3])
        self.agg_funcs = self.fe_config.get("aggregation_functions", ["mean"])
        self.extract_trends = self.fe_config.get("extract_trends", True)

    def create_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates historical lag features for each voice biomarker per subject.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with added lag columns.
        """
        logger.info(f"Generating temporal lags for biomarker columns: {self.lags}")
        df_lags = df.copy()
        
        try:
            # Sort within group by test time before lagging
            df_lags = df_lags.sort_values(by=[self.subject_col, self.test_time_col])
            
            for lag in self.lags:
                for col in self.biomarkers:
                    col_name = f"{col}_lag_{lag}"
                    # Shift within subject group
                    df_lags[col_name] = df_lags.groupby(self.subject_col)[col].shift(lag)
                    
            return df_lags
        except Exception as e:
            raise FeatureEngineeringError(f"Error constructing lag features: {e}")

    def create_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates moving window statistics (mean, std, etc.) for each voice biomarker per subject.
        
        Args:
            df (pd.DataFrame): DataFrame with existing columns.
            
        Returns:
            pd.DataFrame: DataFrame with added rolling window columns.
        """
        logger.info(f"Generating rolling window features: windows={self.windows}, metrics={self.agg_funcs}")
        df_rolling = df.copy()
        
        try:
            df_rolling = df_rolling.sort_values(by=[self.subject_col, self.test_time_col])
            
            for window in self.windows:
                for col in self.biomarkers:
                    # Create a rolling object grouped by subject
                    rolling_obj = (
                        df_rolling.groupby(self.subject_col)[col]
                        .rolling(window=window, min_periods=1)
                    )
                    
                    for agg in self.agg_funcs:
                        col_name = f"{col}_roll_{agg}_w{window}"
                        if agg == "mean":
                            df_rolling[col_name] = rolling_obj.mean().reset_index(level=0, drop=True)
                        elif agg == "std":
                            df_rolling[col_name] = rolling_obj.std().reset_index(level=0, drop=True).fillna(0.0)
                            
            return df_rolling
        except Exception as e:
            raise FeatureEngineeringError(f"Error constructing rolling window features: {e}")

    def create_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates longitudinal trend features (e.g. difference in biomarker values 
        compared to the previous visit) per subject.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with trend features.
        """
        if not self.extract_trends:
            return df
            
        logger.info("Generating visit-to-visit biomarker trend (delta) features...")
        df_trends = df.copy()
        
        try:
            df_trends = df_trends.sort_values(by=[self.subject_col, self.test_time_col])
            
            for col in self.biomarkers:
                col_name = f"{col}_trend_delta"
                # Difference compared to immediate previous visit
                df_trends[col_name] = df_trends.groupby(self.subject_col)[col].diff().fillna(0.0)
                
            return df_trends
        except Exception as e:
            raise FeatureEngineeringError(f"Error constructing trend features: {e}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes all configured feature engineering transformations in sequence.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame.
            
        Returns:
            pd.DataFrame: Engineered DataFrame with temporal features, filled or pruned of NaNs.
        """
        df_lags = self.create_lags(df)
        df_rolling = self.create_rolling_statistics(df_lags)
        df_trends = self.create_trends(df_rolling)
        
        # After lagging, initial visits for each patient will have NaNs for lag columns
        # To maintain usability, we can backfill these NaNs with the patient's first available values
        logger.info("Imputing NaN values introduced by lagging via group-wise backward fill...")
        try:
            # List of newly created lag columns
            lag_cols = [c for c in df_trends.columns if "_lag_" in c]
            if lag_cols:
                # Group-wise backfill, then zero-fill any residual NaN (e.g. if subject has only 1 visit total)
                df_trends[lag_cols] = df_trends.groupby(self.subject_col)[lag_cols].bfill().fillna(0.0)
            return df_trends
        except Exception as e:
            raise FeatureEngineeringError(f"Error filling temporal NaNs: {e}")

def run_feature_engineering_stage(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Orchestrates the feature engineering stage. Builds features, saves the resulting 
    DataFrame to the processed data directory under a new filename.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame from preprocessing stage.
        config (Dict[str, Any]): Loaded project configuration.
        
    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    logger.info("Starting feature engineering pipeline...")
    
    engineer = FeatureEngineer(config)
    df_features = engineer.transform(df)
    
    processed_dir = resolve_path(config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    destination_file = processed_dir / "parkinsons_features.csv"
    try:
        df_features.to_csv(destination_file, index=False)
        logger.info(f"Feature-engineered dataset successfully saved to: {destination_file.as_posix()}")
    except Exception as e:
        raise FeatureEngineeringError(f"Failed to save feature-engineered file to disk: {e}")
        
    return df_features
