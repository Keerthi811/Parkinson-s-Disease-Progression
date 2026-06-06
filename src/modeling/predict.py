"""
Prediction module for generating progression scores on new voice biomarker recordings
using the trained classical machine learning models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception raised for prediction errors."""
    pass

class ModelPredictor:
    """
    ModelPredictor loads a trained model and produces predictions
    for patient voice biomarker entries, validating inputs beforehand.
    """
    def __init__(self, model_path: Union[str, Path]):
        """
        Initializes the predictor by loading the model.
        
        Args:
            model_path (Union[str, Path]): Path to the joblib serialized model.
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        
    def _load_model(self) -> Any:
        """
        Loads the joblib serialized model.
        
        Returns:
            Any: The loaded estimator/model.
            
        Raises:
            PredictionError: If model file is missing or corrupted.
        """
        if not self.model_path.exists():
            raise PredictionError(f"Model file not found at path: {self.model_path.as_posix()}")
            
        try:
            logger.info(f"Loading serialized model from {self.model_path.as_posix()}...")
            return joblib.load(self.model_path)
        except Exception as e:
            raise PredictionError(f"Failed to load model file: {e}")

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Validates the columns and generates predictions for the input DataFrame.
        
        Args:
            df (pd.DataFrame): Dataframe containing the patient records and biomarkers.
            feature_cols (List[str]): Expected list of feature columns in the exact order.
            
        Returns:
            np.ndarray: Generated model predictions.
            
        Raises:
            PredictionError: If input data fails validation.
        """
        logger.info("Validating features for inference...")
        
        # Check missing features
        missing_feats = [col for col in feature_cols if col not in df.columns]
        if missing_feats:
            raise PredictionError(f"Input DataFrame is missing required features: {missing_feats}")
            
        try:
            # Subset and align features to match training order
            X = df[feature_cols]
            
            # Predict
            logger.info("Generating predictions...")
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            raise PredictionError(f"Failed to run predictions: {e}")
