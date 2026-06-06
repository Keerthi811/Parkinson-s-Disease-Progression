#!/usr/bin/env python
"""
Main orchestrator script for the Parkinson's Disease progression prediction project.
Loads configuration, initializes logging, sets reproducibility seeds,
and executes all pipeline stages sequentially.
"""

import argparse
import logging
import sys
import time
from typing import Any, Dict

from src.utils.config_loader import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import set_seeds

# Import pipeline stages
from src.data_validation.validator import run_validation_stage, ValidationError
from src.preprocessing.preprocessor import run_preprocessing_stage, PreprocessingError
from src.feature_engineering.features import run_feature_engineering_stage, FeatureEngineeringError
from src.modeling.train import run_classical_training, ModelingError
from src.deep_learning.train_dl import run_deep_learning_training, DeepLearningError
from src.explainability.explain import run_explainability_stage, ExplainabilityError
from src.visualization.plots import run_visualization_stage, VisualizationError

logger = logging.getLogger("run_pipeline")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Parkinson's Disease progression prediction pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file (default: config.yaml)"
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Skip PyTorch deep learning modeling training stage."
    )
    return parser.parse_args()

def main() -> None:
    """
    Main entry point for the pipeline.
    Executes each stage sequentially with execution tracking and error reporting.
    """
    args = parse_arguments()
    
    # 1. Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration settings: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Setup logging
    try:
        setup_logging(config)
        logger.info("Initializing Parkinson's Disease Progression Pipeline...")
    except Exception as e:
        print(f"CRITICAL: Failed to configure logging setup: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Set random seeds for reproducibility
    seed = config.get("reproducibility", {}).get("seed", 42)
    det_alg = config.get("reproducibility", {}).get("deterministic_algorithms", True)
    set_seeds(seed, det_alg)
    
    start_time = time.time()
    
    try:
        # --- Stage 1: Data Validation ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 1: DATA VALIDATION")
        logger.info("========================================")
        raw_df = run_validation_stage(config)
        logger.info(f"Stage 1 Completed in {time.time() - stage_start:.2f} seconds.")
        
        # --- Stage 2: Data Preprocessing ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 2: PREPROCESSING")
        logger.info("========================================")
        preprocessed_df = run_preprocessing_stage(raw_df, config)
        logger.info(f"Stage 2 Completed in {time.time() - stage_start:.2f} seconds.")
        
        # --- Stage 3: Feature Engineering ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 3: FEATURE ENGINEERING")
        logger.info("========================================")
        features_df = run_feature_engineering_stage(preprocessed_df, config)
        logger.info(f"Stage 3 Completed in {time.time() - stage_start:.2f} seconds.")
        
        # --- Stage 4: Classical Machine Learning ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 4: CLASSICAL ML MODELING")
        logger.info("========================================")
        classical_metrics = run_classical_training(features_df, config)
        logger.info(f"Stage 4 Completed in {time.time() - stage_start:.2f} seconds.")
        
        # --- Stage 5: Deep Learning Sequence Modeling ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 5: DEEP LEARNING SEQUENCE MODELING")
        logger.info("========================================")
        if args.skip_dl:
            logger.info("Deep learning training skipped via command-line argument.")
            dl_metrics = {}
        else:
            dl_metrics = run_deep_learning_training(features_df, config)
            logger.info(f"Stage 5 Completed in {time.time() - stage_start:.2f} seconds.")
            
        # --- Stage 6: Model Explainability ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 6: MODEL EXPLAINABILITY")
        logger.info("========================================")
        run_explainability_stage(features_df, config)
        logger.info(f"Stage 6 Completed in {time.time() - stage_start:.2f} seconds.")
        
        # --- Stage 7: Visualization ---
        stage_start = time.time()
        logger.info("========================================")
        logger.info("STAGE 7: VISUALIZATION")
        logger.info("========================================")
        run_visualization_stage(features_df, config)
        logger.info(f"Stage 7 Completed in {time.time() - stage_start:.2f} seconds.")
        
        logger.info("========================================")
        logger.info(f"Pipeline executed successfully in {time.time() - start_time:.2f} seconds.")
        logger.info("========================================")
        
    except ValidationError as e:
        logger.critical(f"Pipeline failed at Validation stage: {e}")
        sys.exit(1)
    except PreprocessingError as e:
        logger.critical(f"Pipeline failed at Preprocessing stage: {e}")
        sys.exit(1)
    except FeatureEngineeringError as e:
        logger.critical(f"Pipeline failed at Feature Engineering stage: {e}")
        sys.exit(1)
    except ModelingError as e:
        logger.critical(f"Pipeline failed at Modeling stage: {e}")
        sys.exit(1)
    except DeepLearningError as e:
        logger.critical(f"Pipeline failed at Deep Learning stage: {e}")
        sys.exit(1)
    except ExplainabilityError as e:
        logger.critical(f"Pipeline failed at Explainability stage: {e}")
        sys.exit(1)
    except VisualizationError as e:
        logger.critical(f"Pipeline failed at Visualization stage: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline encountered an unexpected global error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
