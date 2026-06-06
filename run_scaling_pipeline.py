#!/usr/bin/env python
"""
Evaluation and scaling runner script for Phase 7.
Loads the temporal dataset, partitions it into a patient-grouped train/test split,
runs patient-grouped cross-validation folds on the train subset, fits standard scaling
parameters strictly on the training set, persists the scaler, and generates evaluation
protocol reports.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.preprocessing.scaling import (
    get_features,
    get_targets,
    split_train_test,
    create_groupkfold,
    scale_features,
    save_scaler
)

logger = logging.getLogger("run_scaling_pipeline")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run leakage-free patient grouping, scaling, and evaluation split strategy."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file (default: config.yaml)"
    )
    return parser.parse_args()

def main() -> None:
    """
    Main execution pipeline for Phase 7.
    """
    args = parse_arguments()
    
    # 1. Load config settings
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load config file: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Setup logging system
    try:
        setup_logging(config)
        logger.info("=========================================")
        logger.info("PHASE 7: FEATURE SCALING AND LEAKAGE-FREE EVALUATION")
        logger.info("=========================================")
    except Exception as e:
        print(f"CRITICAL: Failed to configure logger setup: {e}", file=sys.stderr)
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        # Resolve dataset file paths
        processed_dir = resolve_path(config["paths"]["processed_data_dir"])
        temporal_dataset_path = processed_dir / "parkinsons_temporal.csv"
        
        # Verify temporal dataset exists
        if not temporal_dataset_path.exists():
            msg = f"Temporal dataset not found at expected path: {temporal_dataset_path.as_posix()}. Please run Phase 5 first."
            logger.critical(msg)
            sys.exit(1)
            
        # 3. Load temporal dataset
        logger.info(f"Loading temporal dataset from: {temporal_dataset_path.as_posix()}")
        df_temporal = pd.read_csv(temporal_dataset_path)
        
        # 4. Feature and Target Identification
        features = get_features(df_temporal)
        targets = get_targets()
        
        logger.info(f"Identified {len(features)} predictor features: {features}")
        logger.info(f"Identified targets: {targets}")
        
        # 5. Outer split: Train/Test (80% / 20% patients)
        seed = config.get("reproducibility", {}).get("seed", 42)
        train_df, test_df = split_train_test(df_temporal, test_size=0.2, seed=seed)
        
        # Verify outer split patient leakage
        train_subjects = set(train_df["subject#"].unique())
        test_subjects = set(test_df["subject#"].unique())
        overlap = train_subjects.intersection(test_subjects)
        if overlap:
            raise ValueError(f"Patient leakage detected in outer split! Overlapping patients: {overlap}")
        else:
            logger.info("Outer Split Verification: PASS. Zero patient overlap between Train and Hold-out Test sets.")
            
        # Save outer split datasets
        train_df_path = processed_dir / "parkinsons_train.csv"
        test_df_path = processed_dir / "parkinsons_test.csv"
        logger.info(f"Saving training set to: {train_df_path.as_posix()}")
        train_df.to_csv(train_df_path, index=False)
        logger.info(f"Saving hold-out test set to: {test_df_path.as_posix()}")
        test_df.to_csv(test_df_path, index=False)
        
        # Initialize evaluation split summary record list
        split_records = []
        
        # Append outer split stats
        split_records.append({
            "Fold": "Outer Train/Test Split",
            "Train Patients": len(train_subjects),
            "Validation Patients": len(test_subjects),
            "Train Rows": len(train_df),
            "Validation Rows": len(test_df)
        })
        
        # 6. Inner Splits: 5-Fold CV strictly on the training set
        cv_splits = create_groupkfold(train_df, n_splits=5)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            
            # Verify inner split patient leakage
            fold_train_subs = set(fold_train["subject#"].unique())
            fold_val_subs = set(fold_val["subject#"].unique())
            fold_overlap = fold_train_subs.intersection(fold_val_subs)
            if fold_overlap:
                raise ValueError(f"Patient leakage detected in CV Fold {fold_idx}! Overlap: {fold_overlap}")
            else:
                logger.debug(f"CV Fold {fold_idx} Leakage Check: PASS (no patient overlaps).")
                
            # Perform standard scaling on this fold to test for leakage-free scaling operations
            fold_train_scaled, fold_val_scaled, fold_scaler = scale_features(fold_train, fold_val, features)
            
            # Append fold stats
            split_records.append({
                "Fold": f"CV Fold {fold_idx}",
                "Train Patients": len(fold_train_subs),
                "Validation Patients": len(fold_val_subs),
                "Train Rows": len(fold_train),
                "Validation Rows": len(fold_val)
            })
            
        logger.info("All 5 inner GroupKFold splits generated and verified with zero patient leakage.")
        
        # 7. Fit final StandardScaler on the entire Training Set
        logger.info("Fitting final StandardScaler on complete Training Set features...")
        final_train_scaled, _, final_scaler = scale_features(train_df, pd.DataFrame(), features)
        
        # 8. Save standard scaler to disk
        models_dir = resolve_path(config["paths"]["models_dir"])
        scaler_dir = models_dir / "scalers"
        scaler_path = scaler_dir / "standard_scaler.pkl"
        
        save_scaler(final_scaler, scaler_path)
        
        # 9. Save reports
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        eval_reports_dir = reports_dir / "evaluation_protocol"
        eval_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # A. Split summary report
        split_summary_df = pd.DataFrame(split_records)
        split_summary_path = eval_reports_dir / "data_split_summary.csv"
        logger.info(f"Saving evaluation protocol splits summary to: {split_summary_path.as_posix()}")
        split_summary_df.to_csv(split_summary_path, index=False)
        
        # B. Scaling report
        scaling_report_path = eval_reports_dir / "scaling_report.txt"
        logger.info(f"Saving scaling summary statistics report to: {scaling_report_path.as_posix()}")
        
        with open(scaling_report_path, "w", encoding="utf-8") as f:
            f.write("=========================================================================\n")
            f.write("               PARKINSON'S PREPROCESSING SCALING REPORT\n")
            f.write("=========================================================================\n")
            f.write(f"Number of Features Scaled: {len(features)}\n")
            f.write(f"Saved Scaler File Path: {scaler_path.as_posix()}\n")
            f.write("=========================================================================\n\n")
            f.write("FEATURE MEAN AND STANDARD DEVIATION STATS:\n")
            f.write("-------------------------------------------------------------------------\n")
            f.write(f"{'Feature Name':<25} | {'Mean':<15} | {'Std Dev':<15}\n")
            f.write("-------------------------------------------------------------------------\n")
            for feat_idx, feat in enumerate(features):
                f.write(f"{feat:<25} | {final_scaler.mean_[feat_idx]:<15.6f} | {final_scaler.scale_[feat_idx]:<15.6f}\n")
            f.write("=========================================================================\n")
            
        duration = time.time() - start_time
        logger.info("-----------------------------------------")
        logger.info("SUCCESS: Feature scaling and evaluation splits completed successfully.")
        logger.info(f"Persisted StandardScaler to: {scaler_path.as_posix()}")
        logger.info(f"Total time elapsed: {duration:.2f} seconds.")
        logger.info("-----------------------------------------")
        
        sys.exit(0)
        
    except ValueError as e:
        logger.critical(f"Data partitioning or verification error: {e}")
        sys.exit(1)
    except IOError as e:
        logger.critical(f"File writing or persistence error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected global error in evaluation stage: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
