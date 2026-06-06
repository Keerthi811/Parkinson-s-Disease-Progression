#!/usr/bin/env python
"""
Temporal preprocessing runner script for Phase 5: Longitudinal Temporal Preprocessing.
Loads the cleaned Parkinson's dataset, groups observations by subject to compute 
patient statistics, sorts the dataset chronologically, validates chronological consistency,
generates longitudinal plots, and exports the final patient-aware temporal dataset.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import pandas as pd

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.preprocessing.temporal import TemporalPreprocessor, TemporalError

logger = logging.getLogger("run_temporal_preprocessing")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run longitudinal temporal preprocessing for the Parkinson's prediction project."
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
    Main execution pipeline for Phase 5.
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
        logger.info("PHASE 5: LONGITUDINAL TEMPORAL PREPROCESSING")
        logger.info("=========================================")
    except Exception as e:
        print(f"CRITICAL: Failed to configure logger setup: {e}", file=sys.stderr)
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        # Resolve dataset file paths
        processed_dir = resolve_path(config["paths"]["processed_data_dir"])
        cleaned_dataset_path = processed_dir / "parkinsons_cleaned.csv"
        temporal_dataset_path = processed_dir / "parkinsons_temporal.csv"
        
        # Verify cleaned dataset exists
        if not cleaned_dataset_path.exists():
            msg = f"Cleaned dataset not found at expected path: {cleaned_dataset_path.as_posix()}. Please run Phase 4 first."
            logger.critical(msg)
            sys.exit(1)
            
        # 3. Load cleaned dataset
        logger.info(f"Loading cleaned dataset from: {cleaned_dataset_path.as_posix()}")
        df_cleaned = pd.read_csv(cleaned_dataset_path)
        
        # 4. Instantiate preprocessor orchestrator
        preprocessor = TemporalPreprocessor(config)
        
        # 5. Group patients and compute stats
        patient_stats, summary_stats = preprocessor.group_patients(df_cleaned)
        
        # Resolve reporting directories
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        temporal_reports_dir = reports_dir / "temporal"
        temporal_reports_dir.mkdir(parents=True, exist_ok=True)
        
        patient_stats_csv_path = temporal_reports_dir / "patient_statistics.csv"
        temporal_summary_json_path = temporal_reports_dir / "temporal_summary.json"
        
        # Save patient_statistics.csv
        logger.info(f"Saving patient statistics to: {patient_stats_csv_path.as_posix()}")
        patient_stats.to_csv(patient_stats_csv_path, index=False)
        
        # 6. Chronological sort & validate
        df_sorted, ordering_status = preprocessor.sort_and_validate_chronology(df_cleaned)
        
        # 7. Save temporal dataset
        logger.info(f"Saving sorted temporal dataset to: {temporal_dataset_path.as_posix()}")
        df_sorted.to_csv(temporal_dataset_path, index=False)
        
        # 8. Longitudinal visualizations
        figures_dir = resolve_path(config["paths"]["figures_dir"])
        preprocessor.generate_longitudinal_plots(df_sorted, figures_dir)
        
        # 9. Save temporal summary report
        summary_report = {
            "total_patients": summary_stats["total_patients"],
            "total_observations": summary_stats["total_observations"],
            "avg_visits_per_patient": round(summary_stats["avg_visits_per_patient"], 4),
            "max_visits": summary_stats["max_visits"],
            "min_visits": summary_stats["min_visits"],
            "ordering_status": ordering_status
        }
        
        logger.info(f"Saving temporal summary report to: {temporal_summary_json_path.as_posix()}")
        with open(temporal_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4)
            
        duration = time.time() - start_time
        logger.info("-----------------------------------------")
        logger.info(f"SUCCESS: Longitudinal temporal preprocessing completed (Status: {ordering_status}).")
        logger.info(f"Temporal dataset saved: {df_sorted.shape[0]} rows, {df_sorted.shape[1]} columns.")
        logger.info(f"Total time elapsed: {duration:.2f} seconds.")
        logger.info("-----------------------------------------")
        
        sys.exit(0)
            
    except TemporalError as e:
        logger.critical(f"Temporal preprocessing failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected global error in temporal preprocessing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
