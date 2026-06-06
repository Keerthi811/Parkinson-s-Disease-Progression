#!/usr/bin/env python
"""
Data cleaning runner script for Phase 4: Data Cleaning and Quality Improvement.
Loads the raw Parkinson's Telemonitoring dataset, executes all data cleaning classes
(Deduplication, Imputation, Datatype Standardization, and Integrity checks),
saves the cleaned dataset, and writes structured cleaning reports.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.data_validation.loader import load_dataset, DataLoaderError
from src.preprocessing.cleaner import DataCleaner, CleanerError

logger = logging.getLogger("run_data_cleaning")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run dataset cleaning and quality improvement for the Parkinson's prediction project."
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
    Main entry method to execute data cleaning on the raw dataset.
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
        logger.info("Initializing Parkinson's Telemonitoring Data Cleaning Run...")
    except Exception as e:
        print(f"CRITICAL: Failed to configure logger setup: {e}", file=sys.stderr)
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        # Resolve dataset file paths
        raw_dir = resolve_path(config["paths"]["raw_data_dir"])
        required_files = config["data_validation"]["required_files"]
        if not required_files:
            raise DataLoaderError("No raw files specified in config.yaml data_validation section.")
            
        primary_file = required_files[0]
        dataset_path = raw_dir / primary_file
        
        # 3. Load dataset
        df_raw = load_dataset(dataset_path)
        
        # 4. Instantiate cleaner orchestrator and run data cleaning
        cleaner = DataCleaner(config)
        df_cleaned, report = cleaner.clean(df_raw)
        
        # Resolve output directories
        processed_dir = resolve_path(config["paths"]["processed_data_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        cleaned_dataset_path = processed_dir / "parkinsons_cleaned.csv"
        
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        cleaning_reports_dir = reports_dir / "cleaning"
        cleaning_reports_dir.mkdir(parents=True, exist_ok=True)
        
        json_report_path = cleaning_reports_dir / "cleaning_report.json"
        txt_report_path = cleaning_reports_dir / "cleaning_summary.txt"
        
        # 5. Save cleaned dataset
        logger.info(f"Saving cleaned dataset to: {cleaned_dataset_path.as_posix()}")
        df_cleaned.to_csv(cleaned_dataset_path, index=False)
        
        # 6. Save cleaning reports
        logger.info(f"Saving structured cleaning report to: {json_report_path.as_posix()}")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Saving human-readable summary to: {txt_report_path.as_posix()}")
        summary_text = cleaner.generate_human_readable_summary(report)
        with open(txt_report_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
            
        duration = time.time() - start_time
        logger.info("----------------------------------------")
        logger.info(f"SUCCESS: Data cleaning completed successfully (Status: {report['status']}).")
        logger.info(f"Cleaned dataset saved: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns.")
        logger.info(f"Total time elapsed: {duration:.2f} seconds.")
        logger.info("----------------------------------------")
        
        sys.exit(0)
            
    except DataLoaderError as e:
        logger.critical(f"Data loading failed during cleaning: {e}")
        sys.exit(1)
    except CleanerError as e:
        logger.critical(f"Cleaning process failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected global data cleaning error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
