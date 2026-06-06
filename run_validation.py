#!/usr/bin/env python
"""
Validation runner script for Phase 3: Data Validation Framework.
Loads the raw Parkinson's Telemonitoring dataset, executes all validator classes
(Schema, Missingness, Duplicates, Types, and Ranges), logs the outcome,
and writes structured validation results to disk.
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
from src.data_validation.validator import DatasetValidator, ValidationError

logger = logging.getLogger("run_validation")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run dataset validations for the Parkinson's progression prediction project."
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
    Main entry method to execute validation tests on the raw dataset.
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
        logger.info("Initializing Parkinson's Telemonitoring Data Validation Run...")
    except Exception as e:
        print(f"CRITICAL: Failed to configure logger setup: {e}", file=sys.stderr)
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        # Resolve dataset file path
        raw_dir = resolve_path(config["paths"]["raw_data_dir"])
        required_files = config["data_validation"]["required_files"]
        if not required_files:
            raise DataLoaderError("No raw files specified in config.yaml data_validation section.")
            
        primary_file = required_files[0]
        dataset_path = raw_dir / primary_file
        
        # 3. Load dataset
        df = load_dataset(dataset_path)
        
        # 4. Instantiate validator orchestrator and run validation
        validator = DatasetValidator(config)
        report = validator.validate(df)
        
        # 5. Save reports
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        json_report_path = reports_dir / "validation_report.json"
        txt_report_path = reports_dir / "validation_summary.txt"
        
        # Save validation_report.json
        logger.info(f"Saving structured validation report to: {json_report_path.as_posix()}")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        # Save validation_summary.txt
        logger.info(f"Saving human-readable summary to: {txt_report_path.as_posix()}")
        summary_text = validator.generate_human_readable_summary(report)
        with open(txt_report_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
            
        # Log validation results
        status = report["validation_status"]
        if status == "PASS":
            logger.info(f"SUCCESS: Data validation passed successfully (Overall Status: {status}).")
        elif status == "WARNING":
            logger.warning(f"ATTENTION: Data validation completed with warnings (Overall Status: {status}).")
            for warning in report["range_validation"]["warnings"]:
                logger.warning(f"  - Range Warning: {warning}")
        else:
            logger.error(f"FAILURE: Data validation failed critical checks (Overall Status: {status}).")
            if report["schema_validation"]["status"] == "FAIL":
                logger.error(f"  - Schema Errors: {report['schema_validation']['errors']}")
            if report["datatype_validation"]["status"] == "FAIL":
                logger.error(f"  - Datatype Errors: {report['datatype_validation']['errors']}")
            if report["missing_values"]["status"] == "FAIL":
                logger.error(f"  - Missing Values: columns exceeded 5% limit.")
            if report["duplicates"]["status"] == "FAIL":
                logger.error(f"  - Duplicate Records Errors: {report['duplicates']['errors']}")
                
        duration = time.time() - start_time
        logger.info(f"Validation process completed in {duration:.2f} seconds.")
        
        # Exits cleanly with status code 0 on PASS or WARNING, but exits with 1 on critical FAIL
        if status == "FAIL":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except DataLoaderError as e:
        logger.critical(f"Data loading failed during validation: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.critical(f"Validation stage error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected global validation error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
