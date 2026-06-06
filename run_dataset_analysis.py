#!/usr/bin/env python
"""
Driver script for Phase 2: Dataset Acquisition and Understanding.
Loads the Parkinson's Telemonitoring dataset, analyzes its schema and statistics,
constructs a data dictionary, compiles patient summary tables, and exports EDA plots.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.data_validation.loader import load_dataset, DataLoaderError
from src.data_validation.analyzer import (
    analyze_schema,
    compute_statistics,
    build_data_dictionary,
    generate_summary,
    generate_eda_plots,
    AnalyzerError
)

logger = logging.getLogger("run_dataset_analysis")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Parkinson's Telemonitoring dataset profiling and analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )
    return parser.parse_args()

def main() -> None:
    """
    Main driver method for running end-to-end dataset understanding.
    """
    args = parse_arguments()
    
    # 1. Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load config settings: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Setup logging
    try:
        setup_logging(config)
        logger.info("Initializing Parkinson's Telemonitoring Dataset Analysis Pipeline...")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize logging: {e}", file=sys.stderr)
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
        logger.info("----------------------------------------")
        logger.info("LOADING DATASET")
        logger.info("----------------------------------------")
        df = load_dataset(dataset_path)
        
        # 4. Profile schema
        logger.info("----------------------------------------")
        logger.info("PROFILING SCHEMA")
        logger.info("----------------------------------------")
        schema_df = analyze_schema(df)
        
        # 5. Compute statistics
        logger.info("----------------------------------------")
        logger.info("COMPUTING SUMMARY STATISTICS")
        logger.info("----------------------------------------")
        stats_df = compute_statistics(df)
        
        # 6. Build data dictionary
        logger.info("----------------------------------------")
        logger.info("CONSTRUCTING DATA DICTIONARY")
        logger.info("----------------------------------------")
        data_dict_df = build_data_dictionary(df)
        
        # 7. Compile longitudinal summary
        logger.info("----------------------------------------")
        logger.info("COMPILING LONGITUDINAL SUMMARY")
        logger.info("----------------------------------------")
        summary_df = generate_summary(df, config)
        
        # 8. Export CSV reports
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        summary_out = reports_dir / "dataset_summary.csv"
        stats_out = reports_dir / "dataset_statistics.csv"
        dict_out = reports_dir / "data_dictionary.csv"
        
        logger.info(f"Saving summary report to: {summary_out.as_posix()}")
        summary_df.to_csv(summary_out, index=False)
        
        logger.info(f"Saving statistics table to: {stats_out.as_posix()}")
        stats_df.to_csv(stats_out, index=False)
        
        logger.info(f"Saving data dictionary to: {dict_out.as_posix()}")
        data_dict_df.to_csv(dict_out, index=False)
        
        # 9. Generate and save EDA plots
        figures_dir = resolve_path(config["paths"]["figures_dir"])
        logger.info("----------------------------------------")
        logger.info("GENERATING EXPLORATORY PLOTS")
        logger.info("----------------------------------------")
        generate_eda_plots(df, config, figures_dir)
        
        duration = time.time() - start_time
        logger.info("----------------------------------------")
        logger.info(f"Analysis completed successfully in {duration:.2f} seconds!")
        logger.info("----------------------------------------")
        
    except DataLoaderError as e:
        logger.critical(f"Dataset loading failed: {e}")
        sys.exit(1)
    except AnalyzerError as e:
        logger.critical(f"Analysis profiling failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error running analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
