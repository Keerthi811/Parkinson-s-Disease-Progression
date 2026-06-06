#!/usr/bin/env python
"""
Temporal Feature Engineering runner script for Phase 8.
Loads the temporally ordered Parkinson's dataset, generates patient-aware
sequential features (lags, rolling statistics, rate of change, historical
variability) for all 16 voice biomarkers, resolves introduced NaN values,
saves the enriched dataset, and outputs a feature summary report.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import pandas as pd

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.feature_engineering.temporal_features import TemporalFeatureEngineer, TemporalFeatureError

logger = logging.getLogger("run_temporal_feature_engineering")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run patient-aware temporal feature engineering for the "
            "Parkinson's Disease progression prediction project."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution pipeline for Phase 8.

    Steps:
        1. Load project configuration.
        2. Load the chronologically ordered temporal dataset.
        3. Engineer all temporal features (lags, rolling, rate of change, expanding std).
        4. Validate output integrity (row count, NaN count).
        5. Save the enriched dataset to data/processed/parkinsons_temporal_features.csv.
        6. Save the feature summary report to reports/feature_engineering/feature_summary.csv.
    """
    args = parse_arguments()

    # 1. Load configuration
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"CRITICAL: Failed to load config file: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup logging
    try:
        setup_logging(config)
        logger.info("=========================================")
        logger.info("PHASE 8: TEMPORAL FEATURE ENGINEERING")
        logger.info("=========================================")
    except Exception as exc:
        print(f"CRITICAL: Failed to configure logging: {exc}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()

    try:
        # Resolve input dataset path
        processed_dir = resolve_path(config["paths"]["processed_data_dir"])
        temporal_path = processed_dir / "parkinsons_temporal.csv"

        if not temporal_path.exists():
            logger.critical(
                f"Temporal dataset not found at: {temporal_path.as_posix()}. "
                "Please run Phase 5 (run_temporal_preprocessing.py) first."
            )
            sys.exit(1)

        # 3. Load dataset
        logger.info(f"Loading temporal dataset from: {temporal_path.as_posix()}")
        df = pd.read_csv(temporal_path)
        rows_in, cols_in = df.shape
        logger.info(f"Dataset loaded: {rows_in} rows × {cols_in} columns.")

        # 4. Instantiate engineer and build features
        engineer = TemporalFeatureEngineer(config)
        df_enriched, feature_meta = engineer.engineer_features(df)

        # 5. Integrity validation
        rows_out, cols_out = df_enriched.shape
        remaining_nans = int(df_enriched.isna().sum().sum())

        logger.info("-" * 45)
        logger.info("OUTPUT VALIDATION:")
        logger.info(f"  Input shape  : {rows_in} rows × {cols_in} columns")
        logger.info(f"  Output shape : {rows_out} rows × {cols_out} columns")
        logger.info(f"  New features : {cols_out - cols_in}")
        logger.info(f"  Total NaNs remaining : {remaining_nans}")

        if remaining_nans > 0:
            logger.warning(
                f"{remaining_nans} NaN values remain after resolution. "
                "Inspect the data for patients with insufficient visit counts."
            )
        else:
            logger.info("  NaN check : PASS (zero NaNs remaining)")
        logger.info("-" * 45)

        # 6. Save enriched dataset
        output_path = processed_dir / "parkinsons_temporal_features.csv"
        logger.info(f"Saving enriched dataset to: {output_path.as_posix()}")
        df_enriched.to_csv(output_path, index=False)

        # 7. Save feature summary report
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        fe_report_dir = reports_dir / "feature_engineering"
        engineer.generate_feature_summary(feature_meta, fe_report_dir)

        duration = time.time() - start_time
        logger.info("-----------------------------------------")
        logger.info("SUCCESS: Temporal feature engineering completed.")
        logger.info(
            f"  Enriched dataset : {output_path.as_posix()}"
        )
        logger.info(
            f"  Feature summary  : {(fe_report_dir / 'feature_summary.csv').as_posix()}"
        )
        logger.info(f"  Elapsed time     : {duration:.2f} seconds.")
        logger.info("-----------------------------------------")

        sys.exit(0)

    except TemporalFeatureError as exc:
        logger.critical(f"Temporal feature engineering failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.critical(
            f"Unexpected error in temporal feature engineering: {exc}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
