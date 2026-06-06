#!/usr/bin/env python
"""
Exploratory Longitudinal Data Analysis runner script for Phase 6.
Loads the temporally ordered Parkinson's dataset, executes the ExploratoryAnalyzer
pipeline to produce target analyses, correlations, progression timelines,
patient trajectories, biomarker grids, longitudinal statistics, and feature relationships,
and saves the output figures, tables, and reports.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import pandas as pd

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.visualization.eda import ExploratoryAnalyzer, EDAError

logger = logging.getLogger("run_eda")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run exploratory longitudinal data analysis for the Parkinson's prediction project."
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
    Main execution pipeline for Phase 6.
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
        logger.info("PHASE 6: EXPLORATORY LONGITUDINAL DATA ANALYSIS")
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
            
        # 3. Load dataset
        logger.info(f"Loading temporal dataset from: {temporal_dataset_path.as_posix()}")
        df_temporal = pd.read_csv(temporal_dataset_path)
        
        # 4. Resolve output directories
        figures_dir = resolve_path(config["paths"]["figures_dir"])
        tables_dir = resolve_path(config["paths"]["tables_dir"])
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        eda_reports_dir = reports_dir / "eda"
        
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        eda_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Instantiate ExploratoryAnalyzer and run stages
        analyzer = ExploratoryAnalyzer(config)
        
        # Stage A: Target Analysis
        analyzer.analyze_targets(df_temporal, figures_dir)
        
        # Stage B: Correlation Analysis
        top_correlations = analyzer.analyze_correlations(df_temporal, figures_dir, tables_dir)
        
        # Stage C: Disease Progression Analysis
        analyzer.analyze_disease_progression(df_temporal, figures_dir)
        
        # Stage D: Patient Trajectory Analysis
        analyzer.analyze_patient_trajectories(df_temporal, figures_dir)
        
        # Stage E: Biomarker Analysis & Skewness
        skew_dict = analyzer.analyze_biomarkers(df_temporal, figures_dir)
        
        # Stage F: Longitudinal statistics & slopes
        longitudinal_stats = analyzer.compute_longitudinal_statistics(df_temporal, tables_dir)
        
        # Stage G: Feature relationship analysis
        analyzer.analyze_feature_relationships(df_temporal, figures_dir)
        
        # Stage H: Generate Summary Report
        summary_text = analyzer.generate_summary_report(
            skew_dict=skew_dict,
            top_correlations=top_correlations,
            longitudinal_stats=longitudinal_stats,
            report_dir=eda_reports_dir
        )
        
        duration = time.time() - start_time
        logger.info("-----------------------------------------")
        logger.info("SUCCESS: Exploratory Longitudinal Data Analysis completed successfully.")
        logger.info(f"All figures saved to: {figures_dir.as_posix()}")
        logger.info(f"All tables saved to: {tables_dir.as_posix()}")
        logger.info(f"Summary report written to: {(eda_reports_dir / 'eda_summary.txt').as_posix()}")
        logger.info(f"Total time elapsed: {duration:.2f} seconds.")
        logger.info("-----------------------------------------")
        
        sys.exit(0)
            
    except EDAError as e:
        logger.critical(f"Exploratory analysis failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected global error in exploratory analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
