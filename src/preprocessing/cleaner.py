"""
Data cleaning module for Parkinson's Disease progression prediction.
Contains modular components for missing value imputation, duplicate removal,
datatype standardization, and data integrity checks.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CleanerError(Exception):
    """Custom exception raised when data cleaning operations fail."""
    pass

class MissingValueHandler:
    """Handles missing values in numerical features using median imputation below thresholds."""
    def __init__(self, threshold_pct: float = 5.0):
        self.threshold = threshold_pct / 100.0

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("Initializing Missing Value Handler...")
        df_clean = df.copy()
        n_rows = len(df_clean)
        imputation_stats = {}
        total_imputed_count = 0
        
        try:
            for col in df_clean.columns:
                # Only check numeric columns for median imputation
                if not pd.api.types.is_numeric_dtype(df_clean[col]):
                    continue
                    
                missing_count = int(df_clean[col].isnull().sum())
                if missing_count == 0:
                    continue
                    
                missing_pct = missing_count / n_rows
                
                if missing_pct < self.threshold:
                    median_val = float(df_clean[col].median())
                    # In case the entire column is NaN, median is NaN, we fallback to 0.0
                    if pd.isnull(median_val):
                        median_val = 0.0
                        
                    df_clean[col] = df_clean[col].fillna(median_val)
                    total_imputed_count += missing_count
                    imputation_stats[col] = {
                        "missing_count": missing_count,
                        "imputation_strategy": "median",
                        "imputed_value": median_val
                    }
                    logger.info(
                        f"Imputed {missing_count} missing values in '{col}' "
                        f"using median value: {median_val:.4f}."
                    )
                else:
                    logger.warning(
                        f"Column '{col}' exceeds missingness threshold ({self.threshold * 100}%): "
                        f"{missing_count} missing ({missing_pct * 100:.2f}%). Imputation skipped!"
                    )
                    imputation_stats[col] = {
                        "missing_count": missing_count,
                        "imputation_strategy": "skipped_exceeded_threshold",
                        "imputed_value": None
                    }
                    
            return df_clean, {
                "total_imputed_values": total_imputed_count,
                "imputed_columns": imputation_stats
            }
        except Exception as e:
            raise CleanerError(f"Missing value imputation failed: {e}")


class DuplicateHandler:
    """Handles duplicate records, removing full-row duplicates while retaining visit trials."""
    def __init__(self, config: Dict[str, Any]):
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.subject_col = schema_cfg.get("subject_id_col", "subject#")
        self.test_time_col = schema_cfg.get("test_time_col", "test_time")

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("Initializing Duplicate Handler...")
        try:
            n_rows_before = len(df)
            
            # 1. Remove exact full-row duplicates only
            df_clean = df.drop_duplicates(keep="first")
            n_rows_after = len(df_clean)
            removed_count = n_rows_before - n_rows_after
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} exact full-row duplicate records.")
            else:
                logger.info("No exact full-row duplicates detected.")
                
            # 2. Log details about subject-time duplicates (trials) to confirm they are kept
            trials_count = 0
            if self.subject_col in df_clean.columns and self.test_time_col in df_clean.columns:
                trials_mask = df_clean.duplicated(subset=[self.subject_col, self.test_time_col])
                trials_count = int(trials_mask.sum())
                logger.info(
                    f"Retained {trials_count} distinct subject-time clinical phonation trials "
                    f"for progression modeling."
                )
                
            return df_clean, {
                "exact_duplicates_removed": removed_count,
                "clinical_trials_retained": trials_count
            }
        except Exception as e:
            raise CleanerError(f"Duplicate removal failed: {e}")


class DataTypeStandardizer:
    """Standardizes data columns to expected integer and float datatypes."""
    def __init__(self, config: Dict[str, Any]):
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.subject_col = schema_cfg.get("subject_id_col", "subject#")
        self.sex_col = schema_cfg.get("sex_col", "sex")

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("Initializing DataType Standardizer...")
        df_clean = df.copy()
        corrections = {}
        total_corrections = 0
        
        try:
            # 1. Standardize integer columns (subject# and sex)
            int_cols = [self.subject_col, self.sex_col]
            for col in int_cols:
                if col not in df_clean.columns:
                    continue
                # If not integer, convert
                if not pd.api.types.is_integer_dtype(df_clean[col]):
                    original_type = str(df_clean[col].dtype)
                    # Use rounding then convert to int to handle float float-to-int cases safely
                    df_clean[col] = df_clean[col].round().astype(np.int64)
                    corrections[col] = {
                        "from": original_type,
                        "to": "integer"
                    }
                    total_corrections += 1
                    logger.info(f"Standardized column '{col}' from '{original_type}' to 'integer'.")
                    
            # 2. Standardize all other numeric columns to float
            for col in df_clean.columns:
                if col in int_cols:
                    continue
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if not pd.api.types.is_float_dtype(df_clean[col]):
                        original_type = str(df_clean[col].dtype)
                        df_clean[col] = df_clean[col].astype(np.float64)
                        corrections[col] = {
                            "from": original_type,
                            "to": "float"
                        }
                        total_corrections += 1
                        logger.info(f"Standardized column '{col}' from '{original_type}' to 'float'.")
                        
            return df_clean, {
                "total_datatype_corrections": total_corrections,
                "corrections": corrections
            }
        except Exception as e:
            raise CleanerError(f"Datatype standardization failed: {e}")


class DataCleaner:
    """Orchestrator class coordinating all modular cleaning components."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.missing_val_handler = MissingValueHandler(threshold_pct=5.0)
        self.duplicate_handler = DuplicateHandler(config)
        self.type_standardizer = DataTypeStandardizer(config)
        
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.mandatory_cols = [
            schema_cfg.get("subject_id_col", "subject#"),
            schema_cfg.get("age_col", "age"),
            schema_cfg.get("sex_col", "sex"),
            schema_cfg.get("test_time_col", "test_time"),
            schema_cfg.get("motor_updrs_target", "motor_UPDRS"),
            schema_cfg.get("total_updrs_target", "total_UPDRS")
        ]

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes all cleaning handlers and verifies data integrity.
        
        Args:
            df (pd.DataFrame): Raw DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and structured cleaning report.
        """
        logger.info("Executing end-to-end data cleaning pipeline...")
        n_rows_before = len(df)
        n_cols_before = len(df.columns)
        
        # 1. Duplicate records cleaning
        df_dedup, dup_stats = self.duplicate_handler.clean(df)
        
        # 2. Missing values cleaning
        df_imputed, missing_stats = self.missing_val_handler.clean(df_dedup)
        
        # 3. Datatype standardization
        df_standard, type_stats = self.type_standardizer.clean(df_imputed)
        
        n_rows_after = len(df_standard)
        n_cols_after = len(df_standard.columns)
        
        # 4. Integrity Checks
        logger.info("Running post-cleaning data integrity checks...")
        
        # Check column count is preserved
        if n_cols_before != n_cols_after:
            msg = f"Data integrity error: Column count changed from {n_cols_before} to {n_cols_after} during cleaning."
            logger.error(msg)
            raise CleanerError(msg)
            
        # Verify no mandatory columns were lost
        missing_mandatory = [col for col in self.mandatory_cols if col not in df_standard.columns]
        if missing_mandatory:
            msg = f"Data integrity error: Mandatory columns were lost during cleaning: {missing_mandatory}"
            logger.error(msg)
            raise CleanerError(msg)
            
        logger.info("Integrity checks passed successfully. Clean dataset matches expected specifications.")
        
        # Compile cleaning report
        report = {
            "rows_before": n_rows_before,
            "rows_after": n_rows_after,
            "duplicates_removed": dup_stats["exact_duplicates_removed"],
            "clinical_trials_retained": dup_stats["clinical_trials_retained"],
            "missing_values_imputed": missing_stats["total_imputed_values"],
            "datatype_corrections": type_stats["total_datatype_corrections"],
            "imputation_details": missing_stats["imputed_columns"],
            "corrections_details": type_stats["corrections"],
            "status": "SUCCESS"
        }
        
        return df_standard, report

    def generate_human_readable_summary(self, report: Dict[str, Any]) -> str:
        """
        Generates a human-readable text summary of the cleaning operations.
        
        Args:
            report (Dict[str, Any]): Compiled cleaning report dict.
            
        Returns:
            str: Formatting text summary.
        """
        lines = []
        lines.append("=================================================================")
        lines.append("        PARKINSON'S DISEASE DATASET CLEANING SUMMARY")
        lines.append("=================================================================")
        lines.append(f"Execution Status: {report['status']}")
        lines.append(f"Processed on Local Time")
        lines.append("=================================================================\n")
        
        lines.append("1. ROW AND VOLUME STATISTICS:")
        lines.append(f"  Rows Before Cleaning: {report['rows_before']}")
        lines.append(f"  Rows After Cleaning:  {report['rows_after']}")
        lines.append(f"  Total Rows Removed:   {report['rows_before'] - report['rows_after']}")
        lines.append("")
        
        lines.append("2. DEDUPLICATION HISTORY:")
        lines.append(f"  Exact Full-Row Duplicates Removed: {report['duplicates_removed']}")
        lines.append(f"  Clinical Phonation Trials Kept:    {report['clinical_trials_retained']}")
        lines.append("")
        
        lines.append("3. MISSINGNESS IMPUTATION:")
        lines.append(f"  Total Missing Values Imputed: {report['missing_values_imputed']}")
        imp_details = report["imputation_details"]
        if imp_details:
            lines.append("  Imputed Columns:")
            for col, details in imp_details.items():
                lines.append(
                    f"    - {col}: {details['missing_count']} values filled "
                    f"with {details['imputation_strategy']} ({details['imputed_value']:.4f})"
                )
        else:
            lines.append("  No columns required missing value imputation.")
        lines.append("")
        
        lines.append("4. DATATYPE CORRECTIONS:")
        lines.append(f"  Total Columns Standardized: {report['datatype_corrections']}")
        type_details = report["corrections_details"]
        if type_details:
            lines.append("  Standardized Columns:")
            for col, details in type_details.items():
                lines.append(f"    - {col}: converted '{details['from']}' to '{details['to']}'")
        else:
            lines.append("  All columns matched expected datatypes (no conversions needed).")
            
        lines.append("\n=================================================================")
        
        return "\n".join(lines)
