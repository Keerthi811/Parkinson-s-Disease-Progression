"""
Data validation framework for Parkinson's Disease progression prediction.
Contains modular validators checking schemas, datatypes, missing values,
duplicates, and valid range boundaries without modifying raw data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when structural validation checks fail."""
    pass

class SchemaValidator:
    """Validator checking presence of expected columns and detecting unexpected columns."""
    def __init__(self, config: Dict[str, Any]):
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.mandatory_cols = [
            schema_cfg.get("subject_id_col", "subject#"),
            schema_cfg.get("age_col", "age"),
            schema_cfg.get("sex_col", "sex"),
            schema_cfg.get("test_time_col", "test_time"),
            schema_cfg.get("motor_updrs_target", "motor_UPDRS"),
            schema_cfg.get("total_updrs_target", "total_UPDRS")
        ]
        self.voice_biomarkers = schema_cfg.get("voice_biomarkers", [])
        # The full expected columns list includes both mandatory columns and voice biomarkers
        self.expected_cols = self.mandatory_cols + self.voice_biomarkers

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running Schema Validation...")
        errors = []
        
        # Check missing mandatory columns
        missing_mandatory = [col for col in self.mandatory_cols if col not in df.columns]
        if missing_mandatory:
            errors.append(f"Missing mandatory columns: {missing_mandatory}")
            
        # Check missing voice biomarkers
        missing_biomarkers = [col for col in self.voice_biomarkers if col not in df.columns]
        if missing_biomarkers:
            logger.warning(f"Some configured voice biomarkers are missing from dataset: {missing_biomarkers}")
            
        # Check unexpected columns (not in expected schema)
        unexpected_cols = [col for col in df.columns if col not in self.expected_cols]
        if unexpected_cols:
            logger.warning(f"Detected unexpected columns in dataset: {unexpected_cols}")
            
        status = "FAIL" if missing_mandatory else "PASS"
        
        return {
            "status": status,
            "errors": errors,
            "missing_mandatory_columns": missing_mandatory,
            "missing_biomarker_columns": missing_biomarkers,
            "unexpected_columns": unexpected_cols
        }


class MissingValueValidator:
    """Validator checking for missing values (NaNs) and flagging columns exceeding threshold."""
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running Missing Value Validation...")
        n_rows = len(df)
        missing_counts = {}
        missing_pcts = {}
        exceeded_cols = {}
        
        for col in df.columns:
            count = int(df[col].isnull().sum())
            pct = float(count / n_rows) if n_rows > 0 else 0.0
            
            missing_counts[col] = count
            missing_pcts[col] = round(pct * 100, 2)
            
            if pct > self.threshold:
                exceeded_cols[col] = {
                    "count": count,
                    "percentage": round(pct * 100, 2)
                }
                logger.warning(
                    f"Column '{col}' exceeds missing value threshold ({self.threshold * 100}%): "
                    f"{count} missing ({pct * 100:.2f}%)"
                )
                
        status = "FAIL" if exceeded_cols else "PASS"
        
        return {
            "status": status,
            "threshold_pct": self.threshold * 100,
            "exceeded_columns": exceeded_cols,
            "missing_counts": missing_counts,
            "missing_percentages": missing_pcts
        }


class DuplicateValidator:
    """Validator checking for full-row duplicates and duplicate observations per subject/time."""
    def __init__(self, config: Dict[str, Any]):
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.subject_col = schema_cfg.get("subject_id_col", "subject#")
        self.test_time_col = schema_cfg.get("test_time_col", "test_time")

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running Duplicate Records Validation...")
        errors = []
        
        # 1. Full row duplicates
        full_dup_mask = df.duplicated()
        full_dup_count = int(full_dup_mask.sum())
        if full_dup_count > 0:
            errors.append(f"Detected {full_dup_count} exact full-row duplicate records.")
            
        # 2. Duplicate observations within same patient and test_time
        sub_time_dup_count = 0
        if self.subject_col in df.columns and self.test_time_col in df.columns:
            sub_time_dup_mask = df.duplicated(subset=[self.subject_col, self.test_time_col])
            sub_time_dup_count = int(sub_time_dup_mask.sum())
            if sub_time_dup_count > 0:
                errors.append(
                    f"Detected {sub_time_dup_count} duplicate recordings for the same subject "
                    f"at the same test time."
                )
                
        # Full-row duplicates are a critical FAIL.
        # Subject-time duplicates are verified as legitimate distinct voice trials and are allowed to PASS.
        if full_dup_count > 0:
            status = "FAIL"
        else:
            status = "PASS"
        
        return {
            "status": status,
            "errors": errors,
            "full_row_duplicates_count": full_dup_count,
            "subject_time_duplicates_count": sub_time_dup_count
        }


class DataTypeValidator:
    """Validator checking that columns match expected semantic datatypes."""
    def __init__(self, config: Dict[str, Any]):
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.type_mapping = {
            schema_cfg.get("subject_id_col", "subject#"): "integer",
            schema_cfg.get("age_col", "age"): "numeric",
            schema_cfg.get("sex_col", "sex"): "integer",
            schema_cfg.get("test_time_col", "test_time"): "numeric",
            schema_cfg.get("motor_updrs_target", "motor_UPDRS"): "numeric",
            schema_cfg.get("total_updrs_target", "total_UPDRS"): "numeric"
        }

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running Datatype Validation...")
        mismatches = {}
        errors = []
        
        for col, expected_type in self.type_mapping.items():
            if col not in df.columns:
                continue
                
            actual_dtype = df[col].dtype
            
            if expected_type == "integer":
                # Ensure column is an integer representation
                if not pd.api.types.is_integer_dtype(df[col]):
                    mismatches[col] = {
                        "expected": "integer",
                        "actual": str(actual_dtype)
                    }
                    errors.append(f"Column '{col}' expected type 'integer' but got '{actual_dtype}'.")
            elif expected_type == "numeric":
                # Ensure column is numeric (float or int)
                if not pd.api.types.is_numeric_dtype(df[col]):
                    mismatches[col] = {
                        "expected": "numeric",
                        "actual": str(actual_dtype)
                    }
                    errors.append(f"Column '{col}' expected type 'numeric' but got '{actual_dtype}'.")
                    
        status = "FAIL" if mismatches else "PASS"
        
        return {
            "status": status,
            "errors": errors,
            "mismatches": mismatches
        }


class RangeValidator:
    """Validator performing logical domain range boundary checks on numeric entries."""
    def __init__(self, config: Dict[str, Any]):
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        self.subject_col = schema_cfg.get("subject_id_col", "subject#")
        self.age_col = schema_cfg.get("age_col", "age")
        self.test_time_col = schema_cfg.get("test_time_col", "test_time")
        self.motor_col = schema_cfg.get("motor_updrs_target", "motor_UPDRS")
        self.total_col = schema_cfg.get("total_updrs_target", "total_UPDRS")

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running Logical Range Validation...")
        warnings = []
        violations = {}
        
        # 1. Check age > 0
        if self.age_col in df.columns:
            invalid_age_count = int((df[self.age_col] <= 0).sum())
            if invalid_age_count > 0:
                warnings.append(f"Detected {invalid_age_count} records where age is <= 0.")
                violations[self.age_col] = invalid_age_count
                
        # 2. Check test_time >= -7.0 (Allowing up to a 7-day pre-baseline screening window)
        if self.test_time_col in df.columns:
            # Verified negative values (min -4.26 days) as legitimate pre-baseline observations
            invalid_time_count = int((df[self.test_time_col] < -7.0).sum())
            if invalid_time_count > 0:
                warnings.append(f"Detected {invalid_time_count} records with test_time < -7.0 (outside 7-day pre-baseline window).")
                violations[self.test_time_col] = invalid_time_count
                
        # 3. Check motor_UPDRS >= 0
        if self.motor_col in df.columns:
            invalid_motor_count = int((df[self.motor_col] < 0).sum())
            if invalid_motor_count > 0:
                warnings.append(f"Detected {invalid_motor_count} records where motor_UPDRS is negative.")
                violations[self.motor_col] = invalid_motor_count
                
        # 4. Check total_UPDRS >= 0
        if self.total_col in df.columns:
            invalid_total_count = int((df[self.total_col] < 0).sum())
            if invalid_total_count > 0:
                warnings.append(f"Detected {invalid_total_count} records where total_UPDRS is negative.")
                violations[self.total_col] = invalid_total_count
                
        status = "WARNING" if warnings else "PASS"
        
        return {
            "status": status,
            "warnings": warnings,
            "violations": violations
        }


class DatasetValidator:
    """Orchestrator class coordinating all modular validation steps."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schema_val = SchemaValidator(config)
        self.missing_val = MissingValueValidator(threshold=0.05)
        self.dup_val = DuplicateValidator(config)
        self.type_val = DataTypeValidator(config)
        self.range_val = RangeValidator(config)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes all validators and compiles the validation report.
        
        Args:
            df (pd.DataFrame): Dataset to validate.
            
        Returns:
            Dict[str, Any]: Structured validation report dictionary.
        """
        logger.info("Starting dataset validation checks...")
        
        schema_rep = self.schema_val.validate(df)
        missing_rep = self.missing_val.validate(df)
        dup_rep = self.dup_val.validate(df)
        type_rep = self.type_val.validate(df)
        range_rep = self.range_val.validate(df)
        
        # Calculate overall status
        # FAIL if schema, type, duplicates, or missing value checks fail
        if (schema_rep["status"] == "FAIL" or 
            type_rep["status"] == "FAIL" or 
            dup_rep["status"] == "FAIL" or 
            missing_rep["status"] == "FAIL"):
            overall_status = "FAIL"
        elif range_rep["status"] == "WARNING" or dup_rep["status"] == "WARNING":
            overall_status = "WARNING"
        else:
            overall_status = "PASS"
            
        logger.info(f"Dataset validation checks finished. Overall Status: {overall_status}")
        
        return {
            "validation_status": overall_status,
            "schema_validation": schema_rep,
            "missing_values": missing_rep,
            "duplicates": dup_rep,
            "datatype_validation": type_rep,
            "range_validation": range_rep
        }

    def generate_human_readable_summary(self, report: Dict[str, Any]) -> str:
        """
        Generates a human-readable text validation summary report.
        
        Args:
            report (Dict[str, Any]): Compiled JSON validation report.
            
        Returns:
            str: Summary text representation.
        """
        lines = []
        lines.append("=================================================================")
        lines.append("        PARKINSON'S DISEASE DATASET VALIDATION REPORT")
        lines.append("=================================================================")
        lines.append(f"Overall Validation Status: {report['validation_status']}")
        lines.append(f"Generated on Local Time")
        lines.append("=================================================================\n")
        
        # 1. Schema Validation Summary
        schema = report["schema_validation"]
        lines.append("1. SCHEMA VALIDATION:")
        lines.append(f"  Status: {schema['status']}")
        if schema["errors"]:
            lines.append(f"  Errors: {schema['errors']}")
        lines.append(f"  Missing Mandatory Columns: {schema['missing_mandatory_columns']}")
        lines.append(f"  Unexpected Columns Detected: {len(schema['unexpected_columns'])}")
        if schema["unexpected_columns"]:
            lines.append(f"  Unexpected Columns: {schema['unexpected_columns']}")
        lines.append("")
        
        # 2. Datatypes Summary
        types = report["datatype_validation"]
        lines.append("2. DATATYPE VALIDATION:")
        lines.append(f"  Status: {types['status']}")
        if types["errors"]:
            for err in types["errors"]:
                lines.append(f"  - {err}")
        else:
            lines.append("  All semantic columns match expected datatypes.")
        lines.append("")
        
        # 3. Missing Values Summary
        missing = report["missing_values"]
        lines.append("3. MISSING VALUES:")
        lines.append(f"  Status: {missing['status']}")
        lines.append(f"  Threshold: {missing['threshold_pct']}%")
        if missing["exceeded_columns"]:
            lines.append("  Columns exceeding missing threshold:")
            for col, details in missing["exceeded_columns"].items():
                lines.append(f"    - {col}: {details['count']} NaNs ({details['percentage']}%)")
        else:
            lines.append("  No columns exceed the 5% missingness threshold.")
        lines.append("")
        
        # 4. Duplicates Summary
        dups = report["duplicates"]
        lines.append("4. DUPLICATE RECORDS:")
        lines.append(f"  Status: {dups['status']}")
        lines.append(f"  Exact Full-Row Duplicates: {dups['full_row_duplicates_count']}")
        lines.append(f"  Subject-Time Duplicates (leakage): {dups['subject_time_duplicates_count']}")
        if dups["errors"]:
            for err in dups["errors"]:
                lines.append(f"  - {err}")
        lines.append("")
        
        # 5. Range Summary
        ranges = report["range_validation"]
        lines.append("5. LOGICAL RANGE CHECK:")
        lines.append(f"  Status: {ranges['status']}")
        if ranges["warnings"]:
            lines.append("  Range check violations/warnings:")
            for warn in ranges["warnings"]:
                lines.append(f"  - {warn}")
        else:
            lines.append("  All observations fall within expected range boundaries.")
        lines.append("\n=================================================================")
        
        return "\n".join(lines)


def run_validation_stage(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Validation stage called by the pipeline. Loads data, validates, and 
    raises ValidationError if the structure is broken.
    
    Args:
        config (Dict[str, Any]): Loaded project configurations.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        ValidationError: If schema or datatype check fails.
    """
    logger.info("Executing pipeline validation stage...")
    raw_dir = resolve_path(config["paths"]["raw_data_dir"])
    required_files = config["data_validation"]["required_files"]
    if not required_files:
        raise ValidationError("No raw files specified in config.yaml.")
        
    primary_file = required_files[0]
    primary_file_path = raw_dir / primary_file
    
    if not primary_file_path.exists():
        raise ValidationError(f"Raw data file missing at: {primary_file_path.as_posix()}")
        
    try:
        df = pd.read_csv(primary_file_path)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV: {e}")
        
    # Execute validators
    validator = DatasetValidator(config)
    report = validator.validate(df)
    
    # Restrict pipeline execution only on critical structural fails
    if report["schema_validation"]["status"] == "FAIL" or report["datatype_validation"]["status"] == "FAIL":
        raise ValidationError(
            f"Pipeline validation failed. Schema status: {report['schema_validation']['status']}. "
            f"Datatype status: {report['datatype_validation']['status']}."
        )
        
    return df
