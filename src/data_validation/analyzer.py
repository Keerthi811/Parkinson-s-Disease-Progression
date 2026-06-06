"""
Dataset analysis and profiling module for Parkinson's Disease progression.
Generates schema analyses, statistical metrics, automated data dictionaries,
longitudinal summaries, and exploratory data visualizations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class AnalyzerError(Exception):
    """Custom exception raised when dataset analysis fails."""
    pass

def analyze_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the schema of the DataFrame: row/col count, names, data types, 
    missing value counts, and unique value counts.
    
    Args:
        df (pd.DataFrame): Input dataset.
        
    Returns:
        pd.DataFrame: Table detailing schema analysis.
    """
    logger.info("Analyzing dataset schema...")
    try:
        schema_info = []
        n_rows = len(df)
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_count = int(df[col].isnull().sum())
            unique_count = int(df[col].nunique())
            missing_pct = float((missing_count / n_rows) * 100)
            
            schema_info.append({
                "Column": col,
                "Data Type": dtype,
                "Non-Null Count": n_rows - missing_count,
                "Missing Count": missing_count,
                "Missing Pct (%)": round(missing_pct, 2),
                "Unique Value Count": unique_count
            })
            
        return pd.DataFrame(schema_info)
    except Exception as e:
        raise AnalyzerError(f"Schema analysis failed: {e}")

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics for all numeric columns in the DataFrame:
    Mean, Median, Standard Deviation, Min, Max, Skewness, and Kurtosis.
    
    Args:
        df (pd.DataFrame): Input dataset.
        
    Returns:
        pd.DataFrame: Statistical table.
    """
    logger.info("Computing descriptive statistics (including skewness & kurtosis)...")
    try:
        stats_info = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean_val = float(df[col].mean())
            median_val = float(df[col].median())
            std_val = float(df[col].std())
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # Compute skewness and kurtosis
            skew_val = float(df[col].skew())
            kurt_val = float(df[col].kurt())
            
            stats_info.append({
                "Column": col,
                "Mean": mean_val,
                "Median": median_val,
                "Std Dev": std_val,
                "Min": min_val,
                "Max": max_val,
                "Skewness": skew_val,
                "Kurtosis": kurt_val
            })
            
        return pd.DataFrame(stats_info)
    except Exception as e:
        raise AnalyzerError(f"Statistics computation failed: {e}")

def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically builds a data dictionary for the dataset.
    Lists column names, datatypes, unique value counts, and typical example values.
    
    Args:
        df (pd.DataFrame): Input dataset.
        
    Returns:
        pd.DataFrame: Data dictionary table.
    """
    logger.info("Building automatic data dictionary...")
    try:
        dict_info = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = int(df[col].nunique())
            
            # Get up to 3 example values, drop NaNs, and format as string representation
            examples_list = df[col].dropna().unique()[:3].tolist()
            examples_str = ", ".join([str(val) for val in examples_list])
            
            dict_info.append({
                "Column Name": col,
                "Data Type": dtype,
                "Number of Unique Values": unique_count,
                "Example Values": examples_str
            })
            
        return pd.DataFrame(dict_info)
    except Exception as e:
        raise AnalyzerError(f"Data dictionary generation failed: {e}")

def generate_summary(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates a dataset summary report containing high-level longitudinal stats.
    
    Args:
        df (pd.DataFrame): Input dataset.
        config (Dict[str, Any]): Project configuration.
        
    Returns:
        pd.DataFrame: Summary report table.
    """
    logger.info("Compiling dataset longitudinal summary...")
    try:
        schema_cfg = config.get("data_validation", {}).get("schema", {})
        subject_col = schema_cfg.get("subject_id_col", "subject#")
        test_time_col = schema_cfg.get("test_time_col", "test_time")
        
        if subject_col not in df.columns or test_time_col not in df.columns:
            raise AnalyzerError(f"Missing required columns '{subject_col}' or '{test_time_col}' for summary.")
            
        n_recordings = len(df)
        n_patients = int(df[subject_col].nunique())
        avg_recordings = float(n_recordings / n_patients)
        
        earliest_time = float(df[test_time_col].min())
        latest_time = float(df[test_time_col].max())
        
        summary_data = {
            "Metric": [
                "Number of Patients",
                "Number of Recordings",
                "Average Recordings per Patient",
                "Earliest test_time",
                "Latest test_time"
            ],
            "Value": [
                float(n_patients),
                float(n_recordings),
                round(avg_recordings, 2),
                round(earliest_time, 4),
                round(latest_time, 4)
            ]
        }
        
        return pd.DataFrame(summary_data)
    except Exception as e:
        raise AnalyzerError(f"Dataset summary generation failed: {e}")

def generate_eda_plots(df: pd.DataFrame, config: Dict[str, Any], output_dir: Path) -> None:
    """
    Generates exploratory data visualizations:
    - Distribution of motor_UPDRS
    - Distribution of total_UPDRS
    - Histogram of patient observations
    - Correlation heatmap of voice biomarkers
    
    Args:
        df (pd.DataFrame): Input dataset.
        config (Dict[str, Any]): Project configuration.
        output_dir (Path): Location to save plot figures.
    """
    logger.info(f"Generating EDA plots inside: {output_dir.as_posix()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schema_cfg = config.get("data_validation", {}).get("schema", {})
    subject_col = schema_cfg.get("subject_id_col", "subject#")
    motor_col = schema_cfg.get("motor_updrs_target", "motor_UPDRS")
    total_col = schema_cfg.get("total_updrs_target", "total_UPDRS")
    biomarkers = schema_cfg.get("voice_biomarkers", [])
    
    try:
        # Style configurations
        sns.set_theme(style="whitegrid")
        
        # 1. Distribution of motor_UPDRS
        if motor_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df, x=motor_col, kde=True, color="dodgerblue", bins=30, ax=ax)
            ax.set_title("Distribution of Motor UPDRS Score", fontsize=14)
            ax.set_xlabel("Motor UPDRS")
            ax.set_ylabel("Frequency")
            fig_path = output_dir / "motor_updrs_dist.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {fig_path.name}")
            
        # 2. Distribution of total_UPDRS
        if total_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df, x=total_col, kde=True, color="crimson", bins=30, ax=ax)
            ax.set_title("Distribution of Total UPDRS Score", fontsize=14)
            ax.set_xlabel("Total UPDRS")
            ax.set_ylabel("Frequency")
            fig_path = output_dir / "total_updrs_dist.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {fig_path.name}")
            
        # 3. Histogram of patient observations (recordings per patient)
        if subject_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            # Calculate number of observations per patient
            obs_counts = df[subject_col].value_counts()
            sns.histplot(obs_counts, color="forestgreen", discrete=True, ax=ax)
            ax.set_title("Observations per Patient (Recordings Distribution)", fontsize=14)
            ax.set_xlabel("Number of Recordings")
            ax.set_ylabel("Number of Patients")
            fig_path = output_dir / "observations_per_patient.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {fig_path.name}")
            
        # 4. Correlation heatmap
        available_biomarkers = [b for b in biomarkers if b in df.columns]
        if available_biomarkers:
            fig, ax = plt.subplots(figsize=(14, 12))
            corr_matrix = df[available_biomarkers].corr()
            
            # Mask upper triangle for cleaner look
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(
                corr_matrix, 
                mask=mask, 
                cmap="coolwarm", 
                vmax=1.0, 
                vmin=-1.0, 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .7},
                annot=False, # Large matrix, disable cell text to avoid overlap
                ax=ax
            )
            ax.set_title("Biomarkers Correlation Matrix Heatmap", fontsize=16)
            fig_path = output_dir / "biomarkers_correlation.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {fig_path.name}")
            
    except Exception as e:
        raise AnalyzerError(f"Exploratory plotting failed: {e}")
