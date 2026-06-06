"""
Longitudinal and temporal preprocessing module for Parkinson's Disease voice biomarkers.
Provides modular components to group subjects, chronologically sort visits, validate 
temporal consistency, generate premium plots, and output summary reports.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class TemporalError(Exception):
    """Custom exception raised when longitudinal or temporal processing steps fail."""
    pass

class TemporalPreprocessor:
    """
    Orchestrates longitudinal dataset preparation, including grouping, sorting, 
    validation, statistics gathering, and premium plot generation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the preprocessor with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Full pipeline configuration.
        """
        self.config = config
        self.schema_config = config.get("data_validation", {}).get("schema", {})
        
        # Extracted config variables
        self.subject_col = self.schema_config.get("subject_id_col", "subject#")
        self.test_time_col = self.schema_config.get("test_time_col", "test_time")
        self.motor_updrs_col = self.schema_config.get("motor_updrs_target", "motor_UPDRS")
        self.total_updrs_col = self.schema_config.get("total_updrs_target", "total_UPDRS")

    def group_patients(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Groups observations by subject ID and computes summary statistics.
        
        Args:
            df (pd.DataFrame): Cleaned input DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Grouped patient statistics DataFrame 
                                                 and dictionary of global summary stats.
        """
        logger.info("Grouping observations by patient subject ID...")
        try:
            # Group by subject and count observations
            grouped = df.groupby(self.subject_col)
            patient_stats = grouped.size().reset_index(name="recording_count")
            
            # Compute global statistics
            total_patients = int(patient_stats[self.subject_col].nunique())
            total_observations = int(len(df))
            avg_visits = float(patient_stats["recording_count"].mean())
            min_visits = int(patient_stats["recording_count"].min())
            max_visits = int(patient_stats["recording_count"].max())
            
            summary_stats = {
                "total_patients": total_patients,
                "total_observations": total_observations,
                "avg_visits_per_patient": avg_visits,
                "min_visits": min_visits,
                "max_visits": max_visits
            }
            
            logger.info(
                f"Grouping completed: {total_patients} unique patients, "
                f"avg {avg_visits:.2f} recordings/patient (min={min_visits}, max={max_visits})."
            )
            return patient_stats, summary_stats
        except Exception as e:
            raise TemporalError(f"Failed to group patient observations: {e}")

    def sort_and_validate_chronology(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Sorts dataset chronologically by subject ID and test_time.
        Validates that no ordering violations exist (i.e. test_time is non-decreasing for each patient).
        
        Args:
            df (pd.DataFrame): Dataframe to sort and validate.
            
        Returns:
            Tuple[pd.DataFrame, str]: Sorted DataFrame and verification status string ("PASS" or "FAIL").
        """
        logger.info(f"Sorting observations by {self.subject_col} and {self.test_time_col}...")
        try:
            # Sort observations chronologically (stable sort to maintain raw record input sequence where test_time matches)
            df_sorted = df.sort_values(by=[self.subject_col, self.test_time_col], kind="stable").copy()
            
            # Validate chronological consistency per patient
            ordering_violations = 0
            
            for subject_id, group in df_sorted.groupby(self.subject_col):
                # Check if test_time is monotonically non-decreasing
                times = group[self.test_time_col].values
                diffs = np.diff(times)
                violations = np.sum(diffs < 0)
                if violations > 0:
                    ordering_violations += violations
                    logger.error(
                        f"Chronological ordering violation: Subject {subject_id} has "
                        f"{violations} visits with decreasing test_time."
                    )
            
            if ordering_violations > 0:
                logger.warning(f"Validation failed: {ordering_violations} chronological ordering violations detected.")
                status = "FAIL"
                raise TemporalError(f"Dataset has {ordering_violations} chronological ordering violations after sorting.")
            else:
                logger.info("Chronological consistency check: PASS. All subject visits are sorted properly.")
                status = "PASS"
                
            return df_sorted, status
        except Exception as e:
            raise TemporalError(f"Chronological sorting/validation failed: {e}")

    def generate_longitudinal_plots(self, df: pd.DataFrame, figures_dir: Path) -> None:
        """
        Generates exploratory longitudinal and temporal figures.
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            figures_dir (Path): Directory where figures should be saved.
        """
        logger.info("Generating exploratory longitudinal visualizations...")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure premium plotting styles (aesthetic requirements)
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update({
            "font.sans-serif": ["Arial", "Inter", "DejaVu Sans"],
            "figure.titlesize": 20,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12
        })
        
        # Define high-contrast aesthetic colors
        primary_color = "#4f46e5"  # Modern Indigo
        secondary_color = "#0ea5e9"  # Vibrant Sky Blue
        accent_color = "#f43f5e"  # Deep Rose
        
        try:
            # 1. Patient observation count distribution
            logger.info("Creating patient observation count distribution plot...")
            patient_stats = df.groupby(self.subject_col).size().reset_index(name="recording_count")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=patient_stats,
                x="recording_count",
                kde=True,
                color=primary_color,
                edgecolor="white",
                bins=15,
                alpha=0.85,
                ax=ax
            )
            ax.set_title("Distribution of Recording/Observation Counts per Patient", pad=15, weight="bold")
            ax.set_xlabel("Number of Recordings", labelpad=10)
            ax.set_ylabel("Count of Patients", labelpad=10)
            plt.tight_layout()
            plt.savefig(figures_dir / "patient_observation_count_distribution.png", dpi=300)
            plt.close(fig)
            
            # 2. Patient progression timeline examples (UPDRS progression)
            logger.info("Creating patient progression timeline examples plot...")
            # Pick a subset of 4 patients across the range
            available_subjects = sorted(df[self.subject_col].unique())
            step = max(1, len(available_subjects) // 4)
            sample_subjects = [available_subjects[i * step] for i in range(4)]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False, sharey=True)
            axes_flat = axes.flatten()
            
            for idx, subject in enumerate(sample_subjects):
                ax = axes_flat[idx]
                patient_data = df[df[self.subject_col] == subject].sort_values(by=self.test_time_col)
                
                # Plot total_UPDRS and motor_UPDRS progression over test_time
                ax.plot(
                    patient_data[self.test_time_col],
                    patient_data[self.total_updrs_col],
                    marker="o",
                    linestyle="-",
                    linewidth=2.5,
                    color=primary_color,
                    label="Total UPDRS",
                    markersize=6
                )
                ax.plot(
                    patient_data[self.test_time_col],
                    patient_data[self.motor_updrs_col],
                    marker="s",
                    linestyle="--",
                    linewidth=2,
                    color=accent_color,
                    label="Motor UPDRS",
                    markersize=6
                )
                
                ax.set_title(f"Patient Trajectory: Subject {subject}", weight="bold")
                ax.set_xlabel("Days from Baseline (test_time)")
                if idx in [0, 2]:
                    ax.set_ylabel("UPDRS Score")
                ax.legend(loc="upper left")
                ax.grid(True, linestyle=":", alpha=0.6)
                
            plt.suptitle("Longitudinal UPDRS Progression Examples", weight="bold", y=0.98)
            plt.tight_layout()
            plt.savefig(figures_dir / "patient_progression_timeline_examples.png", dpi=300)
            plt.close(fig)
            
            # 3. test_time distribution plot
            logger.info("Creating test_time distribution plot...")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=df,
                x=self.test_time_col,
                kde=True,
                color=secondary_color,
                edgecolor="white",
                alpha=0.8,
                ax=ax
            )
            ax.set_title("Overall Distribution of Follow-Up test_time (Days)", pad=15, weight="bold")
            ax.set_xlabel("test_time (Days)", labelpad=10)
            ax.set_ylabel("Observation Frequency", labelpad=10)
            plt.tight_layout()
            plt.savefig(figures_dir / "test_time_distribution.png", dpi=300)
            plt.close(fig)
            
            logger.info("All longitudinal visualizations saved successfully.")
        except Exception as e:
            raise TemporalError(f"Failed to generate longitudinal visualizations: {e}")
            
