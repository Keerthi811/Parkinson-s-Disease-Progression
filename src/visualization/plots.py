"""
Visualization module for Parkinson's disease progression analysis.
Provides plotting scripts for longitudinal patient trajectories, 
model regression evaluations, and interpretability graphs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import matplotlib
# Use a non-interactive backend (Agg) to prevent GUI popup loops during command executions
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)

class VisualizationError(Exception):
    """Custom exception raised for plotting pipeline errors."""
    pass

def plot_biomarker_trajectories(
    df: pd.DataFrame, 
    biomarkers: List[str], 
    subject_ids: List[int],
    subject_col: str,
    test_time_col: str,
    save_path: Path
) -> None:
    """
    Plots the longitudinal trajectory of specific voice biomarkers for selected subjects.
    
    Args:
        df (pd.DataFrame): Dataframe containing longitudinal patient visits.
        biomarkers (List[str]): Biomarker names to plot.
        subject_ids (List[int]): Sublist of patient subject IDs to focus on.
        subject_col (str): Column mapping subject ID.
        test_time_col (str): Column mapping visit test time.
        save_path (Path): Destination file path for saving the figure.
    """
    logger.info("Plotting biomarker trajectories for selected subjects...")
    try:
        # Filter dataframe for selected subjects
        sub_df = df[df[subject_col].isin(subject_ids)].copy()
        
        if sub_df.empty:
            logger.warning(f"No records found for subjects {subject_ids}. Trajectory plot skipped.")
            return

        fig, axes = plt.subplots(len(biomarkers), 1, figsize=(10, 3 * len(biomarkers)), sharex=True)
        if len(biomarkers) == 1:
            axes = [axes]
            
        for i, col in enumerate(biomarkers):
            ax = axes[i]
            sns.lineplot(
                data=sub_df,
                x=test_time_col,
                y=col,
                hue=subject_col,
                marker="o",
                palette="tab10",
                ax=ax
            )
            ax.set_title(f"Longitudinal Trajectory of {col}")
            ax.set_ylabel("Value (Scaled)")
            ax.legend(title="Subject ID", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.6)
            
        axes[-1].set_xlabel("Time from Baseline (Days)")
        plt.tight_layout()
        
        # Save and close to prevent leaks
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Trajectory plot successfully saved to: {save_path.as_posix()}")
    except Exception as e:
        raise VisualizationError(f"Failed to generate biomarker trajectories plot: {e}")

def plot_regression_performance(
    predictions_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    save_path: Path
) -> None:
    """
    Plots actual vs predicted values for regression evaluation.
    
    Args:
        predictions_df (pd.DataFrame): Dataframe containing predictions.
        actual_col (str): Column name for the ground truth labels.
        pred_col (str): Column name for model predictions.
        save_path (Path): Destination file path for saving the figure.
    """
    logger.info("Plotting model predictions vs. ground truth regression plot...")
    try:
        if predictions_df.empty:
            logger.warning("Predictions DataFrame is empty. Regression plot skipped.")
            return
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw scatter plot
        sns.scatterplot(
            data=predictions_df,
            x=actual_col,
            y=pred_col,
            alpha=0.6,
            color="teal",
            edgecolor="w",
            ax=ax
        )
        
        # Reference diagonal line
        min_val = min(predictions_df[actual_col].min(), predictions_df[pred_col].min())
        max_val = max(predictions_df[actual_col].max(), predictions_df[pred_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
        
        ax.set_title("Model Progression Predictions vs Ground Truth (UPDRS)")
        ax.set_xlabel("Actual UPDRS Score")
        ax.set_ylabel("Predicted UPDRS Score")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Save and close
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Model performance plot successfully saved to: {save_path.as_posix()}")
    except Exception as e:
        raise VisualizationError(f"Failed to generate regression performance plot: {e}")

def plot_importance_bars(
    importance_df: pd.DataFrame, 
    top_n: int, 
    save_path: Path
) -> None:
    """
    Plots feature importances as a bar chart.
    
    Args:
        importance_df (pd.DataFrame): Table containing features and importance values.
        top_n (int): Number of top features to plot.
        save_path (Path): Destination file path.
    """
    logger.info(f"Plotting top {top_n} feature importances bar chart...")
    try:
        if importance_df.empty:
            logger.warning("Importance DataFrame is empty. Importance plot skipped.")
            return
            
        plot_df = importance_df.head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=plot_df,
            x="importance",
            y="feature",
            hue="feature",
            palette="viridis",
            legend=False,
            ax=ax
        )
        
        ax.set_title(f"Top {top_n} Predictive Biomarker & Engineered Features")
        ax.set_xlabel("Relative Gini Importance")
        ax.set_ylabel("Features")
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Feature importance plot successfully saved to: {save_path.as_posix()}")
    except Exception as e:
        raise VisualizationError(f"Failed to generate feature importance plot: {e}")

def run_visualization_stage(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Orchestrates the visualization stage. Loads preprocessed datasets, 
    trained model predictions, and writes figures to reports/figures.
    
    Args:
        df (pd.DataFrame): The preprocessed/feature engineered DataFrame.
        config (Dict[str, Any]): Loaded project configurations.
    """
    logger.info("Starting visualization pipeline...")
    paths_cfg = config["paths"]
    schema_cfg = config["data_validation"]["schema"]
    
    figures_dir = resolve_path(paths_cfg["figures_dir"])
    eval_dir = resolve_path(paths_cfg["evaluation_dir"])
    tables_dir = resolve_path(paths_cfg["tables_dir"])
    
    subject_col = schema_cfg["subject_id_col"]
    test_time_col = schema_cfg["test_time_col"]
    
    # 1. Plot trajectories for first 3 subjects
    biomarkers_to_plot = schema_cfg["voice_biomarkers"][:3] # Plot top 3 biomarkers
    available_subjects = df[subject_col].unique()[:3].tolist()
    
    traj_plot_path = figures_dir / "subject_biomarker_trajectories.png"
    plot_biomarker_trajectories(
        df=df,
        biomarkers=biomarkers_to_plot,
        subject_ids=available_subjects,
        subject_col=subject_col,
        test_time_col=test_time_col,
        save_path=traj_plot_path
    )
    
    # 2. Plot regression performance
    predictions_file = eval_dir / "test_predictions.csv"
    if predictions_file.exists():
        try:
            pred_df = pd.read_csv(predictions_file)
            reg_plot_path = figures_dir / "model_predictions_regression.png"
            plot_regression_performance(
                predictions_df=pred_df,
                actual_col=schema_cfg["total_updrs_target"],
                pred_col="predicted_UPDRS",
                save_path=reg_plot_path
            )
        except Exception as e:
            logger.error(f"Error drawing regression performance plot: {e}")
            
    # 3. Plot feature importances
    importance_file = tables_dir / "feature_importances.csv"
    if importance_file.exists():
        try:
            imp_df = pd.read_csv(importance_file)
            imp_plot_path = figures_dir / "feature_importances.png"
            top_n = config.get("explainability", {}).get("feature_importance_top_n", 15)
            plot_importance_bars(
                importance_df=imp_df,
                top_n=top_n,
                save_path=imp_plot_path
            )
        except Exception as e:
            logger.error(f"Error drawing feature importance plot: {e}")
            
    logger.info("Visualization pipeline completed successfully.")
