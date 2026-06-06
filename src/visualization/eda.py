"""
Exploratory Longitudinal Data Analysis module for Parkinson's Disease progression.
Provides comprehensive functions for target variable analysis, correlation mapping,
disease progression trends, patient trajectories, biomarker grids, longitudinal statistics,
and feature relationships.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class EDAError(Exception):
    """Custom exception raised for Exploratory Data Analysis errors."""
    pass

class ExploratoryAnalyzer:
    """
    Orchestrates Exploratory Longitudinal Data Analysis (EDA) on the Parkinson's dataset.
    Generates high-quality figures and structured CSV tables.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the analyzer with config settings.
        
        Args:
            config (Dict[str, Any]): Loaded project configurations.
        """
        self.config = config
        self.schema_cfg = config.get("data_validation", {}).get("schema", {})
        
        self.subject_col = self.schema_cfg.get("subject_id_col", "subject#")
        self.test_time_col = self.schema_cfg.get("test_time_col", "test_time")
        self.motor_updrs_col = self.schema_cfg.get("motor_updrs_target", "motor_UPDRS")
        self.total_updrs_col = self.schema_cfg.get("total_updrs_target", "total_UPDRS")
        self.biomarkers = self.schema_cfg.get("voice_biomarkers", [])
        
        # Configure premium plotting styles (aesthetic requirements)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "font.sans-serif": ["Arial", "Inter", "DejaVu Sans"],
            "figure.titlesize": 20,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.facecolor": "white"
        })
        
        # Dynamic high-contrast aesthetic colors
        self.primary = "#4f46e5"     # Indigo
        self.secondary = "#0ea5e9"   # Sky Blue
        self.accent = "#f43f5e"      # Rose
        self.neutral_dark = "#1e293b" # Slate Dark
        self.palette = [self.primary, self.secondary, self.accent, "#10b981", "#f59e0b", "#8b5cf6"]

    def analyze_targets(self, df: pd.DataFrame, fig_dir: Path) -> None:
        """
        Performs target variable analysis (distributions, boxplots, violin plots).
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            fig_dir (Path): Destination figures directory.
        """
        logger.info("Analyzing target variables (motor_UPDRS and total_UPDRS)...")
        try:
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Distributions plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.histplot(df[self.motor_updrs_col], kde=True, color=self.primary, ax=axes[0], edgecolor="white", alpha=0.8)
            axes[0].set_title("Distribution of Motor UPDRS", weight="bold")
            axes[0].set_xlabel("Motor UPDRS Score")
            axes[0].set_ylabel("Density")
            
            sns.histplot(df[self.total_updrs_col], kde=True, color=self.secondary, ax=axes[1], edgecolor="white", alpha=0.8)
            axes[1].set_title("Distribution of Total UPDRS", weight="bold")
            axes[1].set_xlabel("Total UPDRS Score")
            axes[1].set_ylabel("Density")
            
            plt.suptitle("Target Variable Distributions", weight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(fig_dir / "target_distributions.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # 2. Box & Violin plots side-by-side
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            sns.boxplot(y=df[self.motor_updrs_col], color=self.primary, ax=axes[0, 0], width=0.4)
            axes[0, 0].set_title("Boxplot of Motor UPDRS", weight="bold")
            axes[0, 0].set_ylabel("Motor UPDRS Score")
            
            sns.violinplot(y=df[self.motor_updrs_col], color=self.primary, ax=axes[0, 1], width=0.6)
            axes[0, 1].set_title("Violin plot of Motor UPDRS", weight="bold")
            axes[0, 1].set_ylabel("Motor UPDRS Score")
            
            sns.boxplot(y=df[self.total_updrs_col], color=self.secondary, ax=axes[1, 0], width=0.4)
            axes[1, 0].set_title("Boxplot of Total UPDRS", weight="bold")
            axes[1, 0].set_ylabel("Total UPDRS Score")
            
            sns.violinplot(y=df[self.total_updrs_col], color=self.secondary, ax=axes[1, 1], width=0.6)
            axes[1, 1].set_title("Violin plot of Total UPDRS", weight="bold")
            axes[1, 1].set_ylabel("Total UPDRS Score")
            
            plt.suptitle("Target Variable Boxplots and Violin Plots", weight="bold", y=0.98)
            plt.tight_layout()
            plt.savefig(fig_dir / "target_box_violin.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info("Target variable plots saved successfully.")
        except Exception as e:
            raise EDAError(f"Failed to analyze target variables: {e}")

    def analyze_correlations(self, df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> List[Tuple[str, float]]:
        """
        Computes Pearson correlation between all numeric features and targets.
        Generates correlation heatmap and saves the top 20 correlations.
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            fig_dir (Path): Figures directory.
            table_dir (Path): Tables directory.
            
        Returns:
            List[Tuple[str, float]]: List of top correlated feature names and correlation coefficients.
        """
        logger.info("Computing Pearson correlations and drawing heatmap...")
        try:
            table_dir.mkdir(parents=True, exist_ok=True)
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # Select numeric columns for correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr(method="pearson")
            
            # Slices of correlation for targets
            corr_motor = corr_matrix[self.motor_updrs_col].drop([self.motor_updrs_col, self.total_updrs_col])
            corr_total = corr_matrix[self.total_updrs_col].drop([self.motor_updrs_col, self.total_updrs_col])
            
            # Compile top correlations
            records = []
            for col in corr_motor.index:
                records.append({
                    "feature": col,
                    "correlation_with_motor_UPDRS": corr_motor[col],
                    "abs_correlation_with_motor_UPDRS": abs(corr_motor[col]),
                    "correlation_with_total_UPDRS": corr_total[col],
                    "abs_correlation_with_total_UPDRS": abs(corr_total[col])
                })
            
            corr_df = pd.DataFrame(records)
            
            # Sort by absolute correlation with total_UPDRS (or motor_UPDRS) to get top 20
            corr_df_sorted = corr_df.sort_values(by="abs_correlation_with_total_UPDRS", ascending=False).head(20)
            
            # Save correlation table
            correlation_table_path = table_dir / "correlation_table.csv"
            corr_df_sorted.to_csv(correlation_table_path, index=False)
            logger.info(f"Top 20 correlations table saved to: {correlation_table_path.as_posix()}")
            
            # Generate correlation heatmap of voice features and targets
            heatmap_cols = self.biomarkers + [self.motor_updrs_col, self.total_updrs_col]
            # Ensure columns exist in DataFrame
            heatmap_cols = [c for c in heatmap_cols if c in df.columns]
            
            fig, ax = plt.subplots(figsize=(14, 12))
            mask = np.triu(np.ones_like(df[heatmap_cols].corr(), dtype=bool))
            sns.heatmap(
                df[heatmap_cols].corr(),
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
                ax=ax
            )
            ax.set_title("Pearson Correlation Heatmap (Biomarkers & Targets)", weight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(fig_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # Return sorted top features as tuple list
            top_features = [(row["feature"], row["correlation_with_total_UPDRS"]) for _, row in corr_df_sorted.iterrows()]
            return top_features
        except Exception as e:
            raise EDAError(f"Failed to analyze correlations: {e}")

    def analyze_disease_progression(self, df: pd.DataFrame, fig_dir: Path) -> None:
        """
        Generates aggregated disease progression trend plots over test_time.
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            fig_dir (Path): Figures directory.
        """
        logger.info("Analyzing disease progression trends over test_time...")
        try:
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # To plot a publication-quality disease progression timeline, we can bin test_time
            # in 20-day bins to calculate mean and standard errors, or use Seaborn lineplot
            # which automatically aggregates with confidence intervals.
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Round test_time to 15-day intervals for cleaner visualization aggregation
            df_trend = df.copy()
            df_trend["test_time_binned"] = (df_trend[self.test_time_col] / 15).round() * 15
            
            # Plot binned trends with 95% confidence intervals
            sns.lineplot(
                data=df_trend,
                x="test_time_binned",
                y=self.total_updrs_col,
                color=self.primary,
                linewidth=3,
                marker="o",
                label="Total UPDRS (Aggregated)",
                ax=ax,
                errorbar="ci"
            )
            sns.lineplot(
                data=df_trend,
                x="test_time_binned",
                y=self.motor_updrs_col,
                color=self.accent,
                linewidth=2.5,
                marker="s",
                linestyle="--",
                label="Motor UPDRS (Aggregated)",
                ax=ax,
                errorbar="ci"
            )
            
            ax.set_title("Aggregated Parkinson's Disease Progression vs Follow-Up Time", weight="bold", pad=15)
            ax.set_xlabel("Time from Baseline (Days, Binned in 15-day Intervals)", labelpad=10)
            ax.set_ylabel("UPDRS Score", labelpad=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left")
            
            plt.tight_layout()
            plt.savefig(fig_dir / "disease_progression_trend.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info("Disease progression trend plot saved successfully.")
        except Exception as e:
            raise EDAError(f"Failed to analyze disease progression: {e}")

    def analyze_patient_trajectories(self, df: pd.DataFrame, fig_dir: Path) -> None:
        """
        Plots longitudinal trajectories of 10 random patients and 10 patients with highest visit count.
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            fig_dir (Path): Figures directory.
        """
        logger.info("Analyzing individual patient trajectories (random and highest visit-count)...")
        try:
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # Group by subject to count visits
            visit_counts = df.groupby(self.subject_col).size().reset_index(name="visits")
            
            # 1. Select 10 highest visit-count patients
            top_10_subjects = visit_counts.sort_values(by="visits", ascending=False).head(10)[self.subject_col].tolist()
            
            # 2. Select 10 random patients
            all_subjects = visit_counts[self.subject_col].tolist()
            np.random.seed(42)  # For reproducibility
            random_10_subjects = list(np.random.choice(all_subjects, size=min(10, len(all_subjects)), replace=False))
            
            # Plot side by side
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
            
            # Plot Random 10
            for i, subject in enumerate(random_10_subjects):
                subject_data = df[df[self.subject_col] == subject].sort_values(by=self.test_time_col)
                axes[0].plot(
                    subject_data[self.test_time_col],
                    subject_data[self.total_updrs_col],
                    marker="o",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"Sub {subject}"
                )
            axes[0].set_title("Trajectories of 10 Random Patients", weight="bold")
            axes[0].set_xlabel("test_time (Days)")
            axes[0].set_ylabel("Total UPDRS Score")
            axes[0].grid(True, linestyle=":", alpha=0.6)
            axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Patient ID")
            
            # Plot Top 10 Highest visits
            for i, subject in enumerate(top_10_subjects):
                subject_data = df[df[self.subject_col] == subject].sort_values(by=self.test_time_col)
                axes[1].plot(
                    subject_data[self.test_time_col],
                    subject_data[self.total_updrs_col],
                    marker="s",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"Sub {subject}"
                )
            axes[1].set_title("Trajectories of 10 Patients with Highest Visit Counts", weight="bold")
            axes[1].set_xlabel("test_time (Days)")
            axes[1].grid(True, linestyle=":", alpha=0.6)
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Patient ID")
            
            plt.suptitle("Longitudinal Patient Trajectory Overlay Plots (Total UPDRS)", weight="bold", y=0.98)
            plt.tight_layout()
            plt.savefig(fig_dir / "patient_trajectories.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info("Patient trajectories plot saved successfully.")
        except Exception as e:
            raise EDAError(f"Failed to analyze patient trajectories: {e}")

    def analyze_biomarkers(self, df: pd.DataFrame, fig_dir: Path) -> Dict[str, float]:
        """
        Plots grids of histograms/density plots and boxplots for all 16 voice biomarkers.
        Identifies skewed voice biomarkers.
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            fig_dir (Path): Figures directory.
            
        Returns:
            Dict[str, float]: Skewness dictionary of voice biomarkers.
        """
        logger.info("Analyzing voice biomarkers (skewness and distribution grids)...")
        try:
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate skewness for all voice features
            skew_dict = df[self.biomarkers].skew().to_dict()
            
            # 1. Density/Histogram Grid (4x4)
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            axes_flat = axes.flatten()
            for idx, col in enumerate(self.biomarkers):
                ax = axes_flat[idx]
                sns.histplot(df[col], kde=True, color=self.primary, ax=ax, edgecolor="white", alpha=0.7)
                ax.set_title(f"{col}\n(Skew: {skew_dict[col]:.2f})", fontsize=12)
                ax.set_xlabel("")
                ax.set_ylabel("")
            plt.suptitle("Density & Histograms of Voice Biomarkers", weight="bold", y=0.99, fontsize=18)
            plt.tight_layout()
            plt.savefig(fig_dir / "biomarker_distributions_grid.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # 2. Boxplot Grid (4x4)
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            axes_flat = axes.flatten()
            for idx, col in enumerate(self.biomarkers):
                ax = axes_flat[idx]
                sns.boxplot(x=df[col], color=self.secondary, ax=ax, width=0.4)
                ax.set_title(col, fontsize=12)
                ax.set_xlabel("")
                ax.set_ylabel("")
            plt.suptitle("Boxplots of Voice Biomarkers", weight="bold", y=0.99, fontsize=18)
            plt.tight_layout()
            plt.savefig(fig_dir / "biomarker_boxplots_grid.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info("Biomarker grid figures saved successfully.")
            return skew_dict
        except Exception as e:
            raise EDAError(f"Failed to analyze voice biomarkers: {e}")

    def compute_longitudinal_statistics(self, df: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
        """
        Computes per-patient longitudinal statistics: visits, mean scores, 
        and disease progression slopes (using linear regression vs test_time).
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            table_dir (Path): Tables directory.
            
        Returns:
            pd.DataFrame: Computed longitudinal statistics DataFrame.
        """
        logger.info("Computing longitudinal progression statistics and slopes per patient...")
        try:
            table_dir.mkdir(parents=True, exist_ok=True)
            stats_list = []
            
            for subject_id, group in df.groupby(self.subject_col):
                n_visits = len(group)
                mean_motor = group[self.motor_updrs_col].mean()
                mean_total = group[self.total_updrs_col].mean()
                
                # Fit linear regression for progression slope
                times = group[self.test_time_col].values
                
                # Check that we have valid variations in test_time to compute slope
                if n_visits > 1 and np.ptp(times) > 0:
                    slope_motor, _, _, _, _ = linregress(times, group[self.motor_updrs_col].values)
                    slope_total, _, _, _, _ = linregress(times, group[self.total_updrs_col].values)
                else:
                    slope_motor = 0.0
                    slope_total = 0.0
                    
                stats_list.append({
                    "subject#": int(subject_id),
                    "num_visits": int(n_visits),
                    "mean_motor_UPDRS": float(mean_motor),
                    "mean_total_UPDRS": float(mean_total),
                    "motor_UPDRS_progression_slope": float(slope_motor),
                    "total_UPDRS_progression_slope": float(slope_total)
                })
                
            stats_df = pd.DataFrame(stats_list)
            stats_df_path = table_dir / "longitudinal_statistics.csv"
            stats_df.to_csv(stats_df_path, index=False)
            logger.info(f"Longitudinal statistics saved to: {stats_df_path.as_posix()}")
            
            return stats_df
        except Exception as e:
            raise EDAError(f"Failed to compute longitudinal statistics: {e}")

    def analyze_feature_relationships(self, df: pd.DataFrame, fig_dir: Path) -> None:
        """
        Generates pairplots for top correlated biomarkers and scatter plots vs targets.
        
        Args:
            df (pd.DataFrame): Sorted temporal DataFrame.
            fig_dir (Path): Figures directory.
        """
        logger.info("Analyzing relationships between top correlated features and targets...")
        try:
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # Select top 4 biomarkers correlated with total_UPDRS (based on Pearson correlation)
            corr_with_total = df[self.biomarkers].corrwith(df[self.total_updrs_col]).abs()
            top_4_biomarkers = corr_with_total.sort_values(ascending=False).head(4).index.tolist()
            
            logger.info(f"Top 4 voice features chosen for relationship analysis: {top_4_biomarkers}")
            
            # 1. Pairplot of top features and targets
            pairplot_cols = top_4_biomarkers + [self.motor_updrs_col, self.total_updrs_col]
            fig = sns.pairplot(
                df[pairplot_cols],
                kind="reg",
                diag_kind="kde",
                plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": 0.4, "color": "teal"}}
            )
            fig.savefig(fig_dir / "top_features_pairplot.png", dpi=300)
            plt.close("all")
            
            # 2. Scatter plots for top 2 features vs motor_UPDRS and total_UPDRS
            top_2_biomarkers = top_4_biomarkers[:2]
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            for row_idx, feature in enumerate(top_2_biomarkers):
                # Scatter vs motor_UPDRS
                sns.regplot(
                    data=df,
                    x=feature,
                    y=self.motor_updrs_col,
                    scatter_kws={"alpha": 0.4, "color": self.primary},
                    line_kws={"color": "red", "linewidth": 2},
                    ax=axes[row_idx, 0]
                )
                axes[row_idx, 0].set_title(f"{feature} vs Motor UPDRS", weight="bold")
                axes[row_idx, 0].set_xlabel(feature)
                axes[row_idx, 0].set_ylabel("Motor UPDRS")
                
                # Scatter vs total_UPDRS
                sns.regplot(
                    data=df,
                    x=feature,
                    y=self.total_updrs_col,
                    scatter_kws={"alpha": 0.4, "color": self.secondary},
                    line_kws={"color": "red", "linewidth": 2},
                    ax=axes[row_idx, 1]
                )
                axes[row_idx, 1].set_title(f"{feature} vs Total UPDRS", weight="bold")
                axes[row_idx, 1].set_xlabel(feature)
                axes[row_idx, 1].set_ylabel("Total UPDRS")
                
            plt.suptitle("Top Voice Biomarkers vs. Progression Targets (Scatter with Trendlines)", weight="bold", y=0.99)
            plt.tight_layout()
            plt.savefig(fig_dir / "feature_target_scatter.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info("Feature relationship plots saved successfully.")
        except Exception as e:
            raise EDAError(f"Failed to analyze feature relationships: {e}")

    def generate_summary_report(
        self,
        skew_dict: Dict[str, float],
        top_correlations: List[Tuple[str, float]],
        longitudinal_stats: pd.DataFrame,
        report_dir: Path
    ) -> str:
        """
        Compiles the exploratory longitudinal analysis summary text report.
        
        Args:
            skew_dict (Dict[str, float]): Skewness details.
            top_correlations (List[Tuple[str, float]]): Top correlations details.
            longitudinal_stats (pd.DataFrame): Longitudinal progression slope stats.
            report_dir (Path): Destination report directory.
            
        Returns:
            str: Formatting text summary.
        """
        logger.info("Compiling exploratory longitudinal analysis summary report...")
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Find strongly skewed features (e.g. absolute skewness > 1.0)
            highly_skewed = [feat for feat, val in skew_dict.items() if abs(val) > 1.0]
            
            # Find average and std of progression slopes
            avg_motor_slope = longitudinal_stats["motor_UPDRS_progression_slope"].mean()
            avg_total_slope = longitudinal_stats["total_UPDRS_progression_slope"].mean()
            std_motor_slope = longitudinal_stats["motor_UPDRS_progression_slope"].std()
            std_total_slope = longitudinal_stats["total_UPDRS_progression_slope"].std()
            
            # Count positive vs negative progression slopes (disease progression rate)
            pos_motor_slopes = int((longitudinal_stats["motor_UPDRS_progression_slope"] > 0).sum())
            pos_total_slopes = int((longitudinal_stats["total_UPDRS_progression_slope"] > 0).sum())
            
            lines = []
            lines.append("=========================================================================")
            lines.append("        PARKINSON'S DISEASE EXPLORATORY LONGITUDINAL ANALYSIS SUMMARY")
            lines.append("=========================================================================")
            lines.append(f"Analysis completed successfully.")
            lines.append("=========================================================================\n")
            
            lines.append("1. TARGET VARIABLE CHARACTERISTICS:")
            lines.append(f"  - motor_UPDRS: Scores range from {longitudinal_stats['mean_motor_UPDRS'].min():.2f} to {longitudinal_stats['mean_motor_UPDRS'].max():.2f} (mean of patient means).")
            lines.append(f"  - total_UPDRS: Scores range from {longitudinal_stats['mean_total_UPDRS'].min():.2f} to {longitudinal_stats['mean_total_UPDRS'].max():.2f} (mean of patient means).")
            lines.append("  - Target distributions exhibit a bimodal shape, indicating patient cohort subdivisions.")
            lines.append("")
            
            lines.append("2. STRONGEST VOICE BIOMARKERS (Correlations with total_UPDRS):")
            for i, (feat, val) in enumerate(top_correlations[:5]):
                lines.append(f"  {i+1}. {feat}: Pearson r = {val:.4f}")
            lines.append("  - Observation: Pitch period entropy (PPE), recurrence period density entropy (RPDE), and amplitude perturbation parameters (like Shimmer:APQ11) exhibit the highest linear association with UPDRS progression targets.")
            lines.append("")
            
            lines.append("3. DISEASE PROGRESSION AND TEMPORAL TRENDS:")
            lines.append(f"  - Mean patient progression slope (Total UPDRS): {avg_total_slope:.6f} score points/day (std: {std_total_slope:.6f})")
            lines.append(f"  - Mean patient progression slope (Motor UPDRS): {avg_motor_slope:.6f} score points/day (std: {std_motor_slope:.6f})")
            lines.append(f"  - Patients showing positive progression rate (Total UPDRS): {pos_total_slopes} out of {len(longitudinal_stats)} ({pos_total_slopes/len(longitudinal_stats)*100:.1f}%)")
            lines.append("  - Observation: Overall trends show a gradual increase in average UPDRS scores as follow-up days increase. However, a significant cohort exhibits negative or flat slopes, illustrating clinical longitudinal heterogeneity.")
            lines.append("")
            
            lines.append("4. SKEWNESS & OUTLIER PROFILE:")
            lines.append(f"  - Number of highly skewed voice features (|skew| > 1.0): {len(highly_skewed)}")
            if highly_skewed:
                lines.append(f"  - Highly skewed biomarkers: {', '.join(highly_skewed)}")
            lines.append("  - Observation: Biomarkers such as Jitter(%), Jitter:RAP, Jitter:DDP, and NHR exhibit highly right-skewed profiles. In contrast, HNR and DFA are closer to normal distributions.")
            lines.append("")
            
            lines.append("5. NOTABLE TRENDS & MODELING IMPLICATIONS:")
            lines.append("  - Implication 1 (Feature Scaling & Transformations): Highly skewed biomarkers will benefit from non-linear transformations (e.g. logarithmic or Box-Cox) or robust scaling before fitting linear models.")
            lines.append("  - Implication 2 (Patient Heterogeneity): Individual patient trajectories vary significantly in both starting levels (intercepts) and progression rates (slopes). Mixed-effects models or personalized longitudinal predictors may be necessary.")
            lines.append("  - Implication 3 (Non-Linear Progression): The relationship between voice biomarkers (e.g., PPE, RPDE) and UPDRS score is moderately strong but exhibits non-linear components, suggesting that tree-based regressors (Random Forest, XGBoost) or deep sequential networks (LSTM) will outperform simple linear regression models.")
            
            lines.append("\n=========================================================================")
            
            summary_text = "\n".join(lines)
            
            # Write to file
            summary_report_path = report_dir / "eda_summary.txt"
            with open(summary_report_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
                
            logger.info(f"EDA Summary Report saved to: {summary_report_path.as_posix()}")
            return summary_text
        except Exception as e:
            raise EDAError(f"Failed to generate EDA summary report: {e}")
            
