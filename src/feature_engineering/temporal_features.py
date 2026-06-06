"""
Temporal Feature Engineering module for Parkinson's Disease progression.
Generates patient-aware sequential features including lags, rolling aggregations,
rates of change, and expanding historical variability for all voice biomarkers.

All operations are performed strictly within each patient group (grouped by subject#)
sorted chronologically by test_time to guarantee zero cross-patient data leakage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalFeatureError(Exception):
    """Custom exception raised when temporal feature engineering operations fail."""
    pass


class TemporalFeatureEngineer:
    """
    Generates longitudinal temporal features from voice biomarkers.

    All operations are grouped strictly by subject# and sorted by test_time
    to ensure zero cross-patient leakage. The following feature families are created
    for every voice biomarker:
      - Lag features       : _lag_1, _lag_2, _lag_3
      - Rolling mean       : _roll_mean_3
      - Rolling std        : _roll_std_3
      - Rate of change     : _rate_change
      - Historical variab. : _historical_variability
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialises the TemporalFeatureEngineer.

        Args:
            config (Dict[str, Any]): Loaded project configuration dictionary.
        """
        self.config = config
        schema_cfg = config.get("data_validation", {}).get("schema", {})

        self.subject_col = schema_cfg.get("subject_id_col", "subject#")
        self.test_time_col = schema_cfg.get("test_time_col", "test_time")
        self.biomarkers: List[str] = schema_cfg.get("voice_biomarkers", [])

        # Feature engineering parameters
        self.lag_periods: List[int] = [1, 2, 3]
        self.rolling_window: int = 3

    # ------------------------------------------------------------------
    # Internal helpers (retained for backward compatibility)
    # ------------------------------------------------------------------

    def _compute_feature_series(
        self, df_sorted: pd.DataFrame, biomarker: str
    ) -> Tuple[Dict[str, pd.Series], List[Dict[str, str]]]:
        """
        Computes all temporal feature Series for a single biomarker column.

        Returns a dict mapping new column name -> Series and a list of metadata dicts.
        Uses patient-grouped operations (groupby subject#) to prevent cross-patient leakage.

        Args:
            df_sorted (pd.DataFrame): Chronologically sorted working DataFrame.
            biomarker (str): Name of the source voice biomarker column.

        Returns:
            Tuple[Dict[str, Series], List[Dict]]: Mapping of new column names to
            computed Series, and a list of feature metadata records.
        """
        grp = df_sorted.groupby(self.subject_col)[biomarker]
        series_map: Dict[str, pd.Series] = {}
        meta: List[Dict[str, str]] = []

        # Lag features
        for lag in self.lag_periods:
            new_col = f"{biomarker}_lag_{lag}"
            series_map[new_col] = grp.shift(lag)
            meta.append({
                "Feature Name": new_col,
                "Feature Type": f"Lag_{lag}",
                "Source Feature": biomarker,
            })

        # Rolling mean
        col_rm = f"{biomarker}_roll_mean_{self.rolling_window}"
        series_map[col_rm] = grp.transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=1).mean()
        )
        meta.append({"Feature Name": col_rm, "Feature Type": "Rolling_Mean_3", "Source Feature": biomarker})

        # Rolling std
        col_rs = f"{biomarker}_roll_std_{self.rolling_window}"
        series_map[col_rs] = grp.transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=2).std()
        )
        meta.append({"Feature Name": col_rs, "Feature Type": "Rolling_Std_3", "Source Feature": biomarker})

        # Rate of change
        col_rc = f"{biomarker}_rate_change"
        series_map[col_rc] = grp.transform(lambda x: x.diff(1))
        meta.append({"Feature Name": col_rc, "Feature Type": "Rate_of_Change", "Source Feature": biomarker})

        # Historical variability (expanding std)
        col_hv = f"{biomarker}_historical_variability"
        series_map[col_hv] = grp.transform(lambda x: x.expanding(min_periods=2).std())
        meta.append({"Feature Name": col_hv, "Feature Type": "Historical_Variability", "Source Feature": biomarker})

        return series_map, meta

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        """
        Generates all temporal features for every voice biomarker.

        Pipeline:
          1. Sort dataset chronologically (subject#, test_time).
          2. For each biomarker, compute all feature Series (lags, rolling stats,
             rate of change, historical variability) and accumulate them into a dict.
          3. Attach all new feature columns in a single pd.concat call to avoid
             repeated frame fragmentation (eliminates PerformanceWarning).
          4. Handle NaNs created by lag/rolling operations:
             - Group-wise backward fill (bfill) preserves values for the first visit
               of each patient without borrowing from other patients.
             - Residual NaNs (patients with only 1 visit, or columns with all NaNs)
               are zero-filled as a safe neutral fallback.

        Args:
            df (pd.DataFrame): Chronologically ordered temporal dataset.

        Returns:
            Tuple[pd.DataFrame, List[Dict]]: Feature-enriched DataFrame and feature
                                             metadata list for summary reporting.
        """
        if self.subject_col not in df.columns:
            raise TemporalFeatureError(
                f"Required grouping column '{self.subject_col}' not found in dataset."
            )
        if self.test_time_col not in df.columns:
            raise TemporalFeatureError(
                f"Required sort column '{self.test_time_col}' not found in dataset."
            )
        if not self.biomarkers:
            raise TemporalFeatureError(
                "No voice biomarkers defined in config. Check 'data_validation.schema.voice_biomarkers'."
            )

        logger.info(
            f"Sorting dataset chronologically by ({self.subject_col}, {self.test_time_col})..."
        )
        df_sorted = df.sort_values(
            by=[self.subject_col, self.test_time_col], kind="stable"
        ).reset_index(drop=True)

        all_meta: List[Dict[str, str]] = []
        # Accumulate all new feature Series here; join once at the end to
        # avoid repeated frame insertions that cause PerformanceWarning.
        all_new_series: Dict[str, pd.Series] = {}

        logger.info(
            f"Engineering temporal features for {len(self.biomarkers)} voice biomarkers..."
        )

        for biomarker in self.biomarkers:
            if biomarker not in df_sorted.columns:
                logger.warning(f"Biomarker '{biomarker}' not found in dataset - skipping.")
                continue

            logger.debug(f"  Processing biomarker: {biomarker}")
            bio_series, bio_meta = self._compute_feature_series(df_sorted, biomarker)
            all_new_series.update(bio_series)
            all_meta.extend(bio_meta)

        # Attach all new features in a single concat - avoids 112 individual
        # frame insertions that caused the PerformanceWarning.
        new_cols = list(all_new_series.keys())
        df_out = pd.concat(
            [df_sorted, pd.DataFrame(all_new_series, index=df_sorted.index)],
            axis=1,
        )

        logger.info(
            f"Generated {len(new_cols)} temporal feature columns across {len(self.biomarkers)} biomarkers."
        )

        # ----------------------------------------------------------
        # Systematic NaN handling
        # Strategy:
        #   1. Group-wise backward fill (bfill) - copies the earliest known
        #      value back to any NaN that arose from lag/rolling at the start
        #      of a patient's history, without crossing patient boundaries.
        #   2. Zero-fill fallback - catches any residual NaNs (e.g., patients
        #      with a single visit where rolling std cannot be computed).
        # ----------------------------------------------------------
        logger.info(
            "Handling NaNs introduced by lag/rolling operations via group-wise bfill + zero fallback..."
        )
        nan_before = int(df_out[new_cols].isna().sum().sum())
        logger.debug(f"  Total NaNs before resolution: {nan_before}")

        df_out[new_cols] = (
            df_out.groupby(self.subject_col)[new_cols]
            .transform(lambda grp: grp.bfill())
        )
        df_out[new_cols] = df_out[new_cols].fillna(0.0)

        nan_after = int(df_out[new_cols].isna().sum().sum())
        logger.info(
            f"NaN resolution complete: {nan_before} NaNs resolved -> {nan_after} remaining."
        )

        return df_out, all_meta

    def generate_feature_summary(
        self,
        meta: List[Dict[str, str]],
        report_dir: Path
    ) -> pd.DataFrame:
        """
        Generates and saves a CSV summary of all engineered temporal features.

        Args:
            meta (List[Dict[str, str]]): Feature metadata list produced by engineer_features().
            report_dir (Path): Directory to save the feature_summary.csv file.

        Returns:
            pd.DataFrame: Feature summary DataFrame.
        """
        logger.info("Generating feature engineering summary report...")
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            summary_df = pd.DataFrame(meta)
            summary_path = report_dir / "feature_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(
                f"Feature summary saved: {len(summary_df)} features documented -> "
                f"{summary_path.as_posix()}"
            )
            return summary_df
        except Exception as exc:
            raise TemporalFeatureError(
                f"Failed to generate feature summary report: {exc}"
            )
