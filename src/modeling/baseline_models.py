"""
Baseline Machine Learning Models module for Parkinson's Disease progression.
Implements training, evaluation, and reporting for four baseline regression models:
  - DummyRegressor  (naive mean-prediction benchmark)
  - LinearRegression
  - RidgeCV         (built-in alpha search on training data)
  - LassoCV         (built-in alpha search on training data)

All evaluation is leakage-free:
  - Scalers are fitted exclusively on training subsets.
  - GroupKFold ensures no patient appears in both train and validation folds.
  - Holdout test patients are never seen during model selection.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MetricsDict = Dict[str, float]
FoldRecord = Dict[str, Any]
ModelResult = Dict[str, Any]


class BaselineModelError(Exception):
    """Custom exception raised when a baseline modelling operation fails."""
    pass


class BaselineModelTrainer:
    """
    Trains, evaluates, and reports on four baseline regression models for
    Parkinson's disease motor_UPDRS prediction.

    Evaluation protocol:
      - Inner loop : 5-fold GroupKFold CV on the development (train) set.
      - Outer loop : Final scoring on the unseen patient hold-out test set.
      - Scaler     : StandardScaler fitted only on the training portion of
                     each fold / on the full train set for test evaluation.

    Attributes:
        n_splits      (int): Number of GroupKFold cross-validation folds.
        subject_col   (str): Patient grouping column name.
        target_col    (str): Regression target column.
        random_state  (int): Reproducibility seed.
    """

    def __init__(
        self,
        n_splits: int = 5,
        subject_col: str = "subject#",
        target_col: str = "motor_UPDRS",
        random_state: int = 42,
    ) -> None:
        """
        Initialises the BaselineModelTrainer.

        Args:
            n_splits     (int): Number of GroupKFold folds (default 5).
            subject_col  (str): Column name of the patient identifier.
            target_col   (str): Target variable to predict.
            random_state (int): Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.subject_col = subject_col
        self.target_col = target_col
        self.random_state = random_state

    # -----------------------------------------------------------------------
    # 1. Model Construction
    # -----------------------------------------------------------------------

    def build_models(self) -> Dict[str, Any]:
        """
        Constructs all four baseline regression models.

        Returns:
            Dict[str, estimator]: Ordered mapping of model names to sklearn
            estimator instances. DummyRegressor acts as a naive benchmark;
            RidgeCV and LassoCV perform built-in alpha selection internally
            during fit() on training data.
        """
        models = {
            "Dummy": DummyRegressor(strategy="mean"),
            "Linear": LinearRegression(),
            "Ridge": RidgeCV(
                alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                scoring="neg_mean_absolute_error",
            ),
            "Lasso": LassoCV(
                alphas=[0.01, 0.1, 1.0, 10.0],
                max_iter=50000,
                cv=3,
                selection="random",
                random_state=self.random_state,
            ),
        }
        logger.info(f"Constructed {len(models)} baseline models: {list(models.keys())}")
        return models

    # -----------------------------------------------------------------------
    # 2. Leakage-Free Scaling
    # -----------------------------------------------------------------------

    def scale_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Fits StandardScaler on train_df features and applies transformation to
        both train_df and val_df. Returns named DataFrames to preserve feature
        names for downstream coefficient analysis and SHAP compatibility.

        The scaler is NEVER fitted on val_df data, preventing any information
        leakage from the validation set into the standardisation parameters.

        Args:
            train_df     (pd.DataFrame): Training fold DataFrame.
            val_df       (pd.DataFrame): Validation fold DataFrame.
            feature_cols (List[str])   : Feature column names to scale.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
                - Scaled training DataFrame (feature columns replaced).
                - Scaled validation DataFrame (feature columns replaced).
                - Fitted StandardScaler instance.
        """
        scaler = StandardScaler()

        train_scaled = train_df.copy()
        val_scaled = val_df.copy()

        # Fit on train, transform train
        train_scaled[feature_cols] = pd.DataFrame(
            scaler.fit_transform(train_df[feature_cols]),
            columns=feature_cols,
            index=train_df.index,
        )

        # Transform val with train-fitted scaler (no fit on val)
        if not val_df.empty:
            val_scaled[feature_cols] = pd.DataFrame(
                scaler.transform(val_df[feature_cols]),
                columns=feature_cols,
                index=val_df.index,
            )

        return train_scaled, val_scaled, scaler

    # -----------------------------------------------------------------------
    # 3. Metric Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsDict:
        """
        Computes MAE, RMSE, and R² for a set of predictions.

        Args:
            y_true (np.ndarray): Ground truth target values.
            y_pred (np.ndarray): Model predictions.

        Returns:
            Dict[str, float]: Dictionary with keys MAE, RMSE, R2.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    # -----------------------------------------------------------------------
    # 4. Cross-Validation
    # -----------------------------------------------------------------------

    def run_cv(
        self,
        model: Any,
        train_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[List[FoldRecord], MetricsDict]:
        """
        Runs 5-fold GroupKFold cross-validation on the development (train) set.

        Per-fold steps:
          1. Verify zero patient overlap between fold-train and fold-val.
          2. Fit StandardScaler on fold-train features only.
          3. Scale both fold-train and fold-val.
          4. Fit model on scaled fold-train.
          5. Predict on scaled fold-val.
          6. Compute MAE, RMSE, R².

        Args:
            model        (estimator) : Unfitted sklearn-compatible regressor.
            train_df     (pd.DataFrame): Full development training set.
            feature_cols (List[str]) : Predictor feature column names.

        Returns:
            Tuple[List[FoldRecord], MetricsDict]:
                - List of per-fold metric dicts (with Model, Fold, MAE, RMSE, R2).
                - Aggregated dict with CV_MAE_Mean, CV_MAE_STD, CV_RMSE_Mean,
                  CV_RMSE_STD, CV_R2_Mean, CV_R2_STD.

        Raises:
            BaselineModelError: If patient leakage is detected in any fold.
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        groups = train_df[self.subject_col].values
        X_dummy = train_df[feature_cols].values  # Used only for GKF split indexing

        fold_records: List[FoldRecord] = []
        fold_maes, fold_rmses, fold_r2s = [], [], []

        model_name = type(model).__name__

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_dummy, groups=groups), start=1
        ):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]

            # Leakage verification
            train_subs = set(fold_train[self.subject_col].unique())
            val_subs = set(fold_val[self.subject_col].unique())
            overlap = train_subs & val_subs
            if overlap:
                raise BaselineModelError(
                    f"Patient leakage detected in {model_name} CV Fold {fold_idx}: "
                    f"overlapping subjects = {overlap}"
                )
            logger.debug(
                f"  [{model_name}] Fold {fold_idx}: "
                f"{len(train_subs)} train patients | {len(val_subs)} val patients | "
                f"overlap=NONE"
            )

            # Scale (fit on fold-train only)
            fold_train_sc, fold_val_sc, _ = self.scale_fold(
                fold_train, fold_val, feature_cols
            )

            X_tr = fold_train_sc[feature_cols].values
            y_tr = fold_train_sc[self.target_col].values
            X_vl = fold_val_sc[feature_cols].values
            y_vl = fold_val_sc[self.target_col].values

            # Fit and predict
            import sklearn.base as _skbase
            fold_model = _skbase.clone(model)
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_vl)

            metrics = self._compute_metrics(y_vl, y_pred)
            fold_maes.append(metrics["MAE"])
            fold_rmses.append(metrics["RMSE"])
            fold_r2s.append(metrics["R2"])

            fold_records.append({
                "Model": model_name,
                "Fold": fold_idx,
                "MAE": round(metrics["MAE"], 6),
                "RMSE": round(metrics["RMSE"], 6),
                "R2": round(metrics["R2"], 6),
            })

            logger.debug(
                f"  [{model_name}] Fold {fold_idx}: "
                f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f}"
            )

        cv_summary: MetricsDict = {
            "CV_MAE_Mean": float(np.mean(fold_maes)),
            "CV_MAE_STD": float(np.std(fold_maes)),
            "CV_RMSE_Mean": float(np.mean(fold_rmses)),
            "CV_RMSE_STD": float(np.std(fold_rmses)),
            "CV_R2_Mean": float(np.mean(fold_r2s)),
            "CV_R2_STD": float(np.std(fold_r2s)),
        }

        logger.info(
            f"  [{model_name}] CV Summary: "
            f"MAE={cv_summary['CV_MAE_Mean']:.4f}+/-{cv_summary['CV_MAE_STD']:.4f} | "
            f"RMSE={cv_summary['CV_RMSE_Mean']:.4f}+/-{cv_summary['CV_RMSE_STD']:.4f} | "
            f"R2={cv_summary['CV_R2_Mean']:.4f}+/-{cv_summary['CV_R2_STD']:.4f}"
        )

        return fold_records, cv_summary

    # -----------------------------------------------------------------------
    # 5. Final Test Evaluation
    # -----------------------------------------------------------------------

    def evaluate_on_test(
        self,
        model: Any,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[MetricsDict, pd.DataFrame]:
        """
        Fits model on the full development train set and evaluates on the
        unseen hold-out test set.

        Steps:
          1. Fit StandardScaler on full train features only.
          2. Scale train and test with the same scaler.
          3. Fit the model on scaled train.
          4. Predict on scaled test.
          5. Return metrics and prediction DataFrame.

        Args:
            model        (estimator)   : Unfitted sklearn-compatible regressor.
            train_df     (pd.DataFrame): Full development training set.
            test_df      (pd.DataFrame): Unseen hold-out test set.
            feature_cols (List[str])   : Predictor feature column names.

        Returns:
            Tuple[MetricsDict, pd.DataFrame]:
                - Test metrics dict with Test_MAE, Test_RMSE, Test_R2.
                - Predictions DataFrame with columns: Actual, Predicted, Residual.
        """
        model_name = type(model).__name__
        logger.info(f"  [{model_name}] Fitting on full train set and evaluating on hold-out test...")

        # Scale: fit on train, transform both
        train_sc, test_sc, _ = self.scale_fold(train_df, test_df, feature_cols)

        X_train = train_sc[feature_cols].values
        y_train = train_sc[self.target_col].values
        X_test = test_sc[feature_cols].values
        y_test = test_sc[self.target_col].values

        # Fit on full train set (no clone — this is the final model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._compute_metrics(y_test, y_pred)
        test_metrics: MetricsDict = {
            "Test_MAE": metrics["MAE"],
            "Test_RMSE": metrics["RMSE"],
            "Test_R2": metrics["R2"],
        }

        logger.info(
            f"  [{model_name}] Test: "
            f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f}"
        )

        predictions_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred,
            "Residual": y_test - y_pred,
        })

        return test_metrics, predictions_df

    # -----------------------------------------------------------------------
    # 6. Model Persistence
    # -----------------------------------------------------------------------

    def save_model(self, model: Any, name: str, models_dir: Path) -> Path:
        """
        Persists a fitted model to disk using joblib.

        Args:
            model      (estimator): Fitted sklearn model instance.
            name       (str)      : Human-readable model name (e.g. "Linear").
            models_dir (Path)     : Parent directory for model artefacts.

        Returns:
            Path: Absolute path of the saved model file.

        Raises:
            BaselineModelError: If serialisation fails.
        """
        slug = name.lower().replace(" ", "_")
        model_map = {
            "dummy": "dummy_regressor",
            "linear": "linear_regression",
            "ridge": "ridge_regression",
            "lasso": "lasso_regression",
        }
        filename = model_map.get(slug, f"{slug}_model") + ".pkl"
        save_path = models_dir / filename
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, save_path)
            logger.info(f"  [{name}] Model persisted -> {save_path.as_posix()}")
            return save_path
        except Exception as exc:
            raise BaselineModelError(
                f"Failed to save model '{name}' to {save_path}: {exc}"
            )

    # -----------------------------------------------------------------------
    # 7. Predictions Persistence
    # -----------------------------------------------------------------------

    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        name: str,
        pred_dir: Path,
    ) -> Path:
        """
        Saves test-set predictions DataFrame (Actual, Predicted, Residual) to CSV.

        Args:
            predictions_df (pd.DataFrame): Predictions with Actual/Predicted/Residual.
            name           (str)          : Model name.
            pred_dir       (Path)         : Output directory.

        Returns:
            Path: Saved CSV file path.
        """
        slug = name.lower().replace(" ", "_")
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_path = pred_dir / f"{slug}_predictions.csv"
        predictions_df.to_csv(save_path, index=False)
        logger.info(f"  [{name}] Predictions saved -> {save_path.as_posix()}")
        return save_path

    # -----------------------------------------------------------------------
    # 8. Coefficient Analysis
    # -----------------------------------------------------------------------

    def save_coefficients(
        self,
        model: Any,
        name: str,
        feature_cols: List[str],
        coef_dir: Path,
        top_n: int = 20,
    ) -> Optional[Path]:
        """
        Extracts and saves feature coefficients for linear models.

        Computes Absolute_Importance = |Coefficient| and sorts descending.
        Logs the top_n most influential features.
        Silently skips DummyRegressor which has no coefficients.

        Args:
            model        (estimator): Fitted model with optional .coef_ attribute.
            name         (str)      : Model name.
            feature_cols (List[str]): List of feature names matching coef_ order.
            coef_dir     (Path)     : Output directory.
            top_n        (int)      : Number of top features to log.

        Returns:
            Optional[Path]: CSV path if saved, else None.
        """
        if not hasattr(model, "coef_"):
            logger.info(f"  [{name}] No coefficients to extract (DummyRegressor). Skipping.")
            return None

        coef_dir.mkdir(parents=True, exist_ok=True)
        slug = name.lower().replace(" ", "_")
        save_path = coef_dir / f"{slug}_coefficients.csv"

        coef_values = model.coef_.flatten()
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": coef_values,
            "Absolute_Importance": np.abs(coef_values),
        }).sort_values("Absolute_Importance", ascending=False).reset_index(drop=True)

        coef_df.to_csv(save_path, index=False)

        logger.info(f"  [{name}] Coefficient analysis saved -> {save_path.as_posix()}")
        logger.info(f"  [{name}] Top {top_n} most influential features:")
        for i, row in coef_df.head(top_n).iterrows():
            logger.info(
                f"    {i+1:>3}. {row['Feature']:<45} coef={row['Coefficient']:>+.6f}"
            )

        return save_path

    # -----------------------------------------------------------------------
    # 9. Visualisations
    # -----------------------------------------------------------------------

    def generate_visualizations(
        self,
        model_name: str,
        y_actual: np.ndarray,
        y_pred: np.ndarray,
        fig_dir: Path,
    ) -> None:
        """
        Generates and saves three diagnostic figures for a model's test-set predictions:
          1. Actual vs Predicted scatter plot with identity line.
          2. Residuals vs Predicted values plot.
          3. Prediction error distribution histogram.

        All figures are saved as high-resolution PNG files suitable for
        dissertation/publication use.

        Args:
            model_name (str)       : Display name used in plot titles and file names.
            y_actual   (np.ndarray): Ground-truth target values.
            y_pred     (np.ndarray): Model predicted values.
            fig_dir    (Path)      : Directory to save figures.
        """
        fig_dir.mkdir(parents=True, exist_ok=True)
        slug = model_name.lower().replace(" ", "_")
        residuals = y_actual - y_pred

        # Shared style settings
        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        })

        # -- 1. Actual vs Predicted ------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_actual, y_pred, alpha=0.45, s=18, color="#4C72B0", edgecolors="none")
        lims = [
            min(y_actual.min(), y_pred.min()) - 1,
            max(y_actual.max(), y_pred.max()) + 1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual motor_UPDRS", fontsize=12)
        ax.set_ylabel("Predicted motor_UPDRS", fontsize=12)
        ax.set_title(f"{model_name} — Actual vs Predicted", fontsize=13, fontweight="bold")
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        ax.annotate(
            f"MAE = {mae:.3f}\nR² = {r2:.3f}",
            xy=(0.05, 0.88),
            xycoords="axes fraction",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8),
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(fig_dir / f"actual_vs_predicted_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # -- 2. Residuals vs Predicted ----------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_pred, residuals, alpha=0.45, s=18, color="#DD8452", edgecolors="none")
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Predicted motor_UPDRS", fontsize=12)
        ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
        ax.set_title(f"{model_name} — Residual Plot", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / f"residuals_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # -- 3. Error Distribution --------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=40, color="#55A868", edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
        ax.axvline(residuals.mean(), color="navy", linewidth=1.2, linestyle=":", label=f"Mean = {residuals.mean():.2f}")
        ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{model_name} — Prediction Error Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(fig_dir / f"error_distribution_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"  [{model_name}] Visualisations saved to: {fig_dir.as_posix()}")

    # -----------------------------------------------------------------------
    # 10. Results Table
    # -----------------------------------------------------------------------

    def generate_results_table(
        self,
        all_results: List[ModelResult],
        eval_dir: Path,
    ) -> pd.DataFrame:
        """
        Builds and saves the baseline results table with CV mean+/-std metrics,
        test metrics, and generalization gap per model.

        Generalization Gap = Test_MAE - CV_MAE_Mean.
        A large positive gap indicates the model generalises worse on unseen
        patients than suggested by cross-validation.

        Args:
            all_results (List[ModelResult]): Aggregated result dicts per model.
            eval_dir    (Path)             : Output directory.

        Returns:
            pd.DataFrame: Results table DataFrame.
        """
        rows = []
        for res in all_results:
            cv = res["cv_summary"]
            test = res["test_metrics"]
            gen_gap = test["Test_MAE"] - cv["CV_MAE_Mean"]
            rows.append({
                "Model": res["model_name"],
                "CV_MAE_Mean": round(cv["CV_MAE_Mean"], 4),
                "CV_MAE_STD": round(cv["CV_MAE_STD"], 4),
                "CV_RMSE_Mean": round(cv["CV_RMSE_Mean"], 4),
                "CV_RMSE_STD": round(cv["CV_RMSE_STD"], 4),
                "CV_R2_Mean": round(cv["CV_R2_Mean"], 4),
                "CV_R2_STD": round(cv["CV_R2_STD"], 4),
                "Test_MAE": round(test["Test_MAE"], 4),
                "Test_RMSE": round(test["Test_RMSE"], 4),
                "Test_R2": round(test["Test_R2"], 4),
                "Generalization_Gap": round(gen_gap, 4),
            })

        results_df = pd.DataFrame(rows)
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / "baseline_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Baseline results table saved -> {out_path.as_posix()}")
        return results_df

    # -----------------------------------------------------------------------
    # 11. Fold-Level Metrics
    # -----------------------------------------------------------------------

    def save_fold_results(
        self,
        all_fold_records: List[FoldRecord],
        eval_dir: Path,
    ) -> pd.DataFrame:
        """
        Saves per-fold metrics for all models to a CSV file.

        Provides granular insight into fold-to-fold variability and model
        stability across cross-validation splits.

        Args:
            all_fold_records (List[FoldRecord]): Flattened list of fold metric
                dicts (Model, Fold, MAE, RMSE, R2) from all models.
            eval_dir         (Path): Output directory.

        Returns:
            pd.DataFrame: Fold-level results DataFrame.
        """
        fold_df = pd.DataFrame(all_fold_records)
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / "fold_results.csv"
        fold_df.to_csv(out_path, index=False)
        logger.info(f"Fold-level results saved ({len(fold_df)} rows) -> {out_path.as_posix()}")
        return fold_df

    # -----------------------------------------------------------------------
    # 12. Leaderboard
    # -----------------------------------------------------------------------

    def generate_leaderboard(
        self,
        all_results: List[ModelResult],
        eval_dir: Path,
    ) -> pd.DataFrame:
        """
        Generates a ranked model leaderboard sorted by Test_MAE (primary)
        then CV_MAE_Mean (secondary). Lower MAE = better rank.

        The leaderboard is designed to be extended in future phases when
        more advanced models are added.

        Args:
            all_results (List[ModelResult]): Aggregated result dicts per model.
            eval_dir    (Path)             : Output directory.

        Returns:
            pd.DataFrame: Leaderboard DataFrame.
        """
        rows = []
        for res in all_results:
            cv = res["cv_summary"]
            test = res["test_metrics"]
            rows.append({
                "Model": res["model_name"],
                "CV_MAE": round(cv["CV_MAE_Mean"], 4),
                "Test_MAE": round(test["Test_MAE"], 4),
                "Generalization_Gap": round(test["Test_MAE"] - cv["CV_MAE_Mean"], 4),
            })

        lb_df = pd.DataFrame(rows).sort_values(
            ["Test_MAE", "CV_MAE"], ascending=[True, True]
        ).reset_index(drop=True)
        lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))

        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / "model_leaderboard.csv"
        lb_df.to_csv(out_path, index=False)
        logger.info(f"Model leaderboard saved -> {out_path.as_posix()}")
        return lb_df

    # -----------------------------------------------------------------------
    # 13. Narrative Report
    # -----------------------------------------------------------------------

    def generate_report(
        self,
        all_results: List[ModelResult],
        leaderboard_df: pd.DataFrame,
        report_path: Path,
    ) -> None:
        """
        Generates a human-readable baseline modelling report in plain text.

        Includes:
          - Best model identification (by Test_MAE).
          - Full performance table in mean+/-std format.
          - Alpha values selected by RidgeCV and LassoCV.
          - Generalization gap analysis per model.
          - Summary observations suitable for dissertation use.

        Args:
            all_results    (List[ModelResult]): Full aggregated results.
            leaderboard_df (pd.DataFrame)     : Ranked leaderboard.
            report_path    (Path)             : Output path for the .txt file.
        """
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_model_name = leaderboard_df.iloc[0]["Model"]
        best_test_mae = leaderboard_df.iloc[0]["Test_MAE"]

        lines = [
            "=" * 75,
            "  PARKINSON'S DISEASE PROGRESSION — BASELINE MODEL EVALUATION REPORT",
            "=" * 75,
            "",
            f"  Target Variable : motor_UPDRS",
            f"  Evaluation      : 5-Fold GroupKFold CV + Unseen Hold-out Test Set",
            f"  Models          : Dummy, Linear Regression, RidgeCV, LassoCV",
            "",
            "-" * 75,
            "  BEST BASELINE MODEL",
            "-" * 75,
            f"  Model      : {best_model_name}",
            f"  Test MAE   : {best_test_mae:.4f}",
            "",
            "-" * 75,
            "  PERFORMANCE SUMMARY  (CV: mean +/- std across 5 folds)",
            "-" * 75,
            f"  {'Model':<18} {'CV MAE':>16} {'CV RMSE':>16} {'CV R2':>12} {'Test MAE':>10} {'Test R2':>8} {'Gen.Gap':>9}",
            "  " + "-" * 73,
        ]

        for res in all_results:
            cv = res["cv_summary"]
            test = res["test_metrics"]
            gap = test["Test_MAE"] - cv["CV_MAE_Mean"]
            lines.append(
                f"  {res['model_name']:<18} "
                f"{cv['CV_MAE_Mean']:>7.4f}+/-{cv['CV_MAE_STD']:.4f} "
                f"{cv['CV_RMSE_Mean']:>7.4f}+/-{cv['CV_RMSE_STD']:.4f} "
                f"{cv['CV_R2_Mean']:>6.4f}+/-{cv['CV_R2_STD']:.4f} "
                f"{test['Test_MAE']:>10.4f} "
                f"{test['Test_R2']:>8.4f} "
                f"{gap:>+9.4f}"
            )

        lines += [
            "",
            "-" * 75,
            "  HYPERPARAMETER SELECTION (RidgeCV / LassoCV)",
            "-" * 75,
        ]
        for res in all_results:
            model = res["fitted_model"]
            name = res["model_name"]
            if hasattr(model, "alpha_"):
                lines.append(f"  {name:<18} Selected alpha = {model.alpha_:.6f}")
            else:
                lines.append(f"  {name:<18} N/A (no alpha hyperparameter)")

        lines += [
            "",
            "-" * 75,
            "  GENERALIZATION GAP ANALYSIS",
            "-" * 75,
            "  Generalization Gap = Test_MAE - CV_MAE_Mean",
            "  A positive gap indicates worse generalization on unseen patients.",
            "",
        ]
        for res in all_results:
            cv = res["cv_summary"]
            test = res["test_metrics"]
            gap = test["Test_MAE"] - cv["CV_MAE_Mean"]
            interp = "GOOD" if abs(gap) < 0.5 else "MODERATE" if abs(gap) < 1.5 else "HIGH"
            lines.append(
                f"  {res['model_name']:<18} Gap = {gap:>+.4f}  [{interp} generalization]"
            )

        lines += [
            "",
            "-" * 75,
            "  LEADERBOARD (ranked by Test_MAE ascending)",
            "-" * 75,
        ]
        for _, row in leaderboard_df.iterrows():
            lines.append(
                f"  #{int(row['Rank']):<3} {row['Model']:<18} "
                f"Test_MAE={row['Test_MAE']:.4f}  CV_MAE={row['CV_MAE']:.4f}  Gap={row['Generalization_Gap']:+.4f}"
            )

        lines += [
            "",
            "-" * 75,
            "  OBSERVATIONS",
            "-" * 75,
            "  1. The DummyRegressor (mean prediction) establishes the naive floor.",
            "     All trained models should substantially outperform it.",
            "  2. RidgeCV and LassoCV use built-in alpha search on training data,",
            "     ensuring no hyperparameter information leaks from validation sets.",
            "  3. LassoCV performs implicit feature selection (many coefficients -> 0),",
            "     which aids interpretability for dissertation analysis.",
            "  4. The generalization gap reveals how well each model transfers from",
            "     seen patients (CV) to entirely unseen patients (hold-out test).",
            "  5. These baselines serve as reference points for advanced models",
            "     (e.g., Random Forest, LSTM) in subsequent phases.",
            "",
            "=" * 75,
        ]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Baseline report saved -> {report_path.as_posix()}")
