#!/usr/bin/env python
"""
Baseline Machine Learning Models runner script for Phase 9.

Orchestrates the full baseline modelling pipeline:
  1. Loads the temporal feature-engineered dataset (134 columns).
  2. Re-applies the Phase 7 patient-level train/test split from persisted CSVs.
  3. Verifies zero patient overlap between development and hold-out sets.
  4. Trains and evaluates four baseline models with leakage-free GroupKFold CV.
  5. Generates all output artefacts:
     - Trained model .pkl files
     - Prediction CSVs (Actual, Predicted, Residual)
     - Coefficient analysis CSVs (Linear, Ridge, Lasso)
     - Diagnostic visualisations (3 plots per model)
     - Fold-level metrics CSV
     - Baseline results table with mean+/-std and generalization gap
     - Model leaderboard
     - Narrative baseline report
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config, resolve_path
from src.utils.logging_setup import setup_logging
from src.modeling.baseline_models import BaselineModelTrainer, BaselineModelError

logger = logging.getLogger("run_baseline_models")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with --config field.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 9 baseline machine learning modelling pipeline for "
            "Parkinson's Disease motor_UPDRS prediction."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the project configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def load_split_from_phase7(
    features_df: pd.DataFrame,
    train_csv: Path,
    test_csv: Path,
    subject_col: str = "subject#",
) -> tuple:
    """
    Re-applies the Phase 7 patient-level outer split to the feature-enriched
    dataset by extracting subject IDs from the Phase 7 persisted CSVs.

    This avoids re-running the split logic with a different random state and
    guarantees the same patient boundary is honoured across all phases.

    Args:
        features_df (pd.DataFrame): Feature-enriched dataset (134 columns).
        train_csv   (Path)        : Path to Phase 7 parkinsons_train.csv.
        test_csv    (Path)        : Path to Phase 7 parkinsons_test.csv.
        subject_col (str)         : Patient identifier column name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) filtered from
        features_df using the Phase 7 patient-level split.

    Raises:
        FileNotFoundError: If Phase 7 split CSVs do not exist.
        ValueError        : If extracted patient sets overlap.
    """
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Phase 7 train split not found: {train_csv.as_posix()}. "
            "Run Phase 7 (run_scaling_pipeline.py) first."
        )
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Phase 7 test split not found: {test_csv.as_posix()}. "
            "Run Phase 7 (run_scaling_pipeline.py) first."
        )

    train_subjects = set(pd.read_csv(train_csv)[subject_col].unique())
    test_subjects = set(pd.read_csv(test_csv)[subject_col].unique())

    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(
            f"Phase 7 split CSV patient overlap detected: {overlap}. "
            "Re-run Phase 7 to regenerate clean splits."
        )

    all_subjects = set(features_df[subject_col].unique())
    missing = (train_subjects | test_subjects) - all_subjects
    if missing:
        logger.warning(
            f"{len(missing)} subjects from Phase 7 splits not found in "
            f"features dataset (may be expected if dataset was reprocessed)."
        )

    train_df = features_df[features_df[subject_col].isin(train_subjects)].copy()
    test_df = features_df[features_df[subject_col].isin(test_subjects)].copy()

    logger.info(
        f"Phase 7 split applied to feature-enriched dataset: "
        f"Train = {len(train_subjects)} patients ({len(train_df)} rows) | "
        f"Test = {len(test_subjects)} patients ({len(test_df)} rows)"
    )
    return train_df, test_df


def identify_features(
    df: pd.DataFrame,
    exclude_cols: list,
) -> list:
    """
    Returns the list of predictor feature column names by excluding the
    specified non-feature columns.

    Args:
        df           (pd.DataFrame): Full dataset.
        exclude_cols (list)        : Columns to exclude (ID, targets).

    Returns:
        List[str]: Predictor feature column names.
    """
    features = [col for col in df.columns if col not in exclude_cols]
    return features


def main() -> None:
    """
    Main execution pipeline for Phase 9 baseline modelling.

    Steps:
        1.  Load configuration and initialise logging.
        2.  Load parkinsons_temporal_features.csv.
        3.  Re-apply Phase 7 patient-level outer train/test split.
        4.  Verify outer split integrity (zero patient overlap).
        5.  Identify feature columns.
        6.  Instantiate BaselineModelTrainer.
        7.  Build all four baseline models.
        8.  For each model:
            a. Run 5-fold GroupKFold CV (with per-fold leakage checks).
            b. Evaluate on unseen hold-out test set.
            c. Save trained model to models/baseline/.
            d. Save prediction CSV to evaluation/baseline/predictions/.
            e. Save coefficient CSV to evaluation/baseline/coefficients/ (linear models).
            f. Generate 3 diagnostic plots to reports/figures/baseline/.
        9.  Save fold-level metrics CSV.
        10. Save baseline results table (mean+/-std, generalization gap).
        11. Generate ranked model leaderboard.
        12. Generate narrative baseline report.
        13. Log final summary with best model by CV MAE and Test MAE.
    """
    args = parse_arguments()

    # -----------------------------------------------------------------------
    # Step 1: Load config and setup logging
    # -----------------------------------------------------------------------
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"CRITICAL: Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        setup_logging(config)
        logger.info("=" * 55)
        logger.info("PHASE 9: BASELINE MACHINE LEARNING MODELS")
        logger.info("=" * 55)
    except Exception as exc:
        print(f"CRITICAL: Logging setup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()

    try:
        # -------------------------------------------------------------------
        # Step 2: Resolve paths
        # -------------------------------------------------------------------
        processed_dir = resolve_path(config["paths"]["processed_data_dir"])
        models_dir = resolve_path(config["paths"]["models_dir"]) / "baseline"
        eval_dir = resolve_path(config["paths"]["evaluation_dir"]) / "baseline"
        reports_dir = resolve_path(config["paths"]["reports_dir"])
        fig_dir = reports_dir / "figures" / "baseline"

        features_path = processed_dir / "parkinsons_temporal_features.csv"
        train_csv = processed_dir / "parkinsons_train.csv"
        test_csv = processed_dir / "parkinsons_test.csv"

        # -------------------------------------------------------------------
        # Step 3: Load feature-engineered dataset
        # -------------------------------------------------------------------
        if not features_path.exists():
            logger.critical(
                f"Feature-engineered dataset not found: {features_path.as_posix()}. "
                "Run Phase 8 (run_temporal_feature_engineering.py) first."
            )
            sys.exit(1)

        logger.info(f"Loading feature-engineered dataset: {features_path.as_posix()}")
        features_df = pd.read_csv(features_path)
        logger.info(f"Dataset loaded: {features_df.shape[0]} rows x {features_df.shape[1]} columns.")

        # -------------------------------------------------------------------
        # Step 4: Re-apply Phase 7 patient-level split
        # -------------------------------------------------------------------
        logger.info("Re-applying Phase 7 patient-level outer train/test split...")
        train_df, test_df = load_split_from_phase7(
            features_df, train_csv, test_csv, subject_col="subject#"
        )

        # -------------------------------------------------------------------
        # Step 5: Verify outer split integrity
        # -------------------------------------------------------------------
        train_subs = set(train_df["subject#"].unique())
        test_subs = set(test_df["subject#"].unique())
        overlap = train_subs & test_subs
        if overlap:
            raise BaselineModelError(
                f"OUTER SPLIT INTEGRITY FAILURE: {len(overlap)} patients appear "
                f"in both train and test sets: {overlap}"
            )
        logger.info(
            "Outer split verification: PASS — zero patient overlap "
            f"({len(train_subs)} train | {len(test_subs)} test)."
        )

        # -------------------------------------------------------------------
        # Step 6: Identify feature columns
        # -------------------------------------------------------------------
        exclude_cols = ["subject#", "motor_UPDRS", "total_UPDRS"]
        feature_cols = identify_features(features_df, exclude_cols)
        logger.info(
            f"Feature columns identified: {len(feature_cols)} predictors "
            f"(voice biomarkers + temporal features)."
        )

        # -------------------------------------------------------------------
        # Step 7: Initialise trainer and build models
        # -------------------------------------------------------------------
        seed = config.get("reproducibility", {}).get("seed", 42)
        trainer = BaselineModelTrainer(
            n_splits=5,
            subject_col="subject#",
            target_col="motor_UPDRS",
            random_state=seed,
        )
        models = trainer.build_models()

        # -------------------------------------------------------------------
        # Step 8: Train and evaluate each model
        # -------------------------------------------------------------------
        all_results = []
        all_fold_records = []
        pred_dir = eval_dir / "predictions"
        coef_dir = eval_dir / "coefficients"

        for model_name, model in models.items():
            logger.info("-" * 55)
            logger.info(f"MODEL: {model_name}")
            logger.info("-" * 55)

            # (a) GroupKFold cross-validation
            logger.info(f"  [{model_name}] Running 5-fold GroupKFold CV...")
            fold_records, cv_summary = trainer.run_cv(model, train_df, feature_cols)
            all_fold_records.extend(fold_records)

            # (b) Final evaluation on hold-out test set
            test_metrics, predictions_df = trainer.evaluate_on_test(
                model, train_df, test_df, feature_cols
            )

            # (c) Save fitted model
            trainer.save_model(model, model_name, models_dir)

            # (d) Save predictions
            trainer.save_predictions(predictions_df, model_name, pred_dir)

            # (e) Save coefficients (skips Dummy)
            trainer.save_coefficients(
                model, model_name, feature_cols, coef_dir, top_n=20
            )

            # (f) Generate diagnostic visualisations
            trainer.generate_visualizations(
                model_name,
                predictions_df["Actual"].values,
                predictions_df["Predicted"].values,
                fig_dir,
            )

            # Collect aggregated result for reporting
            all_results.append({
                "model_name": model_name,
                "cv_summary": cv_summary,
                "test_metrics": test_metrics,
                "fitted_model": model,
            })

        # -------------------------------------------------------------------
        # Step 9: Save fold-level metrics
        # -------------------------------------------------------------------
        logger.info("=" * 55)
        logger.info("Saving fold-level metrics...")
        trainer.save_fold_results(all_fold_records, eval_dir)

        # -------------------------------------------------------------------
        # Step 10: Save baseline results table
        # -------------------------------------------------------------------
        logger.info("Saving baseline results table...")
        results_df = trainer.generate_results_table(all_results, eval_dir)

        # -------------------------------------------------------------------
        # Step 11: Generate model leaderboard
        # -------------------------------------------------------------------
        logger.info("Generating model leaderboard...")
        leaderboard_df = trainer.generate_leaderboard(all_results, eval_dir)

        # -------------------------------------------------------------------
        # Step 12: Generate narrative report
        # -------------------------------------------------------------------
        logger.info("Generating baseline narrative report...")
        report_path = eval_dir / "baseline_report.txt"
        trainer.generate_report(all_results, leaderboard_df, report_path)

        # -------------------------------------------------------------------
        # Step 13: Final summary log
        # -------------------------------------------------------------------
        duration = time.time() - start_time
        best_by_cv = min(
            all_results, key=lambda r: r["cv_summary"]["CV_MAE_Mean"]
        )
        best_by_test = min(
            all_results, key=lambda r: r["test_metrics"]["Test_MAE"]
        )

        logger.info("=" * 55)
        logger.info("PHASE 9 COMPLETE")
        logger.info("=" * 55)
        logger.info(f"  Best model (CV MAE)   : {best_by_cv['model_name']} "
                    f"(MAE={best_by_cv['cv_summary']['CV_MAE_Mean']:.4f})")
        logger.info(f"  Best model (Test MAE) : {best_by_test['model_name']} "
                    f"(MAE={best_by_test['test_metrics']['Test_MAE']:.4f})")
        logger.info("")
        logger.info("  Output artefacts:")
        logger.info(f"    Models      : {models_dir.as_posix()}")
        logger.info(f"    Predictions : {pred_dir.as_posix()}")
        logger.info(f"    Coefficients: {coef_dir.as_posix()}")
        logger.info(f"    Figures     : {fig_dir.as_posix()}")
        logger.info(f"    Evaluation  : {eval_dir.as_posix()}")
        logger.info(f"    Report      : {report_path.as_posix()}")
        logger.info(f"  Elapsed time  : {duration:.2f}s")
        logger.info("=" * 55)

        sys.exit(0)

    except FileNotFoundError as exc:
        logger.critical(f"Required input file missing: {exc}")
        sys.exit(1)
    except BaselineModelError as exc:
        logger.critical(f"Baseline modelling error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.critical(f"Unexpected error in Phase 9: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
