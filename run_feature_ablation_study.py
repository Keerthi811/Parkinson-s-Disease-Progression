#!/usr/bin/env python
"""
Feature Ablation Study Script — Phase 9 Step 4.

Objective:
    Determine which feature groups improve or degrade performance.
    Train Linear Regression and Ridge Regression under four configurations.
    Generate reports/diagnostics/ablation_study.csv ranked by Test MAE.
"""

import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "reports" / "diagnostics"
FEATURES_CSV = DATA_DIR / "parkinsons_temporal_features.csv"
TRAIN_CSV = DATA_DIR / "parkinsons_train.csv"
TEST_CSV = DATA_DIR / "parkinsons_test.csv"
OUT_CSV = DIAG_DIR / "ablation_study.csv"

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def scale_leakage_free(train_df, val_df, feature_cols):
    scaler = StandardScaler()
    tr = train_df.copy()
    vl = val_df.copy()
    tr[feature_cols] = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols]),
        columns=feature_cols, index=train_df.index
    )
    if not val_df.empty:
        vl[feature_cols] = pd.DataFrame(
            scaler.transform(val_df[feature_cols]),
            columns=feature_cols, index=val_df.index
        )
    return tr, vl

def run_cv(model, train_df, feature_cols, subject_col, target_col, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df[subject_col].values
    X_dummy = train_df[feature_cols].values
    
    maes, rmses, r2s = [], [], []
    
    for tr_idx, vl_idx in gkf.split(X_dummy, groups=groups):
        ft = train_df.iloc[tr_idx]
        fv = train_df.iloc[vl_idx]
        
        ft_sc, fv_sc = scale_leakage_free(ft, fv, feature_cols)
        
        import sklearn.base as _skbase
        fold_model = _skbase.clone(model)
        fold_model.fit(ft_sc[feature_cols].values, ft_sc[target_col].values)
        y_pred = fold_model.predict(fv_sc[feature_cols].values)
        y_true = fv_sc[target_col].values
        
        m = compute_metrics(y_true, y_pred)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        r2s.append(m["R2"])
        
    return {
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "R2": np.mean(r2s)
    }

def run_test(model, train_df, test_df, feature_cols, target_col):
    tr_sc, te_sc = scale_leakage_free(train_df, test_df, feature_cols)
    model.fit(tr_sc[feature_cols].values, tr_sc[target_col].values)
    y_pred = model.predict(te_sc[feature_cols].values)
    y_true = te_sc[target_col].values
    return compute_metrics(y_true, y_pred)

def main():
    print("=" * 60)
    print("FEATURE ABLATION STUDY — Phase 9 Step 4")
    print("=" * 60)

    # 1. Load files
    if not FEATURES_CSV.exists() or not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print("Error: Missing input dataset or split CSVs. Run previous phases first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    train_ph7 = pd.read_csv(TRAIN_CSV)
    test_ph7 = pd.read_csv(TEST_CSV)
    
    train_subs = set(train_ph7["subject#"].unique())
    test_subs = set(test_ph7["subject#"].unique())
    
    train_df = df[df["subject#"].isin(train_subs)].copy()
    test_df = df[df["subject#"].isin(test_subs)].copy()
    
    print(f"Train set: {len(train_subs)} patients ({len(train_df)} rows)")
    print(f"Test set : {len(test_subs)} patients ({len(test_df)} rows)")
    print("-" * 60)

    # 2. Define feature sets
    voice_biomarkers = [
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "PPE"
    ]
    metadata_features = ["age", "sex", "test_time"]
    original_features = voice_biomarkers + metadata_features # 19 features
    
    # Lag features
    lag_features = [c for c in df.columns if "_lag_" in c] # 48 features
    
    # Rolling features
    rolling_features = [c for c in df.columns if "_roll_" in c] # 32 features
    
    # All predictor features (excluding identifiers and target)
    exclude_cols = ["subject#", "motor_UPDRS", "total_UPDRS"]
    all_features = [c for c in df.columns if c not in exclude_cols] # 131 features

    # Define Configurations
    configs = {
        "Experiment A": voice_biomarkers,
        "Experiment B": original_features + lag_features,
        "Experiment C": original_features + lag_features + rolling_features,
        "Experiment D": all_features
    }

    # Models
    models = {
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring="neg_mean_absolute_error")
    }

    results = []

    for exp_name, feat_cols in configs.items():
        feat_count = len(feat_cols)
        print(f"Running {exp_name} with {feat_count} features...")
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            # CV
            cv_metrics = run_cv(model, train_df, feat_cols, "subject#", "motor_UPDRS", n_splits=5)
            # Test
            test_metrics = run_test(model, train_df, test_df, feat_cols, "motor_UPDRS")
            
            results.append({
                "Experiment": f"{exp_name} ({model_name})",
                "Feature_Count": feat_count,
                "CV_MAE": round(cv_metrics["MAE"], 6),
                "CV_RMSE": round(cv_metrics["RMSE"], 6),
                "CV_R2": round(cv_metrics["R2"], 6),
                "Test_MAE": round(test_metrics["MAE"], 6),
                "Test_RMSE": round(test_metrics["RMSE"], 6),
                "Test_R2": round(test_metrics["R2"], 6)
            })
            
    ablation_df = pd.DataFrame(results)
    
    # Rank by Test MAE ascending
    ablation_df = ablation_df.sort_values(by="Test_MAE", ascending=True).reset_index(drop=True)
    
    # Save to CSV
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(OUT_CSV, index=False)
    
    print("-" * 60)
    print("ABLATION STUDY RESULTS:")
    print(ablation_df.to_string(index=False))
    print("-" * 60)
    print(f"Saved ablation study results to {OUT_CSV}")

if __name__ == "__main__":
    main()
