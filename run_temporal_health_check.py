#!/usr/bin/env python
"""
Temporal Feature Health Check Script — Phase 9 Step 3.

Objective:
    Determine whether engineered temporal features are introducing noise.
    Computes statistics (missing %, mean, std, variance) for temporal features by type,
    detects constant/near-constant features, and outputs reports.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "reports" / "diagnostics"
FEATURES_CSV = DATA_DIR / "parkinsons_temporal_features.csv"
OUT_CSV = DIAG_DIR / "temporal_feature_health.csv"
OUT_TXT = DIAG_DIR / "feature_quality_report.txt"

def get_feature_type(col_name: str) -> str:
    if "_lag_" in col_name:
        return "Lag"
    elif "_roll_mean_" in col_name:
        return "Rolling Mean"
    elif "_roll_std_" in col_name:
        return "Rolling Std"
    elif "_rate_change" in col_name:
        return "Rate Change"
    elif "_historical_variability" in col_name:
        return "Historical Variability"
    else:
        return "Unknown"

def main():
    print("=" * 60)
    print("TEMPORAL FEATURE HEALTH CHECK — Phase 9 Step 3")
    print("=" * 60)

    # 1. Load dataset
    if not FEATURES_CSV.exists():
        print(f"Error: dataset not found at {FEATURES_CSV}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Identifiers/targets and baseline features
    exclude_cols = ["subject#", "motor_UPDRS", "total_UPDRS"]
    original_voice_features = [
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "PPE"
    ]
    
    all_predictors = [c for c in df.columns if c not in exclude_cols]
    
    # Identify temporal features
    temporal_features = [c for c in all_predictors if get_feature_type(c) != "Unknown"]
    
    # Original features (non-temporal predictors)
    original_features = [c for c in all_predictors if c not in temporal_features]

    print(f"Total predictor features   : {len(all_predictors)}")
    print(f"Total original features    : {len(original_features)} (including age, sex, test_time)")
    print(f"Original voice features    : {len(original_voice_features)}")
    print(f"Total temporal features    : {len(temporal_features)}")
    print("-" * 60)

    # 2. Compute metrics for each temporal feature
    records = []
    for col in temporal_features:
        feat_type = get_feature_type(col)
        missing_pct = df[col].isna().mean() * 100
        mean_val = df[col].mean()
        std_val = df[col].std()
        var_val = df[col].var()
        
        records.append({
            "Feature": col,
            "Type": feat_type,
            "Missing_Pct": missing_pct,
            "Mean": mean_val,
            "Std": std_val,
            "Variance": var_val
        })
        
    health_df = pd.DataFrame(records)
    
    # 3. Detect constant and near-constant features
    # Constant: unique values <= 1 or var == 0
    constant_features = []
    near_constant_features = []
    
    for col in temporal_features:
        n_unique = df[col].nunique()
        var_val = df[col].var()
        if n_unique <= 1 or var_val == 0.0:
            constant_features.append(col)
        elif var_val < 1e-6:
            near_constant_features.append(col)

    # 4. Save CSV
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    health_df.to_csv(OUT_CSV, index=False)
    print(f"Saved temporal feature health details to {OUT_CSV}")

    # 5. Generate report lines
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 60)
    log("TEMPORAL FEATURE HEALTH CHECK REPORT")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
    log()
    log(f"Total Original Features (including age, sex, test_time): {len(original_features)}")
    log(f"Original Voice Features Only                          : {len(original_voice_features)}")
    log(f"Total Temporal Features                               : {len(temporal_features)}")
    log()
    
    log("=" * 60)
    log("SECTION 1: SUMMARY BY TEMPORAL FEATURE TYPE")
    log("=" * 60)
    
    types_to_check = ["Lag", "Rolling Mean", "Rolling Std", "Rate Change", "Historical Variability"]
    for t in types_to_check:
        t_df = health_df[health_df["Type"] == t]
        log(f"Feature Type: {t}")
        log(f"  Count          : {len(t_df)}")
        log(f"  Avg Missing %  : {t_df['Missing_Pct'].mean():.4f}%")
        log(f"  Avg Mean       : {t_df['Mean'].mean():.6f}")
        log(f"  Avg Std        : {t_df['Std'].mean():.6f}")
        log(f"  Avg Variance   : {t_df['Variance'].mean():.6f}")
        log()
        
    log("=" * 60)
    log("SECTION 2: INSTABILITY & VARIANCE DIAGNOSTICS")
    log("=" * 60)
    
    log(f"Constant Features (nunique <= 1 or var == 0) (Count: {len(constant_features)}):")
    if len(constant_features) > 0:
        for f in constant_features:
            log(f"  - {f}")
    else:
        log("  None")
    log()
    
    log(f"Near-Constant Features (var < 1e-6) (Count: {len(near_constant_features)}):")
    if len(near_constant_features) > 0:
        for f in near_constant_features:
            col_var = df[f].var()
            log(f"  - {f:<55} var = {col_var:.2e}")
    else:
        log("  None")
    log()
    
    log("=" * 60)
    log("SECTION 3: INDIVIDUAL FEATURE HEALTH METRICS")
    log("=" * 60)
    log(f"{'Feature':<60} | {'Type':<22} | {'Missing %':<9} | {'Mean':<12} | {'Std':<12} | {'Variance':<12}")
    log("-" * 135)
    for _, row in health_df.iterrows():
        log(f"{row['Feature']:<60} | {row['Type']:<22} | {row['Missing_Pct']:>8.2f}% | {row['Mean']:>12.6f} | {row['Std']:>12.6f} | {row['Variance']:>12.6f}")
        
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"Saved feature quality report to {OUT_TXT}")

if __name__ == "__main__":
    main()
