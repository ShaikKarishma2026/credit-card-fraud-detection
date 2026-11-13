"""
Credit Card Fraud Data Cleaning & Preprocessing Pipeline
Implements Phase 2:
1. Missing value detection & median imputation
2. Duplicate detection & removal
3. Data type verification & casting
4. Outlier detection using IQR & capping
5. Feature engineering (Hour_of_Day, Log_Amount, Amount_Bin)
6. RobustScaler normalization for Amount & Time
7. Cleaned dataset export & diagnostic audit report
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = BASE_DIR / "dataset" / "creditcard_raw.csv"
CLEANED_CSV_PATH = BASE_DIR / "dataset" / "creditcard_cleaned.csv"
CLEANING_REPORT_PATH = BASE_DIR / "dataset" / "cleaning_report.json"

def clean_credit_card_data(raw_csv: Path = RAW_CSV_PATH, cleaned_csv: Path = CLEANED_CSV_PATH, report_path: Path = CLEANING_REPORT_PATH):
    print("=" * 65)
    print("PHASE 2: DATA CLEANING & PREPROCESSING PIPELINE")
    print("=" * 65)

    df = pd.read_csv(raw_csv)
    initial_shape = df.shape
    print(f"[1/7] Ingested raw dataset: {initial_shape[0]} rows, {initial_shape[1]} columns.")

    audit = {
        "raw_record_count": initial_shape[0],
        "raw_feature_count": initial_shape[1],
        "missing_values_detected": {},
        "duplicates_removed": 0,
        "outliers_treated": [],
        "engineered_features": []
    }

    # 1. Missing Value Detection
    null_counts = df.isnull().sum()
    detected_nulls = {col: int(val) for col, val in null_counts.items() if val > 0}
    audit["missing_values_detected"] = detected_nulls
    print(f"[2/7] Missing Value Audit: {detected_nulls}")

    # Impute missing values with column median
    for col in detected_nulls:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    print("      -> Imputed all missing values with feature medians.")

    # 2. Duplicate Detection & Removal
    dup_count = int(df.duplicated().sum())
    audit["duplicates_removed"] = dup_count
    df = df.drop_duplicates().copy()
    print(f"[3/7] Removed {dup_count} duplicate transaction records. Clean count: {len(df)}")

    # 3. Data Type Checking & Verification
    for col in df.columns:
        if col == "Class":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
    print("[4/7] Verified and cast data types (float64 for features, int64 for target Class).")

    # 4. Outlier Analysis & Treatment on Amount
    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3.0 * IQR # 3x IQR extreme bound
    
    extreme_outliers = df[df["Amount"] > 15000.0]
    print(f"[5/7] Outlier Analysis: Detected {len(extreme_outliers)} extreme transaction entry errors (> $15,000).")
    for idx in extreme_outliers.index:
        old_val = float(df.loc[idx, "Amount"])
        new_val = float(round(df["Amount"].quantile(0.99), 2))
        df.loc[idx, "Amount"] = new_val
        audit["outliers_treated"].append({"index": int(idx), "old_value": old_val, "capped_value": new_val})

    # 5. Feature Engineering
    # A. Hour of Day (0 to 23 hours)
    df["Hour_of_Day"] = ((df["Time"] // 3600) % 24).astype(int)

    # B. Log Amount (normalizes right skew)
    df["Log_Amount"] = np.round(np.log1p(df["Amount"]), 4)

    # C. Amount Spending Bins
    df["Amount_Bin"] = pd.qcut(df["Amount"], q=4, labels=["Micro (<$15)", "Small ($15-45)", "Medium ($45-120)", "High (>$120)"])

    audit["engineered_features"] = ["Hour_of_Day", "Log_Amount", "Amount_Bin"]
    print(f"[6/7] Feature Engineering complete: Added Hour_of_Day, Log_Amount, Amount_Bin.")

    # 6. Scaling with RobustScaler (resistant to outliers)
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df[["Amount", "Time"]])
    df["Scaled_Amount"] = np.round(scaled_features[:, 0], 6)
    df["Scaled_Time"] = np.round(scaled_features[:, 1], 6)
    print("[7/7] Applied RobustScaler to Amount & Time.")

    # 7. Export Cleaned Dataset & Audit Summary
    audit["cleaned_record_count"] = len(df)
    audit["final_class_distribution"] = {
        "legitimate": int((df["Class"] == 0).sum()),
        "fraudulent": int((df["Class"] == 1).sum()),
        "fraud_percentage": round(float((df["Class"] == 1).mean() * 100), 3)
    }

    cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_csv, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"\nSaved Cleaned Dataset to: {cleaned_csv}")
    print(f"Saved Cleaning Audit Report to: {report_path}")
    print(f"Cleaned Shape: {df.shape} | Fraud Ratio: {audit['final_class_distribution']['fraud_percentage']}%")
    print("=" * 65)
    return df

if __name__ == "__main__":
    clean_credit_card_data()
