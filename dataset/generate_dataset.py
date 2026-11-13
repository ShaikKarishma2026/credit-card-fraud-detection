"""
Credit Card Fraud Detection Dataset Generator & Loader
Models the European Credit Card Fraud Detection Dataset:
Features: Time, V1-V28 (PCA components), Amount, Class (0 = Legit, 1 = Fraud)
Incorporates realistic fraud signatures, extreme class imbalance, and raw data anomalies
to rigorously exercise data cleaning and preprocessing.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = BASE_DIR / "dataset" / "creditcard_raw.csv"

def generate_credit_card_dataset(output_path: Path = RAW_CSV_PATH, total_samples: int = 5000, fraud_ratio: float = 0.018):
    print("=" * 65)
    print("PHASE 1: CREDIT CARD FRAUD DATASET GENERATION & COLLECTION")
    print("=" * 65)

    n_fraud = int(total_samples * fraud_ratio)
    n_legit = total_samples - n_fraud

    print(f"Generating {total_samples} total transactions:")
    print(f"  - Legitimate Transactions (Class = 0): {n_legit} ({n_legit/total_samples*100:.2f}%)")
    print(f"  - Fraudulent Transactions (Class = 1): {n_fraud} ({n_fraud/total_samples*100:.2f}%)")

    # 1. Time Feature (seconds elapsed across 48 hours: 0 to 172800)
    # Legitimate transactions peak during daytime hours (diurnal cycle)
    time_legit = np.random.exponential(scale=35, size=n_legit).cumsum()
    time_legit = (time_legit / time_legit.max()) * 172800
    
    # Fraud transactions occur more frequently during night hours (e.g. hours 1-6 and 22-24)
    time_fraud = np.random.uniform(0, 172800, size=n_fraud)
    # skew slightly towards night intervals
    night_mask = (time_fraud % 86400 < 21600) | (time_fraud % 86400 > 79200)
    # replace 40% with night values
    for i in range(len(time_fraud)):
        if np.random.rand() < 0.35:
            day_offset = int(time_fraud[i] // 86400) * 86400
            time_fraud[i] = day_offset + np.random.uniform(3600, 18000)

    # 2. Amount Feature (USD)
    # Legitimate: right-skewed log-normal distribution (median ~$25, mean ~$88)
    amount_legit = np.random.lognormal(mean=3.2, sigma=1.3, size=n_legit)
    amount_legit = np.clip(np.round(amount_legit, 2), 0.50, 4800.0)

    # Fraudulent: dual mode (card testing $1-$10, or high-value exploitation $250-$2200)
    amount_fraud_micro = np.random.uniform(0.99, 9.99, size=int(n_fraud * 0.40))
    amount_fraud_macro = np.random.lognormal(mean=5.5, sigma=0.9, size=n_fraud - len(amount_fraud_micro))
    amount_fraud = np.concatenate([amount_fraud_micro, amount_fraud_macro])
    np.random.shuffle(amount_fraud)
    amount_fraud = np.clip(np.round(amount_fraud, 2), 0.99, 3500.0)

    # 3. PCA Components V1 - V28
    # In European credit card data:
    # V10, V12, V14, V16, V17 are heavily NEGATIVE for fraud.
    # V2, V4, V11 are heavily POSITIVE for fraud.
    # Other V features follow standard standard normal distribution ~ N(0, 1) with slight variance.
    v_features_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    v_features_fraud = np.random.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
    # Key Separating PCA Signals:
    # V14: Strong negative signal (mean ~ -7.5)
    v_features_fraud[:, 13] = np.random.normal(loc=-7.2, scale=2.1, size=n_fraud)
    # V12: Strong negative signal (mean ~ -6.0)
    v_features_fraud[:, 11] = np.random.normal(loc=-6.1, scale=1.9, size=n_fraud)
    # V10: Strong negative signal (mean ~ -5.5)
    v_features_fraud[:, 9] = np.random.normal(loc=-5.4, scale=1.8, size=n_fraud)
    # V17: Strong negative signal (mean ~ -6.8)
    v_features_fraud[:, 16] = np.random.normal(loc=-6.5, scale=2.2, size=n_fraud)
    # V4: Strong positive signal (mean ~ +4.5)
    v_features_fraud[:, 3] = np.random.normal(loc=4.6, scale=1.5, size=n_fraud)
    # V11: Strong positive signal (mean ~ +3.8)
    v_features_fraud[:, 10] = np.random.normal(loc=3.9, scale=1.4, size=n_fraud)
    # V2: Strong positive signal (mean ~ +3.2)
    v_features_fraud[:, 1] = np.random.normal(loc=3.3, scale=1.6, size=n_fraud)

    # Combine Legit and Fraud
    time_all = np.concatenate([time_legit, time_fraud])
    amount_all = np.concatenate([amount_legit, amount_fraud])
    v_all = np.vstack([v_features_legit, v_features_fraud])
    class_all = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Shuffle
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    time_all = time_all[indices]
    amount_all = amount_all[indices]
    v_all = v_all[indices]
    class_all = class_all[indices]

    # Create DataFrame
    data_dict = {
        "Time": np.round(time_all, 1),
    }
    for i in range(1, 29):
        data_dict[f"V{i}"] = np.round(v_all[:, i - 1], 6)
    data_dict["Amount"] = np.round(amount_all, 2)
    data_dict["Class"] = class_all

    df = pd.DataFrame(data_dict)

    # 4. Introduce Raw Anomalies for Cleaning Phase
    # Duplicate rows (8 duplicates)
    dup_samples = df.sample(n=8, random_state=42)
    df = pd.concat([df, dup_samples], ignore_index=True)

    # Missing values (nulls in 12 records)
    null_idx = np.random.choice(df.index, size=12, replace=False)
    for idx in null_idx[:6]:
        df.loc[idx, "Amount"] = np.nan
    for idx in null_idx[6:]:
        df.loc[idx, "V1"] = np.nan

    # Extreme Outlier Amount (e.g. entry error $99,999.00)
    df.loc[15, "Amount"] = 99999.99

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved raw dataset with {len(df)} records and {df.shape[1]} features to:")
    print(f"-> {output_path}")

    # Display dataset info
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", list(df.columns[:8]), "...", list(df.columns[-3:]))
    print("\nClass Breakdown:\n", df["Class"].value_counts())
    print("\nSample Records:")
    print(df[["Time", "V1", "V4", "V12", "V14", "Amount", "Class"]].head(3))
    print("=" * 65)
    return df

if __name__ == "__main__":
    generate_credit_card_dataset()
