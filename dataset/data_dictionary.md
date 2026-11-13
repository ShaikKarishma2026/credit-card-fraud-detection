# Credit Card Fraud Detection Data Dictionary

## Overview
- **Domain:** Financial Fraud Detection / Risk Management
- **Dataset Source:** Modeled on the European Credit Card Fraud Detection Dataset (Worldline & ULB Machine Learning Group)
- **Timeframe:** 48 hours of credit card transactions (172,800 seconds)
- **Target Variable:** `Class` (`0` = Legitimate, `1` = Fraudulent)
- **Class Imbalance:** Extreme imbalance with ~1.8% fraudulent transactions (representative of real-world banking transaction volumes)

---

## Feature Explanations

| Feature Name | Data Type | Units / Range | Description & Fraud Detection Relevance |
| :--- | :--- | :--- | :--- |
| **`Time`** | Float | 0.0 – 172,800.0 | Number of seconds elapsed between the current transaction and the first transaction in the dataset. Used to derive cyclic temporal features such as `Hour_of_Day`. |
| **`V1` – `V28`** | Float | Real numbers ($\sim -15$ to $+10$) | Principal Component Analysis (PCA) components. Due to user privacy and confidential banking data protection, original transaction attributes (cardholder metadata, merchant category code, IP geolocation, terminal ID, device fingerprint) are transformed into orthogonal principal components. |
| **`V14`, `V12`, `V10`, `V17`** | Float | Typically negative ($\sim -7$) for fraud | **Primary Negative Discriminators:** In European card transaction patterns, these components show strong negative divergence during fraudulent activity. |
| **`V4`, `V11`, `V2`** | Float | Typically positive ($\sim +4$) for fraud | **Primary Positive Discriminators:** These components show strong positive divergence for fraudulent transactions. |
| **`Amount`** | Float | $0.50 – $4,800.00 | Transaction value in USD/EUR. Can be used for card-testing micro-transactions (<$10) or high-ticket unauthorized charges. Heavily right-skewed. |
| **`Class`** | Integer | 0 or 1 | **Target Label:** `0` denotes a legitimate, authorized transaction; `1` denotes a fraudulent, unauthorized transaction. |

---

## Engineered Features (Added in Phase 2 Cleaning)

| Engineered Feature | Calculation | Rationale |
| :--- | :--- | :--- |
| **`Hour_of_Day`** | `(Time // 3600) % 24` | Captures circadian transaction velocity (night vs business hours). |
| **`Log_Amount`** | `np.log1p(Amount)` | Normalizes the heavy right skew of transaction values for linear models. |
| **`Amount_Scaled`** | `RobustScaler(Amount)` | Reduces the disruptive effect of extreme spending outliers. |
| **`Time_Scaled`** | `RobustScaler(Time)` | Standardizes the elapsed seconds across the 48-hour window. |
| **`Amount_Bin`** | Categorical (`Low`, `Medium`, `High`, `Very High`) | Discretizes spending for risk segmentation. |
