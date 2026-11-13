# Credit Card Fraud Exploratory Data Analysis (EDA) Report

## 1. Executive Summary
- **Total Portfolio Transactions:** 5,000
- **Legitimate Transactions:** 4,910 (98.20%)
- **Fraudulent Transactions:** 90 (1.80%)
- **Severe Class Imbalance:** 1:55 (1 fraud for every 55 legit transactions)

## 2. Key Statistical Insights
- **Class Imbalance:** Extreme class skew: Fraud comprises only 1.80% (90 cases) against 4910 legitimate transactions. Standard accuracy would yield ~98.2% by naively predicting all legitimate, necessitating Precision-Recall evaluation and SMOTE/class-weight balancing.

- **Amount Distribution:** Fraudulent transactions average $236.76, significantly higher than legitimate transactions ($57.03). Fraud exhibits a bimodal signature: micro card-testing charges (<$10) followed by larger burst amounts ($250-$2,000).

- **Circadian Velocity:** Fraud probability peaks during off-peak night windows (Hours 2:00 - 5:00, reaching 6.9%), when automated fraud scripts exploit delayed cardholder alerting.

- **PCA Discriminators:** Features V14, V12, V10 exhibit strong negative correlations with fraud, while V4, V11, V2 show strong positive correlations.
