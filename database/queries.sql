-- Credit Card Fraud Detection Analytical SQL Queries
-- Database: credit_card_fraud_db

-- Query 1: Total Transactions
SELECT COUNT(*) AS total_transactions 
FROM transactions;

-- Query 2: Fraudulent vs Legitimate Transaction Counts
SELECT 
    CASE WHEN class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_type,
    COUNT(*) AS transaction_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) AS percentage
FROM transactions
GROUP BY class;

-- Query 3: Detailed Financial Amounts by Transaction Class
SELECT 
    CASE WHEN class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_type,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS average_amount,
    ROUND(MIN(amount), 2) AS minimum_amount,
    ROUND(MAX(amount), 2) AS maximum_amount,
    ROUND(STDDEV(amount), 2) AS std_amount
FROM transactions
GROUP BY class;

-- Query 4: Exact Overall Fraud Percentage
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN class = 0 THEN 1 ELSE 0 END) AS legit_transactions,
    ROUND(SUM(CASE WHEN class = 1 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 3) AS fraud_percentage
FROM transactions;

-- Query 5: Hourly Fraud Velocity Analysis (Peak Risk Windows)
SELECT 
    hour_of_day,
    COUNT(*) AS total_volume,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN class = 1 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS hourly_fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_hourly_amount
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_count DESC;
