-- Credit Card Fraud Detection Database Schema
-- Target Database: credit_card_fraud_db (MySQL / MariaDB / ANSI SQL compliant)

CREATE DATABASE IF NOT EXISTS credit_card_fraud_db;
USE credit_card_fraud_db;

-- 1. Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    transaction_id VARCHAR(36) NOT NULL UNIQUE,
    time_step DECIMAL(10, 1) NOT NULL,
    hour_of_day INT NOT NULL,
    v1 DECIMAL(10, 6) NOT NULL,
    v2 DECIMAL(10, 6) NOT NULL,
    v3 DECIMAL(10, 6) NOT NULL,
    v4 DECIMAL(10, 6) NOT NULL,
    v5 DECIMAL(10, 6) NOT NULL,
    v6 DECIMAL(10, 6) NOT NULL,
    v7 DECIMAL(10, 6) NOT NULL,
    v8 DECIMAL(10, 6) NOT NULL,
    v9 DECIMAL(10, 6) NOT NULL,
    v10 DECIMAL(10, 6) NOT NULL,
    v11 DECIMAL(10, 6) NOT NULL,
    v12 DECIMAL(10, 6) NOT NULL,
    v13 DECIMAL(10, 6) NOT NULL,
    v14 DECIMAL(10, 6) NOT NULL,
    v15 DECIMAL(10, 6) NOT NULL,
    v16 DECIMAL(10, 6) NOT NULL,
    v17 DECIMAL(10, 6) NOT NULL,
    v18 DECIMAL(10, 6) NOT NULL,
    v19 DECIMAL(10, 6) NOT NULL,
    v20 DECIMAL(10, 6) NOT NULL,
    v21 DECIMAL(10, 6) NOT NULL,
    v22 DECIMAL(10, 6) NOT NULL,
    v23 DECIMAL(10, 6) NOT NULL,
    v24 DECIMAL(10, 6) NOT NULL,
    v25 DECIMAL(10, 6) NOT NULL,
    v26 DECIMAL(10, 6) NOT NULL,
    v27 DECIMAL(10, 6) NOT NULL,
    v28 DECIMAL(10, 6) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    scaled_amount DECIMAL(10, 6) NOT NULL,
    scaled_time DECIMAL(10, 6) NOT NULL,
    class INT NOT NULL CHECK (class IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Fraud Predictions Table
CREATE TABLE IF NOT EXISTS fraud_predictions (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    prediction_id VARCHAR(36) NOT NULL UNIQUE,
    transaction_ref VARCHAR(36),
    model_name VARCHAR(100) NOT NULL,
    predicted_class INT NOT NULL,
    fraud_probability DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    hour_of_day INT NOT NULL,
    prediction_source VARCHAR(30) DEFAULT 'Manual Entry',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Model Results Table
CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(100) NOT NULL,
    accuracy DECIMAL(6, 4) NOT NULL,
    precision_score DECIMAL(6, 4) NOT NULL,
    recall_score DECIMAL(6, 4) NOT NULL,
    f1_score DECIMAL(6, 4) NOT NULL,
    roc_auc DECIMAL(6, 4) NOT NULL,
    pr_auc DECIMAL(6, 4) NOT NULL,
    is_best_model BOOLEAN DEFAULT FALSE,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for lightning fast analytics
CREATE INDEX IF NOT EXISTS idx_trans_class ON transactions(class);
CREATE INDEX IF NOT EXISTS idx_trans_amount ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_trans_hour ON transactions(hour_of_day);
CREATE INDEX IF NOT EXISTS idx_pred_risk ON fraud_predictions(risk_level);
