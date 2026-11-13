"""
Database Setup & Seeding Script for Credit Card Fraud Detection
Executes relational schema, creates tables, and populates 5,000 transactions
from dataset/creditcard_cleaned.csv into the transactions table.
"""

import uuid
from pathlib import Path
import pandas as pd
from sqlalchemy import text

from database.db_manager import get_db

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_CSV_PATH = BASE_DIR / "dataset" / "creditcard_cleaned.csv"
SCHEMA_SQL_PATH = BASE_DIR / "database" / "schema.sql"

def setup_and_seed_database():
    print("=" * 65)
    print("PHASE 3: DATABASE INITIALIZATION & INGESTION")
    print("=" * 65)

    db = get_db()
    print(f"Connecting to database engine: {db.db_type.upper()}")

    # 1. Create tables using engine
    with db.get_connection() as conn:
        # SQLite / MySQL compatible table creation
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                class INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name VARCHAR(100) NOT NULL,
                accuracy DECIMAL(6, 4) NOT NULL,
                precision_score DECIMAL(6, 4) NOT NULL,
                recall_score DECIMAL(6, 4) NOT NULL,
                f1_score DECIMAL(6, 4) NOT NULL,
                roc_auc DECIMAL(6, 4) NOT NULL,
                pr_auc DECIMAL(6, 4) NOT NULL,
                is_best_model BOOLEAN DEFAULT 0,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    print("[1/3] Created relational tables (transactions, fraud_predictions, model_results).")

    # 2. Ingest cleaned CSV transactions
    df = pd.read_csv(CLEANED_CSV_PATH)
    print(f"[2/3] Loaded {len(df)} transactions from {CLEANED_CSV_PATH} for database ingestion.")

    # Generate unique transaction IDs
    transaction_records = []
    for _, row in df.iterrows():
        tx_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
        record = {
            "transaction_id": tx_id,
            "time_step": float(row["Time"]),
            "hour_of_day": int(row["Hour_of_Day"]),
            "amount": float(row["Amount"]),
            "scaled_amount": float(row["Scaled_Amount"]),
            "scaled_time": float(row["Scaled_Time"]),
            "class": int(row["Class"])
        }
        for i in range(1, 29):
            record[f"v{i}"] = float(row[f"V{i}"])
        transaction_records.append(record)

    # Clear previous seed to allow clean restarts
    with db.get_connection() as conn:
        conn.execute(text("DELETE FROM transactions"))
        conn.commit()

        # Batch insert
        cols = ["transaction_id", "time_step", "hour_of_day", "amount", "scaled_amount", "scaled_time", "class"] + [f"v{i}" for i in range(1, 29)]
        col_names = ", ".join(cols)
        val_placeholders = ", ".join([f":{c}" for c in cols])
        insert_stmt = text(f"INSERT INTO transactions ({col_names}) VALUES ({val_placeholders})")
        
        # Execute in chunks of 500
        chunk_size = 500
        for i in range(0, len(transaction_records), chunk_size):
            chunk = transaction_records[i : i + chunk_size]
            conn.execute(insert_stmt, chunk)
        conn.commit()

    print(f"[3/3] Successfully ingested {len(transaction_records)} transactions into the database!")

    # Verify via KPI query
    kpis = db.get_transaction_kpis()
    print("\nDatabase Verification KPIs:")
    print(f"  - Total Transactions:       {kpis['total_transactions']}")
    print(f"  - Fraudulent Transactions:  {kpis['fraudulent_transactions']} ({kpis['fraud_detection_rate_pct']}%)")
    print(f"  - Legitimate Transactions:  {kpis['legitimate_transactions']}")
    print(f"  - Total Volume Value:       ${kpis['total_transaction_amount']:,.2f}")
    print(f"  - Average Transaction:      ${kpis['avg_transaction_amount']:.2f}")
    print(f"  - Average Fraud Amount:     ${kpis['avg_fraud_amount']:.2f}")
    print(f"  - Average Legit Amount:     ${kpis['avg_legit_amount']:.2f}")
    print("=" * 65)

if __name__ == "__main__":
    setup_and_seed_database()
