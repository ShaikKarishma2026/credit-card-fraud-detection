"""
Database Manager Module for Credit Card Fraud Detection
Supports MySQL (via PyMySQL/SQLAlchemy) with seamless SQLite fallback.
Executes analytical SQL queries, handles data insertion, and logs predictions and model metrics.
"""

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent.parent
DB_SQLITE_PATH = BASE_DIR / "database" / "credit_card_fraud.db"

# MySQL connection string format: mysql+pymysql://<user>:<password>@<host>:<port>/<dbname>
MYSQL_URL = os.environ.get("DATABASE_URL")

def get_engine():
    if MYSQL_URL and MYSQL_URL.startswith("mysql"):
        try:
            engine = create_engine(MYSQL_URL, echo=False)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine, "mysql"
        except Exception as e:
            print(f"[Warning] MySQL connection failed ({e}). Falling back to SQLite.")
    
    # SQLite Fallback Engine
    sqlite_url = f"sqlite:///{DB_SQLITE_PATH}"
    engine = create_engine(sqlite_url, echo=False)
    return engine, "sqlite"

class DatabaseManager:
    def __init__(self):
        self.engine, self.db_type = get_engine()

    def get_connection(self):
        return self.engine.connect()

    def get_transaction_kpis(self) -> Dict[str, Any]:
        """Calculates executive KPI cards directly from the database."""
        with self.get_connection() as conn:
            # Query overall volume and fraud counts
            q1 = text("""
                SELECT 
                    COUNT(*) AS total_transactions,
                    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
                    SUM(CASE WHEN class = 0 THEN 1 ELSE 0 END) AS legit_transactions,
                    AVG(amount) AS avg_amount,
                    SUM(amount) AS total_amount,
                    AVG(CASE WHEN class = 1 THEN amount ELSE NULL END) AS avg_fraud_amount,
                    AVG(CASE WHEN class = 0 THEN amount ELSE NULL END) AS avg_legit_amount
                FROM transactions
            """)
            res = conn.execute(q1).fetchone()
            
            total = int(res[0] or 0)
            fraud = int(res[1] or 0)
            legit = int(res[2] or 0)
            avg_amt = float(res[3] or 0.0)
            total_amt = float(res[4] or 0.0)
            avg_fraud_amt = float(res[5] or 0.0)
            avg_legit_amt = float(res[6] or 0.0)

            fraud_rate = (fraud / total * 100.0) if total > 0 else 0.0

            return {
                "total_transactions": total,
                "fraudulent_transactions": fraud,
                "legitimate_transactions": legit,
                "fraud_detection_rate_pct": round(fraud_rate, 3),
                "avg_transaction_amount": round(avg_amt, 2),
                "total_transaction_amount": round(total_amt, 2),
                "avg_fraud_amount": round(avg_fraud_amt, 2),
                "avg_legit_amount": round(avg_legit_amt, 2),
                "db_engine": self.db_type
            }

    def execute_analytics_queries(self) -> Dict[str, Any]:
        """Executes the 5 analytical SQL queries requested in Phase 3."""
        with self.get_connection() as conn:
            # 1. Total count
            total_trans = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar() or 0

            # 2. Fraud vs Legit breakdown
            q_breakdown = text("""
                SELECT 
                    CASE WHEN class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_type,
                    COUNT(*) AS transaction_count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) AS percentage
                FROM transactions
                GROUP BY class
            """)
            breakdown_rows = [dict(r._mapping) for r in conn.execute(q_breakdown).fetchall()]

            # 3. Transaction amounts by class
            q_amounts = text("""
                SELECT 
                    CASE WHEN class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_type,
                    ROUND(SUM(amount), 2) AS total_amount,
                    ROUND(AVG(amount), 2) AS average_amount,
                    ROUND(MIN(amount), 2) AS minimum_amount,
                    ROUND(MAX(amount), 2) AS maximum_amount
                FROM transactions
                GROUP BY class
            """)
            amounts_rows = [dict(r._mapping) for r in conn.execute(q_amounts).fetchall()]

            # 4. Hourly fraud distribution
            q_hourly = text("""
                SELECT 
                    hour_of_day,
                    COUNT(*) AS total_volume,
                    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_count,
                    ROUND(AVG(amount), 2) AS avg_amount
                FROM transactions
                GROUP BY hour_of_day
                ORDER BY hour_of_day ASC
            """)
            hourly_rows = [dict(r._mapping) for r in conn.execute(q_hourly).fetchall()]

            return {
                "total_transactions": total_trans,
                "class_breakdown": breakdown_rows,
                "amount_analysis": amounts_rows,
                "hourly_distribution": hourly_rows
            }

    def get_recent_transactions(
        self,
        limit: int = 100,
        class_filter: Optional[int] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        hour_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves paginated and filtered transactions."""
        query_str = "SELECT * FROM transactions WHERE 1=1"
        params = {}

        if class_filter is not None and class_filter in [0, 1]:
            query_str += " AND class = :cls"
            params["cls"] = class_filter

        if min_amount is not None:
            query_str += " AND amount >= :min_amt"
            params["min_amt"] = min_amount

        if max_amount is not None:
            query_str += " AND amount <= :max_amt"
            params["max_amt"] = max_amount

        if hour_filter is not None:
            query_str += " AND hour_of_day = :hr"
            params["hr"] = hour_filter

        query_str += " ORDER BY id DESC LIMIT :lim"
        params["lim"] = limit

        with self.get_connection() as conn:
            rows = conn.execute(text(query_str), params).fetchall()
            return [dict(r._mapping) for r in rows]

    def log_prediction(
        self,
        model_name: str,
        predicted_class: int,
        fraud_probability: float,
        risk_level: str,
        amount: float,
        hour_of_day: int,
        transaction_ref: Optional[str] = None,
        source: str = "Manual Entry"
    ) -> str:
        """Logs a model prediction into the fraud_predictions table."""
        pred_id = f"PRED-{uuid.uuid4().hex[:8].upper()}"
        if not transaction_ref:
            transaction_ref = f"TX-{uuid.uuid4().hex[:8].upper()}"

        insert_sql = text("""
            INSERT INTO fraud_predictions (
                prediction_id, transaction_ref, model_name, predicted_class,
                fraud_probability, risk_level, amount, hour_of_day, prediction_source
            ) VALUES (
                :pred_id, :tx_ref, :model, :cls, :prob, :risk, :amt, :hr, :src
            )
        """)

        with self.get_connection() as conn:
            conn.execute(insert_sql, {
                "pred_id": pred_id,
                "tx_ref": transaction_ref,
                "model": model_name,
                "cls": int(predicted_class),
                "prob": float(round(fraud_probability, 4)),
                "risk": risk_level,
                "amt": float(round(amount, 2)),
                "hr": int(hour_of_day),
                "src": source
            })
            conn.commit()

        return pred_id

    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent fraud predictions."""
        with self.get_connection() as conn:
            rows = conn.execute(text("""
                SELECT * FROM fraud_predictions ORDER BY id DESC LIMIT :lim
            """), {"lim": limit}).fetchall()
            return [dict(r._mapping) for r in rows]

    def log_model_results(self, results: List[Dict[str, Any]]):
        """Logs comparative evaluation metrics for all ML models."""
        with self.get_connection() as conn:
            conn.execute(text("DELETE FROM model_results"))
            insert_sql = text("""
                INSERT INTO model_results (
                    model_name, accuracy, precision_score, recall_score, f1_score, roc_auc, pr_auc, is_best_model
                ) VALUES (
                    :name, :acc, :prec, :rec, :f1, :roc, :pr, :best
                )
            """)
            for m in results:
                conn.execute(insert_sql, {
                    "name": m["model_name"],
                    "acc": float(m["accuracy"]),
                    "prec": float(m["precision"]),
                    "rec": float(m["recall"]),
                    "f1": float(m["f1_score"]),
                    "roc": float(m["roc_auc"]),
                    "pr": float(m["pr_auc"]),
                    "best": bool(m.get("is_best", False))
                })
            conn.commit()

    def get_saved_model_results(self) -> List[Dict[str, Any]]:
        """Retrieves model comparison metrics from database."""
        with self.get_connection() as conn:
            rows = conn.execute(text("SELECT * FROM model_results ORDER BY pr_auc DESC")).fetchall()
            return [dict(r._mapping) for r in rows]

# Singleton instance
_db_manager = None

def get_db() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
