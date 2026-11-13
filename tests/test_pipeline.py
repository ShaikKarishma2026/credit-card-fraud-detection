"""
End-to-End Test Suite for Credit Card Fraud Detection Platform
Validates dataset integrity, database queries, ML models, and FastAPI routes.
"""

import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.db_manager import get_db
from models.prediction_service import get_predictor
from backend.main import app, health_check, get_kpis, get_analytical_queries, list_transactions, predict_transaction
from backend.schemas import TransactionFeatures


def test_dataset_exists_and_valid():
    """Verify cleaned dataset exists and has target column."""
    csv_path = BASE_DIR / "dataset" / "creditcard_cleaned.csv"
    assert csv_path.exists(), "Cleaned dataset CSV does not exist."
    df = pd.read_csv(csv_path)
    assert len(df) == 5000, f"Expected 5000 records, got {len(df)}"
    assert "Class" in df.columns, "Class target column missing."
    assert df["Class"].sum() > 0, "No positive fraud cases found in dataset."


def test_database_connectivity_and_kpis():
    """Verify database connection, transaction count, and KPI calculations."""
    db = get_db()
    kpis = db.get_transaction_kpis()
    assert kpis["total_transactions"] == 5000
    assert kpis["fraudulent_transactions"] > 0
    assert kpis["legitimate_transactions"] > 0
    assert kpis["avg_transaction_amount"] > 0


def test_database_analytical_queries():
    """Verify execution of the 5 requested SQL analytics queries."""
    db = get_db()
    results = db.execute_analytics_queries()
    assert "total_transactions" in results
    assert "class_breakdown" in results
    assert len(results["class_breakdown"]) == 2
    assert "amount_analysis" in results
    assert "hourly_distribution" in results
    assert len(results["hourly_distribution"]) == 24


def test_ml_predictor_single_inference():
    """Verify single transaction scoring with explainability and risk tiering."""
    predictor = get_predictor()
    # Test normal transaction
    legit_tx = {"Time": 36000.0, "Amount": 35.0, "V14": 0.5, "V12": -0.2, "V4": -0.1}
    legit_res = predictor.predict_single(legit_tx, model_name="Random Forest")
    assert legit_res["predicted_class"] in [0, 1]
    assert 0.0 <= legit_res["fraud_probability"] <= 1.0
    assert legit_res["risk_level"] in ["Low", "Medium", "High"]
    assert "recommended_action" in legit_res

    # Test extreme anomaly transaction
    fraud_tx = {"Time": 7200.0, "Amount": 1200.0, "V14": -7.5, "V12": -5.5, "V4": 4.5}
    fraud_res = predictor.predict_single(fraud_tx, model_name="Random Forest")
    assert fraud_res["fraud_probability"] > 0.0


def test_ml_predictor_batch_inference():
    """Verify batch dataframe scoring."""
    predictor = get_predictor()
    sample_df = pd.DataFrame([
        {"Time": 100, "Amount": 50.0, "V14": 0.1, "V12": 0.0, "V4": 0.0},
        {"Time": 200, "Amount": 900.0, "V14": -6.0, "V12": -4.0, "V4": 3.0}
    ])
    scored = predictor.predict_batch(sample_df, model_name="Random Forest")
    assert len(scored) == 2
    assert "Predicted_Class" in scored.columns
    assert "Fraud_Probability" in scored.columns
    assert "Risk_Level" in scored.columns


def test_backend_routes():
    """Verify backend functions operate without exceptions."""
    h = health_check()
    assert h["status"] == "healthy"
    assert h["database_connected"] is True

    k = get_kpis()
    assert k["total_transactions"] == 5000

    q = get_analytical_queries()
    assert len(q["class_breakdown"]) == 2

    tx_page = list_transactions(limit=5)
    assert tx_page["count"] == 5

    tf = TransactionFeatures(Time=5000, Amount=120.0, V14=-3.0, V4=2.0)
    p = predict_transaction(tf)
    assert "prediction_id" in p
    assert p["prediction_id"].startswith("PRED-")


if __name__ == "__main__":
    print("Running end-to-end tests...")
    test_dataset_exists_and_valid()
    print("✓ Dataset validation passed.")
    test_database_connectivity_and_kpis()
    print("✓ Database connectivity and KPI validation passed.")
    test_database_analytical_queries()
    print("✓ SQL analytical queries passed.")
    test_ml_predictor_single_inference()
    print("✓ ML single inference passed.")
    test_ml_predictor_batch_inference()
    print("✓ ML batch inference passed.")
    test_backend_routes()
    print("✓ Backend endpoints passed.")
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
