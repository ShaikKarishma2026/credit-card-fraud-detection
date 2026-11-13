"""
FastAPI Backend Application for Credit Card Fraud Detection
Provides RESTful APIs for real-time scoring, batch inference, SQL analytics,
model performance comparison, and historical prediction audits.
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Project path imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.db_manager import get_db
from models.prediction_service import get_predictor
from backend.schemas import TransactionFeatures, PredictionResponse, KPIResponse

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Enterprise Machine Learning REST API for Real-Time Transaction Fraud Prediction and Analytical Intelligence",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Credit Card Fraud Detection API",
        "status": "active",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "/api/health",
            "/api/analytics/kpis",
            "/api/analytics/queries",
            "/api/transactions",
            "/api/models/performance",
            "/api/predict",
            "/api/predict/batch",
            "/api/predict/upload-csv",
            "/api/predictions/history"
        ]
    }


@app.get("/api/health", tags=["System"])
def health_check():
    """Checks system health and database connectivity."""
    db = get_db()
    try:
        kpis = db.get_transaction_kpis()
        return {
            "status": "healthy",
            "database_connected": True,
            "database_engine": db.db_type,
            "total_transactions_in_db": kpis["total_transactions"],
            "models_loaded": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "database_connected": False, "error": str(e)}
        )


@app.get("/api/analytics/kpis", tags=["Analytics"])
def get_kpis():
    """Returns executive KPI metrics calculated directly from the database."""
    try:
        db = get_db()
        return db.get_transaction_kpis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/queries", tags=["Analytics"])
def get_analytical_queries():
    """Executes the 5 SQL analytical business queries requested in Phase 3."""
    try:
        db = get_db()
        return db.execute_analytics_queries()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transactions", tags=["Transactions"])
def list_transactions(
    limit: int = Query(100, ge=1, le=1000),
    class_filter: Optional[int] = Query(None, description="0 for Legit, 1 for Fraud"),
    min_amount: Optional[float] = Query(None, ge=0.0),
    max_amount: Optional[float] = Query(None, ge=0.0),
    hour_filter: Optional[int] = Query(None, ge=0, le=23)
):
    """Retrieves paginated and filtered transactions from the database."""
    try:
        db = get_db()
        # Defensive coercion for direct function calls vs HTTP queries
        lim_val = int(limit) if isinstance(limit, int) else 100
        cls_val = int(class_filter) if isinstance(class_filter, int) else None
        min_amt = float(min_amount) if isinstance(min_amount, (int, float)) else None
        max_amt = float(max_amount) if isinstance(max_amount, (int, float)) else None
        hr_val = int(hour_filter) if isinstance(hour_filter, int) else None

        records = db.get_recent_transactions(
            limit=lim_val,
            class_filter=cls_val,
            min_amount=min_amt,
            max_amount=max_amt,
            hour_filter=hr_val
        )
        return {
            "count": len(records),
            "filters": {
                "class_filter": cls_val,
                "min_amount": min_amt,
                "max_amount": max_amt,
                "hour_filter": hr_val
            },
            "transactions": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/performance", tags=["Machine Learning"])
def get_model_performance():
    """Returns comparative evaluation metrics for all 4 ML models."""
    comparison_file = BASE_DIR / "models" / "model_comparison.json"
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            return json.load(f)
    
    # Fallback to database
    db = get_db()
    saved = db.get_saved_model_results()
    if saved:
        return saved
    raise HTTPException(status_code=404, detail="Model performance results not found.")


@app.get("/api/models/feature-importance", tags=["Machine Learning"])
def get_feature_importance():
    """Returns feature importance ranking for tree-based models."""
    fi_file = BASE_DIR / "models" / "feature_importance.json"
    if fi_file.exists():
        with open(fi_file, "r") as f:
            return json.load(f)
    return {}


@app.get("/api/models/curves", tags=["Machine Learning"])
def get_model_curves():
    """Returns ROC and Precision-Recall curve coordinates for all models."""
    curves_file = BASE_DIR / "models" / "curves_data.json"
    if curves_file.exists():
        with open(curves_file, "r") as f:
            return json.load(f)
    return {}


@app.post("/api/predict", tags=["Prediction"])
def predict_transaction(tx: TransactionFeatures):
    """
    Scores a single credit card transaction in real time.
    Calculates probability, determines risk level, logs prediction in DB,
    and returns actionable fraud detection insights.
    """
    try:
        predictor = get_predictor()
        input_dict = tx.dict()
        model_name = input_dict.pop("model_name", "Random Forest")

        result = predictor.predict_single(input_dict, model_name=model_name)

        # Log prediction to database
        db = get_db()
        hour_val = int((tx.Time / 3600) % 24)
        pred_id = db.log_prediction(
            model_name=result["model_used"],
            predicted_class=result["predicted_class"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            amount=result["amount"],
            hour_of_day=hour_val,
            source="API Single Prediction"
        )
        result["prediction_id"] = pred_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/upload-csv", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query("Random Forest")
):
    """
    Upload a CSV file containing transactions and receive scored predictions,
    risk tiers, and aggregated batch statistics.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        predictor = get_predictor()
        scored_df = predictor.predict_batch(df, model_name=model_name)

        total_tx = len(scored_df)
        fraud_tx = int((scored_df["Predicted_Class"] == 1).sum())
        legit_tx = total_tx - fraud_tx
        high_risk_count = int((scored_df["Risk_Level"] == "High").sum())
        med_risk_count = int((scored_df["Risk_Level"] == "Medium").sum())

        return {
            "filename": file.filename,
            "model_used": model_name,
            "total_scored": total_tx,
            "summary": {
                "fraudulent_detected": fraud_tx,
                "legitimate_detected": legit_tx,
                "high_risk_count": high_risk_count,
                "medium_risk_count": med_risk_count,
                "fraud_rate_pct": round((fraud_tx / total_tx * 100), 2) if total_tx > 0 else 0.0
            },
            "sample_results": scored_df.head(100).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")


@app.get("/api/predictions/history", tags=["Prediction"])
def get_prediction_history(limit: int = Query(50, ge=1, le=200)):
    """Retrieves audit trail of recently scored transactions from the database."""
    try:
        db = get_db()
        return db.get_recent_predictions(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
