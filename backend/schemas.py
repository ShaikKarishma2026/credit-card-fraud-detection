"""
Pydantic Schemas for FastAPI Backend
Defines strict request and response validation for single transaction prediction,
batch inference, analytics metrics, and database records.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    Time: float = Field(0.0, description="Seconds elapsed since first transaction")
    Amount: float = Field(0.0, ge=0.0, description="Transaction amount in currency")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    model_name: Optional[str] = Field("Random Forest", description="Model algorithm to use")


class PredictionResponse(BaseModel):
    prediction_id: Optional[str] = None
    model_used: str
    prediction: str
    predicted_class: int
    fraud_probability: float
    fraud_probability_pct: str
    risk_level: str
    recommended_action: str
    amount: float
    time: float
    key_deviations: List[Dict[str, Any]]


class KPIResponse(BaseModel):
    total_transactions: int
    fraudulent_transactions: int
    legitimate_transactions: int
    fraud_detection_rate_pct: float
    avg_transaction_amount: float
    total_transaction_amount: float
    avg_fraud_amount: float
    avg_legit_amount: float
    db_engine: str


class ModelMetric(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    is_best: bool
