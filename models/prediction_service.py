"""
Prediction Service Module for Credit Card Fraud Detection
Provides real-time single-transaction inference and high-throughput batch scoring.
Includes risk categorization, explainable feature contributions, and recommendations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

class FraudPredictor:
    def __init__(self):
        self.scalers_path = MODELS_DIR / "scalers.joblib"
        self.all_models_path = MODELS_DIR / "all_models.joblib"
        self.best_model_path = MODELS_DIR / "best_model.joblib"

        if not self.scalers_path.exists():
            raise FileNotFoundError("Scaler artifacts not found. Please train models first.")

        scaler_dict = joblib.load(self.scalers_path)
        self.scaler_time = scaler_dict["scaler_time"]
        self.scaler_amount = scaler_dict["scaler_amount"]
        self.feature_names = scaler_dict["feature_names"]

        if self.all_models_path.exists():
            self.models = joblib.load(self.all_models_path)
        else:
            self.models = {"Random Forest": joblib.load(self.best_model_path)}

        self.default_model_name = "Random Forest" if "Random Forest" in self.models else list(self.models.keys())[0]

    def _prepare_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Scales Time and Amount and orders columns to match training features."""
        df = df_input.copy()

        # Handle missing columns if any
        if "Time" not in df.columns:
            df["Time"] = 0.0
        if "Amount" not in df.columns:
            df["Amount"] = 0.0

        for i in range(1, 29):
            col = f"V{i}"
            if col not in df.columns:
                df[col] = 0.0

        df["Scaled_Time"] = self.scaler_time.transform(df[["Time"]])
        df["Scaled_Amount"] = self.scaler_amount.transform(df[["Amount"]])

        return df[self.feature_names]

    def predict_single(
        self,
        features: Dict[str, float],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scores a single transaction dictionary.
        Returns prediction, probability, risk level, and explainable feature impacts.
        """
        model_name = model_name or self.default_model_name
        if model_name not in self.models:
            model_name = self.default_model_name

        model = self.models[model_name]
        df_input = pd.DataFrame([features])
        X = self._prepare_features(df_input)

        if model_name == "Isolation Forest":
            raw_pred = model.predict(X)[0]
            pred_class = 1 if raw_pred == -1 else 0
            score = model.score_samples(X)[0]
            # normalize anomaly score
            prob = float(np.clip(1.0 - (score + 0.6) / 0.8, 0.0, 1.0))
        else:
            pred_class = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0, 1])

        # Determine Risk Level
        if prob >= 0.70:
            risk_level = "High"
            action = "Decline & Flag for Fraud Investigation"
        elif prob >= 0.35:
            risk_level = "Medium"
            action = "Hold for 2FA / Manual Verification"
        else:
            risk_level = "Low"
            action = "Approve Transaction"

        # Explainability: Identify top anomalous PCA deviations
        # Typical fraud PCA markers in European dataset: highly negative V14, V12, V10, V17; positive V4, V11
        feature_deviations = []
        for i in range(1, 29):
            col = f"V{i}"
            val = float(features.get(col, 0.0))
            if abs(val) > 2.0:
                direction = "abnormally high" if val > 0 else "abnormally negative"
                feature_deviations.append({
                    "feature": col,
                    "value": round(val, 3),
                    "significance": f"{direction} (variance: {round(val, 2)})"
                })

        feature_deviations.sort(key=lambda x: abs(x["value"]), reverse=True)

        return {
            "model_used": model_name,
            "prediction": "Fraudulent" if pred_class == 1 else "Legitimate",
            "predicted_class": pred_class,
            "fraud_probability": round(prob, 4),
            "fraud_probability_pct": f"{round(prob * 100, 2)}%",
            "risk_level": risk_level,
            "recommended_action": action,
            "amount": features.get("Amount", 0.0),
            "time": features.get("Time", 0.0),
            "key_deviations": feature_deviations[:5]
        }

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Scores a DataFrame of transactions in vectorized batch mode."""
        model_name = model_name or self.default_model_name
        if model_name not in self.models:
            model_name = self.default_model_name

        model = self.models[model_name]
        X = self._prepare_features(df_batch)

        df_out = df_batch.copy()

        if model_name == "Isolation Forest":
            raw_preds = model.predict(X)
            preds = np.where(raw_preds == -1, 1, 0)
            scores = model.score_samples(X)
            probs = np.clip(1.0 - (scores + 0.6) / 0.8, 0.0, 1.0)
        else:
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

        df_out["Predicted_Class"] = preds
        df_out["Prediction"] = np.where(preds == 1, "Fraudulent", "Legitimate")
        df_out["Fraud_Probability"] = np.round(probs, 4)
        df_out["Risk_Level"] = np.where(probs >= 0.70, "High", np.where(probs >= 0.35, "Medium", "Low"))
        df_out["Model_Used"] = model_name

        return df_out

# Global predictor instance
_predictor = None

def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor
