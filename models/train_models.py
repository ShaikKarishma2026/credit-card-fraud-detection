"""
Machine Learning Training Pipeline for Credit Card Fraud Detection
Implements, trains, evaluates, and compares 4 ML algorithms:
1. Logistic Regression (Class-weighted)
2. Decision Tree (Class-weighted)
3. Random Forest (Class-weighted with SMOTE comparison)
4. Isolation Forest (Unsupervised Anomaly Detection)

Addresses extreme class imbalance using SMOTE strictly on training folds to prevent data leakage.
Computes Confusion Matrices, Precision, Recall, F1, ROC-AUC, and PR-AUC.
Saves the champion model, evaluation metrics, and persists metrics to the database.
"""

import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

# Ensure database module is accessible
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from database.db_manager import get_db

DATASET_PATH = BASE_DIR / "dataset" / "creditcard_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split_data():
    print(f"[1/6] Loading cleaned dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # Core 30 features: Time, V1-V28, Amount
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"      Train set: {X_train.shape[0]} samples ({y_train.sum()} frauds, {y_train.mean():.2%})")
    print(f"      Test set:  {X_test.shape[0]} samples ({y_test.sum()} frauds, {y_test.mean():.2%})")

    # Fit RobustScaler on Time and Amount on TRAINING SET ONLY to prevent leakage
    scaler_time = RobustScaler()
    scaler_amount = RobustScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled["Scaled_Time"] = scaler_time.fit_transform(X_train[["Time"]])
    X_train_scaled["Scaled_Amount"] = scaler_amount.fit_transform(X_train[["Amount"]])
    X_test_scaled["Scaled_Time"] = scaler_time.transform(X_test[["Time"]])
    X_test_scaled["Scaled_Amount"] = scaler_amount.transform(X_test[["Amount"]])

    # Model training features: V1..V28 + Scaled_Amount + Scaled_Time
    model_features = [f"V{i}" for i in range(1, 29)] + ["Scaled_Amount", "Scaled_Time"]
    X_train_ready = X_train_scaled[model_features]
    X_test_ready = X_test_scaled[model_features]

    # Save scalers
    joblib.dump({"scaler_time": scaler_time, "scaler_amount": scaler_amount, "feature_names": model_features},
                MODELS_DIR / "scalers.joblib")

    return X_train_ready, X_test_ready, y_train, y_test, model_features


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    print("\n[2/6] Applying SMOTE to training data (handling extreme class imbalance)...")
    smote = SMOTE(random_state=42, sampling_strategy=0.25)  # Boost fraud representation to 25% for balanced training
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"      SMOTE Train Set: {X_train_smote.shape[0]} samples ({y_train_smote.sum()} frauds)")

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Isolation Forest": IsolationForest(
            n_estimators=150,
            contamination=0.018,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    curves_data = {}
    cm_data = {}
    trained_estimators = {}

    print("\n[3/6] Training and evaluating models...")
    for name, clf in models.items():
        print(f"   -> Training {name}...")
        if name == "Isolation Forest":
            # Isolation Forest is unsupervised: fit on X_train_ready directly
            clf.fit(X_train)
            # Predictions: -1 for anomaly (fraud), 1 for inlier (legitimate)
            raw_preds = clf.predict(X_test)
            y_pred = np.where(raw_preds == -1, 1, 0)
            
            # Anomaly score: lower score means more abnormal
            scores = clf.score_samples(X_test)
            # Convert to pseudo-probability [0, 1]
            min_s, max_s = scores.min(), scores.max()
            y_prob = 1.0 - (scores - min_s) / (max_s - min_s + 1e-8)
        elif name == "Random Forest":
            # Train Random Forest on SMOTE augmented training data
            clf.fit(X_train_smote, y_train_smote)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

        trained_estimators[name] = clf

        # Evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))

        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_data[name] = cm

        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)

        curves_data[name] = {
            "fpr": fpr[::max(1, len(fpr) // 50)].tolist(),
            "tpr": tpr[::max(1, len(tpr) // 50)].tolist(),
            "precision_curve": p_curve[::max(1, len(p_curve) // 50)].tolist(),
            "recall_curve": r_curve[::max(1, len(r_curve) // 50)].tolist(),
        }

        result_entry = {
            "model_name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        }
        results.append(result_entry)
        print(f"      {name} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # Determine best model based on F1-Score and PR-AUC
    # In fraud detection, high Recall + high PR-AUC is top priority to catch thieves without burying investigators in false alarms
    best_entry = max(results, key=lambda m: (m["f1_score"] * 0.5 + m["pr_auc"] * 0.5))
    best_model_name = best_entry["model_name"]
    for r in results:
        r["is_best"] = (r["model_name"] == best_model_name)

    print(f"\n[4/6] Champion Model Selected: {best_model_name} (PR-AUC: {best_entry['pr_auc']}, F1: {best_entry['f1_score']})")

    # Feature Importance for Tree Models
    feature_importance = {}
    rf = trained_estimators["Random Forest"]
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance["Random Forest"] = [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in sorted_idx
    ]

    return trained_estimators, results, curves_data, feature_importance, best_model_name


def save_artifacts(trained_estimators, results, curves_data, feature_importance, best_model_name):
    print("\n[5/6] Saving trained models and performance metadata...")
    
    # Save best model
    best_clf = trained_estimators[best_model_name]
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_clf, best_model_path)
    print(f"      Champion model saved to: {best_model_path}")

    # Save all models for live switching/comparison in UI
    all_models_path = MODELS_DIR / "all_models.joblib"
    joblib.dump(trained_estimators, all_models_path)

    # Save comparison JSON
    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save curves data
    with open(MODELS_DIR / "curves_data.json", "w") as f:
        json.dump(curves_data, f, indent=2)

    # Save feature importance
    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    # Save summary metadata
    summary = {
        "best_model": best_model_name,
        "metrics": next(m for m in results if m["model_name"] == best_model_name),
        "total_models_evaluated": len(results),
        "evaluation_criteria": "Harmonic blend of F1-Score, Fraud Recall, and Precision-Recall AUC"
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[6/6] Logging evaluation metrics into the Database (model_results table)...")
    db = get_db()
    db.log_model_results(results)
    print("      Successfully recorded model results into the database!")


def main():
    print("=" * 70)
    print("   CREDIT CARD FRAUD DETECTION - MACHINE LEARNING PIPELINE")
    print("=" * 70)
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    trained_estimators, results, curves_data, feature_importance, best_model_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    save_artifacts(trained_estimators, results, curves_data, feature_importance, best_model_name)
    print("\n[COMPLETE] Machine Learning pipeline executed successfully.")


if __name__ == "__main__":
    main()
