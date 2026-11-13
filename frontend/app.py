"""
Credit Card Fraud Detection - Enterprise Streamlit Dashboard
An interactive Data Science & Machine Learning platform for fraud intelligence,
real-time transaction scoring, model benchmarking, and SQL transaction forensics.
"""

import io
import json
import os
import sys
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup path to import backend & database modules
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.db_manager import get_db
from models.prediction_service import get_predictor

# Custom CSS for executive look and feel
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-fraud {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-legit {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #dc2626;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    return get_predictor()


@st.cache_data(ttl=60)
def load_db_kpis():
    db = get_db()
    return db.get_transaction_kpis()


@st.cache_data(ttl=60)
def load_db_analytics():
    db = get_db()
    return db.execute_analytics_queries()


@st.cache_data
def load_model_comparison():
    path = BASE_DIR / "models" / "model_comparison.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


@st.cache_data
def load_feature_importance():
    path = BASE_DIR / "models" / "feature_importance.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=64)
st.sidebar.title("FraudShield AI")
st.sidebar.caption("Enterprise Fraud Analytics Platform")

section = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview",
        "2. Transaction Analytics",
        "3. Fraud Detection",
        "4. EDA Visualizations",
        "5. Machine Learning Performance",
        "6. Transaction History",
        "7. Prediction Results"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("System Info")
kpi_data = load_db_kpis()
st.sidebar.write(f"**Database:** {kpi_data['db_engine'].upper()}")
st.sidebar.write(f"**Indexed Transactions:** {kpi_data['total_transactions']:,}")
st.sidebar.write(f"**Fraud Rate:** {kpi_data['fraud_detection_rate_pct']}%")


# ==============================================================
# SECTION 1: OVERVIEW
# ==============================================================
if section == "1. Overview":
    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.markdown("Real-time monitoring, SQL transaction forensics, and Machine Learning fraud intelligence.")

    # KPI Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric(
            label="Total Transactions",
            value=f"{kpi_data['total_transactions']:,}",
            help="Total transaction records ingested into database"
        )
    with k2:
        st.metric(
            label="Fraudulent Transactions",
            value=f"{kpi_data['fraudulent_transactions']:,}",
            delta=f"{kpi_data['fraud_detection_rate_pct']}% of total",
            delta_color="inverse"
        )
    with k3:
        st.metric(
            label="Legitimate Transactions",
            value=f"{kpi_data['legitimate_transactions']:,}",
            delta=f"{round(100 - kpi_data['fraud_detection_rate_pct'], 1)}%"
        )
    with k4:
        st.metric(
            label="Fraud Detection Rate",
            value=f"{kpi_data['fraud_detection_rate_pct']}%",
            help="Benchmark fraud rate across cardholder network"
        )
    with k5:
        st.metric(
            label="Avg Transaction Amount",
            value=f"${kpi_data['avg_transaction_amount']:.2f}",
            help=f"Fraud Avg: ${kpi_data['avg_fraud_amount']:.2f} vs Legit Avg: ${kpi_data['avg_legit_amount']:.2f}"
        )

    st.divider()

    # Visual Insights Row
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("Transaction Class Distribution")
        donut_df = pd.DataFrame({
            "Category": ["Legitimate", "Fraudulent"],
            "Count": [kpi_data["legitimate_transactions"], kpi_data["fraudulent_transactions"]]
        })
        chart_donut = alt.Chart(donut_df).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", scale=alt.Scale(range=["#10b981", "#ef4444"])),
            tooltip=["Category", "Count"]
        ).properties(height=280)
        st.altair_chart(chart_donut, use_container_width=True)

    with col_chart2:
        st.subheader("Average Spending: Fraud vs. Legitimate")
        avg_df = pd.DataFrame({
            "Transaction Type": ["Legitimate", "Fraudulent"],
            "Average Amount ($)": [kpi_data["avg_legit_amount"], kpi_data["avg_fraud_amount"]]
        })
        chart_bar = alt.Chart(avg_df).mark_bar(cornerRadius=6).encode(
            x=alt.X("Transaction Type", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Average Amount ($)"),
            color=alt.Color("Transaction Type", scale=alt.Scale(range=["#3b82f6", "#dc2626"])),
            tooltip=["Transaction Type", "Average Amount ($)"]
        ).properties(height=280)
        st.altair_chart(chart_bar, use_container_width=True)

    st.info(
        "💡 **Key Finding:** Fraudulent transactions exhibit an average ticket size of "
        f"**${kpi_data['avg_fraud_amount']}**, which is **{round(kpi_data['avg_fraud_amount'] / max(1, kpi_data['avg_legit_amount']), 1)}x** "
        f"higher than legitimate transactions (${kpi_data['avg_legit_amount']})."
    )


# ==============================================================
# SECTION 2: TRANSACTION ANALYTICS
# ==============================================================
elif section == "2. Transaction Analytics":
    st.title("📊 Transaction Analytics & Hourly Forensics")
    st.markdown("Analysis of transaction volumes, velocity, and monetary values across diurnal cycles.")

    analytics_data = load_db_analytics()
    hourly_df = pd.DataFrame(analytics_data["hourly_distribution"])

    tab_hourly, tab_amount = st.tabs(["🕒 Hourly Velocity & Risk", "💰 Amount Analysis"])

    with tab_hourly:
        st.subheader("Hourly Transaction Volume & Fraud Distribution")
        
        # Dual line/bar chart for Hourly Volume and Fraud Cases
        base = alt.Chart(hourly_df).encode(x=alt.X("hour_of_day:O", title="Hour of the Day (0-23)"))
        
        bar_vol = base.mark_bar(opacity=0.7, color="#60a5fa").encode(
            y=alt.Y("total_volume:Q", title="Total Transactions"),
            tooltip=["hour_of_day", "total_volume", "fraud_count", "avg_amount"]
        )
        
        line_fraud = base.mark_line(color="#dc2626", strokeWidth=3, point=True).encode(
            y=alt.Y("fraud_count:Q", title="Fraud Count"),
            tooltip=["hour_of_day", "fraud_count"]
        )
        
        st.altair_chart(bar_vol + line_fraud, use_container_width=True)

        st.caption("Blue bars denote total volume; Red line denotes confirmed fraud incidents per hour.")

    with tab_amount:
        st.subheader("Monetary Distribution Statistics by Class")
        amt_df = pd.DataFrame(analytics_data["amount_analysis"])
        st.dataframe(amt_df, use_container_width=True)

        st.write("### 5 Key SQL Analytics Queries Executed")
        st.code("""
-- Query 1: Total Volume
SELECT COUNT(*) FROM transactions;

-- Query 2: Class Breakdown
SELECT class, COUNT(*), ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) AS pct
FROM transactions GROUP BY class;

-- Query 3: Amount Stats
SELECT class, SUM(amount), AVG(amount), MIN(amount), MAX(amount)
FROM transactions GROUP BY class;

-- Query 4: Hourly Distribution
SELECT hour_of_day, COUNT(*), SUM(class) FROM transactions GROUP BY hour_of_day;
        """, language="sql")


# ==============================================================
# SECTION 3: FRAUD DETECTION (REAL-TIME PREDICTION)
# ==============================================================
elif section == "3. Fraud Detection":
    st.title("🛡️ Real-Time Fraud Prediction Engine")
    st.markdown("Score individual transactions manually or process bulk CSV transaction streams.")

    predictor = load_cached_predictor()

    pred_tab1, pred_tab2 = st.tabs(["✍️ Manual Transaction Scoring", "📁 Batch CSV Scoring"])

    with pred_tab1:
        st.subheader("Manual Transaction Input")
        
        # Demo Presets
        preset = st.selectbox(
            "Select a Preset Scenario or customize manually:",
            [
                "Custom Manual Input",
                "Preset: Normal Grocery Store Purchase ($45.50)",
                "Preset: Suspicious Midnight ATM Burst ($850.00)",
                "Preset: High-Risk International E-Commerce Fraud ($1,290.00)"
            ]
        )

        # Preset default feature maps
        default_amt = 45.50
        default_time = 36000.0
        default_v14 = 0.1
        default_v12 = -0.2
        default_v10 = 0.05
        default_v4 = -0.1

        if preset == "Preset: Normal Grocery Store Purchase ($45.50)":
            default_amt = 45.50
            default_time = 43200.0
            default_v14 = 0.2
            default_v12 = -0.1
            default_v10 = 0.1
            default_v4 = -0.3
        elif preset == "Preset: Suspicious Midnight ATM Burst ($850.00)":
            default_amt = 850.00
            default_time = 7200.0  # 2 AM
            default_v14 = -5.8
            default_v12 = -4.2
            default_v10 = -3.5
            default_v4 = 3.8
        elif preset == "Preset: High-Risk International E-Commerce Fraud ($1,290.00)":
            default_amt = 1290.00
            default_time = 14400.0  # 4 AM
            default_v14 = -8.4
            default_v12 = -6.1
            default_v10 = -5.2
            default_v4 = 5.2

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            in_amt = st.number_input("Transaction Amount ($)", value=default_amt, min_value=0.0, step=10.0)
            in_time = st.number_input("Transaction Time (Seconds)", value=default_time, min_value=0.0, step=3600.0)
            in_model = st.selectbox("ML Model Classifier", ["Random Forest", "Logistic Regression", "Decision Tree", "Isolation Forest"])

        with col_in2:
            st.markdown("**Top Fraud Correlation Features:**")
            in_v14 = st.slider("V14 (Strong negative correlation with fraud)", -15.0, 10.0, float(default_v14), step=0.1)
            in_v12 = st.slider("V12 (Strong negative correlation with fraud)", -15.0, 10.0, float(default_v12), step=0.1)
            in_v10 = st.slider("V10 (Negative correlation with fraud)", -15.0, 10.0, float(default_v10), step=0.1)

        with col_in3:
            st.markdown("**Top Positive Fraud Features:**")
            in_v4 = st.slider("V4 (Positive correlation with fraud)", -5.0, 15.0, float(default_v4), step=0.1)
            in_v11 = st.slider("V11 (Positive correlation with fraud)", -5.0, 15.0, 0.0, step=0.1)
            in_v2 = st.slider("V2 (Positive correlation with fraud)", -5.0, 15.0, 0.0, step=0.1)

        with st.expander("⚙️ Fine-Tune Remaining PCA Features (V1 to V28)"):
            all_v_inputs = {}
            cols = st.columns(4)
            for i in range(1, 29):
                col_idx = (i - 1) % 4
                c = cols[col_idx]
                if i in [14, 12, 10, 4, 11, 2]:
                    continue
                all_v_inputs[f"V{i}"] = c.number_input(f"V{i}", value=0.0, step=0.5, key=f"v_{i}")

        if st.button("🚀 Run Fraud Prediction", type="primary", use_container_width=True):
            tx_dict = {
                "Time": in_time,
                "Amount": in_amt,
                "V14": in_v14,
                "V12": in_v12,
                "V10": in_v10,
                "V4": in_v4,
                "V11": in_v11,
                "V2": in_v2,
            }
            tx_dict.update(all_v_inputs)

            result = predictor.predict_single(tx_dict, model_name=in_model)

            # Log to DB
            db = get_db()
            hour_val = int((in_time / 3600) % 24)
            pred_id = db.log_prediction(
                model_name=result["model_used"],
                predicted_class=result["predicted_class"],
                fraud_probability=result["fraud_probability"],
                risk_level=result["risk_level"],
                amount=result["amount"],
                hour_of_day=hour_val,
                source="Streamlit Manual Input"
            )

            st.success(f"Transaction Evaluated! Reference ID: `{pred_id}`")

            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric("Verdict", result["prediction"])
                if result["predicted_class"] == 1:
                    st.error("⚠️ FRAUDULENT PATTERN DETECTED")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")

            with res_c2:
                st.metric("Fraud Probability", result["fraud_probability_pct"])
                st.progress(float(result["fraud_probability"]))

            with res_c3:
                st.metric("Risk Classification", result["risk_level"])
                st.info(f"**Action:** {result['recommended_action']}")

            if result["key_deviations"]:
                st.write("#### 🔍 Explainable Fraud Factors (Top Deviations):")
                st.json(result["key_deviations"])

    with pred_tab2:
        st.subheader("Bulk Batch CSV Transaction Scoring")
        st.markdown("Upload a CSV file containing transactions (`Time, V1..V28, Amount`) to score all records simultaneously.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_upload)}** rows. Columns: {list(df_upload.columns)[:8]}...")

            if st.button("⚡ Score Batch File Now"):
                with st.spinner("Scoring transactions with champion ML model..."):
                    scored_df = predictor.predict_batch(df_upload, model_name="Random Forest")
                    
                    fraud_found = (scored_df["Predicted_Class"] == 1).sum()
                    st.success(f"Processed {len(scored_df)} transactions! Found {fraud_found} suspicious cases.")

                    # Summary cards
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Total Scored", len(scored_df))
                    sc2.metric("Detected Frauds", fraud_found, delta=f"{round(fraud_found / len(scored_df) * 100, 2)}%")
                    sc3.metric("Legitimate", len(scored_df) - fraud_found)

                    st.dataframe(scored_df.head(50), use_container_width=True)

                    # CSV Download Button
                    csv_buffer = io.StringIO()
                    scored_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Scored Transactions CSV",
                        data=csv_buffer.getvalue(),
                        file_name="scored_fraud_predictions.csv",
                        mime="text/csv"
                    )


# ==============================================================
# SECTION 4: EDA VISUALIZATIONS
# ==============================================================
elif section == "4. EDA Visualizations":
    st.title("📈 Exploratory Data Analysis (EDA) Gallery")
    st.markdown("High-resolution analytical charts and statistical breakdowns derived from Phase 4.")

    charts_dir = BASE_DIR / "eda" / "charts"
    
    chart_info = [
        ("1_class_imbalance.png", "1. Class Imbalance Breakdown", "Extreme imbalance: 98.2% Legitimate vs 1.8% Fraudulent transactions. Requires SMOTE and precision-recall optimization."),
        ("2_transaction_amount_distribution.png", "2. Transaction Amount Distribution", "Log-transformed spending profiles. Fraud transactions exhibit a distinct secondary peak at high values."),
        ("3_fraud_percentage_breakdown.png", "3. Fraud Percentage Breakdown", "Proportional pie visualization illustrating the rare event nature of financial fraud."),
        ("4_transaction_time_analysis.png", "4. Diurnal Time Analysis", "Fraud events spike between 01:00 and 05:00 UTC, while legitimate purchases peak during standard daylight business hours."),
        ("5_correlation_heatmap.png", "5. Feature Correlation Heatmap", "Linear relationships among top PCA components. Notice strong inverse correlation on V14, V12, and V10 with fraud."),
        ("6_v_features_distribution_fraud_vs_legit.png", "6. Key Discriminative Feature KDEs", "Distinct bimodal separations for V14, V12, V10, and V4 between fraud and legitimate transactions."),
        ("7_amount_vs_time_scatter.png", "7. Amount vs Time Scatter", "Temporal scatter highlighting concentrated clusters of high-value fraud during low-volume night hours."),
        ("8_outlier_boxplots.png", "8. Outlier Boxplots", "Statistical interquartile ranges (IQR) highlighting significant upper-bound outliers in transaction amounts.")
    ]

    selected_chart = st.selectbox("Select Visual Investigation:", [c[1] for c in chart_info])
    
    match = next(c for c in chart_info if c[1] == selected_chart)
    chart_path = charts_dir / match[0]

    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
        st.info(f"📌 **Key Insight:** {match[2]}")
    else:
        st.warning("Chart image not found. Ensure EDA script has executed.")


# ==============================================================
# SECTION 5: MACHINE LEARNING PERFORMANCE
# ==============================================================
elif section == "5. Machine Learning Performance":
    st.title("🤖 Machine Learning Model Benchmarking")
    st.markdown("Comparative performance metrics across 4 Machine Learning classifiers on unseen test data.")

    models_data = load_model_comparison()
    fi_data = load_feature_importance()

    if models_data:
        df_models = pd.DataFrame(models_data)
        
        # Display comparison table
        st.subheader("Model Evaluation Summary (Holdout Test Set)")
        display_cols = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "is_best"]
        st.dataframe(
            df_models[display_cols].style.highlight_max(subset=["f1_score", "pr_auc", "recall"], color="#dcfce7"),
            use_container_width=True
        )

        # Bar chart comparison
        st.subheader("Performance Metric Comparison (F1, Recall, PR-AUC)")
        melted_df = df_models.melt(
            id_vars=["model_name"],
            value_vars=["precision", "recall", "f1_score", "roc_auc", "pr_auc"],
            var_name="Metric",
            value_name="Score"
        )
        metric_chart = alt.Chart(melted_df).mark_bar().encode(
            x=alt.X("Metric:N"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.8, 1.0])),
            color=alt.Color("model_name:N"),
            column=alt.Column("model_name:N")
        ).properties(width=160, height=250)
        st.altair_chart(metric_chart)

        # Confusion Matrix Section
        st.subheader("Confusion Matrix Forensics")
        selected_model_cm = st.selectbox("Select Model for Confusion Matrix:", [m["model_name"] for m in models_data])
        cm_obj = next(m for m in models_data if m["model_name"] == selected_model_cm)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negatives (Legit Correct)", cm_obj["tn"])
        c2.metric("False Positives (False Alarms)", cm_obj["fp"])
        c3.metric("False Negatives (Missed Frauds)", cm_obj["fn"])
        c4.metric("True Positives (Frauds Caught)", cm_obj["tp"])

        # Feature Importance
        if "Random Forest" in fi_data:
            st.subheader("Top Predictive Features (Random Forest Feature Importance)")
            fi_df = pd.DataFrame(fi_data["Random Forest"][:10])
            fi_chart = alt.Chart(fi_df).mark_bar(color="#6366f1", cornerRadius=4).encode(
                x=alt.X("importance:Q", title="Relative Gini Importance"),
                y=alt.Y("feature:N", sort="-x", title="Feature Name"),
                tooltip=["feature", "importance"]
            ).properties(height=300)
            st.altair_chart(fi_chart, use_container_width=True)


# ==============================================================
# SECTION 6: TRANSACTION HISTORY
# ==============================================================
elif section == "6. Transaction History":
    st.title("📜 Database Transaction History & Search")
    st.markdown("Query the transaction database with interactive filters, search, and CSV export.")

    db = get_db()

    f1, f2, f3 = st.columns(3)
    with f1:
        cls_choice = st.selectbox("Transaction Status", ["All Transactions", "Legitimate Only", "Fraudulent Only"])
        cls_val = None if cls_choice == "All Transactions" else (1 if "Fraudulent" in cls_choice else 0)
    with f2:
        max_amt_filter = st.slider("Max Transaction Amount ($)", 10.0, 3000.0, 3000.0, step=50.0)
    with f3:
        record_limit = st.slider("Record Limit", 25, 500, 100, step=25)

    tx_list = db.get_recent_transactions(
        limit=record_limit,
        class_filter=cls_val,
        max_amount=max_amt_filter
    )

    if tx_list:
        df_tx = pd.DataFrame(tx_list)
        st.write(f"Showing **{len(df_tx)}** records:")
        
        # Display dataframe
        st.dataframe(df_tx, use_container_width=True)

        # Download button
        csv_buffer = io.StringIO()
        df_tx.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Filtered Transactions CSV",
            data=csv_buffer.getvalue(),
            file_name="filtered_transactions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No transactions match the selected filter criteria.")


# ==============================================================
# SECTION 7: PREDICTION RESULTS AUDIT
# ==============================================================
elif section == "7. Prediction Results":
    st.title("📋 Prediction Audit Trail")
    st.markdown("Log of all transaction predictions scored through the UI and REST API.")

    db = get_db()
    preds = db.get_recent_predictions(limit=100)

    if preds:
        df_preds = pd.DataFrame(preds)
        st.dataframe(df_preds, use_container_width=True)

        # CSV Download
        csv_buf = io.StringIO()
        df_preds.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Prediction Audit Log CSV",
            data=csv_buf.getvalue(),
            file_name="prediction_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions recorded in audit log yet. Score a transaction in the Fraud Detection tab to generate entries.")
