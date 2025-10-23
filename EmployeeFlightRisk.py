# app.py — Flight Risk Predictor (Beginner-friendly)
# Upload CSV → pick target → train model → view metrics → download predictions

import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Risk Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Flight Risk Predictor")
st.caption("Upload your employee data CSV, choose the target column (e.g., flight_risk / attrition), and train a model.")

# ----------------------------
# Helpers
# ----------------------------
TRUTHY = {"1","true","yes","y","left","leaver","attrited","churn","quit","resigned","separated"}
FALSY  = {"0","false","no","n","stay","stayed","active","retained"}

def coerce_binary_target(series: pd.Series) -> pd.Series:
    """Map common labels to 0/1."""
    s = series.copy()
    # numeric-like?
    try:
        sn = pd.to_numeric(s, errors="coerce")
        uniq = set(np.unique(sn.dropna()))
        if uniq.issubset({0,1}):
            return sn.fillna(0).astype(int)
    except Exception:
        pass
    s_str = s.astype(str).str.strip().str.lower()

    def map_label(x: str) -> int:
        if x in TRUTHY: return 1
        if x in FALSY:  return 0
        if any(k in x for k in ["left","leave","attrit","quit","resign","terminate","churn","separat"]): return 1
        if any(k in x for k in ["stay","active","retain","present"]): return 0
        return 0
    return s_str.map(map_label).astype(int)

def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Basic cleaning + one-hot using pandas.get_dummies (version-safe)."""
    X = df[feature_cols].copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("Unknown").replace({"nan": "Unknown"})
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X_cat = pd.get_dummies(X[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=X.index)
    X_num = X[num_cols] if num_cols else pd.DataFrame(index=X.index)
    return pd.concat([X_num, X_cat], axis=1)

def top_features(model, feature_names, top_n=15):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature","importance"])
    idx = np.argsort(vals)[::-1][:top_n]
    return pd.DataFrame({"feature": np.array(feature_names)[idx], "importance": vals[idx]})

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
max_rows_preview = st.sidebar.slider("Preview rows", 5, 50, 10, 5)

# ----------------------------
# Upload
# ----------------------------
st.subheader("1) 📂 Upload your CSV")
file = st.file_uploader("Choose a CSV file", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.write("**Data preview:**")
    st.dataframe(df.head(max_rows_preview))

    st.subheader("2) 🎯 Choose Target & Features")
    target_col = st.selectbox("Target (binary: left/stay, yes/no, 0/1, etc.)", options=df.columns.tolist())

    default_exclude = {target_col} | {c for c in df.columns if c.lower() in {"id","emp_id","employee_id"}}
    default_features = [c for c in df.columns if c not in default_exclude]

    feature_cols = st.multiselect(
        "Select feature columns",
        options=[c for c in df.columns if c != target_col],
        default=default_features
    )
    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    st.subheader("3) 🧠 Train the Model")
    if st.button("Train & Evaluate"):
        with st.spinner("Training..."):
            y = coerce_binary_target(df[target_col])
            X = prepare_features(df, feature_cols)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.loc[X.index]

            if y.nunique() < 2:
                st.error("Target column has only one class after mapping. Check your labels (need 0 and 1).")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            if model_choice == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=300, random_state=random_state, n_jobs=-1
                )
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Probabilities
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                try:
                    decision = model.decision_function(X_test)
                    y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
                except Exception:
                    y_proba = np.zeros_like(y_pred, dtype=float)

            # Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
            c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
            c4.metric("F1", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
            try:
                c5.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
            except Exception:
                c5.metric("ROC AUC", "N/A")

            st.divider()
            st.subheader("🧩 Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
            st.pyplot(fig)

            st.subheader("🏅 Top Features")
            feats = top_features(model, X.columns, top_n=15)
            if not feats.empty:
                st.dataframe(feats.reset_index(drop=True))
            else:
                st.info("Model doesn't expose importances; try Random Forest.")

            # Score full dataset & allow download
            st.subheader("⬇️ Download Predictions for Full Data")
            model.fit(X, y)
            if hasattr(model, "predict_proba"):
                proba_full = model.predict_proba(X)[:, 1]
            else:
                try:
                    decision_full = model.decision_function(X)
                    proba_full = (decision_full - decision_full.min())/(decision_full.max()-decision_full.min()+1e-9)
                except Exception:
                    proba_full = np.zeros(len(X))
            pred_full = model.predict(X)

            out = df.copy()
            out["flight_risk_pred"] = pred_full
            out["flight_risk_prob"] = proba_full

            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button(
                "Download CSV with predictions",
                data=buf,
                file_name="flight_risk_scored.csv",
                mime="text/csv"
            )

else:
    st.info("Upload a CSV to begin. Example target values: 0/1, Yes/No, True/False, Left/Stay, etc.")

with st.expander("💡 Tips for better results"):
    st.markdown("""
- Target column should indicate leaving risk (e.g., `flight_risk`, `attrition`, `left`).
- Include numeric features (tenure_months, overtime_hours, pay_change_pct, age).
- Categorical features (department, job_role, location) are auto one-hot encoded.
- Start with **Random Forest** to see top features easily.
- For production: handle class imbalance (class_weight='balanced'), add time-based splits, and calibrate probabilities.
""")
