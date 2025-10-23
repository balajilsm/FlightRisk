# app.py
# --- Streamlit Flight Risk Predictor (Beginner-friendly) ---
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
st.caption("Upload your employee data CSV, choose the target column (e.g., flight_risk / attrition), and train a simple model.")

# ----------------------------
# Helpers
# ----------------------------
TRUTHY = {"1","true","yes","y","left","leaver","attrited","churn","quit","resigned","separated"}
FALSY  = {"0","false","no","n","stay","stayed","active","retained"}

def coerce_binary_target(series: pd.Series) -> pd.Series:
    """
    Convert many common string labels into 0/1.
    - If numeric-like → cast to int(0/1)
    - If text → map truthy to 1, falsy to 0; else try smart guess; unknown -> 0
    """
    s = series.copy()

    # If already numeric-ish, try to cast
    try:
        sn = pd.to_numeric(s, errors="coerce")
        # If values are mostly {0,1}, use them
        unique_non_nan = set(np.unique(sn.dropna()))
        if unique_non_nan.issubset({0,1}):
            return sn.fillna(0).astype(int)
    except Exception:
        pass

    # Lowercase strings for mapping
    s_str = s.astype(str).str.strip().str.lower()

    def map_label(x: str) -> int:
        if x in TRUTHY:
            return 1
        if x in FALSY:
            return 0
        # smart guess: if it contains these words
        if any(k in x for k in ["left","leave","attrit","quit","resign","terminate","churn","separat"]):
            return 1
        if any(k in x for k in ["stay","active","retain","present"]):
            return 0
        # fallback
        return 0

    return s_str.map(map_label).astype(int)

def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df[feature_cols].copy()

    # Basic cleaning
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("Unknown").replace({"nan":"Unknown"})

    # One-hot via pandas (robust across sklearn versions)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols], drop_first=True)
    else:
        X_cat = pd.DataFrame(index=X.index)

    X_num = X[num_cols] if num_cols else pd.DataFrame(index=X.index)
    X_enc = pd.concat([X_num, X_cat], axis=1)
    return X_enc

def top_features(model, feature_names, top_n=15):
    """Return top features by absolute importance/weight."""
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature","importance"])
    idx = np.argsort(vals)[::-1][:top_n]
    return pd.DataFrame({"feature": np.array(feature_names)[idx], "importance": vals[idx]})

# ----------------------------
# Sidebar: Settings
# ----------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])
test_size = st.sidebar.slider("Test size (for evaluation)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
max_rows_preview = st.sidebar.slider("Max preview rows", 5, 50, 10, 5)

# ----------------------------
# File Upload
# ----------------------------
st.subheader("1) 📂 Upload your CSV")
file = st.file_uploader("Choose a CSV file", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.write("**Data preview:**")
    st.dataframe(df.head(max_rows_preview))

    # ----------------------------
    # Target & Features
    # ----------------------------
    st.subheader("2) 🎯 Choose Target & Features")
    target_col = st.selectbox(
        "Select the target column (binary flight risk / attrition indicator)",
        options=df.columns.tolist()
    )

    # Auto-suggest features (exclude target + id-ish)
    default_exclude = {target_col}
    default_exclude |= {c for c in df.columns if c.lower() in {"id","emp_id","employee_id"}}
    default_features = [c for c in df.columns if c not in default_exclude]

    feature_cols = st.multiselect(
        "Select feature columns",
        options=[c for c in df.columns if c != target_col],
        default=default_features
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    # ----------------------------
    # Train
    # ----------------------------
    st.subheader("3) 🧠 Train the Model")
    if st.button("Train & Evaluate"):
        with st.spinner("Training..."):
            # Build X, y
            y_raw = df[target_col]
            y = coerce_binary_target(y_raw)
            X = prepare_features(df, feature_cols)

            # Guard: need both classes present
            if y.nunique() < 2:
                st.error("The target column has only one class after mapping. Please check labels (need 0 and 1).")
                st.stop()

            # Align (drop rows with any NA post-encoding just in case)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.loc[X.index]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Choose model
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, n_jobs=None)
            else:
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=random_state,
                    n_jobs=-1
                )

            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            # Probabilities (if available)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # fallback: decision_function or zeros
                try:
                    decision = model.decision_function(X_test)
                    # scale to 0..1
                    y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
                except Exception:
                    y_proba = np.zeros_like(y_pred, dtype=float)

            # Metrics
            colA, colB, colC, colD, colE = st.columns(5)
            colA.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            colB.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
            colC.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
            colD.metric("F1", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
            try:
                auc = roc_auc_score(y_test, y_proba)
                colE.metric("ROC AUC", f"{auc:.3f}")
            except Exception:
                colE.metric("ROC AUC", "N/A")

            st.divider()

            # Confusion Matrix
            st.subheader("🧩 Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
            st.pyplot(fig)

            # Top Features
            st.subheader("🏅 Top Features")
            feats = top_features(model, X.columns, top_n=15)
            if not feats.empty:
                st.dataframe(feats.reset_index(drop=True))
            else:
                st.info("Model doesn't expose feature importance; try Random Forest.")

            # Score the whole dataset and let user download
            st.subheader("⬇️ Download Predictions (scored full dataset)")
            # Refit on full data for final scoring (simple approach)
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
                label="Download CSV with predictions",
                data=buf,
                file_name="flight_risk_scored.csv",
                mime="text/csv"
            )

            st.success("Done! You can tune features or model type and retrain anytime.")

else:
    st.info("Upload a CSV to begin. Example target values can be 0/1, Yes/No, True/False, Left/Stay, etc.")


# ----------------------------
# Tips
# ----------------------------
with st.expander("💡 Tips for better results"):
    st.markdown("""
- **Target column** should indicate whether the employee left / is at risk (e.g., `flight_risk`, `attrition`, `left`).
- Numeric features (e.g., `tenure_months`, `overtime_hours`, `age`) help a lot.
- Categorical features (e.g., `department`, `job_role`, `location`) are auto one-hot encoded.
- Start with **Random Forest** to see feature importance easily.
- This is a simple baseline. For production, consider time-based splits, class imbalance handling, and calibration.
""")
