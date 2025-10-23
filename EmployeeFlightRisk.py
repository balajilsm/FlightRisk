# EmployeeFlightRisk.py — Fast-mode Flight Risk Predictor
# ✅ Faster cold starts
# ✅ "Fast mode" for big CSVs (sampling + capped categoricals)
# ✅ Toggle between Random Forest (slower, explainable) and Logistic (fast)

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

st.set_page_config(page_title="Flight Risk Predictor (Fast)", page_icon="✈️", layout="wide")
st.title("✈️ Flight Risk Predictor (Fast)")
st.caption("Upload CSV → pick target → train → metrics → download predictions. Includes a faster path for large files.")

# ----------------------------
# Speed/robustness helpers
# ----------------------------
TRUTHY = {"1","true","yes","y","left","leaver","attrited","churn","quit","resigned","separated"}
FALSY  = {"0","false","no","n","stay","stayed","active","retained"}

def coerce_binary_target(series: pd.Series) -> pd.Series:
    # Try numeric first
    sn = pd.to_numeric(series, errors="coerce")
    uniq = set(np.unique(sn.dropna()))
    if uniq and uniq.issubset({0,1}):
        return sn.fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    def m(x: str) -> int:
        if x in TRUTHY: return 1
        if x in FALSY:  return 0
        if any(k in x for k in ["left","leave","attrit","quit","resign","terminate","churn","separat"]): return 1
        if any(k in x for k in ["stay","active","retain","present"]): return 0
        return 0
    return s.map(m).astype(int)

def cap_categorical_cardinality(df: pd.DataFrame, max_uniques=30):
    """Replace rare categories with 'Other' to shrink one-hot size."""
    capped = df.copy()
    for c in capped.columns:
        if not pd.api.types.is_numeric_dtype(capped[c]):
            vc = capped[c].astype(str).value_counts()
            keep = set(vc.head(max_uniques).index)
            capped[c] = capped[c].astype(str).where(capped[c].astype(str).isin(keep), "Other")
    return capped

def prepare_features(df: pd.DataFrame, feature_cols: list, fast_mode=False) -> pd.DataFrame:
    X = df[feature_cols].copy()

    # Basic cleaning
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("Unknown").replace({"nan": "Unknown"})

    # Cap cardinality in fast mode
    if fast_mode:
        X = cap_categorical_cardinality(X, max_uniques=30)

    # One-hot via pandas (fast & version-safe)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X_cat = pd.get_dummies(X[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=X.index)
    X_num = X[num_cols] if num_cols else pd.DataFrame(index=X.index)
    X_enc = pd.concat([X_num, X_cat], axis=1)

    # Remove columns with zero variance (can happen after capping)
    nunique = X_enc.nunique()
    keep_cols = nunique[nunique > 1].index
    return X_enc[keep_cols]

def top_features(model, feature_names, top_n=12):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature","importance"])
    idx = np.argsort(vals)[::-1][:top_n]
    return pd.DataFrame({"feature": np.array(feature_names)[idx], "importance": vals[idx]})

# ----------------------------
# Sidebar tuning (speed-focused)
# ----------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression (fast)", "Random Forest (explainable)"])
fast_mode = st.sidebar.toggle("Fast mode (recommended for large CSVs)", value=True)
sample_rows = st.sidebar.number_input("Max rows in fast mode", min_value=1000, value=5000, step=1000)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
max_rows_preview = st.sidebar.slider("Preview rows", 5, 50, 10, 5)

# RF knobs (used only if selected)
rf_estimators = st.sidebar.slider("RF: n_estimators", 50, 400, 120, 10)
rf_max_depth = st.sidebar.select_slider("RF: max_depth", options=[None, 6, 8, 10, 12], value=None)

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

    # Optional sampling to speed up training
    work_df = df
    if fast_mode and len(df) > sample_rows:
        work_df = df.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)
        st.info(f"Fast mode: sampled {len(work_df):,} rows from {len(df):,} to speed up training.")

    st.subheader("3) 🧠 Train the Model")
    if st.button("Train & Evaluate"):
        with st.spinner("Training (fast)…"):
            y = coerce_binary_target(work_df[target_col])
            X = prepare_features(work_df, feature_cols, fast_mode=fast_mode)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.loc[X.index]

            if y.nunique() < 2:
                st.error("Target column has only one class after mapping. Check your labels (need 0 and 1).")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            if model_choice.startswith("Logistic"):
                model = LogisticRegression(max_iter=600, n_jobs=None)
            else:
                model = RandomForestClassifier(
                    n_estimators=rf_estimators,
                    max_depth=rf_max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )

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
            feats = top_features(model, X.columns, top_n=12)
            if not feats.empty:
                st.dataframe(feats.reset_index(drop=True))
            else:
                st.info("Model doesn't expose importances; choose Random Forest for importances.")

            # Score FULL dataset with the trained model refit on full (fast but practical)
            st.subheader("⬇️ Download Predictions for Full Data")
            # Prepare full X on the full df with same fast-mode rules
            y_full = coerce_binary_target(df[target_col])
            X_full = prepare_features(df, feature_cols, fast_mode=fast_mode)
            X_full = X_full.replace([np.inf, -np.inf],
