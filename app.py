import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import confusion_matrix, classification_report
from notebooks.helper.commoncode import get_metrics
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mobile Price Classification", layout="wide",initial_sidebar_state="expanded")
st.title("Mobile Price Classification Dashboard")

st.write("Validate trained machine learning models on uploaded dataset.")

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------------- LOAD PREPROCESSING FILES ----------------
columns_path = os.path.join(MODEL_DIR, "columns.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(columns_path) or not os.path.exists(scaler_path):
    st.error("Preprocessing files missing. Please export models first.")
    st.stop()

feature_columns = joblib.load(columns_path)
scaler = joblib.load(scaler_path)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

# ---- model selection ----
model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "logistics",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

model_path = os.path.join(MODEL_DIR, f"{model_choice}.pkl")

if not os.path.exists(model_path):
    st.sidebar.error(f"{model_choice}.pkl not found.")
    st.stop()

model = joblib.load(model_path)
st.sidebar.success(f"{model_choice} model ready")

# ---- file upload ----
uploaded_file = st.sidebar.file_uploader(
    "Upload Validation Dataset (must contain price_range)",
    type=["csv"]
)

# ================= MAIN PANEL =================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------- validate target --------
    if "price_range" not in df.columns:
        st.error("Dataset must contain 'price_range' column.")
        st.stop()

    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    # -------- validate features --------
    if list(X.columns) != feature_columns:
        st.error("Feature columns do not match training dataset.")
        st.write("Expected columns:")
        st.write(feature_columns)
        st.stop()

    # -------- preprocess --------
    X_scaled = scaler.transform(X)

    # -------- prediction --------
    y_pred = model.predict(X_scaled)

    try:
        y_prob = model.predict_proba(X_scaled)
    except:
        y_prob = None

    # -------- metrics --------
    metrics = get_metrics(y, y_pred, y_prob)

    # ================= METRICS DASHBOARD =================
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['Precision']:.4f}")
    col3.metric("Recall", f"{metrics['Recall']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    col5.metric("MCC", f"{metrics['MCC']:.4f}")
    col6.metric("AUC", f"{metrics['AUC']:.4f}" if metrics["AUC"] else "N/A")

    # ================= CONFUSION MATRIX =================
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))  # smaller figure

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,  # remove color bar (saves space)
        annot_kws={"size": 10},  # smaller numbers
        linewidths=0.5,
        linecolor='gray',
        xticklabels=["Low", "Med", "High", "V.High"],
        yticklabels=["Low", "Med", "High", "V.High"],
        ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12)

    # make tick labels smaller
    ax.tick_params(axis='both', labelsize=9)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # ================= CLASSIFICATION REPORT =================
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("⬅️ Please select a model and upload a dataset from the left sidebar.")
