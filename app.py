import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

st.title("Income Classification App")

# -------------------------------
# Load trained models
# -------------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/k-nearest_neighbors.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes_gaussian.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "AdaBoost": joblib.load("model/adaboost_ensemble.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# -------------------------------
# a. Dataset upload (CSV only)
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST Dataset (CSV format only)",
    type=["csv"]
)

# -------------------------------
# b. Model selection dropdown
# -------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Split features and target
    X = df.drop("income", axis=1)
    y = df["income"]

    X_scaled = scaler.transform(X)
    model = models[model_name]

    # Predictions
    y_pred = model.predict(X_scaled)

    # -------------------------------
    # c. Evaluation metrics
    # -------------------------------
    st.subheader("Evaluation Metrics")

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"Precision: {prec:.4f}")
    st.write(f"Recall: {rec:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # -------------------------------
    # d. Confusion matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Optional: Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
