st.subheader("Test Dataset Download")

st.markdown(
    "[Click here to download test CSV file](https://raw.githubusercontent.com/2024dc04028-creator/income-classification-app/main/test_income_data.csv)"
)


import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("Income Classification App")

st.write(
    "Upload a CSV test dataset, select a classification model, "
    "and evaluate model performance."
)

# -------------------------------------------------
# Load trained models, scaler, and feature names
# -------------------------------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "AdaBoost": joblib.load("model/adaboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# -------------------------------------------------
# Dataset Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST Dataset (CSV format only)",
    type=["csv"]
)

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------
    # Handle target column safely
    # --------------------------------------------
    if "income" in df.columns:
        X = df.drop("income", axis=1)
        y = df["income"]
        labels_available = True
    else:
        X = df.copy()
        y = None
        labels_available = False

    # --------------------------------------------
    # FORCE NUMERIC DATA (CRITICAL FIX)
    # --------------------------------------------
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # --------------------------------------------
    # ALIGN FEATURES WITH TRAINING DATA
    # --------------------------------------------
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]  # correct order

    # --------------------------------------------
    # Scale features (NumPy only)
    # --------------------------------------------
    X_scaled = scaler.transform(X.values)

    # --------------------------------------------
    # Prediction
    # --------------------------------------------
    model = models[model_name]
    y_pred = model.predict(X_scaled)

    # --------------------------------------------
    # Evaluation
    # --------------------------------------------
    if labels_available:
        st.subheader("Evaluation Metrics")

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

    else:
        st.warning(
            "Target column 'income' not found. "
            "Showing predictions only."
        )

        df["Predicted_income"] = y_pred
        st.subheader("Prediction Output Preview")
        st.dataframe(df.head())
