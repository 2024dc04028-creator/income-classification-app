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
    "This application allows uploading a CSV test dataset, selecting a "
    "classification model, and evaluating model performance."
)

# -------------------------------------------------
# Load trained models
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

# -------------------------------------------------
# a. Dataset Upload Option (CSV)
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST Dataset (CSV format only)",
    type=["csv"]
)

# -------------------------------------------------
# b. Model Selection Dropdown
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

    # -------------------------------------------------
    # Handle target column safely
    # -------------------------------------------------
    if "income" in df.columns:
        X = df.drop("income", axis=1)
        y = df["income"]
        labels_available = True
    else:
        X = df.copy()
        y = None
        labels_available = False

    # -------------------------------------------------
    # Feature scaling
    # -------------------------------------------------
    X_scaled = scaler.transform(X.values)
    # -------------------------------------------------
    # Model prediction
    # -------------------------------------------------
    model = models[model_name]
    y_pred = model.predict(X_scaled)

    # -------------------------------------------------
    # c. Display Evaluation Metrics
    # -------------------------------------------------
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

        # -------------------------------------------------
        # d. Confusion Matrix
        # -------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------
        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

    else:
        st.warning(
            "Target column 'income' not found in uploaded dataset. "
            "Displaying predictions only."
        )

        df["Predicted_income"] = y_pred
        st.subheader("Prediction Output Preview")
        st.dataframe(df.head())
