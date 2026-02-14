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

st.subheader("Test Dataset Download")

st.markdown(
    "[Click here to download test CSV file](https://raw.githubusercontent.com/2024dc04028-creator/income-classification-app/main/test_income_data.csv)"
)

# -------------------------------------------------
# Load Models and Preprocessing Objects
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
# Model Selection Dropdown
# -------------------------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

# -------------------------------------------------
# Load Default Test Dataset Automatically
# -------------------------------------------------
default_data = pd.read_csv("test_income_data.csv")

uploaded_file = st.file_uploader(
    "Upload New Test Dataset (Optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded successfully.")
else:
    df = default_data
    st.info("Default test dataset loaded automatically.")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Handle Target Column
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
# Safe Numeric Conversion
# -------------------------------------------------
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# -------------------------------------------------
# Align Features with Training
# -------------------------------------------------
for col in feature_names:
    if col not in X.columns:
        X[col] = 0

X = X[feature_names]

# -------------------------------------------------
# Scale Features
# -------------------------------------------------
X_scaled = scaler.transform(X.values)

# -------------------------------------------------
# Predict
# -------------------------------------------------
model = models[model_name]
y_pred = model.predict(X_scaled)

# -------------------------------------------------
# Display Results
# -------------------------------------------------
if labels_available:

    st.subheader(f"{model_name} - Evaluation Metrics")

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader(f"{model_name} - Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")
    st.pyplot(fig)

    st.subheader(f"{model_name} - Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.warning(f"{model_name}: No 'income' column found. Showing predictions only.")
    df["Predicted_income"] = y_pred
    st.subheader(f"{model_name} - Prediction Output")
    st.dataframe(df.head())
