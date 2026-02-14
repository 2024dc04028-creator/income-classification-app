import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("Income Classification App")

st.subheader("Test Dataset Download")

st.markdown(
    "[Click here to download test CSV file](https://raw.githubusercontent.com/2024dc04028-creator/income-classification-app/main/test_income_data.csv)"
)

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "AdaBoost"
    ]
)

# -------------------------------------------------
# Load Default Dataset
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
if "income" not in df.columns:
    st.error("Uploaded dataset must contain 'income' column to compute metrics.")
    st.stop()

X = df.drop("income", axis=1)
y = df["income"]

# -------------------------------------------------
# Safe Numeric Conversion
# -------------------------------------------------
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Initialize Models
# -------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}

# -------------------------------------------------
# Train Selected Model
# -------------------------------------------------
model = models[model_name]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------------------------
# Find Best Model (Based on Accuracy)
# -------------------------------------------------
best_model_name = ""
best_accuracy = 0

for name, m in models.items():
    m.fit(X_train, y_train)
    temp_pred = m.predict(X_test)
    temp_acc = accuracy_score(y_test, temp_pred)

    if temp_acc > best_accuracy:
        best_accuracy = temp_acc
        best_model_name = name

st.success(f"🏆 Best Model Based on Accuracy: {best_model_name} ({best_accuracy:.4f})")

# -------------------------------------------------
# Display Metrics
# -------------------------------------------------
st.subheader(f"{model_name} - Evaluation Metrics")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("Precision", f"{prec:.4f}")
col3.metric("Recall", f"{rec:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
st.subheader(f"{model_name} - Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"{model_name} Confusion Matrix")
st.pyplot(fig)

# -------------------------------------------------
# Classification Report
# -------------------------------------------------
st.subheader(f"{model_name} - Classification Report")
st.text(classification_report(y_test, y_pred))
