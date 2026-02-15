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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Income Classification App", layout="wide")

st.title("Income Classification Using Machine Learning")

st.markdown(
    "[Download Test CSV File](https://raw.githubusercontent.com/2024dc04028-creator/income-classification-app/main/test_income_data.csv)"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
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

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (Optional)",
    type=["csv"]
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
default_data = pd.read_csv("test_income_data.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded.")
else:
    df = default_data
    st.sidebar.info("Using default test dataset.")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Check Target Column
# -------------------------------------------------
if "income" not in df.columns:
    st.error("Dataset must contain 'income' column.")
    st.stop()

X = df.drop("income", axis=1)
y = df["income"]

# Convert to numeric safely
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
# Model Dictionary
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
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# Official Best Model (Based on Step 2 Results)
# -------------------------------------------------
st.success(
    "🏆 Official Best Model Based on Step 2 Evaluation: Random Forest (Accuracy: 0.8563)"
)

# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------
st.subheader(f"{model_name} - Evaluation Metrics")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("Precision", f"{prec:.4f}")
col3.metric("Recall", f"{rec:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("AUC Score", f"{auc:.4f}")

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

# -------------------------------------------------
# Model Comparison Chart (Accuracy)
# -------------------------------------------------
st.subheader("Model Comparison (Accuracy)")

results = []

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc_model = accuracy_score(y_test, preds)
    results.append((name, acc_model))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

fig2, ax2 = plt.subplots()
ax2.bar(results_df["Model"], results_df["Accuracy"])
ax2.set_ylabel("Accuracy")
ax2.set_xticklabels(results_df["Model"], rotation=45, ha="right")
st.pyplot(fig2)
