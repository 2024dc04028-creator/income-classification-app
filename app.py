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
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Income Classification App", layout="wide")

st.title("Income Classification Using Machine Learning")

# -------------------------------------------------
# Load Example Dataset FIRST (needed for download button)
# -------------------------------------------------
example_data = pd.read_csv("test_income_data.csv")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Model",
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
    "Upload Dataset (Optional)",
    type=["csv"]
)

st.sidebar.markdown("### Download Example Dataset")

csv_data = example_data.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="Download example_dataset.csv",
    data=csv_data,
    file_name="example_dataset.csv",
    mime="text/csv"
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded.")
else:
    df = example_data
    st.sidebar.info("Using example dataset.")

# -------------------------------------------------
# Display Dataset
# -------------------------------------------------
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

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Models Dictionary
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
# Evaluate All Models (For Comparison)
# -------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)

    results.append((name, acc, auc))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])

# -------------------------------------------------
# Best Model
# -------------------------------------------------
best_model = results_df.sort_values("Accuracy", ascending=False).iloc[0]

st.success(
    f"🏆 Best Model Based on Accuracy: {best_model['Model']} ({best_model['Accuracy']:.4f})"
)

# -------------------------------------------------
# Train Selected Model
# -------------------------------------------------
selected_model = models[model_name]
selected_model.fit(X_train, y_train)

y_pred = selected_model.predict(X_test)
y_prob = selected_model.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# Metrics Display
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
# Model Comparison Chart
# -------------------------------------------------
st.subheader("Model Comparison (Accuracy)")

fig2, ax2 = plt.subplots()
ax2.bar(results_df["Model"], results_df["Accuracy"])
ax2.set_ylabel("Accuracy")
ax2.set_xticklabels(results_df["Model"], rotation=45, ha="right")
st.pyplot(fig2)
