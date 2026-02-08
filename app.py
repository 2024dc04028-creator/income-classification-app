import streamlit as st
import numpy as np
import joblib

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "AdaBoost": joblib.load("model/adaboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

st.title("Income Classification App")

model_choice = st.selectbox("Select Model", list(models.keys()))

age = st.number_input("Age", 18, 100)
education_num = st.number_input("Education Number", 1, 16)
hours_per_week = st.number_input("Hours per Week", 1, 100)
capital_gain = st.number_input("Capital Gain", 0)
capital_loss = st.number_input("Capital Loss", 0)

input_data = np.array([[age, education_num, capital_gain, capital_loss, hours_per_week]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = models[model_choice].predict(input_scaled)

    if prediction[0] == 1:
        st.success("Income > 50K")
    else:
        st.warning("Income ≤ 50K")
