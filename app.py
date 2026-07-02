import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Load Model and Scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Check if files exist
if not os.path.exists(model_path):
    st.error("❌ model.pkl not found!")
    st.stop()

if not os.path.exists(scaler_path):
    st.error("❌ scaler.pkl not found!")
    st.stop()

# Load files
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's medical information below.")

# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=30)

sex = st.selectbox(
    "Sex",
    ("Female", "Male")
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=0.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# Convert Male/Female into numeric value
sex = 1 if sex == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")