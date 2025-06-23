
import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="centered")

# App header
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
<style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 15em;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.write("### Enter Patient Details in the Sidebar:")

# Sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.radio("Sex", ["Female", "Male"])
sex_value = 0 if sex == "Female" else 1

cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", value=200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs_value = 1 if fbs == "Yes" else 0

restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.sidebar.radio("Exercise Induced Angina", ["No", "Yes"])
exang_value = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.number_input("ST Depression Induced (oldpeak)", value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Predict button
if st.sidebar.button("Predict Heart Disease"):
    features = np.array([age, sex_value, cp, trestbps, chol, fbs_value,
                         restecg, thalach, exang_value, oldpeak,
                         slope, ca, thal]).reshape(1, -1)

    prediction = model.predict(features)

    st.write("---")
    if prediction[0] == 1:
        st.error("⚠️ **Prediction: High Risk of Heart Disease!** Please consult a cardiologist.")
    else:
        st.success("✅ **Prediction: Low Risk of Heart Disease.** Stay healthy!")

# Footer
st.write("---")
