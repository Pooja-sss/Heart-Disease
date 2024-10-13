import streamlit as st
import pickle
import pandas as pd

# Load the model using pickle
try:
    with open('heart_disease_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")


# Define a function to make predictions
def predict_heart_disease(sample_patient_features):
    try:
        input_df = pd.DataFrame([sample_patient_features])
        prediction = loaded_model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# Streamlit UI components
st.title("Heart Disease Prediction App")

# User inputs for prediction
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3)

# Button to make prediction
if st.button("Predict"):
    sample_patient_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    result = predict_heart_disease(sample_patient_features)

    # Show the prediction result
    if result == 1:
        st.success("The patient is predicted to have heart disease.")
    else:
        st.success("The patient is not predicted to have heart disease.")