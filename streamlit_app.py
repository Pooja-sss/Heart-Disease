import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Title of the app
st.title("Heart Disease Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=120, value=62)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=140)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=268)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=200, value=160)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", options=[0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=3.6)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0-3)", options=[0, 1, 2, 3])

# Create a button for prediction
if st.button("Predict"):
    # Prepare input data
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    # Change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_reshaped)

    # Display the prediction result
    if prediction[0] == 0:
        st.success('The Person does not have Heart Disease')
    else:
        st.error('The Person has Heart Disease')
