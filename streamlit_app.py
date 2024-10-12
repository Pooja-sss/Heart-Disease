import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define a function to make predictions
def predict_disease(sample_patient_features):
    input_df = pd.DataFrame([sample_patient_features])

    # Make a prediction
    prediction = loaded_model.predict(input_df)
    return prediction

# Streamlit app layout
st.title("Heart Disease Prediction App")

# User input for features
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", [0, 1])  # Assuming 0 = female, 1 = male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # Example chest pain types
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300)
chol = st.number_input("Cholesterol", min_value=0, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
target = st.selectbox("Heart Disease Presence (1 = Yes, 0 = No)", [0, 1])

# Create a button for prediction
if st.button("Predict"):
    sample_patient_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = predict_disease(sample_patient_features)

    # Display the result
    if prediction[0] == 1:
        st.success("You may have heart disease.")
    else:
        st.success("You likely do not have heart disease.")