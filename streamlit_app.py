# app.py
import streamlit as st
import pickle
import numpy as np

# Load your trained model
try:
    loaded_model = pickle.load(open('heart_disease_model.pkl','rb'))
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")


# Define a function to make predictions
def predict_heart_disease(sample_patient_features):
    sample_patient_features = [
        63,  # age
        1,  # sex (1 = male; 0 = female)
        145,  # trestbps
        233,  # chol
        1,  # fbs (1 = true; 0 = false)
        150,  # thalach
        0,  # exang (1 = yes; 0 = no)
        2.3,  # oldpeak
        0,  # slope (0 = upsloping; 1 = flat; 2 = downsloping)
        0,  # ca (number of major vessels)
        0,  # cp_1 (chest pain type 1)
        0,  # cp_2 (chest pain type 2)
        1,  # cp_3 (chest pain type 3)
        0,  # restecg_1
        0,  # restecg_2
        1,  # thal_1
        0,  # thal_2
        0  # thal_3
    ]
    sample_patient_features_as_numpy_array = np.array(sample_patient_features)
    sample_patient_features_reshaped = sample_patient_features_as_numpy_array.reshape(1, -1)

    # Make a prediction
    prediction = loaded_model.predict(sample_patient_features_reshaped)
    prediction = loaded_model.predict(sample_patient_features_reshaped)

    # Output the prediction result
    if prediction[0] == 1:
        return prediction[0]


# Streamlit UI components
st.title("Heart Disease Prediction App")

# User inputs
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
