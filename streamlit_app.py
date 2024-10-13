import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the model
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the trained model
loaded_model = load_model()
st.success("Model loaded successfully.")

# Streamlit UI components
st.title("Heart Disease Prediction App")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
thal = st.selectbox("Thalassemia (0-3)", options=[0, 1, 2, 3])

# Button to make prediction
if st.button("Predict"):
    # Prepare the input data for prediction
    sample_patient_features = [
        age,    # age
        sex,    # sex
        cp,     # chest pain type
        trestbps,  # resting blood pressure
        chol,   # cholesterol
        fbs,    # fasting blood sugar
        restecg,  # resting electrocardiographic results
        thalach,  # maximum heart rate
        exang,  # exercise induced angina
        oldpeak,  # oldpeak
        slope,  # slope
        ca,     # number of major vessels
        thal    # thalassemia
    ]

    input_df = pd.DataFrame([sample_patient_features], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    # Make a prediction
    prediction = loaded_model.predict(input_df)

    # Output the prediction result
    if prediction[0] == 1:
        st.success("The patient is predicted to have heart disease.")
    else:
        st.success("The patient is not predicted to have heart disease.")


    # Show the prediction result
    if result == 1:
        st.success("The patient is predicted to have heart disease.")
    else:
        st.success("The patient is not predicted to have heart disease.")
