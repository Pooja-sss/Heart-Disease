import streamlit as st
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier  # or your specific model
import pickle

# Load your trained model
with open('heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app title
st.title("Heart Disease Prediction")

# Input fields for patient features
age = st.number_input("Age", min_value=0, max_value=120, value=63)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

# Add other input fields for other features
# For example:
cholesterol = st.number_input("Cholesterol Level", min_value=0, value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=120)
# Continue adding input fields for all relevant features...

# Convert input features to DataFrame
input_features = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cholesterol': [cholesterol],
    'blood_pressure': [blood_pressure],
    # Add all other features in the same way...
})

# Prediction
if st.button("Predict"):
    prediction = loaded_model.predict(input_features)
    if prediction[0] == 1:
        st.success("The patient is predicted to have heart disease.")
    else:
        st.success("The patient is not predicted to have heart disease.")

