import streamlit as st
import pickle
import pandas as pd

# Load the model from a .pkl file
loaded_model = "heart_disease_model.pkl"
with open(loaded_model, 'rb') as file:
    loaded_model = pickle.load(file)


# Function to make predictions
def predict_heart_disease(sample_patient_features):
    input_df = pd.DataFrame([sample_patient_features])  # Convert input to numpy array and reshape
    prediction = loaded_model.predict(input_df)
    return prediction[0]


# Streamlit app
st.title("Heart Disease Prediction")

# Input features based on the model
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"))
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)

# Convert categorical variables into numeric for the model
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Button to make a prediction
if st.button("Predict"):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]
    result = predict_heart_disease(input_data)

    if result == 1:
        st.write("The model predicts that this individual **has heart disease**.")
    else:
        st.write("The model predicts that this individual **does not have heart disease**.")
