# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle  # To load the trained m

# Load the trained model
try:
    with open('heart_disease_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        # Check if the loaded model is a list or an actual model
        if isinstance(loaded_model, list):
            st.error("Error: Loaded object is a list. Please load the correct model.")
        else:
            st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to predict heart disease
def predict_heart_disease(sample_patient_features):
    try:
        input_df = pd.DataFrame([sample_patient_features], columns=[ 63,    # age
    1,     # sex (1 = male; 0 = female)
    145,   # trestbps
    233,   # chol
    1,     # fbs (1 = true; 0 = false)
    150,   # thalach
    0,     # exang (1 = yes; 0 = no)
    2.3,   # oldpeak
    0,     # slope (0 = upsloping; 1 = flat; 2 = downsloping)
    0,     # ca (number of major vessels)
    0,     # cp_1 (chest pain type 1)
    0,     # cp_2 (chest pain type 2)
    1,     # cp_3 (chest pain type 3)
    0,     # restecg_1
    0,     # restecg_2
    1,     # thal_1
    0,     # thal_2
    0      # thal_3])  # Add all feature names
        st.write("Input features shape:", input_df.shape)  # Log input shape for debugging

        # Make the prediction if model is not a list
        prediction = loaded_model.predict(input_df)
        return prediction
    except AttributeError as ae:
        st.error(f"Prediction failed: {ae}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")



def main():
    # Streamlit app UI
    st.title("Heart Disease Prediction App")

    # User input form
    with st.form(key='heart_disease_form'):
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, value=120)
        chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=0, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=1.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])

        # Chest Pain Types (cp)
        cp_1 = st.selectbox("Chest Pain Type 1", options=[0, 1])
        cp_2 = st.selectbox("Chest Pain Type 2", options=[0, 1])
        cp_3 = st.selectbox("Chest Pain Type 3", options=[0, 1])

        # Resting Electrocardiographic Results (restecg)
        restecg_1 = st.selectbox("Resting Electrocardiographic Result 1", options=[0, 1])
        restecg_2 = st.selectbox("Resting Electrocardiographic Result 2", options=[0, 1])

        # Thalassemia
        thal_1 = st.selectbox("Thalassemia 1", options=[0, 1])
        thal_2 = st.selectbox("Thalassemia 2", options=[0, 1])
        thal_3 = st.selectbox("Thalassemia 3", options=[0, 1])

        submit_button = st.form_submit_button(label='Predict')

    # When the button is clicked, make a prediction
    if submit_button:
        sample_patient_features = [
            age, sex, trestbps, chol, fbs, thalach,
            exang, oldpeak, slope, ca, cp_1, cp_2,
            cp_3, restecg_1, restecg_2, thal_1, thal_2, thal_3
        ]
         # Reshape the input for prediction
    if st.button("Predict"):
        try:
            prediction = predict_heart_disease(sample_patient_features)
            if prediction is not None:
                if prediction[0] == 1:
                    st.success("The patient is predicted to have heart disease.")
                else:
                    st.success("The patient is not predicted to have heart disease.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == '__main__':
    main()

