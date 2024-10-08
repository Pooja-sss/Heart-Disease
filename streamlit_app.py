# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle  # To load the trained m

# Load the trained KNN model (make sure to adjust the file path as necessary)
loaded_model = pickle.load(open('heart_disease_model.pkl', 'rb'))

def predict_heart_disease(sample_patient_features):

    # Convert the sample to a DataFrame or appropriate input format
    input_df = pd.DataFrame([sample_patient_features])

    # Make a prediction
    prediction = loaded_model.predict(input_df)

    # Output the prediction result
    if prediction[0] == 1:
        return prediction[0]

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
        sample_patient_features_reshaped = np.array(sample_patient_features).reshape(1, -1)

        # Make a prediction and display the result
        prediction = predict_heart_disease(sample_patient_features_reshaped[0])
        # Output the prediction result
        if prediction == 1:
            st.write("The patient is predicted to have heart disease.")
        else:
            st.write("The patient is not predicted to have heart disease.")


if __name__ == '__main__':
    main()