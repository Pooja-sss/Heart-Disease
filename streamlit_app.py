import streamlit as st
import pandas as pd
import pickle



# Load the trained model
def load_model():
    try:
        with open('heart_disease_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")


# Define the prediction function
def predict_heart_disease(model, input_features):
    try:
        input_df = pd.DataFrame([input_features], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")


def main():
    # Load the model
    model = load_model()

    if model is None:
        return  # Exit if the model couldn't be loaded

    # Set title for the app
    st.title("Heart Disease Prediction App")

    # Get user inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=63)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # 0, 1, 2, or 3
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # 0 = false, 1 = true
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # 0, 1, or 2
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=200, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])  # 0 = no, 1 = yes
    oldpeak = st.number_input("Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.0,
                              value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])  # 0, 1, or 2
    ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])  # 0, 1, 2, or 3

    # Create a button to make predictions
    if st.button("Predict"):
        input_features = [
            age,
            1 if sex == "Male" else 0,  # Convert sex to binary
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]

        prediction = predict_heart_disease(model, input_features)

        if prediction == 1:
            st.success("The patient is predicted to have heart disease.")
        else:
            st.success("The patient is not predicted to have heart disease.")


if __name__ == "__main__":
    main()


