# Install necessary libraries (uncomment the below lines if needed)
# !pip install streamlit scikit-learn pandas

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load heart disease dataset
# You can replace this with the actual dataset path
@st.cache_data
def load_data():
    # Sample structure of a heart disease dataset
    data = pd.read_csv("heart.csv")
    return data


# Build a simple machine learning model (Logistic Regression)
@st.cache_resource
def train_model(data):
    # Assuming 'target' is the label and rest are features
    X = data.drop(columns='target')
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy


# User input function for Streamlit form
def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', (0, 1))  # 1: male, 0: female
    cp = st.sidebar.selectbox('Chest Pain Type (CP)', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 130)
    chol = st.sidebar.slider('Cholesterol (chol)', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.sidebar.selectbox('Resting ECG (restecg)', (0, 1, 2))
    thalach = st.sidebar.slider('Max Heart Rate (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.sidebar.slider('Oldpeak (ST depression)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (slope)', (0, 1, 2))
    ca = st.sidebar.slider('Number of Major Vessels (ca)', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3))

    # Create a dictionary for user inputs
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Convert to DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features


# Main function for the Streamlit app
def main():
    st.title("Heart Disease Prediction App")
    st.write("""
    ### Enter the details of a patient to predict the probability of heart disease.
    """)

    # Load data
    data = load_data()

    # Display raw data if the user wants to see it
    if st.checkbox('Show raw data'):
        st.write(data)

    # Train the model
    model, accuracy = train_model(data)

    # Display model accuracy
    st.write(f"Model accuracy: {accuracy:.2f}")

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    user_input = user_input_features()

    # Display user input
    st.write("### User Input Parameters:")
    st.write(user_input)

    # Predict
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # Display prediction and probability
    st.write("### Prediction: ", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
    st.write(f"### Prediction Probability: {prediction_proba[0][1]:.2f}")


if __name__ == '__main__':
    main()
