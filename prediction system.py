
import pandas as pd
import pickle

# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.pkl', 'rb'))

sample_patient_features = [
    63,    # age
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
    0      # thal_3
]

# Convert the sample to a DataFrame or appropriate input format
input_df = pd.DataFrame([sample_patient_features],)

# Make a prediction
prediction = loaded_model.predict(input_df)

# Output the prediction result
if prediction[0] == 1:
    print("The patient is predicted to have heart disease.")
else:
    print("The patient is not predicted to have heart disease.")