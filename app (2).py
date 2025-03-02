import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib') 
scaler = joblib.load('scaler.joblib')

# Define the prediction function (may need adjustment)
def predict_heart_disease(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Streamlit UI components
st.title("Heart Disease Prediction")

# Input fields for heart disease features
age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=0, max_value=1, value=1, step=1)
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0, step=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120, step=1)
chol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200, step=1)
fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", min_value=0, max_value=1, value=0, step=1)
restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=0, step=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150, step=1)
exang = st.number_input("Exercise Induced Angina (1 = Yes, 0 = No)", min_value=0, max_value=1, value=0, step=1)
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=0, step=1)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
thal = st.number_input("Thalassemia (1-3)", min_value=1, max_value=3, value=1, step=1)

# Create the input dictionary for prediction
input_data = {
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

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_heart_disease(input_data)  # Call the heart disease prediction function

        if pred == 1:
            st.error(f"Prediction: Heart Disease with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No Heart Disease with probability {prob:.2f}")
