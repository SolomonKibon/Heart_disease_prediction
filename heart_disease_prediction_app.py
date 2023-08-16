import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

st.subheader('Solomon Kibon: 15/08/2023')
# Load the trained model
model = joblib.load('best_model_heart.pkl')
# Define the app title and layout
st.title("Heart Disease Prediction App")
#define input fields for features
Age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
Resting_Blood_Pressure = st.number_input("RestingBP", min_value=0, max_value=300, value=100, step=1)
Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=100, step=1)
Fasting_blood_sugar = st.selectbox("FastingBS", [0, 1])
Max_heart_rate = st.number_input("MaxHR", min_value=0, max_value=300, value=100, step=1)
Oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
Sex = st.selectbox("Sex", ["M", "F"])
ChestPainType = st.selectbox("ChestPainType", ['ATA','NAP','TA','ASY'])
Resting_electrocardiographic_results = st.selectbox("RestingECG", ['Normal','ST','LVH'])
ExerciseAngina = st.selectbox("ExerciseAngina",['N','Y'])
ST_Slope= st.selectbox("ST_Slope", ['UP','Flat','Down'])
# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "Cholesterol": [Cholesterol],
            "MaxHR": [Max_heart_rate],
            "Oldpeak": [Oldpeak],
            "RestingECG_ST": [1 if Resting_electrocardiographic_results=='ST' else 0],
            "ExerciseAngina_Y":[1 if ExerciseAngina == 'Y' else 0],
            "ST_Slope_Flat":[1 if ST_Slope == 'Flat' else 0],
        }
    )
    
    # Scale input data using the same scaler used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction
    if prediction == 1:
        st.error("High risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")