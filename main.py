# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:30:00 2025

@author: sksou
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("churn_model.sav", "rb"))

# Prediction function
def churn_prediction(input_data):
    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "The customer is likely to stay (Not churn)."
    else:
        return "The customer is likely to churn."


def main():
    # Title
    st.title("Customer Churn Prediction Web App")

    # User input fields (example features from Telco dataset)
    gender = st.selectbox("Gender", ("Male", "Female"))
    SeniorCitizen = st.selectbox("Senior Citizen", (0, 1))
    Partner = st.selectbox("Partner", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ("Yes", "No"))
    tenure = st.text_input("Tenure (in months)")
    PhoneService = st.selectbox("Phone Service", ("Yes", "No"))
    InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    MonthlyCharges = st.text_input("Monthly Charges")
    TotalCharges = st.text_input("Total Charges")

    # Convert categorical inputs into numerical form
    gender_val = 1 if gender == "Male" else 0
    Partner_val = 1 if Partner == "Yes" else 0
    Dependents_val = 1 if Dependents == "Yes" else 0
    PhoneService_val = 1 if PhoneService == "Yes" else 0
    InternetService_val = 0
    if InternetService == "DSL":
        InternetService_val = 1
    elif InternetService == "Fiber optic":
        InternetService_val = 2

    # Prediction result
    diagnosis = ""

    if st.button("Predict Churn"):
        try:
            input_data = [
                gender_val, int(SeniorCitizen), Partner_val, Dependents_val,
                float(tenure), PhoneService_val, InternetService_val,
                float(MonthlyCharges), float(TotalCharges)
            ]
            diagnosis = churn_prediction(input_data)
        except ValueError:
            st.error("Please enter valid numbers for tenure, monthly charges, and total charges.")

    st.success(diagnosis)


if __name__ == "__main__":
    main()
