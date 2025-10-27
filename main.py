# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:30:00 2025

@author: sksou
"""
# -*- coding: utf-8 -*-
"""
Customer Churn Prediction Web App with Spaced Columns
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved churn model
loaded_model = pickle.load(open("churn_model.sav", "rb"))

# Prediction function
def churn_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)

    if prediction[0] == 0:
        return "✅ The customer is likely to stay (Not Churn)."
    else:
        return "⚠️ The customer is likely to Churn."


def main():
    st.title("📊 Customer Churn Prediction Web App")

    # Columns with spacing (ratios: col, space, col, space, col)
    col1, space1, col2, space2, col3 = st.columns([1, 0.2, 1, 0.2, 1])

    with col1:
        gender = st.selectbox("Gender", ("Female", "Male"))
        SeniorCitizen = st.selectbox("Senior Citizen", (0, 1))
        Partner = st.selectbox("Partner", ("Yes", "No"))
        Dependents = st.selectbox("Dependents", ("Yes", "No"))
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, step=1)

    with col2:
        PhoneService = st.selectbox("Phone Service", ("Yes", "No"))
        MultipleLines = st.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
        InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
        OnlineSecurity = st.selectbox("Online Security", ("No", "Yes", "No internet service"))
        OnlineBackup = st.selectbox("Online Backup", ("No", "Yes", "No internet service"))

    with col3:
        DeviceProtection = st.selectbox("Device Protection", ("No", "Yes", "No internet service"))
        TechSupport = st.selectbox("Tech Support", ("No", "Yes", "No internet service"))
        StreamingTV = st.selectbox("Streaming TV", ("No", "Yes", "No internet service"))
        StreamingMovies = st.selectbox("Streaming Movies", ("No", "Yes", "No internet service"))
        Contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
        PaperlessBilling = st.selectbox("Paperless Billing", ("Yes", "No"))
        PaymentMethod = st.selectbox(
            "Payment Method",
            ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    # Prediction result
    diagnosis = ""

    if st.button("🔍 Predict Churn"):
        try:
            # NOTE: Preprocessing (encoding) must match training
            input_data = [
                gender, SeniorCitizen, Partner, Dependents, tenure,
                PhoneService, MultipleLines, InternetService, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                MonthlyCharges, TotalCharges
            ]
            diagnosis = churn_prediction(input_data)
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    st.success(diagnosis)


if __name__ == "__main__":
    main()
