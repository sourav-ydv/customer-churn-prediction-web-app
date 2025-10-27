# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:30:00 2025

@author: sksou
"""


import numpy as np
import pickle
import streamlit as st

# Set wide mode
st.set_page_config(layout="wide")

# Custom CSS to stretch full width
st.markdown(
    """
    <style>
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

    # Create 4 wider columns with spacing
    col1, space1, col2, space2, col3, space3, col4 = st.columns([1.5, 0.2, 1.5, 0.2, 1.5, 0.2, 1.5])

    # Column 1 - Personal info
    with col1:
        gender = st.selectbox("Gender", [ , "Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["Select...", 0, 1])
        Partner = st.selectbox("Partner", ["Select...", "Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Select...", "Yes", "No"])
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, step=1)

    # Column 2 - Phone & Internet
    with col2:
        PhoneService = st.selectbox("Phone Service", ["Select...", "Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Select...", "No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["Select...", "DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Select...", "No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Select...", "No", "Yes", "No internet service"])

    # Column 3 - Services
    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["Select...", "No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Select...", "No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Select...", "No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Select...", "No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract", ["Select...", "Month-to-month", "One year", "Two year"])

    # Column 4 - Billing
    with col4:
        PaperlessBilling = st.selectbox("Paperless Billing", ["Select...", "Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Select...", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    # Prediction result
    diagnosis = ""

    if st.button("🔍 Predict Churn"):
        # Collect all dropdowns to check for unselected fields
        dropdowns = [
            gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
        ]

        if "Select..." in dropdowns:
            st.error("⚠️ Please select a value for all fields before prediction.")
        else:
            try:
                input_data = [
                    gender, SeniorCitizen, Partner, Dependents, tenure,
                    PhoneService, MultipleLines, InternetService, OnlineSecurity,
                    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                    MonthlyCharges, TotalCharges
                ]
                diagnosis = churn_prediction(input_data)
                st.success(diagnosis)
            except Exception as e:
                st.error(f"Error in prediction: {e}")


if __name__ == "__main__":
    main()






