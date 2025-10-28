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
        max-width: 70% !important;
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
    st.title("📊 Customer Churn Prediction (Simplified Features)")

    # Two columns layout for inputs
    col1, space, col2 = st.columns([1.5, 0.2, 1.5])

    with col1:
        Contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            index=None,
            placeholder="Choose Contract"
        )
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, step=1)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    with col2:
        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            index=None,
            placeholder="Choose Option"
        )
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=None,
            placeholder="Choose Payment Method"
        )
        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=None,
            placeholder="Choose Internet Service"
        )
        OnlineSecurity = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            index=None,
            placeholder="Choose Option"
        )
        TechSupport = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            index=None,
            placeholder="Choose Option"
        )

    # Prediction result
    diagnosis = ""

    if st.button("🔍 Predict Churn"):
        dropdowns = [Contract, PaperlessBilling, PaymentMethod, InternetService, OnlineSecurity, TechSupport]

        if any(option is None for option in dropdowns):
            st.error("⚠️ Please select a value for all dropdowns before prediction.")
        else:
            try:
                # === Encoding mappings ===
                contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
                paperless_map = {"No": 0, "Yes": 1}
                payment_map = {
                    "Electronic check": 0,
                    "Mailed check": 1,
                    "Bank transfer (automatic)": 2,
                    "Credit card (automatic)": 3
                }
                internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
                service_map = {"No": 0, "Yes": 1, "No internet service": 2}

                # === Defaults for less important features ===
                defaults = {
                    "gender": 0,               # Female
                    "SeniorCitizen": 0,
                    "Partner": 0,              # No
                    "Dependents": 0,           # No
                    "PhoneService": 1,         # Yes
                    "MultipleLines": 1,        # No
                    "DeviceProtection": 0,     # No
                    "StreamingTV": 0,          # No
                    "StreamingMovies": 0       # No
                }

                # === Build full 19-feature input ===
                input_data = [
                    defaults["gender"],
                    defaults["SeniorCitizen"],
                    defaults["Partner"],
                    defaults["Dependents"],
                    tenure,
                    defaults["PhoneService"],
                    defaults["MultipleLines"],
                    internet_map[InternetService],
                    service_map[OnlineSecurity],
                    service_map["No"],  # OnlineBackup default = "No"
                    defaults["DeviceProtection"],
                    service_map[TechSupport],
                    defaults["StreamingTV"],
                    defaults["StreamingMovies"],
                    contract_map[Contract],
                    paperless_map[PaperlessBilling],
                    payment_map[PaymentMethod],
                    MonthlyCharges,
                    TotalCharges
                ]

                diagnosis = churn_prediction(input_data)
                st.success(diagnosis)

            except Exception as e:
                st.error(f"Error in prediction: {e}")


if __name__ == "__main__":
    main()







