# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:30:00 2025

@author: sksou
"""

import numpy as np
import pickle
import streamlit as st

st.set_page_config(layout="wide")

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

loaded_model = pickle.load(open("churn_model.sav", "rb"))

def churn_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)

    if prediction[0] == 0:
        return "✅ The customer is likely to stay (Not Churn)."
    else:
        return "⚠️ The customer is likely to Churn."


def main():
    st.title("📊 Customer Churn Prediction")

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
        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            index=None,
            placeholder="Choose Option"
        )

    with col2:
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

    diagnosis = ""

    if st.button("Predict Churn"):
        dropdowns = [Contract, PaperlessBilling, PaymentMethod, InternetService, OnlineSecurity, TechSupport]

        if any(option is None for option in dropdowns):
            st.error("⚠️ Please select a value for all dropdowns before prediction.")
        else:
            try:
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

                defaults = {
                    "gender": 0,              
                    "SeniorCitizen": 0,
                    "Partner": 0,              
                    "Dependents": 0,          
                    "PhoneService": 1,         
                    "MultipleLines": 1,       
                    "DeviceProtection": 0,    
                    "StreamingTV": 0,         
                    "StreamingMovies": 0      
                }
                
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
                    service_map["No"],  
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
