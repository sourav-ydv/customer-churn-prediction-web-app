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
        max-width: 90% !important;
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
        return "✅ The customer is likely to stay."
    else:
        return "⚠️ The customer is likely to Churn."


def main():
    st.title("📊 Customer Churn Prediction Web App")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=None, placeholder="Choose Gender")
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], index=None, placeholder="Choose Option")
        Partner = st.selectbox("Partner", ["Yes", "No"], index=None, placeholder="Choose Partner Status")
        Dependents = st.selectbox("Dependents", ["Yes", "No"], index=None, placeholder="Choose Dependents")
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, step=1)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"], index=None, placeholder="Choose Phone Service")
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], index=None, placeholder="Choose Option")
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=None, placeholder="Choose Internet Service")
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=None, placeholder="Choose Option")
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=None, placeholder="Choose Option")

    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=None, placeholder="Choose Option")
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=None, placeholder="Choose Option")
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=None, placeholder="Choose Option")
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=None, placeholder="Choose Option")
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=None, placeholder="Choose Contract")

    with col4:
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], index=None, placeholder="Choose Option")
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=None,
            placeholder="Choose Payment Method"
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    diagnosis = ""

    if st.button("Predict Churn"):
        dropdowns = [
            gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
        ]

        if any(option is None for option in dropdowns):
            st.error("⚠️ Please select a value for all dropdowns before prediction.")
        else:
            try:
                gender_map = {"Female": 0, "Male": 1}
                partner_map = {"No": 0, "Yes": 1}
                dependents_map = {"No": 0, "Yes": 1}
                phone_map = {"No": 0, "Yes": 1}
                multiple_map = {"No phone service": 0, "No": 1, "Yes": 2}
                internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
                service_map = {"No": 0, "Yes": 1, "No internet service": 2}
                contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
                paperless_map = {"No": 0, "Yes": 1}
                payment_map = {
                    "Electronic check": 0,
                    "Mailed check": 1,
                    "Bank transfer (automatic)": 2,
                    "Credit card (automatic)": 3
                }

                input_data = [
                    gender_map[gender],
                    int(SeniorCitizen),
                    partner_map[Partner],
                    dependents_map[Dependents],
                    tenure,
                    phone_map[PhoneService],
                    multiple_map[MultipleLines],
                    internet_map[InternetService],
                    service_map[OnlineSecurity],
                    service_map[OnlineBackup],
                    service_map[DeviceProtection],
                    service_map[TechSupport],
                    service_map[StreamingTV],
                    service_map[StreamingMovies],
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

