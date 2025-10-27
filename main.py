# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:10:02 2025

@author: sksou
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("C:/Users/sksou/OneDrive/Desktop/coding/ML/diabetes_model.sav", "rb"))


# creating a function
def diabetes_prediction(input_data):
    
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
    
    
def main():
    # giving a title
    st.title("Diabetes Prediction Web App")
    
    # getting the input data from the user
    Pregnencies = st.text_input('Number of Pregnencies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        try:
            input_data = [
                float(Pregnencies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            st.error("Please enter valid numbers in all fields.")
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()