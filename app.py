import streamlit as st
import joblib
import numpy as np

# Load the "Brain"
model = joblib.load('knn_model.pkl')

st.title("Social Media Ad Predictor")

# The inputs must match the order of your CSV: Age, then Salary
age = st.slider("Select Age", 18, 100, 30)
salary = st.number_input("Estimated Annual Salary ($)", value=50000)

if st.button("Will they buy?"):
    # Reshape input for the model
    data = np.array([[age, salary]])
    
    # Make prediction
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        st.success("The model predicts: PURCHASE ✅")
    else:
        st.error("The model predicts: NO PURCHASE ❌")