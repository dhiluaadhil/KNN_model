import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Iris Flower Predictor")

@st.cache_resource
def load_assets():
    # Check if files exist before trying to load them
    if not os.path.exists('knn_model.pkl') or not os.path.exists('scaler.pkl'):
        return None, None
    
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

if model is None:
    st.error("⚠️ Error: 'knn_model.pkl' or 'scaler.pkl' not found! Please upload them to your GitHub repository.")
else:
    st.title("🌸 Iris Flower Classifier")
    # ... rest of your input and prediction code ...
    sepal_l = st.number_input("Sepal Length (cm)", value=5.0)
    sepal_w = st.number_input("Sepal Width (cm)", value=3.0)
    petal_l = st.number_input("Petal Length (cm)", value=1.5)
    petal_w = st.number_input("Petal Width (cm)", value=0.2)

    if st.button("Predict"):
        new_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        
        species = ['Setosa', 'Versicolor', 'Virginica']
        st.success(f"Result: {species[prediction[0]]}")