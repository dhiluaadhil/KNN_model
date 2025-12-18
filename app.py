import streamlit as st
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Iris Flower Predictor", layout="centered")

# 1. Load the saved model and scaler
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_assets():
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. UI Elements
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements below to predict the species.")

col1, col2 = st.columns(2)

with col1:
    sepal_l = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_w = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)

with col2:
    petal_l = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
    petal_w = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# 3. Prediction Logic
if st.button("Predict Species"):
    # Create array of inputs
    new_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    # IMPORTANT: Scale the input data using the saved scaler
    new_data_scaled = scaler.transform(new_data)
    
    # Get prediction
    prediction = model.predict(new_data_scaled)
    
    # Map back to flower names
    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[prediction[0]]
    
    st.success(f"The predicted species is: **{result}**")