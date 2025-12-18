import app as st
import joblib
import numpy as np

# 1. Load the saved model
model = joblib.load('knn_model.pkl')

st.title("KNN Classification App")
st.write("Enter the features below to get a prediction.")

# 2. Create input fields for your data
# Adjust these based on your specific dataset features
feature_1 = st.number_input("Feature 1 (e.g., Sepal Length)", value=0.0)
feature_2 = st.number_input("Feature 2 (e.g., Sepal Width)", value=0.0)

# 3. Predict button
if st.button("Predict"):
    # Reshape input for the model
    input_data = np.array([[feature_1, feature_2]])
    prediction = model.predict(input_data)
    
    st.success(f"The predicted class is: {prediction[0]}")