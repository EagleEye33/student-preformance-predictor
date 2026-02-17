import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Student Score Predictor", layout="centered")

# Load the model with caching to save memory
@st.cache_resource
def load_model():
    data = joblib.load('student_model.pkl')
    return data['model'], data['features']

model, features = load_model()

# Header Section
st.title("🎓 Student Performance Analytics")
st.markdown("""
Predict exam scores based on study habits and historical performance. 
Adjust the parameters in the sidebar to see the prediction update.
""")

# Sidebar for Inputs
st.sidebar.header("Student Parameters")

def get_user_input():
    inputs = []
    # Dynamic sliders based on trained features
    for feature in features:
        # Check if feature is attendance to set 0-100 range
        if 'attendance' in feature.lower():
            val = st.sidebar.slider(f"{feature.replace('_', ' ').title()}", 0, 100, 75)
        else:
            val = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}", value=0)
        inputs.append(val)
    return np.array([inputs])

user_data = get_user_input()

# Main Prediction Logic
st.subheader("Model Prediction")

if st.button("Calculate Prediction"):
    prediction = model.predict(user_data)[0]
    
    # Display Result with Metrics
    col1, col2 = st.columns(2)
    col1.metric("Predicted Score", f"{prediction:.2f}%")
    
    if prediction >= 75:
        col2.success("Status: High Achiever")
    elif prediction >= 40:
        col2.info("Status: Pass")
    else:
        col2.error("Status: Academic Risk")
        
    # Extra: Show the input data used
    st.write("Current Input vector:", pd.DataFrame(user_data, columns=features))
else:
    st.info("Adjust the values on the left and click 'Calculate' to see the result.")