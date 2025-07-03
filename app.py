import streamlit as st
import numpy as np
import joblib

# Load models
model_position = joblib.load("model_peak_position.pkl")
model_intensity = joblib.load("model_peak_intensity.pkl")

# Title
st.title("Spectral Peak Predictor (XRD)")
st.markdown("AI model to predict peak position (2θ) and intensity based on synthesis parameters.")

# User Inputs
material = st.selectbox("Material", ["ZnO", "NiO", "ZnO:NiO"])
dopant = st.slider("Dopant Concentration (%)", 0.0, 50.0, 5.0)
method = st.selectbox("Synthesis Method", ["Sol-Gel", "Sputtering", "Ball Milling"])
temp = st.slider("Annealing Temp (°C)", 100, 1000, 400)

# One-hot encode input
def encode_input(material, method):
    mat_cols = ['Material_NiO', 'Material_ZnO', 'Material_ZnO:NiO']
    meth_cols = ['Method_Ball Milling', 'Method_Sol-Gel', 'Method_Sputtering']
    mat_encoding = [1 if f"Material_{material}" == col else 0 for col in mat_cols]
    meth_encoding = [1 if f"Method_{method}" == col else 0 for col in meth_cols]
    return mat_encoding + meth_encoding

features = [dopant, temp] + encode_input(material, method)

# Prediction
if st.button("Predict Peak"):
    pos = model_position.predict([features])[0]
    intensity = model_intensity.predict([features])[0]
    st.success(f"Predicted 2θ Peak Position: {pos:.2f}°")
    st.success(f"Predicted Intensity: {intensity:.2f}")
