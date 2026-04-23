import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random

# Page config
st.set_page_config(page_title="Stock Movement Predictor", page_icon="📈", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: #ffffff;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        color: #ffffff;
        transform: scale(1.05);
    }
    h1 {
        color: #0d6efd;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .result-box-up {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
        color: #0f5132;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-box-down {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #842029;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("📈 Stock Movement Predictor")
st.markdown("<p style='text-align: center; color: #6c757d;'>Using Ensemble Learning (Random Forest) to predict next day's stock movement.</p>", unsafe_allow_html=True)
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.markdown("### Enter Today's Market Data")
    
    col1, col2 = st.columns(2)
    with col1:
        close_price = st.number_input("Closing Price ($)", min_value=1.0, value=150.0, step=0.1)
        volume = st.number_input("Trading Volume", min_value=1000, value=50000000, step=1000)
        returns = st.number_input("Daily Returns (%)", min_value=-20.0, max_value=20.0, value=0.5, step=0.1) / 100.0
    with col2:
        ma_10 = st.number_input("10-Day Moving Average ($)", min_value=1.0, value=148.0, step=0.1)
        ma_50 = st.number_input("50-Day Moving Average ($)", min_value=1.0, value=140.0, step=0.1)

    if st.button("Predict Movement"):
        # Prepare feature array
        features = np.array([[close_price, volume, returns, ma_10, ma_50]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        st.markdown("### Prediction Result")
        if prediction == 1:
            st.markdown(
                "<div class='result-box-up'><h4>🚀 Prediction: UP</h4><p>The model predicts the stock price will go UP tomorrow.</p></div>", 
                unsafe_allow_html=True
            )
            
            # Gold coins falling animation
            coin_html = "<style>"
            coin_html += "@keyframes fall { 0% { top: -10vh; transform: rotate(0deg); opacity: 1; } 100% { top: 110vh; transform: rotate(360deg); opacity: 0; } }"
            coin_html += ".coin { position: fixed; font-size: 2.5rem; z-index: 9999; pointer-events: none; animation: fall linear forwards; text-shadow: 0 0 10px rgba(255, 215, 0, 0.5); }"
            coin_html += "</style>"
            for _ in range(40):
                left = random.randint(0, 100)
                duration = random.uniform(1.5, 4.0)
                delay = random.uniform(0, 1.5)
                coin_html += f"<div class='coin' style='left: {left}vw; animation-duration: {duration}s; animation-delay: {delay}s;'>🪙</div>"
            st.markdown(coin_html, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='result-box-down'><h4>📉 Prediction: DOWN</h4><p>The model predicts the stock price will go DOWN tomorrow.</p></div>", 
                unsafe_allow_html=True
            )
else:
    st.warning("Model file not found. Please train the model first.")
