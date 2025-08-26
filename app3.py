# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Cached Model Loading Functions
# -------------------------
@st.cache_resource
def load_model(model_name):
    """Load only the selected model."""
    if model_name == "Logistic Regression":
        with open("log_reg_model.pkl", "rb") as f:
            return pickle.load(f)
    elif model_name == "XGBoost":
        with open("xgb_model.pkl", "rb") as f:
            return pickle.load(f)
    else:  # Random Forest
        with open("rf_model.pkl", "rb") as f:
            return pickle.load(f)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Online Payment Fraud Detection", layout="wide")
st.title("💳 Online Payment Fraud Detection")

st.sidebar.header("Model Selection & Input Options")

model_choice = st.sidebar.selectbox("Choose model", ["Logistic Regression", "XGBoost", "Random Forest"])

# User input fields
st.subheader("Enter Transaction Details")
step = st.number_input("Step", min_value=0)
amount = st.number_input("Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, format="%.2f")
type_ = st.selectbox("Transaction Type (encoded)", [0, 1, 2, 3])  # adjust according to your encoding

# Predict button
if st.button("Predict Fraud"):
    input_df = pd.DataFrame([[step, type_, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]],
                            columns=['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'])
    
    # Load only the selected model
    model = load_model(model_choice)
    prediction = model.predict(input_df)[0]
    
    # Show result
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
