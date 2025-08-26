# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load trained models
# -------------------------
with open("log_reg_model.pkl", "rb") as f:
    log_reg_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Online Payment Fraud Detection", layout="wide")
st.title("?? Online Payment Fraud Detection")

st.sidebar.header("Model Selection & Input Options")


model_choice = st.sidebar.selectbox("Choose model", ["Logistic Regression", "XGBoost", "Random Forest"])

# User input fields
st.subheader("Enter Transaction Details")
step = st.number_input("Step", min_value=0)
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)
type_ = st.selectbox("Transaction Type (encoded)", [0, 1, 2, 3])  # adjust according to your encoding

# Predict button
if st.button("Predict Fraud"):
    # Create input dataframe
    input_df = pd.DataFrame([[step, type_, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]],
                            columns=['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'])
    
    # Choose model
    if model_choice == "Logistic Regression":
        prediction = log_reg_model.predict(input_df)[0]
    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(input_df)[0]
    else:  # Random Forest
        prediction = rf_model.predict(input_df)[0]
    
    # Show result
    if prediction == 1:
        st.error("?? Fraudulent Transaction Detected!")
    else:
        st.success("? Transaction is Legitimate.")
