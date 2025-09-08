# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle

st.cache_data.clear()
st.cache_resource.clear()


@st.cache_resource
def load_model(model_name):
   
    if model_name == "Logistic Regression":
        with open("log_reg_model.pkl", "rb") as f:
            return pickle.load(f)
    elif model_name == "XGBoost":
        with open("xgb_model.pkl", "rb") as f:
            return pickle.load(f)
    else:  # Random Forest
        with open("rf_model.pkl", "rb") as f:
            return pickle.load(f)


st.set_page_config(page_title="Online Payment Fraud Detection", layout="wide")
st.title("💳 Online Payment Fraud Detection")

st.sidebar.header("Model Selection & Input Options")
model_choice = st.sidebar.selectbox("Choose model", ["Logistic Regression", "XGBoost", "Random Forest"])

st.subheader("Enter Transaction Details")
step = st.number_input("Step", min_value=0)
amount = st.number_input("Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, format="%.2f")


type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))

transaction_type_encoded = type_mapping[transaction_type]

st.write(f"You selected: {transaction_type} → Encoded as: {transaction_type_encoded}")

if st.button("Predict Fraud"):
    input_df = pd.DataFrame([[
        step,
        transaction_type_encoded,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ]],
    columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

    model = load_model(model_choice)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Type: {transaction_type})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Type: {transaction_type})")
