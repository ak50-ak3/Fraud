import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("/Users/admin/Desktop/FOMLPROJECT/decision_tree_fraud.pkl")

# Load the label encoder for 'type' feature
label_encoder = joblib.load("/Users/admin/Desktop/FOMLPROJECT/label_encoder.pkl")

# Function to predict fraud
def predict_fraud(transaction):
    transaction = np.array(transaction).reshape(1, -1)
    prediction = model.predict(transaction)
    return prediction[0]

# Streamlit UI
st.title("Financial Transaction Fraud Detection")

# User inputs
amount = st.slider("Amount", min_value=0.0, max_value=100000.0, value=0.0)
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, max_value=100000.0, value=0.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, max_value=100000.0, value=0.0)
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, max_value=100000.0, value=0.0)
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, max_value=100000.0, value=0.0)

# Dropdown for transaction type
transaction_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"])

# Convert transaction type to numerical value using label encoder
encoded_type = label_encoder.transform([transaction_type])[0]

# Calculate amount to old balance ratio
amount_balance_ratio = amount / (oldbalanceOrg + 1)

# Prepare the transaction for prediction
transaction_data = [
    encoded_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    amount_balance_ratio,
]

# Make prediction on button click
if st.button("Predict"):
    prediction = predict_fraud(transaction_data)
    if prediction == 1:
        st.write("The transaction is predicted to be **Fraudulent**.")
        st.image("/Users/admin/Desktop/FOMLPROJECT/fraud.png")  # Display fraud image
    else:
        st.write("The transaction is predicted to be **Not Fraudulent**.")
        st.image("/Users/admin/Desktop/FOMLPROJECT/NotFraud.png")  # Display non-fraud image
