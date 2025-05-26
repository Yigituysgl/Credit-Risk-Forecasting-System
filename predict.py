import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("Credit_XGBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")  # expected feature names


st.title("Credit Risk Forecast")
st.markdown("This app predicts the risk of loan default using key financial and credit features.")


st.sidebar.header(" Applicant Information")

loan_amnt = st.sidebar.number_input("Loan Amount", 0, 50000, 10000)
term = st.sidebar.selectbox("Loan Term (months)", [36, 60])
int_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, 13.0)
installment = st.sidebar.number_input("Installment Amount", 0, 2000, 300)
annual_inc = st.sidebar.number_input("Annual Income", 0, 200000, 50000)
dti = st.sidebar.number_input("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 18.0)
grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

input_df = pd.DataFrame({
    "loan_amnt": [loan_amnt],
    "term": [term],
    "int_rate": [int_rate],
    "installment": [installment],
    "annual_inc": [annual_inc],
    "dti": [dti],
    "grade": [grade],
    "home_ownership": [home_ownership],
})


input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)


input_scaled = scaler.transform(input_encoded)


prediction = model.predict_proba(input_scaled)[0][1]  

st.subheader("🔎 Prediction Result")
st.write(f"**Estimated Default Risk:** {prediction:.2%}")


if prediction > 0.5:
    st.error("⚠️ High risk of default")
else:
    st.success("✅ Low risk of default")

