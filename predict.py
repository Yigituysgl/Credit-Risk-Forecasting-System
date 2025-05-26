
import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("Credit_XGBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")  

st.title("💳 Credit Risk Forecast")
st.markdown("This app predicts the risk of loan default using key financial and credit features.")

st.sidebar.header("🧾 Applicant Information")


loan_amnt = st.sidebar.number_input("Loan Amount", 0, 100000, 10000)
term = st.sidebar.selectbox("Loan Term", ["36 months", "60 months"])
int_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, 10.0)
installment = st.sidebar.number_input("Installment Amount", 0, 1000, 300)
annual_inc = st.sidebar.number_input("Annual Income", 0, 200000, 50000)
dti = st.sidebar.number_input("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 18.0)
grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER", "NONE"])


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


input_df['grade'] = pd.Categorical(input_df['grade'], categories=["A", "B", "C", "D", "E", "F", "G"])
input_df['home_ownership'] = pd.Categorical(input_df['home_ownership'], categories=["RENT", "OWN", "MORTGAGE", "OTHER", "NONE"])
input_df['term'] = pd.Categorical(input_df['term'], categories=["36 months", "60 months"])


input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)


input_scaled = scaler.transform(input_encoded)


prediction = model.predict_proba(input_scaled)[0][1]

st.subheader(" Prediction Result")
st.write(f"**Estimated Default Risk:** {prediction:.2%}")

if prediction > 0.5:
    st.error("⚠️ High risk of default")
else:
    st.success("✅ Low risk of default")

