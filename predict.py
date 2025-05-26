import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("Credit_XGBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Risk Forecast")
st.markdown("""
This app predicts the risk of loan default using key financial and credit features.
""")


selected_features = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc',
    'dti', 'open_acc', 'revol_util', 'grade', 'home_ownership'
]

user_input = {}
st.sidebar.header("📋 Applicant Information")

grade_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
home_ownership_options = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']

for feature in selected_features:
    if feature in ['int_rate', 'revol_util']:
        user_input[feature] = st.sidebar.slider(f"{feature}", 0.0, 30.0, 10.0)
    elif feature == 'term':
        user_input[feature] = st.sidebar.selectbox("Loan Term (months)", [36, 60])
    elif feature == 'grade':
        user_input[feature] = st.sidebar.selectbox("Grade", grade_options)
    elif feature == 'home_ownership':
        user_input[feature] = st.sidebar.selectbox("Home Ownership", home_ownership_options)
    else:
        user_input[feature] = st.sidebar.number_input(f"{feature}", step=1.0)


input_df = pd.DataFrame([user_input])
input_df['term'] = input_df['term'].astype(str)
input_df = pd.get_dummies(input_df)


input_df = input_df.reindex(columns=model_features, fill_value=0)


input_scaled = scaler.transform(input_df)


prediction = model.predict_proba(input_scaled)[0][1]

st.subheader("Default Risk Probability:")
st.metric(label="Risk Score (0 to 1)", value=f"{prediction:.2f}")

if prediction > 0.5:
    st.error("⚠️ High risk of default")
else:
    st.success("✅ Low risk of default")

