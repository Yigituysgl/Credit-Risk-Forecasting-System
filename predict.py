import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load('Credit_XGBoost_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('model_features.pkl')

st.title(" Credit Risk Predictor")
st.markdown("""
This app uses a trained XGBoost model to predict the likelihood of loan default based on user-provided financial and credit features.
""")

user_input = {}
st.sidebar.header("Enter Applicant Information")

for feature in features:
    if 'int_rate' in feature or 'revol_util' in feature:
        user_input[feature] = st.sidebar.slider(f"{feature}", 0.0, 30.0, 10.0)
    elif 'term' in feature or 'emp_length' in feature:
        user_input[feature] = st.sidebar.number_input(f"{feature}", step=1)
    elif 'grade' in feature or 'sub_grade' in feature or 'home_ownership' in feature:
        user_input[feature] = st.sidebar.text_input(f"{feature}")
    else:
        user_input[feature] = st.sidebar.number_input(f"{feature}", step=1.0)


input_df = pd.DataFrame([user_input])


input_df = input_df[features]


input_scaled = scaler.transform(input_df)


if st.button("Predict Credit Risk"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f" !! High Risk of Default (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {prob:.2f})")
