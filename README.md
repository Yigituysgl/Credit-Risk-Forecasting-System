# Credit-Risk-Forecasting-System


This project builds a robust machine learning pipeline to predict the likelihood of loan default using borrower credit and financial data. It involves exploratory data analysis (EDA), feature engineering, encoding, normalization, and training with two models: Random Forest and XGBoost. The models are evaluated and compared, with a focus on identifying the most important predictive features.

A Streamlit web app is being developed to make predictions interactively.

 Dataset

- All Lending Club Loan Data (public dataset)
- Contains detailed information about loan applications including loan amount, term, interest rate, income, credit scores, and repayment behavior.

---

# Objectives

- Perform comprehensive data cleaning and preprocessing
- Engineer relevant features and handle missing values
- Encode categorical variables and scale numerical values
- Train and compare XGBoost and Random Forest models
- Analyze feature importance
- Deploy a Streamlit app for user-friendly prediction (in progress)

---

# Exploratory Data Analysis (EDA)

- Distribution analysis for key variables
- Correlation matrix and pairwise plots
- Detection and handling of outliers
- Visual comparison of defaults vs. non-defaults

---

##  Machine Learning Models

| Metric         | Random Forest    | XGBoost |
|----------------|------------------|-----------|
| Accuracy       |  0.99            |  0.98     |
| Recall         |  1.00            |  0.98     |
| F1-score       |  0.99            |  0.99     |
Confusion Matrix	[[TN: 283, FP: 4], [FN: 5, TP: 708]]	[[TN: 284, FP: 3], [FN: 3, TP: 710]]
AUC Score :        0.9950               0.9968


 While Random Forest achieved perfect recall, XGBoost was ultimately selected for deployment due to its superior AUC score and fewer total misclassifications, offering a more balanced and generalizable performance.






