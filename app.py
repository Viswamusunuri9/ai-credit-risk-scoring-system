import streamlit as st
import pandas as pd
from src.predict import predict_risk

st.set_page_config(page_title="Credit Risk System", layout="centered")

st.title("💳 AI Credit Risk Assessment System")
st.caption("Predict loan default risk using machine learning")

st.subheader("📋 Enter Applicant Details")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "Unknown"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "Unknown"])
credit_amount = st.number_input("Credit Amount", value=2000)
duration = st.number_input("Duration (months)", value=12)
purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "education", "business", "domestic appliances"])

# Prediction button
if st.button("🔍 Assess Risk"):
    input_data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose,
    }

    prob, risk = predict_risk(input_data)

    st.subheader("📊 Result")

    if risk == "High Risk":
        st.error(f"⚠️ High Risk | Probability: {prob:.2f}")
        st.write("Recommendation: Reject or further review")
    else:
        st.success(f"✅ Low Risk | Probability: {prob:.2f}")
        st.write("Recommendation: Approve loan")
    
    if prob > 0.7:
        st.warning("🚨 Very high default probability")
    elif prob > 0.4:
        st.info("⚠️ Moderate risk — review carefully")
    else:
        st.success("✅ Safe applicant")
    
    st.subheader("📊 Risk Score")

    st.progress(float(prob))

st.markdown("### 🧠 Model Insight")
st.write("This model considers credit burden (credit amount per month) as a key risk indicator.")
st.markdown("---")
st.caption("Built using Machine Learning • XGBoost • Streamlit")