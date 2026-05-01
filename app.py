import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")
st.markdown("**B.Tech Gen AI | Harshil Parmar | KU2507U0511**")
st.markdown("---")
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=70, value=30)
    frequent_flyer = st.selectbox("Frequent Flyer?", ["Yes", "No"])
    annual_income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])

with col2:
    services_opted = st.slider("Services Opted", min_value=1, max_value=6, value=3)
    social_media = st.selectbox("Account Synced to Social Media?", ["Yes", "No"])
    booked_hotel = st.selectbox("Booked Hotel or Not?", ["Yes", "No"])

st.markdown("---")

# Encode inputs
def encode_inputs(age, frequent_flyer, annual_income, services_opted, social_media, booked_hotel):
    ff_map = {"Yes": 1, "No": 0}
    income_map = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
    yn_map = {"Yes": 1, "No": 0}

    return np.array([[
        age,
        ff_map[frequent_flyer],
        income_map[annual_income],
        services_opted,
        yn_map[social_media],
        yn_map[booked_hotel]
    ]])

if st.button("🔍 Predict Churn", use_container_width=True):
    input_data = encode_inputs(age, frequent_flyer, annual_income, services_opted, social_media, booked_hotel)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN**")
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
        st.markdown("**Recommendation:** Offer a discount, loyalty reward, or personalized retention strategy.")
    else:
        st.success(f"✅ **This customer is NOT likely to churn**")
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
        st.markdown("**Recommendation:** Continue providing quality service to maintain customer satisfaction.")

    st.markdown("---")
    st.subheader("Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income Class", "Services Opted", "Social Media Synced", "Booked Hotel"],
        "Value": [age, frequent_flyer, annual_income, services_opted, social_media, booked_hotel]
    })
    st.table(summary)

st.markdown("---")
st.caption("Customer Churn Prediction App | Random Forest Model | Deployed via Streamlit Cloud")
