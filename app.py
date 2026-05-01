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
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Title */
h1 {
    background: linear-gradient(90deg, #a78bfa, #f472b6, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #a78bfa !important;
}

/* Student badge */
.stMarkdown p strong {
    color: #c4b5fd;
}

/* Column cards */
div[data-testid="column"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px;
    padding: 1rem !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: transform 0.2s, opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-2px) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #db2777) !important;
}

/* Metric box */
div[data-testid="metric-container"] {
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

/* Success box */
div[data-testid="stAlert"][kind="success"], .stSuccess {
    background: rgba(52,211,153,0.10) !important;
    border-left: 4px solid #34d399 !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}

/* Error box */
div[data-testid="stAlert"][kind="error"], .stError {
    background: rgba(244,114,182,0.10) !important;
    border-left: 4px solid #f472b6 !important;
    border-radius: 10px !important;
    color: #f9a8d4 !important;
}

/* Divider */
hr {
    border-color: rgba(167,139,250,0.2) !important;
}

/* Table */
table {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}
th {
    background: rgba(124,58,237,0.3) !important;
    color: #c4b5fd !important;
}

/* Caption */
.stCaption {
    color: rgba(167,139,250,0.6) !important;
    text-align: center !important;
}

/* General text */
p, label, .stMarkdown {
    color: #d1d5f0 !important;
}
</style>
""", unsafe_allow_html=True))

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
