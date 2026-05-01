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

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}
h1 {
    background: linear-gradient(90deg, #a78bfa, #f472b6, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    text-align: center;
}
h2, h3 {
    color: #a78bfa !important;
}
div[data-testid="column"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px;
    padding: 1rem !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #db2777) !important;
}
div[data-testid="metric-container"] {
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
hr { border-color: rgba(167,139,250,0.2) !important; }
th { background: rgba(124,58,237,0.3) !important; color: #c4b5fd !important; }
td { color: #e0e0ff !important; }
.stCaption { color: rgba(167,139,250,0.6) !important; text-align: center !important; }
p, label, .stMarkdown { color: #d1d5f0 !important; }

/* Section divider label */
.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6b7aab;
    margin: 1.2rem 0 0.5rem;
    padding-left: 2px;
}

/* Result boxes */
.result-churn {
    background: linear-gradient(135deg, #3b0a1f, #1f0a2e);
    border: 1px solid #f472b6;
    border-radius: 20px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(244,114,182,0.2);
}
.result-safe {
    background: linear-gradient(135deg, #0a2e1f, #0a1f2e);
    border: 1px solid #34d399;
    border-radius: 20px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(52,211,153,0.2);
}
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-title-churn { font-size: 1.7rem; font-weight: 800; color: #f472b6; margin-bottom: 0.3rem; }
.result-title-safe  { font-size: 1.7rem; font-weight: 800; color: #34d399; margin-bottom: 0.3rem; }
.result-subtitle { font-size: 0.88rem; color: #9ca3af; margin-bottom: 1.2rem; }
.prob-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.prob-value-churn { font-size: 3rem; font-weight: 900; color: #f472b6; line-height: 1; }
.prob-value-safe  { font-size: 3rem; font-weight: 900; color: #34d399; line-height: 1; }
.progress-track {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 14px;
    margin: 0.9rem 0;
    overflow: hidden;
}
.progress-fill-churn { height:100%; border-radius:100px; background: linear-gradient(90deg,#f472b6,#db2777); }
.progress-fill-safe  { height:100%; border-radius:100px; background: linear-gradient(90deg,#34d399,#059669); }
.rec-box-churn {
    background: rgba(244,114,182,0.08);
    border: 1px solid rgba(244,114,182,0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-top: 1rem;
    color: #f9a8d4;
    font-size: 0.88rem;
    text-align: left;
}
.rec-box-safe {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-top: 1rem;
    color: #6ee7b7;
    font-size: 0.88rem;
    text-align: left;
}

/* Summary card */
.summary-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-top: 1.2rem;
}
.summary-title {
    color: #a78bfa;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.8rem;
    font-weight: 600;
}
.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: #9ca3af; }
.summary-val { color: #e0e0ff; font-weight: 600; }
.badge-visual {
    font-size: 0.72rem;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    color: #a78bfa;
    border-radius: 100px;
    padding: 1px 8px;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")

st.markdown("""
<div style='text-align:center; margin-bottom:0.8rem;'>
  <span style='background:rgba(167,139,250,0.15); border:1px solid rgba(167,139,250,0.3);
  border-radius:100px; padding:5px 18px; font-size:0.82rem; color:#a78bfa; letter-spacing:1px;'>
  ✦ &nbsp;B.Tech Gen AI &nbsp;|&nbsp; Harshil Parmar &nbsp;|&nbsp; KU2507U0511&nbsp; ✦
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 1: Core Inputs (used by model) ───────────────────────────────────
st.markdown("<div class='section-label'>🔵 Core Customer Info</div>", unsafe_allow_html=True)
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

# ── SECTION 2: Extra Visual Inputs (display only, not used by model) ─────────
st.markdown("<div class='section-label'>🟣 Additional Customer Profile</div>", unsafe_allow_html=True)
st.subheader("Extended Details")

col3, col4 = st.columns(2)
with col3:
    flight_frequency = st.slider("Flights per Year", min_value=0, max_value=50, value=10)
    travel_class = st.selectbox("Preferred Travel Class", ["Economy", "Business", "First Class"])
    loyalty_points = st.slider("Loyalty Points (in thousands)", min_value=0, max_value=100, value=25)
with col4:
    complaints = st.selectbox("Complaints Raised?", ["No", "Yes"])
    satisfaction = st.slider("Satisfaction Score (1-10)", min_value=1, max_value=10, value=7)
    membership = st.selectbox("Membership Tier", ["Silver", "Gold", "Platinum"])

st.markdown("---")

# ── ENCODE ───────────────────────────────────────────────────────────────────
def encode_inputs(age, frequent_flyer, annual_income, services_opted, social_media, booked_hotel):
    ff_map     = {"Yes": 1, "No": 0}
    income_map = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
    yn_map     = {"Yes": 1, "No": 0}
    return np.array([[
        age,
        ff_map[frequent_flyer],
        income_map[annual_income],
        services_opted,
        yn_map[social_media],
        yn_map[booked_hotel]
    ]])

# ── PREDICT ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):
    input_data  = encode_inputs(age, frequent_flyer, annual_income, services_opted, social_media, booked_hotel)
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        churn_pct = round(probability[1] * 100, 1)
        st.markdown(f"""
        <div class="result-churn">
            <div class="result-emoji">🚨</div>
            <div class="result-title-churn">Customer Will CHURN</div>
            <div class="result-subtitle">This customer is at high risk of leaving the service</div>
            <div class="prob-label">Churn Probability</div>
            <div class="prob-value-churn">{churn_pct}%</div>
            <div class="progress-track">
                <div class="progress-fill-churn" style="width:{churn_pct}%"></div>
            </div>
            <div class="rec-box-churn">
                💡 <strong>Recommendation:</strong> Offer a personalized discount, loyalty reward points,
                or a dedicated support callback to retain this customer before they leave.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        retain_pct = round(probability[0] * 100, 1)
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-emoji">✅</div>
            <div class="result-title-safe">Customer is SAFE</div>
            <div class="result-subtitle">This customer is likely to continue using the service</div>
            <div class="prob-label">Retention Probability</div>
            <div class="prob-value-safe">{retain_pct}%</div>
            <div class="progress-track">
                <div class="progress-fill-safe" style="width:{retain_pct}%"></div>
            </div>
            <div class="rec-box-safe">
                💡 <strong>Recommendation:</strong> Continue providing quality service and
                occasional loyalty perks to maintain long-term customer satisfaction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── INPUT SUMMARY CARD ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">📋 Full Input Summary</div>

        <div class="summary-row">
            <span class="summary-key">Age</span>
            <span class="summary-val">{age} yrs</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Frequent Flyer</span>
            <span class="summary-val">{frequent_flyer}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Annual Income Class</span>
            <span class="summary-val">{annual_income}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Services Opted</span>
            <span class="summary-val">{services_opted}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Social Media Synced</span>
            <span class="summary-val">{social_media}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Booked Hotel</span>
            <span class="summary-val">{booked_hotel}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Flights per Year <span class="badge-visual">visual</span></span>
            <span class="summary-val">{flight_frequency}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Travel Class <span class="badge-visual">visual</span></span>
            <span class="summary-val">{travel_class}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Loyalty Points <span class="badge-visual">visual</span></span>
            <span class="summary-val">{loyalty_points}K pts</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Complaints Raised <span class="badge-visual">visual</span></span>
            <span class="summary-val">{complaints}</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Satisfaction Score <span class="badge-visual">visual</span></span>
            <span class="summary-val">{satisfaction} / 10</span>
        </div>
        <div class="summary-row">
            <span class="summary-key">Membership Tier <span class="badge-visual">visual</span></span>
            <span class="summary-val">{membership}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Customer Churn Prediction App | Random Forest Model | Deployed via Streamlit Cloud")
