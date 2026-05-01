import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
h2, h3 { color: #a78bfa !important; }
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
hr { border-color: rgba(167,139,250,0.2) !important; }
th { background: rgba(124,58,237,0.3) !important; color: #c4b5fd !important; }
td { color: #e0e0ff !important; }
.stCaption { color: rgba(167,139,250,0.6) !important; text-align: center !important; }
p, label, .stMarkdown { color: #d1d5f0 !important; }
.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6b7aab;
    margin: 1.2rem 0 0.5rem;
    padding-left: 2px;
}

/* RESULT BOX */
.result-churn {
    background: linear-gradient(135deg, #3b0a1f, #1f0a2e);
    border: 1px solid #f472b6;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(244,114,182,0.2);
}
.result-safe {
    background: linear-gradient(135deg, #0a2e1f, #0a1f2e);
    border: 1px solid #34d399;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(52,211,153,0.2);
}
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-title-churn { font-size: 1.7rem; font-weight: 800; color: #f472b6; margin-bottom: 0.3rem; }
.result-title-safe  { font-size: 1.7rem; font-weight: 800; color: #34d399; margin-bottom: 0.3rem; }
.result-subtitle { font-size: 0.88rem; color: #9ca3af; margin-bottom: 1.2rem; }
.prob-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.prob-value-churn { font-size: 3.2rem; font-weight: 900; color: #f472b6; line-height: 1; }
.prob-value-safe  { font-size: 3.2rem; font-weight: 900; color: #34d399; line-height: 1; }
.progress-track {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 14px;
    margin: 0.9rem 0;
    overflow: hidden;
}
.progress-fill-churn { height:100%; border-radius:100px; background: linear-gradient(90deg,#f472b6,#db2777); }
.progress-fill-safe  { height:100%; border-radius:100px; background: linear-gradient(90deg,#34d399,#059669); }

/* STATS ROW */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    min-width: 90px;
}
.stat-pill-label { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
.stat-pill-val-churn { font-size: 1.1rem; font-weight: 700; color: #f472b6; }
.stat-pill-val-safe  { font-size: 1.1rem; font-weight: 700; color: #34d399; }

/* RISK METER */
.risk-meter-wrap { margin: 0.8rem auto; max-width: 320px; }
.risk-meter-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #6b7aab;
    margin-bottom: 4px;
}
.risk-track {
    height: 10px;
    border-radius: 100px;
    background: linear-gradient(90deg, #34d399, #f7c06a, #f472b6);
    position: relative;
}
.risk-pointer {
    position: absolute;
    top: -5px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: white;
    border: 3px solid #1a1a2e;
    transform: translateX(-50%);
    box-shadow: 0 0 8px rgba(255,255,255,0.4);
}

/* INSIGHTS */
.insights-wrap {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 1.2rem 0;
    text-align: left;
}
.insight-card-churn {
    background: rgba(244,114,182,0.07);
    border: 1px solid rgba(244,114,182,0.2);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
}
.insight-card-safe {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
}
.insight-icon { font-size: 1.2rem; margin-bottom: 4px; }
.insight-title-churn { font-size: 0.78rem; font-weight: 700; color: #f9a8d4; margin-bottom: 2px; }
.insight-title-safe  { font-size: 0.78rem; font-weight: 700; color: #6ee7b7; margin-bottom: 2px; }
.insight-text { font-size: 0.74rem; color: #9ca3af; line-height: 1.4; }

/* RECOMMENDATION */
.rec-box-churn {
    background: rgba(244,114,182,0.08);
    border: 1px solid rgba(244,114,182,0.25);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-top: 0.5rem;
    color: #f9a8d4;
    font-size: 0.88rem;
    text-align: left;
}
.rec-box-safe {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-top: 0.5rem;
    color: #6ee7b7;
    font-size: 0.88rem;
    text-align: left;
}
.rec-title { font-weight: 700; font-size: 0.82rem; margin-bottom: 0.4rem; }
.rec-item { margin: 0.25rem 0; font-size: 0.82rem; }
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

# ── SECTION 1: Core Inputs ───────────────────────────────────────────────────
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

# ── SECTION 2: Extra Visual Inputs ───────────────────────────────────────────
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
        churn_pct  = round(probability[1] * 100, 1)
        retain_pct = round(probability[0] * 100, 1)
        pointer_pos = churn_pct

        # risk level label
        if churn_pct >= 70:
            risk_level = "🔴 Very High Risk"
        elif churn_pct >= 50:
            risk_level = "🟠 High Risk"
        else:
            risk_level = "🟡 Moderate Risk"

        st.markdown(f"""
        <div class="result-churn">
            <div class="result-emoji">🚨</div>
            <div class="result-title-churn">Customer Will CHURN</div>
            <div class="result-subtitle">This customer is at risk of leaving the service</div>

            <div class="prob-label">Churn Probability</div>
            <div class="prob-value-churn">{churn_pct}%</div>
            <div class="progress-track">
                <div class="progress-fill-churn" style="width:{churn_pct}%"></div>
            </div>

            <div class="stats-row">
                <div class="stat-pill">
                    <div class="stat-pill-label">Risk Level</div>
                    <div class="stat-pill-val-churn">{risk_level}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Stay Chance</div>
                    <div class="stat-pill-val-churn">{retain_pct}%</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Satisfaction</div>
                    <div class="stat-pill-val-churn">{satisfaction}/10</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Complaints</div>
                    <div class="stat-pill-val-churn">{complaints}</div>
                </div>
            </div>

            <div class="risk-meter-wrap">
                <div class="risk-meter-label"><span>Low</span><span>Medium</span><span>High</span></div>
                <div class="risk-track">
                    <div class="risk-pointer" style="left:{pointer_pos}%"></div>
                </div>
            </div>

            <div class="insights-wrap">
                <div class="insight-card-churn">
                    <div class="insight-icon">📉</div>
                    <div class="insight-title-churn">Low Retention</div>
                    <div class="insight-text">Customer shows signs of disengagement based on usage pattern.</div>
                </div>
                <div class="insight-card-churn">
                    <div class="insight-icon">⚠️</div>
                    <div class="insight-title-churn">Income Sensitivity</div>
                    <div class="insight-text">{annual_income} customers are more price-sensitive to service changes.</div>
                </div>
                <div class="insight-card-churn">
                    <div class="insight-icon">✈️</div>
                    <div class="insight-title-churn">Flyer Status</div>
                    <div class="insight-text">{"Frequent flyers tend to compare offers more actively." if frequent_flyer == "Yes" else "Non-frequent flyers have lower brand loyalty."}</div>
                </div>
                <div class="insight-card-churn">
                    <div class="insight-icon">🛎️</div>
                    <div class="insight-title-churn">Services Used</div>
                    <div class="insight-text">Only {services_opted} out of 6 services opted — low engagement level.</div>
                </div>
            </div>

            <div class="rec-box-churn">
                <div class="rec-title">💡 Retention Recommendations</div>
                <div class="rec-item">🎁 Offer a personalized discount or cashback on next booking</div>
                <div class="rec-item">📞 Assign a dedicated customer support agent</div>
                <div class="rec-item">⭐ Upgrade membership tier to increase loyalty</div>
                <div class="rec-item">📧 Send re-engagement email with exclusive offers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        retain_pct = round(probability[0] * 100, 1)
        churn_pct  = round(probability[1] * 100, 1)
        pointer_pos = churn_pct

        if retain_pct >= 80:
            loyalty_level = "🟢 Very Loyal"
        elif retain_pct >= 60:
            loyalty_level = "🟡 Fairly Loyal"
        else:
            loyalty_level = "🟠 Moderately Loyal"

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

            <div class="stats-row">
                <div class="stat-pill">
                    <div class="stat-pill-label">Loyalty Level</div>
                    <div class="stat-pill-val-safe">{loyalty_level}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Churn Risk</div>
                    <div class="stat-pill-val-safe">{churn_pct}%</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Satisfaction</div>
                    <div class="stat-pill-val-safe">{satisfaction}/10</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Membership</div>
                    <div class="stat-pill-val-safe">{membership}</div>
                </div>
            </div>

            <div class="risk-meter-wrap">
                <div class="risk-meter-label"><span>Low</span><span>Medium</span><span>High</span></div>
                <div class="risk-track">
                    <div class="risk-pointer" style="left:{pointer_pos}%"></div>
                </div>
            </div>

            <div class="insights-wrap">
                <div class="insight-card-safe">
                    <div class="insight-icon">💚</div>
                    <div class="insight-title-safe">Strong Retention</div>
                    <div class="insight-text">Customer engagement and usage pattern look healthy and stable.</div>
                </div>
                <div class="insight-card-safe">
                    <div class="insight-icon">💰</div>
                    <div class="insight-title-safe">Income Class</div>
                    <div class="insight-text">{annual_income} customers show stable spending behaviour with the service.</div>
                </div>
                <div class="insight-card-safe">
                    <div class="insight-icon">✈️</div>
                    <div class="insight-title-safe">Flyer Status</div>
                    <div class="insight-text">{"Frequent flyer status adds strong loyalty to the brand." if frequent_flyer == "Yes" else "Customer is consistent even as a non-frequent flyer."}</div>
                </div>
                <div class="insight-card-safe">
                    <div class="insight-icon">🛎️</div>
                    <div class="insight-title-safe">Services Used</div>
                    <div class="insight-text">{services_opted} out of 6 services opted — {"high" if services_opted >= 4 else "moderate"} engagement level.</div>
                </div>
            </div>

            <div class="rec-box-safe">
                <div class="rec-title">💡 Retention Recommendations</div>
                <div class="rec-item">🌟 Reward with loyalty points to maintain engagement</div>
                <div class="rec-item">📬 Send personalized thank-you offers periodically</div>
                <div class="rec-item">🚀 Upsell premium services based on their usage</div>
                <div class="rec-item">📊 Monitor satisfaction score regularly to stay proactive</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Customer Churn Prediction App | Random Forest Model | Deployed via Streamlit Cloud")
