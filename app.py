import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components

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

# ── RESULT HTML BUILDER ───────────────────────────────────────────────────────
def build_result_html(is_churn, churn_pct, retain_pct, risk_or_loyalty_label,
                      satisfaction, complaints, membership, annual_income,
                      frequent_flyer, services_opted):
    pointer_pos = churn_pct

    if is_churn:
        color_main = "#f472b6"
        color_grad = "linear-gradient(90deg,#f472b6,#db2777)"
        box_bg = "linear-gradient(135deg, #3b0a1f, #1f0a2e)"
        box_border = "#f472b6"
        box_shadow = "rgba(244,114,182,0.2)"
        title_text = "Customer Will CHURN"
        subtitle_text = "This customer is at risk of leaving the service"
        emoji = "🚨"
        prob_label = "Churn Probability"
        prob_value = churn_pct
        stat1_label = "Risk Level"
        stat1_val = risk_or_loyalty_label
        stat2_label = "Stay Chance"
        stat2_val = f"{retain_pct}%"
        insight_bg = "rgba(244,114,182,0.07)"
        insight_border = "rgba(244,114,182,0.2)"
        insight_title_color = "#f9a8d4"
        rec_bg = "rgba(244,114,182,0.08)"
        rec_border = "rgba(244,114,182,0.25)"
        rec_color = "#f9a8d4"
        flyer_text = ("Frequent flyers tend to compare offers more actively."
                      if frequent_flyer == "Yes" else "Non-frequent flyers have lower brand loyalty.")
        services_text = f"Only {services_opted} out of 6 services opted — low engagement level."
        income_insight = f"{annual_income} customers are more price-sensitive to service changes."
        rec_items = [
            "🎁 Offer a personalized discount or cashback on next booking",
            "📞 Assign a dedicated customer support agent",
            "⭐ Upgrade membership tier to increase loyalty",
            "📧 Send re-engagement email with exclusive offers"
        ]
    else:
        color_main = "#34d399"
        color_grad = "linear-gradient(90deg,#34d399,#059669)"
        box_bg = "linear-gradient(135deg, #0a2e1f, #0a1f2e)"
        box_border = "#34d399"
        box_shadow = "rgba(52,211,153,0.2)"
        title_text = "Customer is SAFE"
        subtitle_text = "This customer is likely to continue using the service"
        emoji = "✅"
        prob_label = "Retention Probability"
        prob_value = retain_pct
        stat1_label = "Loyalty Level"
        stat1_val = risk_or_loyalty_label
        stat2_label = "Churn Risk"
        stat2_val = f"{churn_pct}%"
        insight_bg = "rgba(52,211,153,0.07)"
        insight_border = "rgba(52,211,153,0.2)"
        insight_title_color = "#6ee7b7"
        rec_bg = "rgba(52,211,153,0.08)"
        rec_border = "rgba(52,211,153,0.25)"
        rec_color = "#6ee7b7"
        flyer_text = ("Frequent flyer status adds strong loyalty to the brand."
                      if frequent_flyer == "Yes" else "Customer is consistent even as a non-frequent flyer.")
        engagement = "high" if services_opted >= 4 else "moderate"
        services_text = f"{services_opted} out of 6 services opted — {engagement} engagement level."
        income_insight = f"{annual_income} customers show stable spending behaviour with the service."
        rec_items = [
            "🌟 Reward with loyalty points to maintain engagement",
            "📬 Send personalized thank-you offers periodically",
            "🚀 Upsell premium services based on their usage",
            "📊 Monitor satisfaction score regularly to stay proactive"
        ]

    rec_html = "".join(
        f'<div style="margin:0.3rem 0;font-size:0.84rem;color:{rec_color};">{item}</div>'
        for item in rec_items
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
      body {{ background: transparent; padding: 0; }}
      .card {{
        background: {box_bg};
        border: 1px solid {box_border};
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 0 40px {box_shadow};
      }}
      .emoji {{ font-size: 3rem; margin-bottom: 0.4rem; }}
      .title {{ font-size: 1.7rem; font-weight: 800; color: {color_main}; margin-bottom: 0.2rem; }}
      .subtitle {{ font-size: 0.88rem; color: #9ca3af; margin-bottom: 1.2rem; }}
      .prob-label {{ font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }}
      .prob-value {{ font-size: 3.2rem; font-weight: 900; color: {color_main}; line-height: 1; }}
      .progress-track {{
        background: rgba(255,255,255,0.08);
        border-radius: 100px;
        height: 14px;
        margin: 0.9rem 0;
        overflow: hidden;
      }}
      .progress-fill {{
        height: 100%;
        border-radius: 100px;
        background: {color_grad};
        width: {prob_value}%;
      }}
      .stats-row {{
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 1.2rem 0;
        flex-wrap: wrap;
      }}
      .stat-pill {{
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        min-width: 85px;
      }}
      .stat-pill-label {{ font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }}
      .stat-pill-val {{ font-size: 1rem; font-weight: 700; color: {color_main}; margin-top: 2px; }}
      .risk-meter-wrap {{ margin: 0.8rem auto; max-width: 320px; }}
      .risk-meter-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #6b7aab;
        margin-bottom: 4px;
      }}
      .risk-track {{
        height: 10px;
        border-radius: 100px;
        background: linear-gradient(90deg, #34d399, #f7c06a, #f472b6);
        position: relative;
      }}
      .risk-pointer {{
        position: absolute;
        top: -5px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: white;
        border: 3px solid #1a1a2e;
        transform: translateX(-50%);
        left: {pointer_pos}%;
        box-shadow: 0 0 8px rgba(255,255,255,0.4);
      }}
      .insights-wrap {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 1.2rem 0;
        text-align: left;
      }}
      .insight-card {{
        background: {insight_bg};
        border: 1px solid {insight_border};
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
      }}
      .insight-icon {{ font-size: 1.2rem; margin-bottom: 4px; }}
      .insight-title {{ font-size: 0.78rem; font-weight: 700; color: {insight_title_color}; margin-bottom: 2px; }}
      .insight-text {{ font-size: 0.74rem; color: #9ca3af; line-height: 1.4; }}
      .rec-box {{
        background: {rec_bg};
        border: 1px solid {rec_border};
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 0.5rem;
        text-align: left;
      }}
      .rec-title {{ font-weight: 700; font-size: 0.85rem; color: {rec_color}; margin-bottom: 0.5rem; }}
    </style>
    </head>
    <body>
    <div class="card">
      <div class="emoji">{emoji}</div>
      <div class="title">{title_text}</div>
      <div class="subtitle">{subtitle_text}</div>
      <div class="prob-label">{prob_label}</div>
      <div class="prob-value">{prob_value}%</div>
      <div class="progress-track"><div class="progress-fill"></div></div>

      <div class="stats-row">
        <div class="stat-pill">
          <div class="stat-pill-label">{stat1_label}</div>
          <div class="stat-pill-val">{stat1_val}</div>
        </div>
        <div class="stat-pill">
          <div class="stat-pill-label">{stat2_label}</div>
          <div class="stat-pill-val">{stat2_val}</div>
        </div>
        <div class="stat-pill">
          <div class="stat-pill-label">Satisfaction</div>
          <div class="stat-pill-val">{satisfaction}/10</div>
        </div>
        <div class="stat-pill">
          <div class="stat-pill-label">{"Complaints" if is_churn else "Membership"}</div>
          <div class="stat-pill-val">{complaints if is_churn else membership}</div>
        </div>
      </div>

      <div class="risk-meter-wrap">
        <div class="risk-meter-label"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="risk-track"><div class="risk-pointer"></div></div>
      </div>

      <div class="insights-wrap">
        <div class="insight-card">
          <div class="insight-icon">{"📉" if is_churn else "💚"}</div>
          <div class="insight-title">{"Low Retention" if is_churn else "Strong Retention"}</div>
          <div class="insight-text">{"Customer shows signs of disengagement based on usage pattern." if is_churn else "Customer engagement and usage pattern look healthy and stable."}</div>
        </div>
        <div class="insight-card">
          <div class="insight-icon">{"⚠️" if is_churn else "💰"}</div>
          <div class="insight-title">{"Income Sensitivity" if is_churn else "Income Class"}</div>
          <div class="insight-text">{income_insight}</div>
        </div>
        <div class="insight-card">
          <div class="insight-icon">✈️</div>
          <div class="insight-title">Flyer Status</div>
          <div class="insight-text">{flyer_text}</div>
        </div>
        <div class="insight-card">
          <div class="insight-icon">🛎️</div>
          <div class="insight-title">Services Used</div>
          <div class="insight-text">{services_text}</div>
        </div>
      </div>

      <div class="rec-box">
        <div class="rec-title">💡 Retention Recommendations</div>
        {rec_html}
      </div>
    </div>
    </body>
    </html>
    """
    return html


# ── PREDICT ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):
    input_data = encode_inputs(age, frequent_flyer, annual_income, services_opted, social_media, booked_hotel)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    churn_pct = round(probability[1] * 100, 1)
    retain_pct = round(probability[0] * 100, 1)

    if prediction == 1:
        if churn_pct >= 70:
            risk_label = "🔴 Very High Risk"
        elif churn_pct >= 50:
            risk_label = "🟠 High Risk"
        else:
            risk_label = "🟡 Moderate Risk"

        result_html = build_result_html(
            is_churn=True,
            churn_pct=churn_pct,
            retain_pct=retain_pct,
            risk_or_loyalty_label=risk_label,
            satisfaction=satisfaction,
            complaints=complaints,
            membership=membership,
            annual_income=annual_income,
            frequent_flyer=frequent_flyer,
            services_opted=services_opted
        )
    else:
        if retain_pct >= 80:
            loyalty_label = "🟢 Very Loyal"
        elif retain_pct >= 60:
            loyalty_label = "🟡 Fairly Loyal"
        else:
            loyalty_label = "🟠 Moderately Loyal"

        result_html = build_result_html(
            is_churn=False,
            churn_pct=churn_pct,
            retain_pct=retain_pct,
            risk_or_loyalty_label=loyalty_label,
            satisfaction=satisfaction,
            complaints=complaints,
            membership=membership,
            annual_income=annual_income,
            frequent_flyer=frequent_flyer,
            services_opted=services_opted
        )

    # ✅ KEY FIX: Use components.html instead of st.markdown to render HTML properly
    components.html(result_html, height=780, scrolling=False)

st.markdown("---")
st.caption("Customer Churn Prediction App | Random Forest Model | Deployed via Streamlit Cloud")
