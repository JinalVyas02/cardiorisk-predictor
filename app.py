import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="CardioRisk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  GLOBAL STYLING — indigo sidebar + lavender canvas + cards
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: #eef0fb;
    }

    /* ---- Sidebar (dark indigo) ---- */
    [data-testid="stSidebar"] {
        background: #2b2a6e;
    }
    [data-testid="stSidebar"] * {
        color: #b9b7e8 !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
    }

    /* ---- Greeting card ---- */
    .greeting-card {
        background: #ffffff;
        border-radius: 22px;
        padding: 22px 30px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }
    .greeting-card h1 {
        font-size: 22px;
        color: #2b2a6e;
        font-weight: 800;
        margin: 0;
    }
    .greeting-card h1 span { color: #ff6b81; }
    .greeting-card p {
        color: #9a9ab8;
        font-size: 13px;
        margin-top: 6px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 700;
        color: #7d7da0;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin: 18px 0 8px 4px;
    }

    /* ---- Colored vitals cards (via container key) ---- */
    div[class*="st-key-card_age"],
    div[class*="st-key-card_height"],
    div[class*="st-key-card_weight"],
    div[class*="st-key-card_aphi"],
    div[class*="st-key-card_aplo"],
    div[class*="st-key-card_gender"],
    div[class*="st-key-card_chol"],
    div[class*="st-key-card_gluc"],
    div[class*="st-key-card_smoke"],
    div[class*="st-key-card_alco"],
    div[class*="st-key-card_active"] {
        border-radius: 18px;
        padding: 14px 16px 6px 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
    div[class*="st-key-card_age"]    { background: linear-gradient(135deg,#63c7f0,#3ea8e0); }
    div[class*="st-key-card_height"] { background: linear-gradient(135deg,#7d78e8,#5b53d6); }
    div[class*="st-key-card_weight"] { background: linear-gradient(135deg,#5fe3c0,#2bc5a0); }
    div[class*="st-key-card_aphi"]   { background: linear-gradient(135deg,#ff8a9e,#ff6b81); }
    div[class*="st-key-card_aplo"]   { background: linear-gradient(135deg,#ff9fd6,#ff6fc3); }
    div[class*="st-key-card_gender"] { background: linear-gradient(135deg,#ffd76b,#ffb648); }

    div[class*="st-key-card_age"] label,
    div[class*="st-key-card_height"] label,
    div[class*="st-key-card_weight"] label,
    div[class*="st-key-card_aphi"] label,
    div[class*="st-key-card_aplo"] label,
    div[class*="st-key-card_gender"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 12px !important;
    }
    div[class*="st-key-card_age"] input,
    div[class*="st-key-card_height"] input,
    div[class*="st-key-card_weight"] input,
    div[class*="st-key-card_aphi"] input,
    div[class*="st-key-card_aplo"] input {
        background: rgba(255,255,255,0.28) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
    }
    div[class*="st-key-card_gender"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.28) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
    }

    /* ---- White lifestyle cards ---- */
    div[class*="st-key-card_chol"],
    div[class*="st-key-card_gluc"],
    div[class*="st-key-card_smoke"],
    div[class*="st-key-card_alco"],
    div[class*="st-key-card_active"] {
        background: #ffffff;
    }
    div[class*="st-key-card_chol"] label,
    div[class*="st-key-card_gluc"] label,
    div[class*="st-key-card_smoke"] label,
    div[class*="st-key-card_alco"] label,
    div[class*="st-key-card_active"] label {
        color: #7d7da0 !important;
        font-weight: 700 !important;
        font-size: 12px !important;
    }

    /* ---- Predict button ---- */
    div[class*="st-key-predict_btn"] button {
        background: linear-gradient(135deg,#ff6b81,#ff4d6d) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 12px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 20px rgba(255,107,129,0.35);
        width: 100%;
    }

    /* ---- Right result panel ---- */
    div[class*="st-key-result_panel"] {
        background: #2b2a6e;
        border-radius: 22px;
        padding: 24px 20px;
        color: #ffffff;
    }
    div[class*="st-key-result_panel"] * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR (decorative nav, matches reference image)
# ============================================================
with st.sidebar:
    st.markdown("## ❤️")
    st.markdown("🏠 &nbsp; Dashboard")
    st.markdown("📊 &nbsp; Reports")
    st.markdown("👤 &nbsp; Profile")
    st.markdown("⚙️ &nbsp; Settings")

# ============================================================
#  LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# ============================================================
#  MAIN LAYOUT: content (left) + result panel (right)
# ============================================================
main_col, right_col = st.columns([3, 1], gap="medium")

with main_col:
    st.markdown(
        '<div class="greeting-card"><h1>Hello, <span>Patient</span> 👋</h1>'
        '<p>Fill in your health details below to check your cardiovascular risk.</p></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
    v1, v2, v3, v4, v5, v6 = st.columns(6)

    with v1:
        with st.container(key="card_age"):
            age_years = st.number_input("🎂 Age", min_value=1, max_value=120, value=50, label_visibility="visible")
    with v2:
        with st.container(key="card_height"):
            height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=165)
    with v3:
        with st.container(key="card_weight"):
            weight = st.number_input("⚖️ Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
    with v4:
        with st.container(key="card_aphi"):
            ap_hi = st.number_input("❤️ Systolic BP", min_value=50, max_value=250, value=120)
    with v5:
        with st.container(key="card_aplo"):
            ap_lo = st.number_input("💓 Diastolic BP", min_value=30, max_value=200, value=80)
    with v6:
        with st.container(key="card_gender"):
            gender_label = st.selectbox("⚧ Gender", ["Female", "Male"])

    st.markdown('<div class="section-label">Lifestyle & Labs</div>', unsafe_allow_html=True)
    l1, l2, l3, l4, l5 = st.columns(5)

    with l1:
        with st.container(key="card_chol"):
            cholesterol_label = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    with l2:
        with st.container(key="card_gluc"):
            gluc_label = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    with l3:
        with st.container(key="card_smoke"):
            smoke_label = st.selectbox("Smokes?", ["No", "Yes"])
    with l4:
        with st.container(key="card_alco"):
            alco_label = st.selectbox("Alcohol?", ["No", "Yes"])
    with l5:
        with st.container(key="card_active"):
            active_label = st.selectbox("Active?", ["Yes", "No"])

    st.write("")
    with st.container(key="predict_btn"):
        submitted = st.button("🔍 Predict My Risk", use_container_width=True)

with right_col:
    with st.container(key="result_panel"):
        st.markdown("#### 🧑‍⚕️ Prediction Result")
        st.caption("Based on your inputs")

        if not model_ready:
            st.warning("model.pkl / scaler.pkl not found. Run train_model.py first.")
        elif submitted:
            gender = 1 if gender_label == "Female" else 2
            chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            cholesterol = chol_map[cholesterol_label]
            gluc = chol_map[gluc_label]
            smoke = 1 if smoke_label == "Yes" else 0
            alco = 1 if alco_label == "Yes" else 0
            active = 1 if active_label == "Yes" else 0
            age_days = age_years * 365.25

            input_data = np.array([[
                age_days, gender, height, weight,
                ap_hi, ap_lo, cholesterol, gluc,
                smoke, alco, active
            ]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            risk_word = "High" if prediction == 1 else "Low"
            st.markdown(f"**Estimated Risk**")
            st.markdown(f"## {risk_word} — {probability*100:.1f}%")
            st.progress(min(max(probability, 0.0), 1.0))

            st.markdown("---")
            st.markdown(f"Age: **{age_years} yrs**  \nGender: **{gender_label}**")
            st.markdown(f"Height: **{height}cm**  \nWeight: **{weight}kg**")
            st.markdown(f"BP: **{ap_hi}/{ap_lo}**")
            st.markdown(f"Cholesterol: **{cholesterol_label}**  \nGlucose: **{gluc_label}**")
            st.markdown(f"Smoke: **{smoke_label}**  \nAlcohol: **{alco_label}**")
            st.markdown(f"Active: **{active_label}**")
        else:
            st.info("Fill in the form and click **Predict My Risk** to see your result here.")

        st.caption("This is a statistical estimate, not a medical diagnosis. Consult a healthcare professional.")
