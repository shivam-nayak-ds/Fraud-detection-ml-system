# fraud-detection-api/streamlit_app.py
# ── Streamlit UI for Fraud Detection ─────────────────────────────────────────

import streamlit as st
import requests
import json

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { max-width: 800px; margin: 0 auto; }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
    }
    .fraud-card {
        background: linear-gradient(135deg, #FF4B4B22, #FF4B4B11);
        border: 1px solid #FF4B4B;
    }
    .safe-card {
        background: linear-gradient(135deg, #00CC6622, #00CC6611);
        border: 1px solid #00CC66;
    }
    .metric-value {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔐 Fraud Detection System")
st.markdown("Enter transaction details to get real-time fraud prediction.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("fraud_form"):
    st.markdown("### 💳 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        TransactionAmt = st.number_input(
            "Transaction Amount ($)", min_value=0.01, value=100.0, step=10.0
        )
        card2 = st.number_input("Card 2 (ID)", value=111.0)
        card6 = st.selectbox("Card Type", ["credit", "debit"])
        P_emaildomain = st.selectbox(
            "Email Domain",
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "other"]
        )
        DeviceInfo = st.text_input("Device Info", "Windows")

    with col2:
        C2  = st.number_input("C2",  value=0.0, step=1.0)
        C4  = st.number_input("C4",  value=0.0, step=1.0)
        C7  = st.number_input("C7",  value=0.0, step=1.0)
        C8  = st.number_input("C8",  value=0.0, step=1.0)
        C10 = st.number_input("C10", value=0.0, step=1.0)

    st.markdown("### 🔢 Additional Features")
    col3, col4, col5 = st.columns(3)

    with col3:
        C11  = st.number_input("C11",  value=0.0, step=1.0)
        C13  = st.number_input("C13",  value=0.0, step=1.0)
        C14  = st.number_input("C14",  value=0.0, step=1.0)

    with col4:
        M4 = st.selectbox("M4", ["M0", "M1", "M2", "M3"])
        M5 = st.selectbox("M5", ["F", "T"])
        M6 = st.selectbox("M6", ["F", "T"])

    with col5:
        V102 = st.number_input("V102", value=0.0, step=0.1)
        V280 = st.number_input("V280", value=0.0, step=0.1)
        V283 = st.number_input("V283", value=0.0, step=0.1)
        V294 = st.number_input("V294", value=0.0, step=0.1)

    submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "TransactionAmt": TransactionAmt,
        "card2": card2,
        "card6": card6,
        "P_emaildomain": P_emaildomain,
        "C2": C2, "C4": C4, "C7": C7, "C8": C8, "C10": C10,
        "C11": C11, "C13": C13, "C14": C14,
        "M4": M4, "M5": M5, "M6": M6,
        "V102": V102, "V280": V280, "V283": V283, "V294": V294,
        "DeviceInfo": DeviceInfo,
    }

    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=10
            )
            result = response.json()

            st.markdown("---")

            if result["result"] == "FRAUD":
                st.markdown(f"""
                <div class="result-card fraud-card">
                    <h2>⚠️ FRAUD DETECTED</h2>
                    <div class="metric-value" style="color: #FF4B4B;">
                        {result['fraud_probability']*100:.1f}%
                    </div>
                    <p>Fraud Probability</p>
                </div>
                """, unsafe_allow_html=True)
                st.error("🚨 This transaction is flagged as **potentially fraudulent**. Recommend blocking.")
            else:
                st.markdown(f"""
                <div class="result-card safe-card">
                    <h2>✅ SAFE TRANSACTION</h2>
                    <div class="metric-value" style="color: #00CC66;">
                        {(1 - result['fraud_probability'])*100:.1f}%
                    </div>
                    <p>Safety Score</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ This transaction appears **legitimate**. Safe to process.")

            # Details expander
            with st.expander("📊 Detailed Response"):
                st.json(result)

        except requests.exceptions.ConnectionError:
            st.error("❌ **API not reachable.** Make sure FastAPI is running on port 8000.")
            st.code("uvicorn main:app --reload", language="bash")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>Built with FastAPI + Streamlit | Fraud Detection ML System</p>",
    unsafe_allow_html=True
)