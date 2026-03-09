import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")

st.write("Enter transaction details to check if it is Fraud or Not.")

# -------- Input Fields -------- #

TransactionAmt = st.number_input("Transaction Amount", min_value=1.0)

ProductCD = st.selectbox("ProductCD", ["W", "C", "H", "S"])

card1 = st.number_input("Card1", step=1)

card2 = st.number_input("Card2")

card3 = st.number_input("Card3")

card4 = st.selectbox(
    "Card Type",
    ["visa", "mastercard", "american express", "discover"]
)

card5 = st.number_input("Card5")

card6 = st.selectbox("Card6", ["credit", "debit"])

P_emaildomain = st.text_input("Email Domain", "gmail.com")

DeviceType = st.selectbox("Device Type", ["desktop", "mobile"])

DeviceInfo = st.text_input("Device Info", "Windows")

# -------- Prediction Button -------- #

if st.button("Check Fraud"):

    data = {
        "TransactionAmt": TransactionAmt,
        "ProductCD": ProductCD,
        "card1": int(card1),
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "P_emaildomain": P_emaildomain,
        "DeviceType": DeviceType,
        "DeviceInfo": DeviceInfo
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = response.json()

        st.subheader("Prediction Result")

        if result["result"] == "FRAUD":
            st.error("⚠️ Fraud Transaction Detected")
        else:
            st.success("✅ Safe Transaction")

        st.write("Fraud Probability:", result["fraud_probability"])

    except Exception as e:
        st.error("API connection failed. Make sure FastAPI is running.")
        st.write(e)