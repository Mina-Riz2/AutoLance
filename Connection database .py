import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import time
import os

# 1. Page settings and making them responsive
st.set_page_config(page_title="AutoLance - Job Hourly Rate Predictor", page_icon="💰", layout="centered")

# 2. Professional CSS implementation, including enlarging the word "AUTOLANCE" and injecting
# the encoded background to ensure 100% visibility.
st.markdown("""
    <style>
    /* إلغاء أي قيود مساحة من Streamlit لجعل الصفحة تمتد بالكامل */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* حقن الخلفية المتدرجة الأصلية مدمج معها كافة الأشكال والجرافيكس مشفرة بالكامل */
    .stApp {
        background-color: #e2f4ed !important;
        background-image: 
            /* 1. Subtle grid dots pattern */
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMCIgaGVpZ2h0PSIzMCI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjEuNSIgZmlsbD0iIzBhNWU0OCIgb3BhY2l0eT0iMC4wNiIvPjwvc3ZnPg=="),
            /* 2. TOP LEFT: bar chart + trend line + donut */
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iNDAwIj48ZyBvcGFjaXR5PSIwLjIyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMCwyMCkiPjxyZWN0IHg9IjAiIHk9IjcyIiB3aWR0aD0iMTgiIGhlaWdodD0iMzgiIHJ4PSI0IiBmaWxsPSIjMWQ5ZTc1Ii8+PHJlY3QgeD0iMjQiIHk9IjUwIiB3aWR0aD0iMTgiIGhlaWdodD0iNjAiIHJ4PSI0IiBmaWxsPSIjMGY2ZTU2Ii8+PHJlY3QgeD0iNDgiIHk9IjYwIiB3aWR0aD0iMTgiIGhlaWdodD0iNTAiIHJ4PSI0IiBmaWxsPSIjNWRjYWE1Ii8+PHJlY3QgeD0iNzIiIHk9IjMwIiB3aWR0aD0iMTgiIGhlaWdodD0iODAiIHJ4PSI0IiBmaWxsPSIjMWQ5ZTc1Ii8+PHJlY3QgeD0iOTYiIHk9IjQ0IiB3aWR0aD0iMTgiIGhlaWdodD0iNjYiIHJ4PSI0IiBmaWxsPSIjMGY2ZTU2Ii8+PGxpbmUgeDE9IjAiIHkxPSIxMTIiIHgyPSIxMTgiIHkyPSIxMTIiIHN0cm9rZT0iIzBhNWU0OCIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9nPjxnIG9wYWNpdH09IjAuMTgiIHRyYW5zb6Zmcm09InRyYW5zbGF0ZSgxNSwxNTApIj48cG9seWxpbmUgcG9pbnRzPSIwLDU1IDIyLDM4IDQ0LDQyIDY2LDIwIDg4LDI1IDExMCw4IiBmaWxsPSJub25lIiBzdHJva2U9IiMxZDllNzUiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PGNpcmNsZSBjeD0iMTEwIiBjeT0iOCIgcj0iNSIgZmlsbD0iIzBmNmU1NiIvPkdjaXJjbGUgY3g9IjY2IiBjeT0iMjAiIHI9IjQiIGZpbGw9IiM1ZGNhYTUiLz48Lz48ZyBvcGFjaXR5PSIwLjIiIHRyYW5zb2Zvcm09InRyYW5zbGF0ZSgyNSwyNTApIj48Y2lyY2xlIGN4PSIzOCIgY3k9IjM4IiByZD0iMzIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2QwZWRlNiIgc3Ryb2tlLXdpZHRoPSIxMyIvPjxwYXRoIGQ9Ik0zOCw2IEEzMiwzMiAwIDAgLDEgNzAsMzgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzFkOWU3NSIgc3Ryb2tlLXdpZHRoPSIxMyIvPjxwYXRoIGQ9Ik03MCwzOCBBMzIsMzIgMCAwLDEgMjAsNjIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzVkY2FhNSIgc3Ryb2tlLXdpZHRoPSIxMyIvPjwvZz48L3N2Zz4="),
            /* 3. TOP RIGHT: line chart with area */
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNTAiIGhlaWdodD0iaGVpZ2h0Ij48ZyBvcGFjaXR5PSIwLjIiIHRyYW5zb2Zvcm09InRyYW5zbGF0ZSgyMCwyMCkiPjxwb2x5bGluZSBwb2ludHM9IjAsODAgMTgsNTUgMzYsNjIgNTQsMzUgNzIsNDIgOTAsMTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBmNmU1NiIgc3Ryb2tlLXdpZHRoPSIyLjUiLz48Y2lyY2xlIGN4PSI5MCIgY3k9IjE4IiByPSI1IiBmaWxsPSIjMGE1ZTQ4Ii8+PC9nPjwvc3ZnPg=="),
            /* 4. BOTTOM RIGHT: pie chart */
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNTAiIGhlaWdodD0iMTUwIj48ZyBvcGFjaXR5PSIwLjIiIHRyYW5zb2Zvcm09InRyYW5zbGF0ZSg0MCw0MCkiPkdjaXJjbGUgY3g9IjM0IiBjeT0iMzQiIHI9IjMwIiBmaWxsPSJub25lIiBzdHJva2U9IiNkMGVkZTYiIHN0cm9rZS13aWR0aD0iMTIiLz48cGF0aCBkPSJNMzQsNCBBMzAsMzAgMCAwLDEgNjQsMzQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzFkOWU3NSIgc3Ryb2tlLXdpZHRoPSIxMiIvPjwvZz48L3N2Zz4="),
            /* 5. BOTTOM LEFT: scatter dots */
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNTAiIGhlaWdodD0iMTUwIj48ZyBvcGFjaXR5PSIwLjE3IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMCwyMCkiPjxjaXJjbGUgY3g9IjEwIiBjeT0iNjAiIHI9IjYiIGZpbGw9IiMxZDllNzUiLz48Y2lyY2xlIGN4PSIyOCIgY3k9IjQwIiByPSI1IiBmaWxsPSIjNWRjYWE1Ii8+<circleIGN4PSI1MCIgY3k9IjUwIiByPSI3IiBmaWxsPSIjMGY2ZTU2Ii8+PGNpcmNsZSBjeT0iNzAiIGN5PSIyNSIgcj0iNSIgZmlsbD0iIzFkOWU3NSIvPjxwb2x5bGluZSBwb2ludHM9IjEwLDYwIDI4LDQwIDUwLDUwIDcwLDI1IiBmaWxsPSJub25lIiBzdHJva2U9IiMxZDllNzUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI0IDMiLz48L2c+PC9zdmc+"),
            /* 6. Original Gradient Background */
            linear-gradient(140deg, #e2f4ed 0%, #cce8df 35%, #c0dff0 70%, #d4eaf8 100%) !important;

        background-position: left top, left top, right top, right bottom, left bottom, center !important;
        background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat !important;
        background-size: auto, auto, auto, auto, auto, cover !important;
        background-attachment: fixed !important;
        min-height: 100vh !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* تكبير وتفخيم كلمة AUTO LANCE بناءً على طلبك */
    .brand-bar {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .brand-name {
        font-size: 52px; /* كبرنا الخط جداً */
        font-weight: 900;
        letter-spacing: 10px; /* زيادة المسافات لتكون فخمة ومطابقة للبراندات العالمية */
        color: #0a5e48;
        text-transform: uppercase;
        text-shadow: 0 4px 12px rgba(15,110,86,0.2);
        margin-bottom: 0px;
    }
    .brand-name span {
        color: #1d9e75;
    }
    .brand-tagline {
        font-size: 13px;
        color: #3b7a70;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 6px;
        margin-bottom: 25px;
        font-weight: 600;
    }

    /* العنوان الفرعي للكارت */
    .main-title-custom {
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #1a3a30;
        margin-bottom: 18px;
    }

    /* تصميم الـ Form Card الأبيض الشفاف الشيك */
    div[data-testid="stContainer"] {
        background: rgba(255, 255, 255, 0.94) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 16px !important;
        padding: 30px 35px !important;
        border: 1px solid rgba(157,217,196,0.5) !important;
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.06), 0 10px 15px -5px rgba(0, 0, 0, 0.04) !important;
        max-width: 540px !important;
        margin: 0 auto !important;
    }

    /* تخصيص الـ Labels الجانبية */
    label {
        font-size: 13px !important;
        color: #444444 !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }

    /* تصميم الـ Inputs */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background-color: #f7fbf9 !important;
        border: 1px solid #c8ddd8 !important;
        border-radius: 8px !important;
        color: #222222 !important;
    }

    /* زرار التنبؤ المتدرج الرائع */
    div.stButton > button {
        width: 100% !important;
        padding: 14px !important;
        background: linear-gradient(90deg, #1d9e75, #0a5e48) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 12px rgba(10,94,72,0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        opacity: 0.92 !important;
        color: #ffffff !important;
    }

    /* صندوق عرض النتيجة */
    .result-box-custom {
        margin-top: 20px;
        background: #f0faf6;
        border: 1.5px solid #9fd9c4;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        font-size: 16px;
        font-weight: 700;
        color: #1a3a30;
    }
    .result-box-custom .rate {
        color: #0a5e48;
        font-size: 26px;
        font-weight: 800;
    }

    /* الفوتر السفلي */
    .footer-bar-custom {
        text-align: center;
        margin-top: 25px;
        font-size: 11px;
        color: #3b7a70;
    }
    </style>
""", unsafe_allow_html=True)

# Default fallback values ​​to prevent errors.
available_categories = ["CODING", "GENERAL", "LANGUAGE", "OTHER", "STEM"]
available_locations = ["EG", "US", "GB", "CA", "FR"]


# 3. Loading the smart model
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "salary_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


try:
    components = load_model()
    lin_reg = components["model"]
    tfidf = components["tfidf"]
    dummy_columns = components["dummy_columns"]

    if "categories" in components:
        available_categories = components["categories"]
    if "locations" in components:
        available_locations = components["locations"]

except Exception as e:
    st.error(f"❌ مشكلة في تحميل الموديل: {e}")
    st.stop()

# 4. Building the User Interface
st.markdown(
    '<div class="brand-bar"><div class="brand-name">AUTO<span>LANCE</span></div><div class="brand-tagline">Freelance Intelligence Platform</div></div>',
    unsafe_allow_html=True)
st.markdown('<div class="main-title-custom">💰 Job Hourly Rate Predictor</div>', unsafe_allow_html=True)

with st.container():
    st.markdown(
        '<h3 style="font-size: 14px; font-weight: 600; color: #1a3a30; margin-bottom: 14px;">Enter Job Details Below for Prediction</h3>',
        unsafe_allow_html=True)

    title_jop = st.text_input("Job Title", placeholder="e.g., Python Developer")
    category = st.selectbox("Category", options=sorted(available_categories), index=0)
    location = st.selectbox("Location", options=sorted(available_locations), index=0)
    Description_jop = st.text_area("Job Description", placeholder="Type or paste detailed text")

    predict_button = st.button("Predict Hourly Rate")

# 5. Processing and displaying the result in a container that exactly matches the original file.
if predict_button:
    if title_jop.strip() and Description_jop.strip():
        new_text = title_jop + " " + Description_jop
        new_text_tfidf = tfidf.transform([new_text])

        input_dummies = pd.DataFrame(0, index=[0], columns=dummy_columns)

        cat_col = f"Category_{category}"
        loc_col = f"Location_{location}"

        if cat_col in input_dummies.columns:
            input_dummies[cat_col] = 1
        if loc_col in input_dummies.columns:
            input_dummies[loc_col] = 1

        final_input = sp.hstack((new_text_tfidf, input_dummies.values))

        with st.spinner("Calculating... ⏳"):
            time.sleep(0.4)
            prediction = lin_reg.predict(final_input)
            prediction = np.clip(float(prediction.item()), 15.0, 150.0)

        st.markdown(f"""
            <div class="result-box-custom">
                >> The predicted hourly rate is: <span class="rate">${prediction:.2f}/hr</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please fill in Job Title and Job Description.")

st.markdown(
    '<div class="footer-bar-custom">Powered by AutoLance and ML &nbsp;·&nbsp; <a href="#" style="color:#0a5e48;">Documentation</a></div>',
    unsafe_allow_html=True)

