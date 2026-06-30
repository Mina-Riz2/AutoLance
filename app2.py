import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import time
import os

# 1.Page settings and modern design inspired by HTML.
st.set_page_config(page_title="AutoLance - Job Hourly Rate Predictor", page_icon="💰", layout="centered")

# Integrating HTML and CSS designs into Streamlit
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.47.0/tabler-icons.min.css">

    <style>
    /* تغيير خلفية التطبيق بالكامل وتطبيق التدرج اللوني المريح */
    .stApp {
        background: linear-gradient(140deg, #e2f4ed 0%, #cce8df 35%, #c0dff0 70%, #d4eaf8 100%) !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* تصميم شعار الهوية AutoLance */
    .brand-title {
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        letter-spacing: 6px;
        color: #0a5e48;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    .brand-title span {
        color: #1d9e75;
    }
    .brand-subtitle {
        text-align: center;
        font-size: 11px;
        color: #3b7a70;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 20px;
        font-weight: 600;
    }

    /* عنوان الأداة الرئيسي */
    .main-title {
        text-align: center;
        font-size: 21px;
        font-weight: 700;
        color: #1a3a30;
        margin-bottom: 18px;
    }

    /* تعديل شكل حاوية المدخلات (Form Card) لتصبح بيضاء شفافة بـ Blur */
    div[data-testid="stForm"], div.stElementContainer div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(6px);
        border-radius: 16px !important;
        border: 1px solid rgba(157, 217, 196, 0.4) !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
    }

    /* تخصيص العناوين الجانبية للمدخلات */
    label[data-testid="stWidgetLabel"] p {
        font-size: 13px !important;
        color: #555 !important;
        font-weight: 600 !important;
    }

    /* تخصيص صناديق الإدخال والقوائم المنسدلة */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
        background-color: #f7fbf9 !important;
        border: 1px solid #c8ddd8 !important;
        border-radius: 8px !important;
        color: #222 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #1d9e75 !important;
        box-shadow: 0 0 0 1px #1d9e75 !important;
    }

    /* تصميم زر التنبؤ الاحترافي المتدرج */
    div.stButton > button {
        background: linear-gradient(90deg, #1d9e75, #0a5e48) !important;
        color: white !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 12px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        transition: all 0.15s ease;
        box-shadow: 0 4px 15px rgba(29, 158, 117, 0.2);
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        color: white !important;
    }
    div.stButton > button:active {
        transform: scale(0.98);
    }

    /* صندوق النتيجة النهائي المقتبس من التصميم */
    .result-box {
        margin-top: 20px;
        background: #f0faf6;
        border: 1.5px solid #9fd9c4;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: #1a3a30;
    }
    .result-box .rate {
        color: #0a5e48;
        font-size: 24px;
        font-weight: 800;
    }

    /* تذييل الصفحة */
    .footer-bar {
        text-align: center;
        margin-top: 25px;
        font-size: 11px;
        color: #3b7a70;
    }
    .footer-bar a { color: #0a5e48; text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

# Default fallback values ​​to prevent errors.
available_categories = ["CODING", "GENERAL", "LANGUAGE", "OTHER", "STEM"]
available_locations = ["EG", "US", "GB", "CA", "FR"]


# 2. Loading the actual model 
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

# 3. User interface with a new, localized, and consistent design.
st.markdown('<div class="brand-title">AUTO<span>LANCE</span></div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Freelance Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">💰 Job Hourly Rate Predictor · متنبئ سعر الساعة</div>', unsafe_allow_html=True)

# Using the container to which the custom CSS has been applied to create the look of a semi-transparent white card.
with st.container():
    st.markdown(
        "<h3 style='font-size:14px; font-weight:600; color:#1a3a30; margin-bottom:15px;'>Enter Job Details Below for Prediction</h3>",
        unsafe_allow_html=True)

    title_jop = st.text_input("Job Title / عنوان الوظيفة:", placeholder="e.g., Python Developer")
    category = st.selectbox("Category / التصنيف:", options=sorted(available_categories))
    location = st.selectbox("Location / الموقع (رمز الدولة):", options=sorted(available_locations))
    Description_jop = st.text_area("Job Description / وصف الوظيفة:", placeholder="Type or paste detailed text...")

    predict_button = st.button("🚀 Predict Hourly Rate / تنبؤ سعر الساعة", use_container_width=True)

# 4. Processing real data and calculating predictions using the invoked model 
if predict_button:
    if title_jop.strip() and Description_jop.strip():
        # Merging texts and preparing the actual TF-IDF for the model.
        new_text = title_jop + " " + Description_jop
        new_text_tfidf = tfidf.transform([new_text])

        # Preparing dummy variables for departments and countries based on the model's actual columns.
        input_dummies = pd.DataFrame(0, index=[0], columns=dummy_columns)

        cat_col = f"Category_{category}"
        loc_col = f"Location_{location}"

        if cat_col in input_dummies.columns:
            input_dummies[cat_col] = 1
        if loc_col in input_dummies.columns:
            input_dummies[loc_col] = 1

        # Merging the text array with the array of countries and categories.
        final_input = sp.hstack((new_text_tfidf, input_dummies.values))

        # Calculating the actual prediction using the model's 'predict' function.
        with st.spinner("Calculating via AutoLance ML Model... ⏳"):
            time.sleep(0.6)  # محاكاة بسيطة لوقت المعالجة ليعطي انطباعاً احترافياً كالتصميم الأول
            prediction = lin_reg.predict(final_input)
            prediction = np.clip(float(prediction.item()), 15.0, 150.0)

        # Display the actual result within the original, enhanced design box.
        st.markdown(f"""
            <div class="result-box">
                >> The predicted hourly rate is: <br>
                <span class="rate">${prediction:.2f}/hr</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please fill in Job Title and Job Description first / من فضلك املأ خانة العنوان والوصف أولاً.")

# Page footer consistent with the attached interface
st.markdown("""
    <div class="footer-bar">
      Powered by AutoLance and ML &nbsp;·&nbsp; <a href="#">Documentation</a>
    </div>
""", unsafe_allow_html=True)







# key :  python -m streamlit run app2.py

