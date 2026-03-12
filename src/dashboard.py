import streamlit as st
import requests
from PIL import Image
import io
import os  # <--- NEW IMPORT

# Config: Read from Docker env, fallback to localhost for testing
API_URL = os.getenv("API_URL", "http://localhost:8001/predict")

st.set_page_config(page_title="OvoScan Operator", page_icon="🥚", layout="wide")

st.title("🥚 OvoScan: Intelligent Quality Control")
st.markdown(f"**System Status:** Connecting to Backend at `{API_URL}`") # Debug info

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Image Acquisition")
    uploaded_file = st.file_uploader("Upload Egg Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Scanning..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        st.session_state['result'] = response.json()
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

with col2:
    st.subheader("2. Diagnostic Report")
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        if res.get("status") == "error":
            st.error("❌ Analysis Failed")
            st.warning(res.get('message'))
        else:
            pred = res['prediction'].upper()
            conf = res['confidence'] * 100
            
            if pred == "FERTILE":
                st.success(f"## {pred}")
            else:
                st.error(f"## {pred}")
            
            st.metric("Confidence Score", f"{conf:.2f}%")
            st.info(res['technical_report'])