import streamlit as st
import requests
from PIL import Image
import io

# Config
API_URL = "http://localhost:8001/predict"

st.set_page_config(page_title="OvoScan Operator", page_icon="🥚", layout="wide")

# Header
st.title("🥚 OvoScan: Intelligent Quality Control")
st.markdown("---")

# Sidebar
st.sidebar.header("System Status")
st.sidebar.success("✅ API Connected")
st.sidebar.info("🧠 Model: YOLOv8 + Llama 3")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Image Acquisition")
    uploaded_file = st.file_uploader("Upload Egg Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Show image
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        # Action Button
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Scanning & Consulting Knowledge Base..."):
                try:
                    # Prepare file for API
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store in session state to display in col2
                        st.session_state['result'] = result
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

with col2:
    st.subheader("2. Diagnostic Report")
    
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        # Prediction Badge
        pred = res['prediction'].upper()
        conf = res['confidence'] * 100
        
        if pred == "FERTILE":
            st.success(f"## {pred}")
        else:
            st.error(f"## {pred}")
            
        st.metric("Confidence Score", f"{conf:.2f}%")
        
        # The RAG Report
        st.markdown("### 📋 Technical Assessment")
        st.info(res['technical_report'])
        
        # Audit Log (JSON)
        with st.expander("View Raw System Logs"):
            st.json(res)
    else:
        st.info("Waiting for analysis...")