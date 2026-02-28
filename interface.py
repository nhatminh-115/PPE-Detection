import os
import io
import base64
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from supabase import create_client, Client

# 1. System Configuration & Initialization
st.set_page_config(page_title="PPE MLOps Dashboard", layout="wide")

FLASK_API_URL = "http://localhost:8000/detect"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

@st.cache_resource
def init_database() -> Client:
    """
    Sử dụng Singleton Pattern thông qua @st.cache_resource để ngăn chặn 
    việc khởi tạo lại kết nối CSDL mỗi khi giao diện render lại.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.sidebar.error(f"Database Connection Error: {e}")
        return None

supabase_client = init_database()

# 2. UI Layout - Header & Sidebar
st.title("PPE Detection System - Operations Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("System Metadata")
    st.info("Inference Engine: YOLOv11m\nArchitecture: Flask API + Streamlit")
    st.markdown("### Repositories")
    st.write("[Model Registry (Dagshub)](https://dagshub.com/nhatminh-115/PPE-Detection.mlflow/)")
    
    if st.button("Refresh Telemetry"):
        st.rerun()

# 3. Main Operational Pipeline (Live Inference)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Stream")
    uploaded_file = st.file_uploader("Upload inspection image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Raw Input", use_container_width=True)
        
        if st.button("Execute Inference"):
            with st.spinner("Processing via Remote API..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    # Thiết lập timeout để tránh treo luồng giao diện (UI Thread Blocking)
                    response = requests.post(FLASK_API_URL, files=files, timeout=15)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.session_state['latest_result'] = res_data
                    else:
                        st.error(f"API Error (HTTP {response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Refused: Flask API is not responding. Please check port 8000.")
                except requests.exceptions.Timeout:
                    st.error("Request Timeout: Inference engine is overloaded or cold-starting.")

# Render Inference Results
with col2:
    st.subheader("Inference Output")
    if 'latest_result' in st.session_state:
        res_data = st.session_state['latest_result']
        
        try:
            img_bytes = base64.b64decode(res_data['image_base64'])
            st.image(img_bytes, caption="Annotated Output", use_container_width=True)
            
            if res_data['violations_count'] > 0:
                st.warning(f"Detected {res_data['violations_count']} compliance violations.")
                df_violations = pd.DataFrame(res_data['details'])
                st.table(df_violations)
            else:
                st.success("Status: Compliant. No violations detected.")
        except Exception as e:
            st.error(f"Data Parsing Error: {e}")
    else:
        st.info("Awaiting input data...")

# 4. Telemetry & Analytics Dashboard
st.markdown("---")
st.subheader("Real-time Violations Telemetry")

if supabase_client:
    try:
        # Tối ưu hóa truy vấn CSDL: Chỉ lấy những trường cần thiết để giảm Payload
        response = supabase_client.table("ppe_violations").select("id, created_at, status").order("created_at", desc=True).limit(100).execute()
        data = response.data

        if data:
            df = pd.DataFrame(data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Khởi tạo chuỗi thời gian (Time-series Aggregation)
            time_series = df.set_index('created_at').resample('h').size()
            st.line_chart(time_series, use_container_width=True)
            
            with st.expander("Raw Database Logs"):
                st.dataframe(df, use_container_width=True)
        else:
            st.write("No historical data available in the registry.")
    except Exception as e:
        st.error(f"Failed to fetch telemetry data: {e}")
else:
    st.warning("Database client is not initialized. Please verify environment variables.")