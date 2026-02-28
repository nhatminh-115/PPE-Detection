import os
import base64
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from supabase import create_client, Client

st.set_page_config(page_title="PPE Detection MLOps", layout="wide")

FLASK_API_URL = os.environ.get("API_URL", "http://localhost:8000/detect")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "") 

@st.cache_resource
def init_database() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.sidebar.error(f"Database Initialization Error: {e}")
        return None

supabase_client = init_database()

st.title("PPE Detection - Operations Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("System Architecture")
    st.info("Inference Engine: YOLOv11m\nServing: Flask WSGI\nFrontend: Streamlit")
    st.markdown("### Repositories")
    st.write("[Model Registry (Dagshub)](https://dagshub.com/nhatminh-115/PPE-Detection.mlflow/)")
    if st.button("Refresh Telemetry"):
        st.rerun()

tab_inference, tab_telemetry = st.tabs(["Live Inference Pipeline", "Telemetry & Database Log"])

with tab_inference:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Input Stream")
        uploaded_file = st.file_uploader("Upload inspection artifact (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Raw Artifact", use_container_width=True)
            
            if st.button("Execute Model"):
                with st.spinner("Invoking Remote API..."):
                    try:
                        files = {"file": uploaded_file.getvalue()}
                        response = requests.post(FLASK_API_URL, files=files, timeout=15)
                        
                        if response.status_code == 200:
                            st.session_state['latest_result'] = response.json()
                        else:
                            st.error(f"Gateway Error (HTTP {response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Microservice Connection Failure: {e}")

    with col2:
        st.subheader("Inference Engine Output")
        if 'latest_result' in st.session_state:
            res_data = st.session_state['latest_result']
            try:
                img_bytes = base64.b64decode(res_data['image_base64'])
                st.image(img_bytes, caption="AI Annotated Result", use_container_width=True)
                
                if res_data['violations_count'] > 0:
                    st.warning(f"Detected {res_data['violations_count']} compliance breaches.")
                    st.table(pd.DataFrame(res_data['details']))
                else:
                    st.success("System Status: Compliant. Zero violations.")
            except Exception as e:
                st.error(f"Decoding Error: {e}")
        else:
            st.info("Standing by for input payload...")

with tab_telemetry:
    st.subheader("Global Violations Registry (Supabase Sync)")
    
    if supabase_client:
        try:
            response = supabase_client.table("ppe_violations").select("*").order("created_at", desc=True).limit(100).execute()
            data = response.data

            if data:
                df = pd.DataFrame(data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                df['violation_details'] = df['violations'].apply(
                    lambda x: ", ".join([v['violation_type'] for v in x]) if isinstance(x, list) else "Unknown"
                )
                display_df = df[['id', 'created_at', 'violation_details', 'status']]
                display_df.rename(columns={
                    'id': 'Log ID', 
                    'created_at': 'Timestamp', 
                    'violation_details': 'Detected Missing PPE', 
                    'status': 'Severity'
                }, inplace=True)

                st.dataframe(display_df, use_container_width=True)
                
                st.markdown("### Temporal Distribution")
                time_series = df.set_index('created_at').resample('h').size()
                st.line_chart(time_series, use_container_width=True)
                
            else:
                st.write("Database is fully initialized but currently holds no violation records.")
        except Exception as e:
            st.error(f"Data Fetch Exception: {e}")
    else:
        st.warning("Database client is detached. Ensure SUPABASE_URL and SUPABASE_KEY are injected via environment variables.")