import streamlit as st
from pathlib import Path
import sys
import os

# Set working directory to app folder
app_dir = Path(__file__).parent / "yolov8" / "app"
os.chdir(app_dir)
sys.path.insert(0, str(app_dir))

# Import app modules
import settings
import helper
import PIL.Image

# Streamlit configuration
st.set_page_config(
    page_title="Smart Bin AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("♻️ Smart Bin AI")

# Sidebar configuration
st.sidebar.header("Model Configuration")
model_options = ["best.pt", "yoloooo.pt", "yolov8n.pt"]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=1)
confidence = st.sidebar.slider("Confidence Threshold", 25, 100, 50) / 100

# Load model
try:
    model_path = Path(settings.WEIGHTS_DIR) / selected_model
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model: {model_path}")
    st.stop()

# Input source selection
st.sidebar.header("Input Source")
source = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

# Image detection
if source == settings.IMAGE:
    uploaded_image = st.sidebar.file_uploader("Upload Image", type=("jpg", "jpeg", "png", "bmp", "webp"))
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_image:
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", width="stretch")
        else:
            st.info("Upload an image to start detection")
    
    with col2:
        if uploaded_image and st.sidebar.button("Detect Objects"):
            results = model.predict(image, conf=confidence, imgsz=1280)
            plotted = results[0].plot()[:, :, ::-1]
            st.image(plotted, caption="Detected Image", width="stretch")

# Video processing
elif source == settings.VIDEO:
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=("mp4", "avi", "mov", "mkv"))
    if uploaded_video:
        st.subheader("Uploaded Video")
        st.video(uploaded_video)
        st.subheader("Detected Video")
        if st.sidebar.button("Detect Video"):
            helper.play_uploaded_video(uploaded_video, confidence, model)
    else:
        st.info("Upload a video to start detection")

# Webcam detection
elif source == settings.WEBCAM:
    helper.play_webcam(confidence, model)
else:
    st.warning("Please select a valid source")
