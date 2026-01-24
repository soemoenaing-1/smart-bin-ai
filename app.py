from pathlib import Path
import PIL.Image
import streamlit as st
import sys
import os

# Add the app directory to Python path
app_dir = Path(__file__).parent / "yolov8" / "app"
sys.path.insert(0, str(app_dir))
os.chdir(app_dir)

import settings
import helper

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Bin AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("♻️ Smart Bin AI")

# -------------------------------------------------
# Sidebar - Model Config
# -------------------------------------------------
st.sidebar.header("Model Configuration")

# Model selection
model_options = ["best.pt", "yoloooo.pt", "yolov8n.pt"]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=1)  # default yoloooo.pt

confidence = st.sidebar.slider(
    "Confidence Threshold",
    25, 100, 50  # increased default
) / 100

# -------------------------------------------------
# Load model
# -------------------------------------------------
model_path = Path(settings.WEIGHTS_DIR) / selected_model

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model: {model_path}")
    st.stop()

# -------------------------------------------------
# Sidebar - Source selection
# -------------------------------------------------
st.sidebar.header("Input Source")
source = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

# -------------------------------------------------
# IMAGE
# -------------------------------------------------
if source == settings.IMAGE:
    uploaded_image = st.sidebar.file_uploader(
        "Upload Image",
        type=("jpg", "jpeg", "png", "bmp", "webp")
    )

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

# -------------------------------------------------
# VIDEO
# -------------------------------------------------
elif source == settings.VIDEO:
    uploaded_video = st.sidebar.file_uploader(
        "Upload Video",
        type=("mp4", "avi", "mov", "mkv")
    )

    if uploaded_video:
        st.subheader("Uploaded Video")
        st.video(uploaded_video)
        st.subheader("Detected Video")
        if st.sidebar.button("Detect Video"):
            helper.play_uploaded_video(uploaded_video, confidence, model)
    else:
        st.info("Upload a video to start detection")

# -------------------------------------------------
# WEBCAM
# -------------------------------------------------
elif source == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.warning("Please select a valid source")
