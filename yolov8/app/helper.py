from ultralytics import YOLO
import streamlit as st
import cv2
import settings

# -------------------------------------------------
# Load YOLOv8 model
# -------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# -------------------------------------------------
# Tracker options
# -------------------------------------------------
def display_tracker_options():
    display_tracker = st.sidebar.radio("Display Tracker", ("Yes", "No"))
    if display_tracker == "Yes":
        tracker_type = st.sidebar.radio(
            "Tracker",
            ("bytetrack.yaml", "botsort.yaml")
        )
        return True, tracker_type
    return False, None

# -------------------------------------------------
# Display detected frames
# -------------------------------------------------
def display_detected_frames(conf, model, st_frame, image, show_tracker, tracker, show_waste=True):
    image = cv2.resize(image, (720, int(720 * 9 / 16)))

    if show_tracker:
        results = model.track(
            image, conf=conf, persist=True, tracker=tracker, imgsz=1280
        )
    else:
        results = model.predict(image, conf=conf, imgsz=1280)

    plotted = results[0].plot()

    # Display detected objects
    if show_waste and results[0].boxes is not None:
        classes = results[0].names
        detected_classes = [classes[int(c)] for c in results[0].boxes.cls]
        unique_waste = list(set(detected_classes))
        st.text(f"Detected waste types: {', '.join(unique_waste)}")

    st_frame.image(plotted, channels="BGR", width=800)

    return plotted

# -------------------------------------------------
# Webcam detection
# -------------------------------------------------
def play_webcam(conf, model):
    cap = cv2.VideoCapture(settings.WEBCAM_PATH)
    st_frame = st.empty()
    stop_button = st.button("Stop Webcam")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or stop_button:
            break

        # Resize frame for faster inference
        frame = cv2.resize(frame, (720, int(720 * 9 / 16)))

        # Run YOLO model prediction
        results = model.predict(frame, conf=conf, imgsz=1280)
        plotted = results[0].plot()

        st_frame.image(plotted, channels="BGR", width=800)

    cap.release()
    st.success("Webcam stopped.")

# -------------------------------------------------
# Uploaded video detection
# -------------------------------------------------
def play_uploaded_video(video_file, conf, model):
    show_tracker, tracker = display_tracker_options()

    # Save uploaded video temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("detected_video.mp4", fourcc, fps, (width, height))

    st_frame = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Rotate frame 90 degrees counterclockwise to fix orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Update output dimensions if rotated
        h, w = frame.shape[:2]
        if w != width or h != height:
            out = cv2.VideoWriter("detected_video.mp4", fourcc, fps, (w, h))
            width, height = w, h

        plotted = display_detected_frames(
            conf, model, st_frame, frame, show_tracker, tracker, show_waste=False
        )

        # Write the plotted frame to output video
        out.write(plotted)

    cap.release()
    out.release()

    # Display the detected video
    st.video("detected_video.mp4")
