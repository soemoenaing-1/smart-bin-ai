# Copilot instructions — Smart Bin AI (YOLO)

Purpose: help AI coding agents be productive quickly in this repository.

- Run: the app is a Streamlit app; start with:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

- Key files:
  - [app.py](app.py) — Streamlit entrypoint that appends `yolov8/app` to `sys.path` and builds the UI.
  - [yolov8/app/helper.py](yolov8/app/helper.py) — model loading, webcam/video loops, and image/video processing.
  - [yolov8/app/settings.py](yolov8/app/settings.py) — path constants (weights/images/videos), source names, webcam index.
  - [requirements.txt](requirements.txt) — required packages (notably `ultralytics`, `torch`, `opencv-python`).
  - [yolov8/app/weights](yolov8/app/weights) — place trained `.pt` models here; `app.py` expects filenames listed in the sidebar options.

- Architecture / big picture:
  - Single-process Streamlit UI ([app.py]) drives three input flows: Image, Video, Webcam (see `SOURCES_LIST` in settings).
  - The ML primitives are in `helper.py`: `load_model()` returns an `ultralytics.YOLO` model, `display_detected_frames()` centralizes predict/track logic.
  - Video pipeline writes `temp_video.mp4` (uploaded) and `detected_video.mp4` (output) in the working directory — agents should expect file I/O side effects.
  - Model tracking uses `model.track(...)` with tracker yaml names (`bytetrack.yaml`, `botsort.yaml`) selectable in the sidebar.

- Important repo-specific conventions & patterns:
  - `app.py` modifies `sys.path` to import the `yolov8/app` package; do not remove that unless you reorganize imports.
  - `helper.load_model` is decorated with `@st.cache_resource` so model objects are cached across reruns — when changing model-loading logic, consider cache invalidation.
  - Confidence slider in the UI is defined as integers (25–100) then divided by 100 in code; UI values are stored in `confidence` (float).
  - Images are resized to width 720 with 16:9 aspect ratio before inference; `model.predict(..., imgsz=1280)` is used for inference.
  - `play_uploaded_video` rotates frames 90° CCW to correct orientation — check this logic when debugging strange orientation results.

- Running & debugging notes:
  - If model loading fails, `app.py` shows `st.error` and calls `st.stop()` — check `yolov8/app/weights` and `settings.WEIGHTS_DIR` if that happens.
  - For webcam debugging, `helper.play_webcam()` uses `cv2.VideoCapture(settings.WEBCAM_PATH)` and a Streamlit `Stop Webcam` button; running locally with a camera requires appropriate permissions.
  - For video processing, output uses `cv2.VideoWriter` with `mp4v` fourcc — ensure OpenCV build supports this codec on the platform.

- Making common edits:
  - Adding a new model: copy a `.pt` into [yolov8/app/weights] and add its filename into `model_options` in [app.py].
  - Change default model: modify the `index` argument of `st.sidebar.selectbox` in [app.py].
  - To toggle tracker behavior, edit `display_tracker_options()` in [yolov8/app/helper.py].

- Performance & environment:
  - The app depends on `torch` and `ultralytics`; GPU speeds up inference but the code runs on CPU. Mismatched CUDA/torch versions cause runtime errors — use the versions listed in `requirements.txt` when possible.

- Where to look for common fixes / investigation tips:
  - Model load errors: check path in [yolov8/app/settings.py] and actual `.pt` files under [yolov8/app/weights].
  - Video codec or writing errors: verify OpenCV installation supports `mp4v` on Windows, or change fourcc to a supported codec.
  - Streamlit re-run issues: remember model caching; restart Streamlit when replacing `.pt` files if the cached model remains.

If anything here is unclear or you want this file expanded with run/CI scripts or test harnesses, tell me which area to expand.
