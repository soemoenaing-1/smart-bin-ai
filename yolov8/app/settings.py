from pathlib import Path

# -------------------------------------------------
# Project Root
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent

# -------------------------------------------------
# Sources
# -------------------------------------------------
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# -------------------------------------------------
# Images config
# -------------------------------------------------
IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "def.jfif"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "def1.jpg"

# -------------------------------------------------
# Videos config (optional local videos)
# -------------------------------------------------
VIDEO_DIR = ROOT / "videos"

# -------------------------------------------------
# ML Model config
# -------------------------------------------------
WEIGHTS_DIR = ROOT / "weights"
DETECTION_MODEL = WEIGHTS_DIR / "best.pt"  # must be .pt

# -------------------------------------------------
# Webcam
# -------------------------------------------------
WEBCAM_PATH = 0
