---
title: Smart Bin AI
emoji: 🤖
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.23.1
app_file: streamlit_app.py
pinned: false
license: mit
---

# Smart Bin AI - YOLO Waste Detection

## 🤖 Application Features
- **Image Detection**: Upload images for waste classification
- **Video Processing**: Analyze videos with object tracking
- **Real-time Webcam**: Live waste detection
- **Multiple Models**: Switch between different YOLO models
- **Adjustable Confidence**: Fine-tune detection sensitivity

## 🚀 Deployment
- **Local**: `streamlit run app.py`
- **Cloud**: Deployed on Streamlit Community Cloud
- **Repository**: https://github.com/soemoenaing-1/smart-bin-ai

## 📁 Project Structure
```
smart-bin-ai/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── yolov8/app/              # Application modules
│   ├── settings.py          # Configuration
│   ├── helper.py            # Utility functions
│   ├── weights/             # YOLO model files
│   ├── images/              # Sample images
│   └── results/             # Detection results
└── README.md                # This file
```

## 🛠️ Technologies Used
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- Pillow
- NumPy