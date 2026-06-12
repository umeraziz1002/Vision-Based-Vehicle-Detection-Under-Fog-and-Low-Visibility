# FogVision — Vision-Based Vehicle Detection under Fog and Low Visibility Conditions
 
> AI-powered vehicle detection system using YOLOv8 deep learning

---

## Project Overview

FogVision is a Flask-based web application that detects vehicles in foggy and low-visibility road environments using a custom-trained YOLOv8 model. It supports image upload, video processing, and live webcam detection, with a professional dashboard and REST API.

---

## Features

- **Image Detection** — Upload JPG/PNG images, get annotated output with bounding boxes
- **Video Detection** — Process MP4/AVI videos frame-by-frame
- **Live Webcam** — Real-time camera feed with frame capture and server-side inference
- **Dashboard** — Detection history, statistics, and class analytics
- **REST API** — Full API for mobile/external integration
- **Download Results** — Save annotated images and videos

---

## Project Structure

```
FoggyWebsite/
├── run.py                    # App entry point
├── best.pt                   # YOLOv8 trained model weights
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py           # App factory
    ├── config.py             # Configuration
    ├── models/
    │   └── detector.py       # YOLOv8 detector class
    ├── routes/
    │   ├── main.py           # Home & About routes
    │   ├── detection.py      # Upload & detection routes
    │   └── dashboard.py      # Dashboard route
    ├── api/
    │   └── endpoints.py      # REST API endpoints
    ├── utils/
    │   └── file_utils.py     # File handling & history logging
    ├── templates/
    │   ├── base.html
    │   ├── index.html
    │   ├── detection.html
    │   ├── result.html
    │   ├── dashboard.html
    │   ├── webcam.html
    │   └── about.html
    └── static/
        ├── css/style.css
        ├── js/
        │   ├── main.js
        │   ├── detection.js
        │   └── webcam.js
        ├── uploads/          # Uploaded files (auto-created)
        ├── processed/        # Detection outputs (auto-created)
        └── logs/             # Detection history JSON (auto-created)
```

---

## Setup & Installation

### 1. Clone / navigate to the project folder

```bash
cd "d:\Semester 7\FYP\FoggyWebsite"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure model weights are present

The file `best.pt` should be in the root directory (`FoggyWebsite/best.pt`).

### 5. Run the application

```bash
python run.py
```

Open your browser at: **http://localhost:5000**

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/detect/image` | Detect vehicles in image |
| POST | `/api/v1/detect/video` | Detect vehicles in video |
| GET | `/api/v1/stats` | Detection statistics |
| GET | `/api/v1/history` | Detection history |

### Example API call (Python)

```python
import requests

with open('foggy_road.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/v1/detect/image',
        files={'file': f},
        data={'confidence': 0.25, 'iou': 0.45}
    )
    print(response.json())
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask 3.0 (Python) |
| Deep Learning | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Frontend | Bootstrap 5, HTML5, CSS3, JavaScript |
| API | REST (JSON) |

---

## Configuration

Edit `app/config.py` to adjust:

- `CONFIDENCE_THRESHOLD` — Default detection confidence (0.25)
- `IOU_THRESHOLD` — Default IoU for NMS (0.45)
- `MAX_CONTENT_LENGTH` — Max upload size (100 MB)
- `MODEL_PATH` — Path to YOLOv8 weights

---

## License

Final Year Project — For academic use only.
