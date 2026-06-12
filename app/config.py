"""
app/config.py - Application Configuration
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fyp-foggy-detection-secret-2024'

    # File Upload Settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'processed')
    LOG_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'logs')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload

    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

    # YOLOv8 Model Path
    MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

    # Detection Settings
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

    # ── DehazeFormer Settings ─────────────────────────────────────────────────
    # Path to pretrained DehazeFormer-S weights (.pth file).
    # Download from: https://github.com/IDKiro/DehazeFormer
    # Place the file at: FoggyWebsite/dehazeformer_s.pth
    DEHAZER_WEIGHTS_PATH = os.path.join(BASE_DIR, 'dehazeformer_s.pth')

    # Master on/off switch — set to False to skip dehazing entirely.
    # Can also be toggled at runtime via the UI without restarting the server.
    DEHAZER_ENABLED = True

    # Folder for dehazed intermediate images (before detection)
    DEHAZED_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'dehazed')

    # History file
    HISTORY_FILE = os.path.join(BASE_DIR, 'app', 'static', 'logs', 'history.json')
