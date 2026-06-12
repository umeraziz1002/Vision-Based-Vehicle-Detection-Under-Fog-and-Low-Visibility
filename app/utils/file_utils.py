"""
app/utils/file_utils.py - File handling utilities
"""

import os
import uuid
import json
from datetime import datetime
from flask import current_app
from werkzeug.utils import secure_filename


def allowed_file(filename: str, file_type: str = 'image') -> bool:
    """Check if a filename has an allowed extension."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in current_app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return ext in current_app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False


def get_file_type(filename: str) -> str:
    """Determine if a file is an image or video based on extension."""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in current_app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return 'image'
    elif ext in current_app.config['ALLOWED_VIDEO_EXTENSIONS']:
        return 'video'
    return 'unknown'


def save_upload(file) -> tuple:
    """
    Save an uploaded file with a unique name.

    Returns:
        (unique_filename, full_path)
    """
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit('.', 1)[1].lower() if '.' in original_name else 'bin'
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_name)
    file.save(full_path)
    return unique_name, full_path


def get_output_path(input_filename: str, suffix: str = '_detected') -> tuple:
    """
    Generate an output file path for processed results.

    Returns:
        (output_filename, full_output_path)
    """
    name, ext = os.path.splitext(input_filename)
    output_name = f"{name}{suffix}{ext}"
    full_path = os.path.join(current_app.config['PROCESSED_FOLDER'], output_name)
    return output_name, full_path


def get_dehazed_path(input_filename: str) -> tuple:
    """
    Generate a file path for the dehazed intermediate image.
    Stored in the DEHAZED_FOLDER, separate from final processed results.

    Returns:
        (dehazed_filename, full_dehazed_path)
    """
    name, ext = os.path.splitext(input_filename)
    dehazed_name = f"{name}_dehazed{ext}"
    full_path = os.path.join(current_app.config['DEHAZED_FOLDER'], dehazed_name)
    return dehazed_name, full_path


def log_detection(entry: dict):
    """Append a detection entry to the history JSON log."""
    history_file = current_app.config['HISTORY_FILE']

    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    # Add timestamp
    entry['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history.insert(0, entry)  # newest first

    # Keep only last 100 entries
    history = history[:100]

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def load_history() -> list:
    """Load detection history from the JSON log."""
    history_file = current_app.config['HISTORY_FILE']
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_stats() -> dict:
    """Compute aggregate statistics from detection history."""
    history = load_history()
    total_files = len(history)
    total_detections = sum(h.get('total_detections', 0) for h in history)
    images = sum(1 for h in history if h.get('file_type') == 'image')
    videos = sum(1 for h in history if h.get('file_type') == 'video')

    # Class frequency across all detections
    class_totals = {}
    for h in history:
        for cls, count in h.get('class_counts', {}).items():
            class_totals[cls] = class_totals.get(cls, 0) + count

    return {
        'total_files': total_files,
        'total_detections': total_detections,
        'images_processed': images,
        'videos_processed': videos,
        'class_totals': class_totals,
        'recent': history[:10]
    }
