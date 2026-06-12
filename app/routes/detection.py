"""
app/routes/detection.py
========================
Detection Routes
----------------
Upload → YOLOv8 Detection → Show results.

Dehazing code is preserved in app/models/dehazer.py and
app/models/dehazeformer.py but is NOT called from any route here.
To re-enable it later, see the git history or the dehaze_only()
and upload_and_detect_with_dehazing() functions in the archive comment below.
"""

import os
from flask import (Blueprint, render_template, request,
                   flash, redirect, url_for, current_app,
                   send_from_directory, jsonify)

from ..models.detector import detector
from ..utils.file_utils import (
    get_file_type, save_upload, get_output_path, log_detection
)

detection_bp = Blueprint('detection', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

@detection_bp.route('/')
def detect_home():
    """Render the upload / detection page."""
    return render_template('detection.html')


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD + DETECT  (no dehazing)
# ─────────────────────────────────────────────────────────────────────────────

@detection_bp.route('/upload', methods=['POST'])
def upload_and_detect():
    """
    Save the uploaded file and run YOLOv8 detection directly.
    Dehazing is intentionally skipped — call dehazer.dehaze_image()
    here if you want to re-enable it in future.
    """
    if 'file' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('detection.detect_home'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('detection.detect_home'))

    file_type = get_file_type(file.filename)
    if file_type == 'unknown':
        flash('Unsupported file type. Please upload JPG, PNG, MP4, or AVI.', 'warning')
        return redirect(url_for('detection.detect_home'))

    try:
        unique_name, upload_path = save_upload(file)
    except Exception as e:
        flash(f'Upload failed: {str(e)}', 'danger')
        return redirect(url_for('detection.detect_home'))

    conf = float(request.form.get('confidence', current_app.config['CONFIDENCE_THRESHOLD']))
    iou  = float(request.form.get('iou',        current_app.config['IOU_THRESHOLD']))

    output_name, output_path = get_output_path(unique_name)

    try:
        if file_type == 'image':
            results = detector.detect_image(upload_path, output_path, conf=conf, iou=iou)
        else:
            results = detector.detect_video(upload_path, output_path, conf=conf, iou=iou)
    except Exception as e:
        flash(f'Detection failed: {str(e)}', 'danger')
        return redirect(url_for('detection.detect_home'))

    # Dehazing not used — set neutral placeholder so result.html doesn't break
    results['dehazing'] = {'used': False, 'inference_ms': 0, 'dehazed_file': None}

    log_detection({
        'file_type':        file_type,
        'original_file':    unique_name,
        'output_file':      output_name,
        'dehazing_used':    False,
        'total_detections': results.get('total_detections', 0),
        'class_counts':     results.get('class_counts', {}),
        'mode':             'detect_only',
    })

    flash(f'Detection complete! Found {results.get("total_detections", 0)} vehicle(s).', 'success')

    return render_template(
        'result.html',
        file_type=file_type,
        original_file=unique_name,
        output_file=output_name,
        results=results
    )


# ─────────────────────────────────────────────────────────────────────────────
# OTHER ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@detection_bp.route('/webcam')
def webcam():
    return render_template('webcam.html')


@detection_bp.route('/download/<filename>')
def download_result(filename):
    return send_from_directory(
        current_app.config['PROCESSED_FOLDER'],
        filename,
        as_attachment=True
    )


@detection_bp.route('/download/dehazed/<filename>')
def download_dehazed(filename):
    """Kept for dehaze_result.html links — dehazing UI is hidden but route works."""
    return send_from_directory(
        current_app.config['DEHAZED_FOLDER'],
        filename,
        as_attachment=True
    )
