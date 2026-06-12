"""
app/api/endpoints.py - REST API endpoints for mobile/external integration
"""

import os
from flask import Blueprint, request, jsonify, current_app, url_for
from ..models.detector import detector
from ..utils.file_utils import (allowed_file, get_file_type,
                                 save_upload, get_output_path,
                                 log_detection, get_stats, load_history)

api_bp = Blueprint('api', __name__)


def api_error(message: str, status: int = 400):
    return jsonify({'success': False, 'error': message}), status


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'success': True, 'status': 'running', 'version': '1.0.0'})


@api_bp.route('/detect/image', methods=['POST'])
def api_detect_image():
    """
    REST API: Detect vehicles in an uploaded image.

    Form data:
        file        : image file (JPG/PNG)
        confidence  : float (optional, default 0.25)
        iou         : float (optional, default 0.45)
        use_dehazer : '1' to dehaze before detection (optional, default '0')

    Returns JSON with detection results and output URL.
    """
    from ..models.dehazer import dehazer
    from ..utils.file_utils import get_dehazed_path

    if 'file' not in request.files:
        return api_error('No file provided.')

    file = request.files['file']
    if file.filename == '':
        return api_error('Empty filename.')

    if get_file_type(file.filename) != 'image':
        return api_error('File must be an image (JPG, PNG, BMP, WEBP).')

    try:
        unique_name, upload_path = save_upload(file)
        output_name, output_path = get_output_path(unique_name)

        conf = float(request.form.get('confidence', current_app.config['CONFIDENCE_THRESHOLD']))
        iou  = float(request.form.get('iou', current_app.config['IOU_THRESHOLD']))
        use_dehazer_flag = request.form.get('use_dehazer', '0') == '1'

        # Optional dehazing step
        dehaze_info = {'skipped': True, 'inference_ms': 0}
        detection_input = upload_path

        if use_dehazer_flag:
            dh_name, dh_path = get_dehazed_path(unique_name)
            dehaze_info = dehazer.dehaze_image(upload_path, dh_path)
            if not dehaze_info.get('skipped'):
                detection_input = dh_path

        results = detector.detect_image(detection_input, output_path, conf=conf, iou=iou)
        results['dehazing'] = {
            'used': use_dehazer_flag and not dehaze_info.get('skipped', True),
            'inference_ms': dehaze_info.get('inference_ms', 0),
        }

        log_detection({
            'file_type': 'image',
            'original_file': unique_name,
            'output_file': output_name,
            'dehazing_used': results['dehazing']['used'],
            'total_detections': results.get('total_detections', 0),
            'class_counts': results.get('class_counts', {}),
            'source': 'api'
        })

        return jsonify({
            'success': True,
            'file_type': 'image',
            'original_file': unique_name,
            'output_file': output_name,
            'output_url': f"/static/processed/{output_name}",
            'results': results
        })

    except Exception as e:
        return api_error(str(e), 500)


@api_bp.route('/detect/video', methods=['POST'])
def api_detect_video():
    """
    REST API: Detect vehicles in an uploaded video.

    Form data:
        file: video file (MP4, AVI, MOV)
        confidence: float (optional)
        iou: float (optional)
    """
    if 'file' not in request.files:
        return api_error('No file provided.')

    file = request.files['file']
    if file.filename == '':
        return api_error('Empty filename.')

    if get_file_type(file.filename) != 'video':
        return api_error('File must be a video (MP4, AVI, MOV, MKV).')

    try:
        unique_name, upload_path = save_upload(file)
        output_name, output_path = get_output_path(unique_name)

        conf = float(request.form.get('confidence', current_app.config['CONFIDENCE_THRESHOLD']))
        iou = float(request.form.get('iou', current_app.config['IOU_THRESHOLD']))

        results = detector.detect_video(upload_path, output_path, conf=conf, iou=iou)

        log_detection({
            'file_type': 'video',
            'original_file': unique_name,
            'output_file': output_name,
            'total_detections': results.get('total_detections', 0),
            'class_counts': results.get('class_counts', {}),
            'source': 'api'
        })

        return jsonify({
            'success': True,
            'file_type': 'video',
            'original_file': unique_name,
            'output_file': output_name,
            'output_url': f"/static/processed/{output_name}",
            'results': results
        })

    except Exception as e:
        return api_error(str(e), 500)


@api_bp.route('/stats', methods=['GET'])
def api_stats():
    """REST API: Get detection statistics."""
    return jsonify({'success': True, 'stats': get_stats()})


@api_bp.route('/history', methods=['GET'])
def api_history():
    """REST API: Get detection history."""
    limit = int(request.args.get('limit', 20))
    history = load_history()[:limit]
    return jsonify({'success': True, 'history': history, 'count': len(history)})
