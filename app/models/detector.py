"""
app/models/detector.py - YOLOv8 Vehicle Detector
Handles model loading and inference for both images and videos.
"""

import os
import cv2
import time
import numpy as np
from flask import current_app

# ── PyTorch 2.6 compatibility fix ────────────────────────────────────────────
# PyTorch 2.6 defaults torch.load to weights_only=True which blocks Ultralytics
# model classes. Since best.pt is our own trusted trained model, we patch
# torch.load to always use weights_only=False before YOLO is imported.
import torch

_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None,
                        weights_only=False, mmap=None, **kwargs):
    """Force weights_only=False for trusted local YOLOv8 checkpoints."""
    if pickle_module is not None:
        return _original_torch_load(f, map_location=map_location,
                                    pickle_module=pickle_module,
                                    weights_only=False, **kwargs)
    return _original_torch_load(f, map_location=map_location,
                                weights_only=False, **kwargs)

torch.load = _patched_torch_load
print("[Detector] Patched torch.load — weights_only=False for trusted model.")

from ultralytics import YOLO
# ─────────────────────────────────────────────────────────────────────────────


class VehicleDetector:
    """Singleton-style YOLOv8 detector for vehicle detection."""

    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        """Return the singleton detector instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str):
        """Load YOLOv8 model from the given path."""
        if self._model is None:
            print(f"[Detector] Loading model from: {model_path}")
            self._model = YOLO(model_path)
            print("[Detector] Model loaded successfully.")
        return self._model

    def get_model(self):
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            model_path = current_app.config['MODEL_PATH']
            self.load_model(model_path)
        return self._model

    def detect_image(self, image_path: str, output_path: str,
                     conf: float = 0.25, iou: float = 0.45) -> dict:
        """
        Run detection on a single image.

        Args:
            image_path: Path to input image
            output_path: Path to save annotated output image
            conf: Confidence threshold
            iou: IoU threshold for NMS

        Returns:
            dict with detection results and metadata
        """
        model = self.get_model()
        start_time = time.time()

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = img.shape[:2]

        # Run inference
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            verbose=False
        )

        inference_time = time.time() - start_time

        # Parse detections
        detections = []
        annotated_img = img.copy()

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names.get(class_id, f"class_{class_id}")

                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence * 100, 2),
                        'bbox': [x1, y1, x2, y2]
                    })

                    # Draw bounding box
                    color = self._get_color(class_id)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    label = f"{class_name} {confidence:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_img, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                    cv2.putText(annotated_img, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save annotated image
        cv2.imwrite(output_path, annotated_img)

        # Count by class
        class_counts = {}
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            'total_detections': len(detections),
            'detections': detections,
            'class_counts': class_counts,
            'inference_time': round(inference_time * 1000, 2),  # ms
            'image_size': {'width': w, 'height': h}
        }

    def detect_video(self, video_path: str, output_path: str,
                     conf: float = 0.25, iou: float = 0.45) -> dict:
        """
        Run detection on a video file, frame by frame.

        Args:
            video_path: Path to input video
            output_path: Path to save annotated output video
            conf: Confidence threshold
            iou: IoU threshold

        Returns:
            dict with aggregated detection results
        """
        model = self.get_model()
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output writer — use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []
        class_counts = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run inference on frame
            results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
            annotated_frame = frame.copy()

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names.get(class_id, f"class_{class_id}")

                        all_detections.append({
                            'frame': frame_count,
                            'class': class_name,
                            'confidence': round(confidence * 100, 2)
                        })
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        # Draw bounding box
                        color = self._get_color(class_id)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {confidence:.0%}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Overlay frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(annotated_frame)

        cap.release()
        out.release()

        total_time = time.time() - start_time

        return {
            'total_detections': len(all_detections),
            'class_counts': class_counts,
            'frames_processed': frame_count,
            'total_frames': total_frames,
            'processing_time': round(total_time, 2),
            'fps_processed': round(frame_count / total_time, 2) if total_time > 0 else 0,
            'video_info': {'width': width, 'height': height, 'fps': fps}
        }

    @staticmethod
    def _get_color(class_id: int) -> tuple:
        """Generate a consistent BGR color for a given class ID."""
        colors = [
            (0, 255, 0),    # Green
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Blue
            (255, 0, 128),  # Pink
            (128, 255, 0),  # Lime
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
            (128, 0, 255),  # Purple
        ]
        return colors[class_id % len(colors)]


# Module-level detector instance
detector = VehicleDetector.get_instance()
