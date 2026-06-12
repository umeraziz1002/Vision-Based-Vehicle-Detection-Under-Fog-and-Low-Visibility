/**
 * webcam.js — Live webcam feed with frame capture
 * Provides camera access and snapshot capture for the webcam detection page.
 */

document.addEventListener('DOMContentLoaded', () => {

  const startBtn        = document.getElementById('startBtn');
  const stopBtn         = document.getElementById('stopBtn');
  const captureBtn      = document.getElementById('captureBtn');
  const webcamVideo     = document.getElementById('webcamVideo');
  const webcamCanvas    = document.getElementById('webcamCanvas');
  const webcamPlaceholder = document.getElementById('webcamPlaceholder');
  const liveBadge       = document.getElementById('liveBadge');
  const webcamStatus    = document.getElementById('webcamStatus');
  const statStatus      = document.getElementById('statStatus');
  const statFrames      = document.getElementById('statFrames');
  const statDetections  = document.getElementById('statDetections');

  let stream = null;
  let frameCount = 0;
  let frameTimer = null;

  // ── Start Camera ──────────────────────────────────────────────
  startBtn?.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false
      });

      webcamVideo.srcObject = stream;
      webcamVideo.classList.remove('d-none');
      webcamPlaceholder.classList.add('d-none');

      // Show controls
      startBtn.classList.add('d-none');
      stopBtn.classList.remove('d-none');
      captureBtn.classList.remove('d-none');
      liveBadge.classList.remove('d-none');

      webcamStatus.textContent = 'Camera active — point at a road scene';
      statStatus.textContent = 'Active';
      statStatus.style.color = 'var(--success)';

      // Count frames
      frameTimer = setInterval(() => {
        frameCount++;
        statFrames.textContent = frameCount;
      }, 100);

    } catch (err) {
      let msg = 'Could not access camera.';
      if (err.name === 'NotAllowedError') msg = 'Camera permission denied. Please allow camera access.';
      else if (err.name === 'NotFoundError') msg = 'No camera found on this device.';
      else if (err.name === 'NotReadableError') msg = 'Camera is in use by another application.';

      webcamStatus.textContent = msg;
      showAlert(msg, 'danger');
    }
  });

  // ── Stop Camera ───────────────────────────────────────────────
  stopBtn?.addEventListener('click', () => {
    stopCamera();
  });

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    clearInterval(frameTimer);
    frameCount = 0;

    webcamVideo.srcObject = null;
    webcamVideo.classList.add('d-none');
    webcamCanvas.classList.add('d-none');
    webcamPlaceholder.classList.remove('d-none');

    startBtn.classList.remove('d-none');
    stopBtn.classList.add('d-none');
    captureBtn.classList.add('d-none');
    liveBadge.classList.add('d-none');

    webcamStatus.textContent = 'Camera stopped';
    statStatus.textContent = 'Idle';
    statStatus.style.color = '';
    statFrames.textContent = '0';
  }

  // ── Capture Frame ─────────────────────────────────────────────
  captureBtn?.addEventListener('click', () => {
    if (!stream || !webcamVideo.videoWidth) return;

    // Draw current frame to canvas
    webcamCanvas.width  = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0);

    // Convert to blob and send to server for detection
    webcamCanvas.toBlob(async (blob) => {
      if (!blob) return;

      captureBtn.disabled = true;
      captureBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Detecting...';

      const formData = new FormData();
      formData.append('file', blob, 'webcam_capture.jpg');
      formData.append('confidence', '0.25');
      formData.append('iou', '0.45');

      try {
        const response = await fetch('/api/v1/detect/image', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.success) {
          const count = data.results.total_detections;
          statDetections.textContent = parseInt(statDetections.textContent || '0') + count;

          // Show result overlay on canvas
          showDetectionOverlay(ctx, data.results.detections || []);
          webcamCanvas.classList.remove('d-none');

          showAlert(
            `Detected <strong>${count}</strong> vehicle(s) in captured frame.`,
            count > 0 ? 'success' : 'info'
          );

          // Offer download
          if (data.output_url) {
            offerDownload(data.output_url, 'webcam_detection.jpg');
          }
        } else {
          showAlert('Detection failed: ' + (data.error || 'Unknown error'), 'danger');
        }
      } catch (err) {
        showAlert('Server error during detection.', 'danger');
        console.error(err);
      } finally {
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<i class="bi bi-camera-fill me-2"></i>Capture Frame';
      }
    }, 'image/jpeg', 0.92);
  });

  // ── Draw detection boxes on canvas ───────────────────────────
  function showDetectionOverlay(ctx, detections) {
    const colors = ['#00d4ff', '#00e676', '#ffab40', '#ff5252', '#7b2ff7'];
    detections.forEach((det, i) => {
      const [x1, y1, x2, y2] = det.bbox;
      const color = colors[i % colors.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${det.class} ${det.confidence}%`;
      ctx.font = 'bold 14px Inter, sans-serif';
      const tw = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 22, tw + 10, 22);

      ctx.fillStyle = '#000';
      ctx.fillText(label, x1 + 5, y1 - 5);
    });
  }

  // ── Offer file download ───────────────────────────────────────
  function offerDownload(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // ── Alert helper ──────────────────────────────────────────────
  function showAlert(message, type = 'info') {
    const container = document.querySelector('.flash-container') || (() => {
      const div = document.createElement('div');
      div.className = 'flash-container';
      document.body.appendChild(div);
      return div;
    })();

    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show flash-alert`;
    alert.innerHTML = `${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    container.appendChild(alert);
    setTimeout(() => bootstrap.Alert.getOrCreateInstance(alert)?.close(), 5000);
  }

  // ── Cleanup on page unload ────────────────────────────────────
  window.addEventListener('beforeunload', () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
  });

});
