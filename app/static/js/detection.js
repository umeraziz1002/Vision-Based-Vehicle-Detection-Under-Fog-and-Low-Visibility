/**
 * detection.js — Upload & Detection page logic
 * Handles drag-and-drop, file preview, range sliders, and form submission.
 * Two-button mode: "Dehaze Image" and "Run Detection"
 */

document.addEventListener('DOMContentLoaded', () => {

  const dropZone       = document.getElementById('dropZone');
  const fileInput      = document.getElementById('fileInput');
  const dropContent    = document.getElementById('dropContent');
  const dropPreview    = document.getElementById('dropPreview');
  const previewImg     = document.getElementById('previewImg');
  const previewVid     = document.getElementById('previewVid');
  const previewName    = document.getElementById('previewName');
  const previewSize    = document.getElementById('previewSize');
  const clearBtn       = document.getElementById('clearFile');
  const submitBtn      = document.getElementById('submitBtn');       // Detect button
  const dehazeOnlyBtn  = document.getElementById('dehazeOnlyBtn');   // Dehaze Only button
  const uploadForm     = document.getElementById('uploadForm');
  const dehazeOnlyForm = document.getElementById('dehazeOnlyForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingMsg     = document.getElementById('loadingMsg');
  const detectHint     = document.getElementById('detectHint');
  const detectBtnLabel = document.getElementById('detectBtnLabel');

  // ── Range sliders ─────────────────────────────────────────────
  const confRange = document.getElementById('confRange');
  const iouRange  = document.getElementById('iouRange');
  const confVal   = document.getElementById('confVal');
  const iouVal    = document.getElementById('iouVal');

  if (confRange) {
    confRange.addEventListener('input', () => {
      confVal.textContent = parseFloat(confRange.value).toFixed(2);
    });
  }

  if (iouRange) {
    iouRange.addEventListener('input', () => {
      iouVal.textContent = parseFloat(iouRange.value).toFixed(2);
    });
  }

  // ── Click to open file dialog ─────────────────────────────────
  if (dropZone) {
    dropZone.addEventListener('click', (e) => {
      // Don't trigger if clicking the clear button
      if (e.target.closest('#clearFile')) return;
      fileInput.click();
    });
  }

  // ── Drag & Drop events ────────────────────────────────────────
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone?.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone?.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone?.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  // ── File input change ─────────────────────────────────────────
  fileInput?.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  // ── Clear file ────────────────────────────────────────────────
  clearBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    resetDropZone();
  });

  // ── Handle selected file ──────────────────────────────────────
  function handleFile(file) {
    const imageTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
    const videoTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
    const allAllowed = [...imageTypes, ...videoTypes];

    if (!allAllowed.includes(file.type) && !isAllowedByExtension(file.name)) {
      showToast('Unsupported file type. Please use JPG, PNG, MP4, or AVI.', 'warning');
      return;
    }

    // Transfer to the actual form input
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    // Show preview
    dropContent.classList.add('d-none');
    dropPreview.classList.remove('d-none');

    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);

    if (imageTypes.includes(file.type) || isImage(file.name)) {
      previewImg.src = URL.createObjectURL(file);
      previewImg.classList.remove('d-none');
      previewVid.classList.add('d-none');
    } else {
      previewVid.src = URL.createObjectURL(file);
      previewVid.classList.remove('d-none');
      previewImg.classList.add('d-none');
    }

    submitBtn.disabled = false;

    // Enable Dehaze Only button only for images (not videos)
    if (dehazeOnlyBtn) {
      const isImg = imageTypes.includes(file.type) || isImage(file.name);
      dehazeOnlyBtn.disabled = !isImg;
      dehazeOnlyBtn.title = isImg ? '' : 'Dehaze Only is available for images only';
    }

    // Update detect button label based on dehazer toggle state
    updateDetectBtnLabel();
  }

  function resetDropZone() {
    fileInput.value = '';
    dropContent.classList.remove('d-none');
    dropPreview.classList.add('d-none');
    previewImg.src = '';
    previewVid.src = '';
    previewImg.classList.add('d-none');
    previewVid.classList.add('d-none');
    submitBtn.disabled = true;
    if (dehazeOnlyBtn) dehazeOnlyBtn.disabled = true;
  }

  // ── Form submission with loading overlay ──────────────────────
  uploadForm?.addEventListener('submit', (e) => {
    if (!fileInput.files.length) {
      e.preventDefault();
      showToast('Please select a file first.', 'warning');
      return;
    }

    const file = fileInput.files[0];
    const isVideo = isVideoFile(file.name);
    const dehazeSwitch = document.getElementById('dehazeSwitchForm');
    const isDehazeOn = dehazeSwitch && dehazeSwitch.checked;

    // Show loading overlay
    loadingOverlay.classList.remove('d-none');

    if (isVideo) {
      loadingMsg.textContent = 'Processing video frames with YOLOv8...';
    } else if (isDehazeOn) {
      loadingMsg.textContent = 'Step 1/2: DehazeFormer enhancing image...';
    } else {
      loadingMsg.textContent = 'Running YOLOv8 inference...';
    }

    // Disable submit button
    const normalSpan  = submitBtn.querySelector('.btn-normal');
    const loadingSpan = submitBtn.querySelector('.btn-loading');
    normalSpan.classList.add('d-none');
    loadingSpan.classList.remove('d-none');
    submitBtn.disabled = true;
    if (dehazeOnlyBtn) dehazeOnlyBtn.disabled = true;
  });

  // ── Dehaze Only button ────────────────────────────────────────
  dehazeOnlyBtn?.addEventListener('click', () => {
    if (!fileInput.files.length) {
      showToast('Please select an image first.', 'warning');
      return;
    }

    // Build a real FormData and POST to /detect/dehaze
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show loading overlay
    loadingOverlay.classList.remove('d-none');
    if (loadingMsg) loadingMsg.textContent = 'DehazeFormer enhancing image...';

    // Disable both buttons
    dehazeOnlyBtn.disabled = true;
    submitBtn.disabled = true;
    const normalSpan  = dehazeOnlyBtn.querySelector('.btn-normal');
    const loadingSpan = dehazeOnlyBtn.querySelector('.btn-loading');
    normalSpan.classList.add('d-none');
    loadingSpan.classList.remove('d-none');

    // Submit via fetch then redirect to the returned URL
    fetch('/detect/dehaze', { method: 'POST', body: formData })
      .then(res => {
        // Flask returns a redirect (302) — follow it
        if (res.redirected) {
          window.location.href = res.url;
        } else if (res.ok) {
          // If server returned HTML directly, replace the page
          return res.text().then(html => {
            document.open(); document.write(html); document.close();
          });
        } else {
          return res.text().then(t => { throw new Error(t); });
        }
      })
      .catch(err => {
        loadingOverlay.classList.add('d-none');
        normalSpan.classList.remove('d-none');
        loadingSpan.classList.add('d-none');
        dehazeOnlyBtn.disabled = false;
        submitBtn.disabled = false;
        showToast('Dehazing failed. Please try again.', 'danger');
        console.error(err);
      });
  });

  // ── Update detect button label based on dehazer toggle ────────
  function updateDetectBtnLabel() {
    if (!detectBtnLabel) return;
    const dehazeSwitch = document.getElementById('dehazeSwitchForm');
    const isOn = dehazeSwitch && dehazeSwitch.checked;
    const hasFile = fileInput && fileInput.files.length > 0;
    const isImg = hasFile && isImage(fileInput.files[0].name);

    if (isOn && isImg) {
      detectBtnLabel.innerHTML = '<i class="bi bi-stars me-1"></i><i class="bi bi-lightning-charge-fill me-2"></i>Dehaze + Detect';
      if (detectHint) detectHint.textContent = 'DehazeFormer will enhance the image before detection';
    } else {
      detectBtnLabel.innerHTML = '<i class="bi bi-lightning-charge-fill me-2"></i>Run Detection';
      if (detectHint) detectHint.textContent = 'Detection only — toggle DehazeFormer above to enhance first';
    }
  }

  // ── Helpers ───────────────────────────────────────────────────
  function isAllowedByExtension(name) {
    const allowed = ['jpg','jpeg','png','bmp','webp','mp4','avi','mov','mkv'];
    const ext = name.split('.').pop().toLowerCase();
    return allowed.includes(ext);
  }

  function isImage(name) {
    return ['jpg','jpeg','png','bmp','webp'].includes(name.split('.').pop().toLowerCase());
  }

  function isVideoFile(name) {
    return ['mp4','avi','mov','mkv'].includes(name.split('.').pop().toLowerCase());
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function showToast(message, type = 'info') {
    const container = document.querySelector('.flash-container') || createFlashContainer();
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show flash-alert`;
    alert.innerHTML = `
      <i class="bi bi-exclamation-triangle-fill me-2"></i>${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    container.appendChild(alert);
    setTimeout(() => bootstrap.Alert.getOrCreateInstance(alert).close(), 4000);
  }

  function createFlashContainer() {
    const div = document.createElement('div');
    div.className = 'flash-container';
    document.body.appendChild(div);
    return div;
  }

});

// ── DehazeFormer toggle switch ────────────────────────────────────────────────
// Updates the label and status pill instantly when the user flips the switch.
// Also notifies the server via AJAX so the runtime state stays in sync.
document.addEventListener('DOMContentLoaded', () => {
  const dehazeSwitch = document.getElementById('dehazeSwitchForm');
  const dehazeLabel  = document.getElementById('dehazeSwitchLabel');
  const statusPill   = document.getElementById('dehazerStatusPill');
  const loadingMsg   = document.getElementById('loadingMsg');

  if (!dehazeSwitch) return;

  dehazeSwitch.addEventListener('change', () => {
    const isOn = dehazeSwitch.checked;

    // Update label text and colour
    dehazeLabel.textContent = isOn ? 'ON' : 'OFF';
    dehazeLabel.style.color = isOn ? 'var(--accent)' : '#9aa3b8';

    // Update status pill
    if (statusPill) {
      statusPill.innerHTML = isOn
        ? '<span class="pill-on"><i class="bi bi-stars me-1"></i>Dehazing will be applied before detection</span>'
        : '<span class="pill-off"><i class="bi bi-slash-circle me-1"></i>Dehazing disabled — using original image</span>';
    }

    // Update loading message so user knows what's happening
    if (loadingMsg) {
      loadingMsg.textContent = isOn
        ? 'Step 1/2: DehazeFormer enhancing image...'
        : 'Running YOLOv8 inference...';
    }

    // Notify server (fire-and-forget — form submission is the real trigger)
    fetch('/detect/dehazer/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: isOn })
    }).catch(() => {}); // silently ignore network errors
  });

  // Update loading message on form submit based on current toggle state
  const uploadForm = document.getElementById('uploadForm');
  if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
      const isOn = dehazeSwitch && dehazeSwitch.checked;
      if (loadingMsg) {
        loadingMsg.textContent = isOn
          ? 'Step 1/2: DehazeFormer enhancing image...'
          : 'Running YOLOv8 inference...';
      }
    }, { capture: true });
  }
});
