/**
 * split_compare.js
 * ================
 * Interactive before/after image comparison slider.
 * Drag the handle left/right to reveal original vs dehazed image.
 */

document.addEventListener('DOMContentLoaded', () => {
  const wrap   = document.getElementById('splitWrap');
  const handle = document.getElementById('splitHandle');
  const after  = document.getElementById('splitAfter');   // dehazed (top layer)

  if (!wrap || !handle || !after) return;

  let isDragging = false;
  let startPercent = 50;

  // Set initial clip at 50%
  setClip(50);

  // ── Mouse events ──────────────────────────────────────────────────────────
  handle.addEventListener('mousedown', (e) => {
    isDragging = true;
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    updateFromEvent(e.clientX);
  });

  document.addEventListener('mouseup', () => { isDragging = false; });

  // ── Touch events ──────────────────────────────────────────────────────────
  handle.addEventListener('touchstart', (e) => {
    isDragging = true;
    e.preventDefault();
  }, { passive: false });

  document.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    updateFromEvent(e.touches[0].clientX);
  }, { passive: true });

  document.addEventListener('touchend', () => { isDragging = false; });

  // ── Click anywhere on wrap to jump ────────────────────────────────────────
  wrap.addEventListener('click', (e) => {
    if (e.target === handle || handle.contains(e.target)) return;
    updateFromEvent(e.clientX);
  });

  // ── Core update function ──────────────────────────────────────────────────
  function updateFromEvent(clientX) {
    const rect = wrap.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(2, Math.min(98, pct));   // clamp 2%–98%
    setClip(pct);
  }

  function setClip(pct) {
    // Clip the "after" (dehazed) image to show only the right portion
    after.style.clipPath = `inset(0 0 0 ${pct}%)`;
    // Move the handle
    handle.style.left = `${pct}%`;
  }
});
