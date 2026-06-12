"""
app/models/dehazer.py
=====================
Dehazer Wrapper
---------------
This module wraps the DehazeFormer architecture into a simple, reusable
class that the rest of the application can call without knowing the
internals of the model.

Key design decisions for FYP:
  - Singleton pattern (same as detector.py) — model loaded once, reused
  - ENABLED flag — can be toggled at runtime without restarting the server
  - Graceful fallback — if weights not found, returns original image unchanged
  - Padding trick — handles images whose H/W are not divisible by 8
  - CPU-only — no GPU required for FYP demo
"""

import os
import cv2
import time
import numpy as np
import torch
from flask import current_app

from .dehazeformer import build_dehazeformer_s


class Dehazer:
    """
    Singleton wrapper around DehazeFormer.

    Usage:
        from app.models.dehazer import dehazer

        # Dehaze a single image file
        dehazed_path = dehazer.dehaze_image(input_path, output_path)

        # Check / toggle the dehazer
        dehazer.enabled          # True / False
        dehazer.set_enabled(False)   # disable at runtime
    """

    _instance = None
    _model    = None

    # ── Runtime toggle ────────────────────────────────────────────────────────
    # This flag controls whether dehazing is applied.
    # Set to False to skip dehazing and pass the original image through.
    enabled = True

    @classmethod
    def get_instance(cls):
        """Return the singleton Dehazer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, weights_path: str):
        """
        Load DehazeFormer-S weights from disk.

        If the weights file does not exist, the model is still built but
        with random weights — dehazing will run but produce poor results.
        A warning is printed so the developer knows what happened.
        """
        if self._model is not None:
            return self._model

        print("[Dehazer] Building DehazeFormer-S architecture...")
        model = build_dehazeformer_s()

        if os.path.exists(weights_path):
            print(f"[Dehazer] Loading weights from: {weights_path}")
            try:
                # Load checkpoint — weights_only=False needed for older .pth files
                checkpoint = torch.load(weights_path, map_location='cpu',
                                        weights_only=False)

                # Checkpoints can be saved as plain state_dict or wrapped in a dict
                if isinstance(checkpoint, dict):
                    state = (checkpoint.get('state_dict')
                             or checkpoint.get('model')
                             or checkpoint.get('params')
                             or checkpoint)
                else:
                    state = checkpoint

                # Remove 'module.' prefix if model was saved with DataParallel
                state = {k.replace('module.', ''): v for k, v in state.items()}

                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f"[Dehazer] Missing keys ({len(missing)}): {missing[:3]}...")
                if unexpected:
                    print(f"[Dehazer] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
                print("[Dehazer] Weights loaded successfully.")

            except Exception as e:
                print(f"[Dehazer] WARNING — could not load weights: {e}")
                print("[Dehazer] Running with random weights (dehazing quality will be poor).")
        else:
            print(f"[Dehazer] WARNING — weights not found at: {weights_path}")
            print("[Dehazer] Running with random weights. Download pretrained weights for good results.")
            print("[Dehazer] See README for download instructions.")

        model.eval()   # inference mode — disables dropout / batch norm updates
        self._model = model
        return self._model

    def get_model(self):
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            weights_path = current_app.config.get('DEHAZER_WEIGHTS_PATH', '')
            self.load_model(weights_path)
        return self._model

    # ── Enable / Disable toggle ───────────────────────────────────────────────

    def set_enabled(self, value: bool):
        """
        Enable or disable dehazing at runtime.

        When disabled, dehaze_image() returns the original image path
        immediately without running any inference — zero overhead.
        """
        self.enabled = bool(value)
        state = "ENABLED" if self.enabled else "DISABLED"
        print(f"[Dehazer] Dehazing {state}")

    # ── Core inference ────────────────────────────────────────────────────────

    def dehaze_image(self, input_path: str, output_path: str) -> dict:
        """
        Dehaze a single image and save the result.

        Args:
            input_path:  Path to the hazy input image (any OpenCV-readable format)
            output_path: Path where the dehazed image will be saved

        Returns:
            dict with keys:
                dehazed_path  — output_path if dehazing ran, else input_path
                skipped       — True if dehazer was disabled
                inference_ms  — inference time in milliseconds (0 if skipped)
        """
        # ── Fast path: dehazer disabled ───────────────────────────────────
        if not self.enabled:
            return {
                'dehazed_path': input_path,
                'skipped': True,
                'inference_ms': 0
            }

        # ── Load image ────────────────────────────────────────────────────
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            raise ValueError(f"[Dehazer] Cannot read image: {input_path}")

        # Convert BGR → RGB and normalise to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        # ── Prepare tensor ────────────────────────────────────────────────
        # (H, W, 3) → (1, 3, H, W)
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

        # Pad to multiple of 8 (required by the 3-level U-Net downsampling)
        tensor, (pad_h, pad_w) = self._pad_to_multiple(tensor, multiple=8)

        # ── Run inference ─────────────────────────────────────────────────
        model = self.get_model()
        start = time.time()

        with torch.no_grad():
            output = model(tensor)          # (1, 3, H_padded, W_padded)

        inference_ms = round((time.time() - start) * 1000, 1)

        # ── Remove padding ────────────────────────────────────────────────
        H_orig, W_orig = img_bgr.shape[:2]
        output = output[:, :, :H_orig, :W_orig]

        # ── Convert back to uint8 BGR image ───────────────────────────────
        out_np = output.squeeze(0).permute(1, 2, 0).numpy()   # (H, W, 3) RGB
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        # ── Save ──────────────────────────────────────────────────────────
        cv2.imwrite(output_path, out_bgr)
        print(f"[Dehazer] Done in {inference_ms} ms → {output_path}")

        return {
            'dehazed_path': output_path,
            'skipped': False,
            'inference_ms': inference_ms
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 8):
        """
        Pad a (B, C, H, W) tensor so H and W are divisible by `multiple`.
        Uses reflection padding to avoid border artifacts.

        Returns:
            padded tensor, (pad_h, pad_w) — amount of padding added
        """
        _, _, H, W = tensor.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            # F.pad order: (left, right, top, bottom)
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode='reflect'
            )
        return tensor, (pad_h, pad_w)


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this in routes/api — do NOT instantiate Dehazer() directly elsewhere.
dehazer = Dehazer.get_instance()
