"""
MolmoClient — drop-in replacement for MolmoPointDetector that queries a
running molmo_server.py instance instead of loading the model locally.

Usage:
    detector = MolmoClient(server_url="http://127.0.0.1:8765")
    # identical API to MolmoPointDetector
    ips = detector.get_interaction_points(...)
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .molmo_point_detector import (
    MolmoPointDetector,
    _extract_points,
    _transform_cam_to_world,
)
from .object_registry import InteractionPoint
from .utils.coordinates import compute_3d_position, pixel_to_normalized

logger = logging.getLogger(__name__)


class MolmoClient(MolmoPointDetector):
    """MolmoPointDetector that delegates inference to a remote molmo_server.

    Args:
        server_url: Base URL of the running molmo_server, e.g.
                    ``"http://127.0.0.1:8765"``.
        timeout:    Per-request timeout in seconds (default 60).
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8765",
        timeout: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Don't call super().__init__() — we never load weights locally.
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        # Satisfy attribute checks used by the base class public API.
        self._model = None
        self._processor = None
        self._exec_device = None

    # ------------------------------------------------------------------
    # Override: skip local model loading entirely
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        pass

    def load(self) -> None:
        self._check_health()

    def _check_health(self) -> None:
        import urllib.request
        url = f"{self._server_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"health check returned HTTP {resp.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Molmo server not reachable at {self._server_url} ({exc}). "
                "Start it with: python scripts/molmo_server.py"
            ) from exc

    # ------------------------------------------------------------------
    # Override: replace torch inference with an HTTP POST
    # ------------------------------------------------------------------

    def _query_single(
        self,
        crop_rgb: np.ndarray,
        crop_depth: Optional[np.ndarray],
        full_depth: Optional[np.ndarray],
        camera_intrinsics: Optional[Any],
        full_image_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        object_type: str,
        action: str,
        robot_state: Optional[Dict[str, Any]],
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Optional[InteractionPoint], Optional[bytes]]:
        from PIL import Image as _PIL

        if custom_prompt is not None:
            from .molmo_point_detector import _make_action_prompt
            prompt_text = custom_prompt
        else:
            from .molmo_point_detector import _make_action_prompt
            prompt_text = _make_action_prompt(action, object_type)

        # Encode crop as PNG for both the server payload and debug output.
        pil_image = _PIL.fromarray(crop_rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        input_image_bytes = buf.getvalue()
        image_b64 = base64.b64encode(input_image_bytes).decode()

        crop_h, crop_w = crop_rgb.shape[:2]

        generated_text, srv_w, srv_h = self._call_server(prompt_text, image_b64)
        # Server returns the image dimensions it decoded; use them if available.
        if srv_w and srv_h:
            crop_w, crop_h = srv_w, srv_h

        self.logger.info("Molmo server output for '%s': %s", action, generated_text)

        pts = _extract_points(generated_text, crop_w, crop_h)
        if not pts:
            self.logger.warning(
                "Molmo server returned no points for action '%s' | prompt: %r | output: %r",
                action, prompt_text, generated_text,
            )
            return None, input_image_bytes

        full_h, full_w = full_image_shape
        crop_y_off, crop_x_off = crop_offset

        def _to_full(x_px: float, y_px: float) -> Tuple[int, int]:
            px = max(0, min(full_w - 1, int(x_px) + crop_x_off))
            py = max(0, min(full_h - 1, int(y_px) + crop_y_off))
            return px, py

        px0, py0 = _to_full(*pts[0])
        norm_2d = pixel_to_normalized((py0, px0), full_image_shape)

        alternative_points = []
        for x_px, y_px in pts[1:]:
            apx, apy = _to_full(x_px, y_px)
            alternative_points.append(
                {"position_2d": pixel_to_normalized((apy, apx), full_image_shape)}
            )

        position_3d = None
        if full_depth is not None and camera_intrinsics is not None:
            cam_pos = compute_3d_position(norm_2d, full_depth, camera_intrinsics)
            if cam_pos is not None:
                world_pos = _transform_cam_to_world(cam_pos, robot_state)
                position_3d = world_pos if world_pos is not None else cam_pos

        return InteractionPoint(
            position_2d=norm_2d,
            position_3d=position_3d,
            alternative_points=alternative_points,
        ), input_image_bytes

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_server(self, prompt: str, image_b64: str) -> Tuple[str, int, int]:
        import json
        import urllib.request

        payload = json.dumps({"prompt": prompt, "image_b64": image_b64}).encode()
        req = urllib.request.Request(
            f"{self._server_url}/point",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"Molmo server request failed: {exc}") from exc

        if "error" in body:
            raise RuntimeError(f"Molmo server error: {body['error']}")

        return body.get("text", ""), body.get("image_w", 0), body.get("image_h", 0)
