"""
MolmoPointDetector

General-purpose Molmo2-4B pointing interface that predicts semantically
meaningful interaction points for detected objects.  For each robot action, a
prompt like "Point to the best place to pick up the <object>" is sent alongside
the RGB crop; Molmo2 returns <points coords="..."/> XML tags that are parsed
into pixel coordinates, then back-projected to 3D via the depth frame and
converted to world frame.

Usage:
    detector = MolmoPointDetector()
    interaction_points = detector.get_interaction_points(
        rgb_image=rgb_np,
        depth_frame=depth_np,
        camera_intrinsics=intrinsics,
        object_id="rubber_duck_1",
        object_type="rubber_duck",
        bounding_box_2d=[y1_px, x1_px, y2_px, x2_px],  # 0-1000 normalized
        actions={"pick", "push-aside"},
        robot_state=robot_state_dict,
    )
    # returns Dict[str, InteractionPoint]
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .object_registry import InteractionPoint
from .utils.coordinates import compute_3d_position, pixel_to_normalized

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action → natural-language pointing prompt
# ---------------------------------------------------------------------------

# Default actions run on every detected object.
DEFAULT_ACTIONS: Set[str] = {"pick", "push-aside", "displace"}

from ptp.perception.pointing_prompts import build_prompt as _build_prompt

def _make_action_prompt(action: str, object_type: str) -> str:
    """Map a built-in action key to a semantically grounded pointing prompt."""
    _ACTION_GOAL_MAP = {
        "pick":        ("grasp",      None),
        "push-aside":  ("push_aside", None),
        "displace":    ("grasp",      "displace"),
        "place-on-top":("grasp",      "place"),
        "open":        ("grasp",      "open"),
        "close":       ("grasp",      "close"),
        "pour":        ("grasp",      "pour"),
    }
    query_type, goal = _ACTION_GOAL_MAP.get(action, ("grasp", None))
    return _build_prompt(query_type, object_type, action_goal=goal)

_MODEL_CHECKPOINT = "allenai/Molmo2-4B"

# Molmo2 output format: <points coords="<frame> <x> <y>; ..."/>
# coords are in [0, 1000] scale; frame id is 1-indexed float.
_COORD_REGEX = re.compile(r'<(?:points|tracks)[^>]*coords="([0-9\t:;, .]+)"')
_FRAME_REGEX = re.compile(r'(?:^|\t|:|,|;)\s*([0-9.]+)\s+([0-9. ]+)')
_POINTS_REGEX = re.compile(r'([0-9]+)\s+([0-9]{3,4})\s+([0-9]{3,4})')


def _extract_points(text: str, image_w: int, image_h: int) -> List[Tuple[float, float]]:
    """Parse Molmo2 <points coords="..."/> output into (x_px, y_px) pixel coords."""
    results = []
    for coord_match in _COORD_REGEX.finditer(text):
        for pt_match in _POINTS_REGEX.finditer(coord_match.group(1)):
            x_norm = float(pt_match.group(2))
            y_norm = float(pt_match.group(3))
            # coords are 0-1000 scale
            x_px = x_norm / 1000.0 * image_w
            y_px = y_norm / 1000.0 * image_h
            if 0 <= x_px <= image_w and 0 <= y_px <= image_h:
                results.append((x_px, y_px))
    return results


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class MolmoPointDetector:
    """
    General-purpose Molmo2-4B pointing interface that produces per-action
    InteractionPoint objects.

    Args:
        checkpoint: HuggingFace model ID or local path (default: allenai/Molmo2-4B).
        device: torch device string; None auto-selects cuda if available.
        logger: Optional logger.

    Example:
        >>> detector = MolmoPointDetector()
        >>> ips = detector.get_interaction_points(
        ...     rgb_image=rgb_np,
        ...     depth_frame=depth_np,
        ...     camera_intrinsics=intrinsics,
        ...     object_id="cup_1",
        ...     object_type="cup",
        ...     bounding_box_2d=[100, 200, 300, 400],
        ...     actions={"pick", "push-aside"},
        ... )
    """

    def __init__(
        self,
        checkpoint: str = _MODEL_CHECKPOINT,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device
        self.logger = logger or logging.getLogger(__name__)
        self._model = None
        self._processor = None
        self._exec_device = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Eagerly load the model and processor. Safe to call multiple times."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self.logger.info("Loading Molmo2-4B from '%s'…", self._checkpoint)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        # 4B in bfloat16 ≈ 8 GB — right at the RTX 4060 limit.
        # 4-bit NF4 brings it to ~2.5 GB.  The vision backbone must be excluded
        # via llm_int8_skip_modules so its LayerNorm stays in bfloat16.
        # llm_int8_enable_fp32_cpu_offload=True lets any layers that don't fit in
        # VRAM (e.g. when GSAM2 is already loaded) offload to CPU in fp32 rather
        # than raising an error.
        quant_cfg = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_backbone", "lm_head"],
                llm_int8_enable_fp32_cpu_offload=True,
            )
            if torch.cuda.is_available() else None
        )

        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "6GiB", "cpu": "48GiB"},
        )
        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg
        else:
            load_kwargs["dtype"] = "auto"

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._checkpoint, **load_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(
                self._checkpoint,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            self._model = None
            self._processor = None
            raise

        self._exec_device = next(
            (p.device for p in self._model.parameters() if p.device.type == "cuda"),
            torch.device("cpu"),
        )
        self.logger.info("Molmo2-4B loaded on %s (8-bit).", self._exec_device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_interaction_points(
        self,
        rgb_image: np.ndarray,
        depth_frame: Optional[np.ndarray],
        camera_intrinsics: Optional[Any],
        object_id: str,
        object_type: str,
        bounding_box_2d: Optional[List[int]] = None,
        actions: Optional[Set[str]] = None,
        robot_state: Optional[Dict[str, Any]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        clearance_profile: Optional[Any] = None,
    ) -> Dict[str, InteractionPoint]:
        """
        Return a Dict[action → InteractionPoint] for a single detected object.

        Args:
            rgb_image: Full-frame RGB image as HxWx3 uint8 ndarray.
            depth_frame: Aligned depth image in metres (same HxW), or None.
            camera_intrinsics: Camera intrinsics with fx/fy/ppx/ppy attributes.
            object_id: Object identifier string (for logging).
            object_type: Human-readable object category (e.g. "rubber_duck").
            bounding_box_2d: [y1, x1, y2, x2] in 0-1000 normalized scale.
            actions: Set of robot action strings to query.  Defaults to DEFAULT_ACTIONS.
            robot_state: Robot state dict with "camera" key for world-frame transform.
            custom_prompts: Optional dict mapping action keys to custom pointing prompt
                strings.  When provided for an action, the custom string is used instead
                of the built-in _ACTION_PROMPTS lookup.
            clearance_profile: Optional ClearanceProfile for the object.  When provided,
                approach_orientation and approach_vector are computed from
                clearance_profile.best_approach_dirs() and stored on each InteractionPoint.

        Returns:
            Dict mapping each queried action to its InteractionPoint, or an
            empty dict if the model call fails.
        """
        self._ensure_loaded()

        actions = actions or DEFAULT_ACTIONS
        custom_prompts = custom_prompts or {}
        h, w = rgb_image.shape[:2]

        # Determine crop region. bounding_box_2d is [y1, x1, y2, x2] in 0-1000
        # normalized scale (same as position_2d); convert to pixels before slicing.
        if bounding_box_2d is not None and len(bounding_box_2d) == 4:
            ny1, nx1, ny2, nx2 = bounding_box_2d
            y1 = max(0, int(ny1 * h / 1000.0))
            x1 = max(0, int(nx1 * w / 1000.0))
            y2 = min(h, int(ny2 * h / 1000.0))
            x2 = min(w, int(nx2 * w / 1000.0))
            crop_rgb = rgb_image[y1:y2, x1:x2]
            crop_depth = depth_frame[y1:y2, x1:x2] if depth_frame is not None else None
            crop_offset = (y1, x1)
        else:
            crop_rgb = rgb_image
            crop_depth = depth_frame
            crop_offset = (0, 0)

        if crop_rgb.size == 0:
            crop_rgb = rgb_image
            crop_depth = depth_frame
            crop_offset = (0, 0)

        # Compute approach orientation from clearance profile once (shared across actions)
        approach_orientation: Optional[str] = None
        approach_vector: Optional[np.ndarray] = None
        if clearance_profile is not None:
            try:
                best_dirs = clearance_profile.best_approach_dirs
                if best_dirs:
                    best_dir = np.asarray(best_dirs[0], dtype=float)
                    approach_vector = best_dir
                    # Check if direction is close to top-down [0, 0, -1] (within 45 deg)
                    top_down = np.array([0.0, 0.0, -1.0])
                    cos_angle = float(np.dot(best_dir, top_down) / (
                        np.linalg.norm(best_dir) * np.linalg.norm(top_down) + 1e-9
                    ))
                    # cos(45 deg) ≈ 0.707
                    approach_orientation = "top_down" if cos_angle >= 0.707 else "side"
            except Exception as exc:
                self.logger.warning(
                    "Failed to compute approach orientation from clearance for %s: %s",
                    object_id, exc,
                )

        result: Dict[str, InteractionPoint] = {}

        for action in actions:
            custom_prompt = custom_prompts.get(action)
            try:
                ip = self._query_single(
                    crop_rgb=crop_rgb,
                    crop_depth=crop_depth,
                    full_depth=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    full_image_shape=(h, w),
                    crop_offset=crop_offset,
                    object_type=object_type,
                    action=action,
                    robot_state=robot_state,
                    custom_prompt=custom_prompt,
                )
                if ip is not None:
                    ip.approach_orientation = approach_orientation
                    ip.approach_vector = approach_vector
                    result[action] = ip
            except Exception as exc:
                self.logger.warning(
                    "Molmo2 query failed for %s/%s: %s", object_id, action, exc
                )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
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
    ) -> Optional[InteractionPoint]:
        """Run one Molmo2 forward pass for a single action, return InteractionPoint."""
        import torch
        from PIL import Image as _PIL

        if custom_prompt is not None:
            prompt_text = custom_prompt
        else:
            prompt_text = _make_action_prompt(action, object_type)

        pil_image = _PIL.fromarray(crop_rgb.astype(np.uint8))
        crop_h, crop_w = crop_rgb.shape[:2]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": prompt_text},
                    {"type": "image", "image": pil_image},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = self._exec_device
        inputs = {
            k: (v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=200)

        generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
        generated_text = self._processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        self.logger.info("Molmo2 raw output for '%s': %s", action, generated_text)

        pts = _extract_points(generated_text, crop_w, crop_h)

        if not pts:
            self.logger.warning(
                "Molmo2 returned no points for action '%s' | prompt: %r | raw output: %r",
                action,
                prompt_text,
                generated_text,
            )
            return None

        # First point → primary; remainder → alternatives.
        # Convert crop-relative pixel coords back to full-frame pixel coords.
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

        # 3-D back-projection.
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
        )


# ---------------------------------------------------------------------------
# World-frame transform (mirrors gsam2_object_tracker._transform_cam_to_world)
# ---------------------------------------------------------------------------

def _transform_cam_to_world(
    cam_pos: np.ndarray,
    robot_state: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    if robot_state is None:
        return None
    cam_tf = robot_state.get("camera")
    if cam_tf is None:
        return None
    try:
        from scipy.spatial.transform import Rotation
        cam_origin = np.array(cam_tf["position"], dtype=float)
        cam_rot    = Rotation.from_quat(cam_tf["quaternion_xyzw"])
        return cam_rot.apply(cam_pos) + cam_origin
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

MolmoInteractionPointDetector = MolmoPointDetector
