"""GSAM2-based object tracker: RAM+ → GroundingDINO → SAM2.

Provides:
  GSAM2ObjectTracker         — single-frame async detect_objects()
  GSAM2ContinuousObjectTracker — background continuous tracking loop

Molmo is NOT owned here; it is loaded once by the caller and passed to
PrimitiveExecutor for execution-time grounding.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import io
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

import os
import numpy as np
from PIL import Image

from ptp.perception.clearance import GripperGeometry, compute_clearance_profile
from ptp.perception.contact_graph import compute_contact_graph
from ptp.perception.occlusion import CameraPose, ObservationRecord, compute_occlusion_map
from ptp.perception.gsam2 import IncrementalObjectTracker, OpenAITagger
from ptp.perception.object_registry import DetectedObject, DetectedObjectRegistry, InteractionPoint
from ptp.perception.utils.coordinates import compute_3d_position, pixel_to_normalized
from ptp.utils.logging_utils import get_structured_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAM2_CFG  = os.environ.get("SAM2_CFG",  "configs/sam2.1/sam2.1_hiera_b+.yaml")
_SAM2_CKPT = os.environ.get("SAM2_CKPT", "./checkpoints/sam2.1_hiera_base_plus.pt")

_JUNK_LABELS = frozenset({
    "image", "photo", "picture", "frame", "background", "scene", "view",
    "object", "thing", "item", "area", "region", "part", "surface",
    "unknown", "none", "null", "",
})

_STOPWORDS = frozenset({
    "a", "an", "the", "put", "place", "move", "pick", "up", "on", "onto",
    "in", "into", "to", "from", "and", "or", "of", "with", "get", "take",
    "stack", "grasp", "bring", "is", "are", "it", "its", "that", "this",
})

_DEBUG_PALETTE = [
    (255,  80,  80, 200), ( 80, 120, 255, 200), ( 80, 220,  80, 200),
    (255, 200,  50, 200), (220,  80, 220, 200), ( 80, 220, 220, 200),
    (255, 160,  50, 200),
]


# ---------------------------------------------------------------------------
# Inline support types (no dependency on object_tracker.py)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrackingStats:
    total_frames: int = 0
    total_detections: int = 0
    skipped_frames: int = 0
    avg_detection_time: float = 0.0
    last_detection_time: float = 0.0
    cache_hit_rate: float = 0.0
    is_running: bool = False


def save_debug_frame(
    png_bytes: bytes,
    objects: List[Any],
    frame_index: int,
    save_dir: Path,
    lock: threading.Lock,
) -> None:
    """Render bounding boxes + labels onto a PNG and write to save_dir."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img, "RGBA")
        for i, obj in enumerate(objects):
            bbox = getattr(obj, "bounding_box_2d", None)
            if not bbox or len(bbox) < 4:
                continue
            ny1, nx1, ny2, nx2 = bbox
            x1, y1 = int(nx1 * w / 1000), int(ny1 * h / 1000)
            x2, y2 = int(nx2 * w / 1000), int(ny2 * h / 1000)
            colour = _DEBUG_PALETTE[i % len(_DEBUG_PALETTE)]
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=2, fill=(*colour[:3], 40))
            draw.text((x1 + 2, y1 + 2), obj.object_id, fill=(255, 255, 255))

        out = save_dir / f"frame_{frame_index:06d}.png"
        with lock:
            img.save(out)
            latest = save_dir / "latest.png"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(out.name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
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
        origin = np.array(cam_tf["position"], dtype=float)
        rot    = Rotation.from_quat(cam_tf["quaternion_xyzw"])
        return rot.apply(cam_pos) + origin
    except Exception:
        return None


def _extract_noun_phrases(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    phrases: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _STOPWORDS:
            i += 1
            continue
        if i + 1 < len(words) and words[i + 1] not in _STOPWORDS:
            phrases.append(f"{w} {words[i + 1]}")
            i += 2
        else:
            phrases.append(w)
            i += 1
    return phrases


# ---------------------------------------------------------------------------
# GSAM2ObjectTracker
# ---------------------------------------------------------------------------

class GSAM2ObjectTracker:
    """Single-frame async object tracker: GroundingDINO → SAM2 → DetectedObjectRegistry.

    Args:
        grounding_model_id: HuggingFace model ID for GroundingDINO.
        sam2_model_cfg: SAM2 config file path.
        sam2_ckpt_path: SAM2 checkpoint path.
        openai_api_key: Key for the OpenAI tagger (falls back to OPENAI_API_KEY env var).
        tagger_model: OpenAI model for scene tagging.
        detection_interval: Frames between GroundingDINO re-detections.
        score_threshold: Confidence threshold for detections.
        device: torch device string.
        llm_client: Optional LLMClient for affordance / predicate inference.
        robot_interface: Optional robot interface for live camera FK.
        compute_clearances: Whether to run depth-based clearance profiles.
        compute_contacts: Whether to compute object contact graph.
        compute_occlusion: Whether to maintain occlusion history map.
    """

    def __init__(
        self,
        grounding_model_id: str = os.environ.get("DINO_CKPT", "IDEA-Research/grounding-dino-tiny"),
        sam2_model_cfg: str = _SAM2_CFG,
        sam2_ckpt_path: str = _SAM2_CKPT,
        detection_interval: int = 20,
        score_threshold: float = 0.5,
        overlap_iou_threshold: float = 0.5,
        device: str = "cuda",
        tag_interval: int = 1,
        llm_client: Optional[Any] = None,
        llm_mode: str = "nl",
        robot_interface: Optional[Any] = None,
        compute_clearances: bool = True,
        gripper: Optional[GripperGeometry] = None,
        compute_contacts: bool = True,
        contact_threshold_m: float = 0.005,
        compute_occlusion: bool = True,
        occlusion_history_len: int = 10,
        occlusion_update_interval: int = 1,
        logger: Optional[Any] = None,
    ) -> None:
        self.logger = logger or get_structured_logger("GSAM2ObjectTracker")
        self.device = device
        self.tag_interval = tag_interval
        self.registry = DetectedObjectRegistry()

        self._llm_client = llm_client
        self._llm_mode = llm_mode
        self._compute_clearances = compute_clearances
        self._gripper = gripper or GripperGeometry()
        self._compute_contacts = compute_contacts
        self._contact_threshold_m = contact_threshold_m
        self._compute_occlusion = compute_occlusion
        self._occlusion_update_interval = occlusion_update_interval
        self._robot_interface = robot_interface

        self._obs_history: Deque[ObservationRecord] = collections.deque(maxlen=occlusion_history_len)
        self._affordance_cache: Dict[str, Set[str]] = {}
        self._last_masks: Dict[str, np.ndarray] = {}
        self.llm_debug_images: List[bytes] = []

        self._timing_gsam2_s: float = 0.0
        self._timing_clearance_s: float = 0.0
        self._timing_contact_graph_s: float = 0.0
        self._timing_occlusion_s: float = 0.0
        self._timing_calls: Dict[str, int] = {
            "gsam2": 0, "clearance": 0, "contact_graph": 0, "occlusion": 0,
        }

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.logger.info("Loading GroundingDINO (%s) + SAM2 on %s…", grounding_model_id, device)
        self._gsam2 = IncrementalObjectTracker(
            grounding_model_id=grounding_model_id,
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device,
            prompt_text="object.",
            detection_interval=detection_interval,
            score_threshold=score_threshold,
            overlap_iou_threshold=overlap_iou_threshold,
        )
        self.logger.info("GroundingDINO + SAM2 loaded.")

        self._tagger = OpenAITagger(llm_client=llm_client) if llm_client is not None else None
        self._current_prompt: str = "object."
        self._frame_count: int = 0
        self._extra_tags: List[str] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_tagger(self, tagger_callable: Any) -> None:
        self._tagger = tagger_callable

    def set_extra_tags(self, tags: List[str]) -> None:
        self._extra_tags = [t.strip().lower() for t in tags if t.strip()]
        self._current_prompt = self._merge_prompt(self._current_prompt, self._extra_tags)
        self._gsam2.set_prompt(self._current_prompt)

    @staticmethod
    def _merge_prompt(ram_prompt: str, extra_tags: List[str]) -> str:
        existing = {t.rstrip(".").strip() for t in ram_prompt.split() if t.strip()}
        additions = [t for t in extra_tags if t not in existing]
        if not additions:
            return ram_prompt
        return (ram_prompt.rstrip() + " " + " ".join(t + "." for t in additions)).strip()

    # ------------------------------------------------------------------
    # Camera transform
    # ------------------------------------------------------------------

    def _get_camera_transform(
        self, robot_state: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Any]:
        from scipy.spatial.transform import Rotation
        if self._robot_interface is not None:
            try:
                pos, rot = self._robot_interface.get_camera_transform()
                if pos is not None and rot is not None:
                    return np.asarray(pos, dtype=float), rot
            except Exception as exc:
                self.logger.warning("get_camera_transform failed: %s", exc)
        if robot_state is not None:
            cam_tf = robot_state.get("camera")
            if cam_tf is not None:
                try:
                    return (
                        np.array(cam_tf["position"], dtype=float),
                        Rotation.from_quat(cam_tf["quaternion_xyzw"]),
                    )
                except Exception:
                    pass
        return None, None

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    async def detect_objects(
        self,
        color_frame: Union[np.ndarray, Image.Image],
        depth_frame: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Any] = None,
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> List[DetectedObject]:
        """Run RAM+ tagging → GroundingDINO → SAM2 for one frame.

        Returns a list of DetectedObject instances added to self.registry.
        """
        _t_detect_start = time.perf_counter()
        rgb_np = np.array(color_frame) if isinstance(color_frame, Image.Image) else color_frame
        h, w = rgb_np.shape[:2]
        loop = asyncio.get_event_loop()

        # Tagging
        _t_tag_s = 0.0
        if self._tagger is not None and self._frame_count % self.tag_interval == 0:
            _t0 = time.perf_counter()
            new_prompt, _ = await loop.run_in_executor(
                None, self._tagger, rgb_np, self._extra_tags or None
            )
            _t_tag_s = time.perf_counter() - _t0
            if new_prompt:
                merged = self._merge_prompt(new_prompt, self._extra_tags)
                if merged != self._current_prompt:
                    self._current_prompt = merged
                    self._gsam2.set_prompt(merged)

        # Grounding + segmentation (GroundingDINO → SAM2)
        t0 = time.perf_counter()
        await loop.run_in_executor(None, self._gsam2.add_image, rgb_np)
        _t_gsam2_s = time.perf_counter() - t0
        self._timing_gsam2_s += _t_gsam2_s
        self._timing_calls["gsam2"] += 1
        self._frame_count += 1

        mask_dict = self._gsam2.last_mask_dict
        if mask_dict is None or not mask_dict.labels:
            return []

        detected: List[DetectedObject] = []
        obj_masks: Dict[str, np.ndarray] = {}

        _t_localize_s = 0.0
        for track_id, obj_info in mask_dict.labels.items():
            class_name = obj_info.class_name or "object"
            words = class_name.strip().lower().split()
            seen: set = set()
            safe_name = "_".join(w for w in words if not (seen.add(w) or w in seen))  # type: ignore[func-returns-value]
            # simpler dedup
            seen2: set = set()
            deduped = []
            for word in class_name.strip().lower().split():
                if word not in seen2:
                    seen2.add(word)
                    deduped.append(word)
            safe_name = "_".join(deduped)

            if safe_name in _JUNK_LABELS:
                continue

            object_id = f"{safe_name}_{track_id}"

            x1 = int(obj_info.x1) if obj_info.x1 is not None else 0
            y1 = int(obj_info.y1) if obj_info.y1 is not None else 0
            x2 = int(obj_info.x2) if obj_info.x2 is not None else w
            y2 = int(obj_info.y2) if obj_info.y2 is not None else h

            mask = obj_info.mask
            if mask is not None:
                import torch
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
            if mask is not None and mask.any():
                ys, xs = np.where(mask)
                cy, cx = int(ys.mean()), int(xs.mean())
            else:
                cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

            position_2d = pixel_to_normalized((cy, cx), (h, w))
            bbox_2d = [
                int(y1 / h * 1000), int(x1 / w * 1000),
                int(y2 / h * 1000), int(x2 / w * 1000),
            ]

            _t0 = time.perf_counter()
            position_3d = None
            if depth_frame is not None and camera_intrinsics is not None:
                cam_pos = compute_3d_position(position_2d, depth_frame, camera_intrinsics)
                if cam_pos is not None:
                    world_pos = _transform_cam_to_world(cam_pos, robot_state)
                    position_3d = world_pos if world_pos is not None else cam_pos
            _t_localize_s += time.perf_counter() - _t0

            affordances: Set[str]
            if self._llm_client is not None:
                crop = rgb_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                affordances = await self._infer_affordances(safe_name, crop if crop.size else rgb_np)
            else:
                affordances = self._affordance_cache.get(safe_name, {"graspable"})

            centroid_ip = InteractionPoint(position_2d=position_2d, position_3d=position_3d)

            obj = DetectedObject(
                object_type=safe_name,
                object_id=object_id,
                affordances=affordances,
                interaction_points={"pick": centroid_ip},
                position_2d=position_2d,
                position_3d=position_3d,
                bounding_box_2d=bbox_2d,
            )
            self.registry.add_object(obj)
            detected.append(obj)

            bool_mask = mask.astype(bool) if isinstance(mask, np.ndarray) else (
                mask.cpu().numpy().astype(bool) if mask is not None else np.zeros((h, w), bool)
            )
            if not isinstance(mask, np.ndarray) and mask is None:
                bool_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            obj_masks[object_id] = bool_mask

        self._last_masks = obj_masks

        cam_position, cam_rotation = self._get_camera_transform(robot_state)

        # Clearance profiles
        if self._compute_clearances and depth_frame is not None and camera_intrinsics is not None:
            t0 = time.perf_counter()
            obj_types = {o.object_id: o.object_type for o in detected}
            for obj in detected:
                target_mask = obj_masks.get(obj.object_id)
                if target_mask is None:
                    continue
                try:
                    obj.clearance_profile = compute_clearance_profile(
                        target_mask=target_mask,
                        depth_frame=depth_frame,
                        camera_intrinsics=camera_intrinsics,
                        all_masks={oid: m for oid, m in obj_masks.items() if oid != obj.object_id},
                        gripper=self._gripper,
                        cam_position=cam_position,
                        camera_quaternion_xyzw=cam_rotation.as_quat() if cam_rotation is not None else None,
                        object_types=obj_types,
                    )
                    self.registry.update_object(obj.object_id, obj)
                except Exception as exc:
                    self.logger.warning("Clearance failed for '%s': %s", obj.object_id, exc)
            self._timing_clearance_s += time.perf_counter() - t0
            self._timing_calls["clearance"] += 1

        # Contact graph
        if self._compute_contacts and depth_frame is not None and camera_intrinsics is not None and len(detected) >= 2:
            t0 = time.perf_counter()
            try:
                self.registry.contact_graph = compute_contact_graph(
                    objects=detected,
                    obj_masks=obj_masks,
                    depth_frame=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    contact_threshold_m=self._contact_threshold_m,
                    camera_position=cam_position,
                    camera_rotation=cam_rotation,
                    llm_client=self._llm_client,
                    llm_mode=self._llm_mode,
                    llm_debug_image_out=self.llm_debug_images,
                )
            except Exception as exc:
                self.logger.warning("Contact graph failed: %s", exc)
            self._timing_contact_graph_s += time.perf_counter() - t0
            self._timing_calls["contact_graph"] += 1

        # Occlusion map
        if self._compute_occlusion and depth_frame is not None and camera_intrinsics is not None:
            self._obs_history.append(ObservationRecord(
                depth_frame=depth_frame,
                camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose.from_robot_state(robot_state),
                obj_masks=dict(obj_masks),
            ))
            if self._frame_count % self._occlusion_update_interval == 0:
                t0 = time.perf_counter()
                try:
                    self.registry.occlusion_map = compute_occlusion_map(
                        observations=list(self._obs_history),
                        object_ids=[o.object_id for o in detected],
                    )
                except Exception as exc:
                    self.logger.warning("Occlusion map failed: %s", exc)
                self._timing_occlusion_s += time.perf_counter() - t0
                self._timing_calls["occlusion"] += 1

        # Evict stale tracks
        active_ids: Set[int] = set(mask_dict.labels.keys())
        for reg_id in list(self.registry._objects.keys()):
            parts = reg_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) not in active_ids:
                self.registry.remove_object(reg_id)

        _t_detect_total = time.perf_counter() - _t_detect_start
        self.last_detect_timings = {
            "tagging_s":      round(_t_tag_s, 4),
            "grounding_segmentation_s": round(_t_gsam2_s, 4),
            "localization_s": round(_t_localize_s, 4),
            "total_s":        round(_t_detect_total, 4),
        }
        self.logger.info(
            "Detection timings — tagging=%.2fs  grounding+seg=%.2fs  "
            "localization=%.2fs  total=%.2fs  objects=%d",
            _t_tag_s, _t_gsam2_s, _t_localize_s, _t_detect_total, len(detected),
        )
        self.logger.debug("Detected %d objects: %s", len(detected), [o.object_id for o in detected])
        return detected

    # ------------------------------------------------------------------
    # LLM affordance inference
    # ------------------------------------------------------------------

    async def _infer_affordances(self, object_type: str, crop: np.ndarray) -> Set[str]:
        if object_type in self._affordance_cache:
            return self._affordance_cache[object_type]
        try:
            from ptp.llm_interface.base import GenerateConfig, ImagePart
            buf = io.BytesIO()
            Image.fromarray(crop).save(buf, format="PNG")
            response = await self._llm_client.generate_async(
                [
                    (
                        f"List affordances for a '{object_type}' in a robot manipulation context. "
                        "Common: graspable, placeable_on, openable, closeable, pushable, containable, movable, fixed. "
                        'Return JSON: {"affordances": ["label1", ...]}'
                    ),
                    ImagePart(data=buf.getvalue(), mime_type="image/png"),
                ],
                config=GenerateConfig(temperature=0.2, max_output_tokens=256,
                                      response_mime_type="application/json"),
            )
            affordances = set(json.loads(response.text).get("affordances", ["graspable"])) or {"graspable"}
            self._affordance_cache[object_type] = affordances
            return affordances
        except Exception as exc:
            self.logger.warning("Affordance inference failed for '%s': %s", object_type, exc)
            return {"graspable"}

    # ------------------------------------------------------------------
    # Geometry recompute (called after manipulation)
    # ------------------------------------------------------------------

    def recompute_geometry(
        self,
        obj_masks: Dict[str, np.ndarray],
        depth_frame: np.ndarray,
        camera_intrinsics: Any,
        robot_state: Optional[Dict[str, Any]] = None,
        affected_ids: Optional[List[str]] = None,
        force_occlusion: bool = False,
        color_frame: Optional[np.ndarray] = None,
    ) -> None:
        objects = self.registry.get_all_objects()
        if not objects:
            return
        cam_position, cam_rotation = self._get_camera_transform(robot_state)
        obj_types = {o.object_id: o.object_type for o in objects}

        if self._compute_clearances:
            for obj in objects:
                if affected_ids is not None and obj.object_id not in affected_ids:
                    continue
                mask = obj_masks.get(obj.object_id)
                if mask is None:
                    continue
                try:
                    obj.clearance_profile = compute_clearance_profile(
                        target_mask=mask,
                        depth_frame=depth_frame,
                        camera_intrinsics=camera_intrinsics,
                        all_masks={oid: m for oid, m in obj_masks.items() if oid != obj.object_id},
                        gripper=self._gripper,
                        cam_position=cam_position,
                        camera_quaternion_xyzw=cam_rotation.as_quat() if cam_rotation is not None else None,
                        object_types=obj_types,
                    )
                    self.registry.update_object(obj.object_id, obj)
                except Exception as exc:
                    self.logger.warning("Clearance recompute failed for '%s': %s", obj.object_id, exc)

        if self._compute_contacts and len(objects) >= 2:
            try:
                self.registry.contact_graph = compute_contact_graph(
                    objects=objects, obj_masks=obj_masks, depth_frame=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    contact_threshold_m=self._contact_threshold_m,
                    camera_position=cam_position, camera_rotation=cam_rotation,
                    color_image=color_frame, llm_client=self._llm_client,
                    llm_mode=self._llm_mode, llm_debug_image_out=self.llm_debug_images,
                )
            except Exception as exc:
                self.logger.warning("Contact graph recompute failed: %s", exc)

        if self._compute_occlusion and force_occlusion:
            self._obs_history.append(ObservationRecord(
                depth_frame=depth_frame, camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose.from_robot_state(robot_state),
                obj_masks=dict(obj_masks),
            ))
            try:
                self.registry.occlusion_map = compute_occlusion_map(
                    observations=list(self._obs_history),
                    object_ids=[o.object_id for o in objects],
                )
            except Exception as exc:
                self.logger.warning("Occlusion map recompute failed: %s", exc)


# ---------------------------------------------------------------------------
# GSAM2ContinuousObjectTracker
# ---------------------------------------------------------------------------

class GSAM2ContinuousObjectTracker:
    """Background continuous tracker wrapping GSAM2ObjectTracker.

    Call set_frame_provider() then start() to begin the async tracking loop.
    The registry is populated continuously and safe to read from any thread.
    """

    def __init__(
        self,
        grounding_model_id: str = "IDEA-Research/grounding-dino-base",
        sam2_model_cfg: str = _SAM2_CFG,
        sam2_ckpt_path: str = _SAM2_CKPT,
        detection_interval: int = 20,
        score_threshold: float = 0.5,
        overlap_iou_threshold: float = 0.5,
        device: str = "cuda",
        tag_interval: int = 1,
        update_interval: float = 0.0,
        on_detection_complete: Optional[Callable[[int], None]] = None,
        llm_client: Optional[Any] = None,
        llm_mode: str = "nl",
        robot_interface: Optional[Any] = None,
        compute_clearances: bool = True,
        gripper: Optional[GripperGeometry] = None,
        compute_contacts: bool = True,
        contact_threshold_m: float = 0.005,
        compute_occlusion: bool = True,
        occlusion_history_len: int = 10,
        occlusion_update_interval: int = 1,
        t1_budget_s: float = 2.0,
        debug_save_dir: Optional[Union[str, Path]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._tracker = GSAM2ObjectTracker(
            grounding_model_id=grounding_model_id,
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            detection_interval=detection_interval,
            score_threshold=score_threshold,
            overlap_iou_threshold=overlap_iou_threshold,
            device=device,
            tag_interval=tag_interval,
            llm_client=llm_client,
            llm_mode=llm_mode,
            robot_interface=robot_interface,
            compute_clearances=compute_clearances,
            gripper=gripper,
            compute_contacts=compute_contacts,
            contact_threshold_m=contact_threshold_m,
            compute_occlusion=compute_occlusion,
            occlusion_history_len=occlusion_history_len,
            occlusion_update_interval=occlusion_update_interval,
            logger=logger,
        )
        self.registry = self._tracker.registry
        self.logger = logger or get_structured_logger("GSAM2ContinuousObjectTracker")
        self.stats = TrackingStats()
        self.update_interval = update_interval
        self.on_detection_complete = on_detection_complete
        self._t1_budget_s = t1_budget_s
        self._t1_throttled = False

        self._frame_provider: Optional[Callable] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_color_frame: Optional[np.ndarray] = None

        self._last_bundle_lock = threading.Lock()
        self._last_bundle: Optional[Dict[str, Any]] = None

        self._debug_save_dir: Optional[Path] = Path(debug_save_dir) if debug_save_dir else None
        self._debug_frame_index: int = 0
        self._debug_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties / delegation
    # ------------------------------------------------------------------

    @property
    def last_color_frame(self) -> Optional[np.ndarray]:
        return self._last_color_frame

    @property
    def _last_masks(self) -> Dict[str, np.ndarray]:
        return self._tracker._last_masks

    @property
    def llm_debug_images(self) -> List[bytes]:
        return self._tracker.llm_debug_images

    def set_tagger(self, tagger_callable: Any) -> None:
        self._tracker.set_tagger(tagger_callable)

    def set_task_context(
        self,
        task_description: Optional[str] = None,
        goal_objects: Optional[List[str]] = None,
        **_kwargs: Any,
    ) -> None:
        hints: List[str] = list(goal_objects or [])
        if task_description:
            hints.extend(_extract_noun_phrases(task_description))
        if hints:
            self._tracker.set_extra_tags(hints)
            self.logger.info("Task hints injected into GroundingDINO: %s", hints)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_frame_provider(self, provider: Callable[[], tuple]) -> None:
        self._frame_provider = provider

    def start(self) -> None:
        if self._running:
            return
        if self._frame_provider is None:
            raise ValueError("Call set_frame_provider() before start().")
        self._running = True
        self.stats.is_running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._tracking_loop())
        self.logger.info("GSAM2ContinuousObjectTracker started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.stats.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("GSAM2ContinuousObjectTracker stopped")

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------

    async def _tracking_loop(self) -> None:
        while self._running:
            loop_start = time.time()
            try:
                provided = self._frame_provider()
                if isinstance(provided, tuple) and len(provided) == 4:
                    color_frame, depth_frame, intrinsics, robot_state = provided
                else:
                    color_frame, depth_frame, intrinsics = provided
                    robot_state = None

                if isinstance(color_frame, np.ndarray):
                    self._last_color_frame = color_frame

                t_det = time.time()
                detected = await self._tracker.detect_objects(
                    color_frame, depth_frame, intrinsics, robot_state=robot_state
                )
                det_time = time.time() - t_det

                self.stats.total_frames += 1
                self.stats.total_detections += len(detected)
                self.stats.last_detection_time = det_time
                self.stats.avg_detection_time = (
                    0.1 * det_time + 0.9 * self.stats.avg_detection_time
                )

                buf = io.BytesIO()
                frame_img = color_frame if isinstance(color_frame, Image.Image) else Image.fromarray(color_frame)
                frame_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                with self._last_bundle_lock:
                    self._last_bundle = {
                        "timestamp": time.time(),
                        "color_png": png_bytes,
                        "depth": np.array(depth_frame, copy=True) if depth_frame is not None else None,
                        "intrinsics": intrinsics,
                        "objects": list(detected),
                        "robot_state": robot_state,
                    }

                if self.on_detection_complete:
                    if asyncio.iscoroutinefunction(self.on_detection_complete):
                        await self.on_detection_complete(len(detected))
                    else:
                        self.on_detection_complete(len(detected))

                if self._debug_save_dir is not None:
                    idx = self._debug_frame_index
                    self._debug_frame_index += 1
                    threading.Thread(
                        target=save_debug_frame,
                        args=(png_bytes, list(detected), idx, self._debug_save_dir, self._debug_lock),
                        daemon=True,
                    ).start()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Tracking loop error: %s", exc, exc_info=True)

            elapsed = time.time() - loop_start
            if self._t1_budget_s > 0 and elapsed > self._t1_budget_s:
                if not self._t1_throttled:
                    self._t1_throttled = True
                    self.logger.warning(
                        "Detection cycle %.0f ms > budget %.0f ms — throttling to 0.5 Hz",
                        elapsed * 1000, self._t1_budget_s * 1000,
                    )
                target_interval = max(self.update_interval, 2.0)
            else:
                if self._t1_throttled:
                    self._t1_throttled = False
                    self.logger.info("Detection cycle back within budget")
                target_interval = self.update_interval

            if target_interval > elapsed:
                await asyncio.sleep(target_interval - elapsed)

    # ------------------------------------------------------------------
    # Snapshot / geometry
    # ------------------------------------------------------------------

    def get_last_detection_bundle(self) -> Optional[Dict[str, Any]]:
        with self._last_bundle_lock:
            return dict(self._last_bundle) if self._last_bundle is not None else None

    def get_all_objects(self) -> List[DetectedObject]:
        return self.registry.get_all_objects()

    def trigger_geometry_recompute(
        self,
        affected_ids: Optional[List[str]] = None,
        force_occlusion: bool = False,
    ) -> bool:
        with self._last_bundle_lock:
            bundle = self._last_bundle
        if bundle is None:
            return False
        depth = bundle.get("depth")
        intr = bundle.get("intrinsics")
        if depth is None or intr is None:
            return False
        obj_masks: Dict[str, np.ndarray] = {}
        if self._tracker._obs_history:
            obj_masks = self._tracker._obs_history[-1].obj_masks
        self._tracker.recompute_geometry(
            obj_masks=obj_masks,
            depth_frame=depth,
            camera_intrinsics=intr,
            robot_state=bundle.get("robot_state"),
            affected_ids=affected_ids,
            force_occlusion=force_occlusion,
        )
        return True

    def get_component_timings(self) -> Dict[str, Dict[str, float]]:
        t = self._tracker

        def _avg(total: float, key: str) -> float:
            n = t._timing_calls.get(key, 0)
            return total / n if n > 0 else 0.0

        return {
            k: {"total_s": getattr(t, f"_timing_{k}_s"), "calls": t._timing_calls[k],
                "avg_s": _avg(getattr(t, f"_timing_{k}_s"), k)}
            for k in ("gsam2", "clearance", "contact_graph", "occlusion")
        }
