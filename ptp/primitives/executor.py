"""Primitive executor — translates LLM plan parameters before calling the motion backend.

LLM outputs reference image-grounded cues (pixel [y, x] pointers, normals, standoffs).
This executor back-projects those cues into metric coordinates using the latest snapshot
depth and camera intrinsics, validates the plan, and optionally drives the configured
motion planner.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ptp.perception.utils.coordinates import compute_3d_position
from ptp.perception.surface_normal import compute_surface_normal, transform_normal_to_base
from ptp.perception.pointing_prompts import build_prompt, build_hinge_prompt
from ptp.primitives.types import SkillPlan
from ptp.primitives.library import PRIMITIVE_LIBRARY
from ptp.primitives.snapshot_utils import SnapshotCache, load_snapshot_artifacts
from ptp.utils.logging_utils import get_structured_logger, RunTimer
from ptp.grasp.grasp_planner import GraspPlanner


@dataclass
class SnapshotCameraPose:
    position: np.ndarray
    rotation: Rotation


@dataclass
class ExecutedPrimitive:
    """Record of a single executed primitive step."""

    index: int
    name: str
    parameters: Dict[str, Any]
    references: Dict[str, Any]
    success: bool
    result: Any


@dataclass
class PrimitiveExecutionResult:
    """Return payload for executor runs."""

    executed: bool
    primitive_results: List[Any] = field(default_factory=list)
    executed_primitives: List[ExecutedPrimitive] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class PrimitiveExecutor:
    """Translate and execute primitive plans against the configured primitives interface."""

    def __init__(
        self,
        primitives: Optional[Any],
        perception_pool_dir: Path,
        logger: Optional[logging.Logger] = None,
        orchestrator: Optional[Any] = None,
        molmo: Optional[Any] = None,
        masks: Optional[Dict[str, Any]] = None,
    ):
        self.primitives = primitives
        self.perception_pool_dir = Path(perception_pool_dir)
        self._snapshot_cache = SnapshotCache()
        self.logger = logger or get_structured_logger("PrimitiveExecutor")
        self.orchestrator = orchestrator
        self._molmo = molmo
        self._masks: Dict[str, Any] = masks or {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
        dry_run: bool = False,
    ) -> PrimitiveExecutionResult:
        """Translate (and optionally execute) a primitive plan."""
        self.logger.info("Executing plan: %s", plan)
        timer = RunTimer()

        with timer.measure("prepare_plan"):
            translated_plan = self.prepare_plan(plan, world_state)
        # propagate per-primitive grounding timings recorded during prepare_plan
        for k, v in getattr(translated_plan, "_grounding_timings", {}).items():
            timer.record(k, v)

        if dry_run:
            self.logger.info("Dry run requested; execution skipped.")
            timer.log_summary(self.logger)
            return PrimitiveExecutionResult(executed=False, timings=timer.to_dict())
        if self.primitives is None:
            raise RuntimeError("Primitives interface is required for execution (dry_run=False).")

        primitive_results: List[Any] = []
        executed_primitives: List[ExecutedPrimitive] = []
        for idx, primitive in enumerate(translated_plan.primitives):
            self.logger.info(
                "Executing primitive [%d/%d]: %s with parameters %s",
                idx + 1,
                len(translated_plan.primitives),
                primitive.name,
                primitive.parameters,
            )
            schema = PRIMITIVE_LIBRARY.get(primitive.name)
            if schema is not None and "execute" in schema.optional_params:
                primitive.parameters.setdefault("execute", True)
            method = getattr(self.primitives, primitive.name, None)
            if not callable(method):
                raise AttributeError(f"Primitives interface missing primitive '{primitive.name}'")
            # Inject references.object_id as a kwarg so move_gripper_to_pose can build
            # ignore_labels and mask the target object out of collision checking.
            call_params = dict(primitive.parameters)
            # Translate discrete turn_amount labels to rotation_angle_deg for twist.
            if primitive.name == "twist" and "turn_amount" in call_params:
                _TURN_DEGREES = {
                    "quarter_turn": 90.0,
                    "half_turn": 180.0,
                    "three_quarter_turn": 270.0,
                    "full_turn": 360.0,
                }
                turn_label = call_params.pop("turn_amount")
                call_params["rotation_angle_deg"] = _TURN_DEGREES.get(turn_label, 360.0)
            ref_obj_id = primitive.references.get("object_id")
            if ref_obj_id:
                call_params.setdefault("object_id", ref_obj_id)
            # Pass metadata through for primitives that consume it (e.g. push_pull
            # uses surface_normal_base to determine push direction).
            if primitive.metadata:
                call_params.setdefault("metadata", dict(primitive.metadata))
            self.logger.debug("Calling '%s' with parameters: %s", primitive.name, call_params)
            success = True
            try:
                with timer.measure(f"execute.{primitive.name}[{idx}]"):
                    raw_result = method(**call_params)
                if isinstance(raw_result, dict) and raw_result.get("success") is False:
                    success = False
                    self.logger.error(
                        "Primitive '%s' returned failure: %s",
                        primitive.name, raw_result.get("reason", raw_result),
                    )
            except Exception as exc:
                success = False
                raw_result = str(exc)
                self.logger.error("Primitive '%s' raised: %s", primitive.name, exc)
            result = self._json_safe(raw_result)
            primitive_results.append(result)

            record = ExecutedPrimitive(
                index=idx,
                name=primitive.name,
                parameters=dict(primitive.parameters),
                references=dict(primitive.references),
                success=success,
                result=result,
            )
            executed_primitives.append(record)

            summary_lines = [
                f"  [{i}] {'OK' if r.success else 'FAIL'} {r.name}"
                + (f" → obj={r.references.get('object_id')}" if r.references.get("object_id") else "")
                for i, r in enumerate(executed_primitives)
            ]
            self.logger.info(
                "Executed primitives so far (%d/%d):\n%s",
                len(executed_primitives),
                len(translated_plan.primitives),
                "\n".join(summary_lines),
            )

            if success:
                time.sleep(0.5)
            else:
                break

        timer.log_summary(self.logger)
        return PrimitiveExecutionResult(
            executed=True,
            primitive_results=primitive_results,
            executed_primitives=executed_primitives,
            timings=timer.to_dict(),
        )

    def prepare_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
    ) -> SkillPlan:
        """Translate parameters and validate the plan without executing it."""
        self.logger.info("[prepare_plan] Translating %d primitives", len(plan.primitives))

        artifacts = load_snapshot_artifacts(
            world_state,
            self.perception_pool_dir,
            cache=self._snapshot_cache,
            snapshot_id=getattr(plan, "source_snapshot_id", None),
        )
        self.logger.debug("[prepare_plan] Snapshot artifacts loaded (snapshot_id=%s)", artifacts.snapshot_id)

        if getattr(plan, "source_snapshot_id", None) and plan.source_snapshot_id != artifacts.snapshot_id:
            self.logger.warning(
                "Plan snapshot %s missing; using %s",
                plan.source_snapshot_id,
                artifacts.snapshot_id or "latest",
            )

        cam_pose: Optional[SnapshotCameraPose] = None
        joints = (artifacts.robot_state or {}).get("joints")
        if self.primitives:
            helper = getattr(self.primitives, "camera_pose_from_joints", None)
            if helper:
                pos, rot = helper(joints)
                cam_pose = SnapshotCameraPose(position=np.asarray(pos, dtype=float), rotation=rot)
            else:
                self.logger.info("[prepare_plan] Primitives interface missing camera_pose_from_joints; skipping transform")
        else:
            self.logger.info("[prepare_plan] No primitives interface; skipping base-frame transform")

        # Seed preset_orientation from the LLM's approach_direction on every
        # move_gripper_to_pose before any grounding pass runs.  This ensures the
        # grasp planner always has the right seed even when pointing_guidance
        # grounding fails or no point cloud is available for antipodal sampling.
        for primitive in plan.primitives:
            if primitive.name == "move_gripper_to_pose":
                approach_dir = primitive.metadata.get("approach_direction", "top_down")
                preset = "side" if approach_dir == "side" else "top_down"
                primitive.parameters.setdefault("preset_orientation", preset)

        # Ground any deferred pointing_guidance via Molmo before coordinate translation.
        _grounding_timings: Dict[str, float] = {}
        for idx, primitive in enumerate(plan.primitives):
            if primitive.name == "move_gripper_to_pose" and primitive.metadata.get("pointing_guidance"):
                _t0 = time.perf_counter()
                self._resolve_pointing_guidance(primitive, artifacts, plan, idx, world_state)
                _grounding_timings[f"molmo_grounding.move_gripper_to_pose[{idx}]"] = round(
                    time.perf_counter() - _t0, 4
                )

        # For push/pull: run Molmo on the surface_label to get an interaction
        # point, then compute the surface normal at that point from depth.
        for idx, primitive in enumerate(plan.primitives):
            if primitive.name in ("push", "pull"):
                _t0 = time.perf_counter()
                self._resolve_push_pull_surface(primitive, artifacts, cam_pose, plan, idx, world_state)
                _grounding_timings[f"molmo_grounding.{primitive.name}[{idx}]"] = round(
                    time.perf_counter() - _t0, 4
                )

        plan._grounding_timings = _grounding_timings  # type: ignore[attr-defined]

        # Translate each primitive (pixel → 3D, camera → base frame).
        for idx, primitive in enumerate(plan.primitives):
            self.logger.debug(
                "[prepare_plan] [%d/%d] %s params=%s refs=%s",
                idx + 1,
                len(plan.primitives),
                primitive.name,
                primitive.parameters,
                primitive.references,
            )

            pixel = primitive.parameters.pop("target_pixel_yx", None)
            if pixel is not None:
                if artifacts.depth is None or artifacts.intrinsics is None:
                    self.logger.warning(
                        "%s: cannot back-project pixel %s (missing depth/intrinsics)",
                        primitive.name, pixel,
                    )
                else:
                    point = compute_3d_position(
                        [float(pixel[0]), float(pixel[1])], artifacts.depth, artifacts.intrinsics
                    )
                    if point is None:
                        self.logger.warning(
                            "%s: back-projection returned no point for %s", primitive.name, pixel
                        )
                    else:
                        depth_offset = float(primitive.parameters.get("depth_offset_m", 0.0) or 0.0)
                        if depth_offset:
                            point = [point[0], point[1], point[2] + depth_offset]
                        primitive.parameters["target_position"] = point
                        primitive.parameters.pop("depth_offset_m", None)

            # If no target_position, inject point_label from references for registry lookup.
            if "target_position" not in primitive.parameters:
                obj_id = primitive.references.get("object_id")
                if obj_id and "point_label" not in primitive.parameters:
                    primitive.parameters["point_label"] = obj_id

            # Transform camera-frame coordinates to base frame.
            # Skip primitives already in base frame (Molmo-grounded positions are world-frame).
            already_in_base = primitive.frame == "base" or primitive.metadata.get("molmo_grounded")
            if cam_pose and not already_in_base:
                for key in ("target_position", "pivot_point"):
                    if key not in primitive.parameters:
                        continue
                    pos = primitive.parameters[key]
                    base_pos = cam_pose.rotation.apply(pos) + cam_pose.position
                    primitive.parameters[key] = [float(v) for v in base_pos]

        # Antipodal grasp sampling — refine target_position and set target_orientation
        # for move_gripper_to_pose grasps (not places).
        _planner_iface = None
        if self.primitives is not None:
            _planner_iface = (
                getattr(self.primitives, "_planner", None)
                or self.primitives
            )

        for idx, primitive in enumerate(plan.primitives):
            if primitive.name != "move_gripper_to_pose":
                continue
            if primitive.parameters.get("is_place"):
                continue
            target_pos = primitive.parameters.get("target_position")
            if target_pos is None:
                continue

            contact_point = np.asarray(target_pos, dtype=float)
            preset = primitive.parameters.get("preset_orientation", "top_down")
            ref_id = primitive.references.get("object_id")
            obj_pts = self._get_object_point_cloud(ref_id, artifacts)

            if obj_pts is not None and len(obj_pts) >= 10:
                planner = GraspPlanner(_planner_iface)
                ignore_labels = {ref_id} if ref_id else None
                candidate = planner.plan(
                    contact_position=contact_point,
                    object_points=obj_pts,
                    seed_orientation=preset if preset in ("top_down", "side") else "top_down",
                    ignore_labels=ignore_labels,
                )
                if candidate is not None:
                    primitive.parameters["target_position"] = candidate.position.tolist()
                    primitive.parameters["target_orientation"] = candidate.orientation.tolist()
                    primitive.parameters.pop("preset_orientation", None)
                    primitive.metadata["antipodal_grounded"] = True
                    self.logger.info(
                        "[prepare_plan] [%d] antipodal grasp: pos=%s quat=%s (ref=%s, %d pts)",
                        idx,
                        [f"{v:.3f}" for v in candidate.position],
                        [f"{v:.3f}" for v in candidate.orientation],
                        ref_id,
                        len(obj_pts),
                    )
                else:
                    self.logger.warning(
                        "[prepare_plan] [%d] antipodal grasp failed for %r; keeping Molmo-grounded pose",
                        idx, ref_id,
                    )
            else:
                self.logger.warning(
                    "[prepare_plan] [%d] antipodal grasp skipped — no point cloud for %r (%s pts)",
                    idx, ref_id, len(obj_pts) if obj_pts is not None else 0,
                )

        # Strip null-valued parameters and metadata fields (strict-schema output uses null
        # as a placeholder for unused optional fields).
        for primitive in plan.primitives:
            for k in [k for k, v in list(primitive.parameters.items()) if v is None]:
                primitive.parameters.pop(k)
            for k in [k for k, v in list(primitive.metadata.items()) if v is None]:
                primitive.metadata.pop(k)

        # Strip unknown parameters (LLM cross-contamination fallback).
        for idx, primitive in enumerate(plan.primitives):
            schema = PRIMITIVE_LIBRARY.get(primitive.name)
            if schema is None:
                continue
            allowed = set(schema.required_params) | set(schema.optional_params)
            unknown = [k for k in list(primitive.parameters) if k not in allowed]
            for k in unknown:
                self.logger.warning("[prepare_plan] [%d] stripping unknown param '%s' from %s", idx, k, primitive.name)
                primitive.parameters.pop(k)
                plan.diagnostics.warnings.append(f"[{idx}] stripped unknown parameter '{k}' from {primitive.name}")

        validation_errors = plan.validate(PRIMITIVE_LIBRARY)
        if validation_errors:
            raise ValueError(f"Plan validation failed: {validation_errors}")
        self.logger.info("[prepare_plan] Validation passed")

        return plan

    # ------------------------------------------------------------------ #
    # Point cloud extraction
    # ------------------------------------------------------------------ #
    def _get_object_point_cloud(
        self,
        object_id: Optional[str],
        artifacts: Any,
    ) -> Optional[np.ndarray]:
        """Return world-frame (N, 3) point cloud for object_id using its GSAM2 mask + depth."""
        if object_id is None or artifacts.depth is None or artifacts.intrinsics is None:
            return None

        masks: Dict[str, Any] = dict(self._masks)
        if not masks and self.orchestrator is not None:
            tracker = getattr(self.orchestrator, "tracker", None)
            raw = getattr(tracker, "_last_masks", None) or {}
            if not raw:
                inner = getattr(tracker, "_tracker", None)
                raw = getattr(inner, "_last_masks", None) or {}
            masks = dict(raw)

        mask = masks.get(object_id)
        if mask is None or not np.any(mask):
            return None

        depth = artifacts.depth
        intr = artifacts.intrinsics
        h, w = depth.shape
        if mask.shape != (h, w):
            return None

        fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
        stride = 2
        rows = np.arange(0, h, stride)
        cols = np.arange(0, w, stride)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        sampled_mask = mask[rr, cc].astype(bool)
        d = depth[rr, cc].astype(float)
        valid = sampled_mask & (d > 0.05) & (d < 3.0)
        if not np.any(valid):
            return None

        d_v = d[valid]
        x = (cc[valid].astype(float) - cx) * d_v / fx
        y = (rr[valid].astype(float) - cy) * d_v / fy
        pts_cam = np.stack([x, y, d_v], axis=1)

        cam_pos: Optional[np.ndarray] = None
        cam_rot: Optional[Rotation] = None

        # Try primitives interface first (syncs from real robot if joints not provided).
        if self.primitives is not None:
            helper = getattr(self.primitives, "camera_pose_from_joints", None)
            if helper is None:
                helper = getattr(self.primitives, "get_camera_transform", None)
            if helper is not None:
                try:
                    joints = (artifacts.robot_state or {}).get("joints")
                    result = helper(joints) if joints is not None else helper(None)
                    if result and result[0] is not None:
                        cam_pos, cam_rot = result
                except Exception:
                    pass

        if cam_pos is None or cam_rot is None and self.orchestrator is not None:
            robot = getattr(getattr(self.orchestrator, "config", None), "robot", None)
            if robot is not None:
                try:
                    cam_pos, cam_rot = robot.get_camera_transform()
                except Exception:
                    pass

        if cam_pos is None or cam_rot is None:
            self.logger.warning(
                "_get_object_point_cloud: camera transform unavailable for %r — "
                "returning camera-frame cloud (base-frame grasp planning will be wrong)",
                object_id,
            )
            return pts_cam.astype(np.float32)

        pts_world = cam_rot.apply(pts_cam) + cam_pos
        return pts_world.astype(np.float32)

    # ------------------------------------------------------------------ #
    # push_pull surface normal grounding
    # ------------------------------------------------------------------ #
    def _resolve_push_pull_surface(
        self,
        primitive: Any,
        artifacts: Any,
        cam_pose: Optional[Any],
        plan: Any,
        step_idx: int,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ground push_pull surface_label via Molmo, then compute the surface normal.

        Pipeline:
        1. Ask Molmo to point at the surface named by surface_label.
        2. Back-project the returned 2-D point to find the pixel location.
        3. Compute the surface normal from a depth patch centred on that pixel
           (PCA plane fit over a 40-px radius).
        4. Rotate the camera-frame normal into the robot base frame using the
           snapshot camera pose.
        5. Store the result in primitive.metadata["surface_normal_base"] so the
           primitives interface can use it as the push/pull direction.

        Falls back gracefully (logs a warning, leaves normal unset) on any error.
        """
        import io
        from PIL import Image as _PIL

        surface_label: str = primitive.parameters.get("surface_label", "")
        has_pivot: bool = bool(primitive.parameters.get("has_pivot", False))
        action_goal: Optional[str] = primitive.parameters.get("action_goal") or None
        ref_id = primitive.references.get("object_id") or surface_label

        # ---- 1. locate Molmo detector ----
        detector = self._molmo
        if detector is None and self.orchestrator is not None:
            tracker = getattr(self.orchestrator, "tracker", None)
            detector = getattr(tracker, "_molmo", None)
            if detector is None:
                inner = getattr(tracker, "_tracker", None)
                detector = getattr(inner, "_molmo", None)

        if detector is None or not artifacts.color_bytes:
            self.logger.info(
                "[prepare_plan] [%d] push_pull: no Molmo detector or image — "
                "skipping surface normal / hinge grounding for %r",
                step_idx, surface_label,
            )
            return

        try:
            rgb = np.array(_PIL.open(io.BytesIO(artifacts.color_bytes)).convert("RGB"))
            depth = artifacts.depth
            intr = artifacts.intrinsics
            if depth is None or intr is None:
                raise ValueError("depth or intrinsics unavailable")

            h, w = rgb.shape[:2]

            # ---- 2a. Molmo: point at the push/pull surface contact point ----
            surface_prompt = build_prompt("push_pull", surface_label, action_goal=action_goal)
            primitive.metadata["surface_prompt"] = surface_prompt

            object_mask = self._masks.get(ref_id) if ref_id else None

            # Look up the target object's bounding box from detections so Molmo
            # receives a crop centred on the object, preventing it from pointing
            # at the wrong hinged object in a scene with multiple articulated items.
            bbox_2d = None
            if ref_id and world_state:
                det_map = {
                    d.get("object_id"): d
                    for d in (world_state.get("latest_detections") or [])
                }
                det = det_map.get(ref_id)
                if det:
                    bbox_2d = det.get("bounding_box_2d")

            surface_results = detector.get_interaction_points(
                rgb_image=rgb,
                depth_frame=depth,
                camera_intrinsics=intr,
                object_id=ref_id,
                object_type=surface_label,
                bounding_box_2d=bbox_2d,
                actions={"_surface"},
                robot_state=artifacts.robot_state,
                custom_prompts={"_surface": surface_prompt},
                object_mask=object_mask,
                all_masks=self._masks,
            )

            ip_surface = surface_results.get("_surface")
            if ip_surface is None:
                raise ValueError("Molmo returned no surface point")

            if ip_surface.input_image_bytes is not None:
                primitive.metadata["molmo_input_image_bytes"] = ip_surface.input_image_bytes

            # ip.position_2d is normalised [y, x] in 0-1000; convert to pixels.
            pos2d = ip_surface.position_2d  # [norm_y, norm_x]
            px_row = float(pos2d[0]) / 1000.0 * h
            px_col = float(pos2d[1]) / 1000.0 * w

            self.logger.info(
                "[prepare_plan] [%d] push_pull: Molmo surface point at pixel (row=%.1f, col=%.1f) "
                "for %r",
                step_idx, px_row, px_col, surface_label,
            )

            # ---- 3. Compute surface normal in camera frame ----
            normal_cam, confidence = compute_surface_normal(
                depth=depth,
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
                center_yx=(px_row, px_col),
                radius_px=40.0,
                method="pca",
            )

            if normal_cam is None:
                raise ValueError("surface normal estimation failed (too few depth points)")

            self.logger.info(
                "[prepare_plan] [%d] push_pull: camera-frame normal=%s confidence=%.2f",
                step_idx,
                [f"{v:.3f}" for v in normal_cam],
                confidence,
            )

            # ---- 4. Rotate normal into base frame ----
            if cam_pose is not None:
                normal_base = transform_normal_to_base(normal_cam, cam_pose.rotation)
            else:
                normal_base = normal_cam
                self.logger.warning(
                    "[prepare_plan] [%d] push_pull: no camera pose — normal stays in camera frame",
                    step_idx,
                )

            self.logger.info(
                "[prepare_plan] [%d] push_pull: base-frame normal=%s for %r",
                step_idx,
                [f"{v:.3f}" for v in normal_base],
                surface_label,
            )

            # ---- 5. Store surface normal result ----
            primitive.metadata["surface_normal_base"] = normal_base.tolist()
            primitive.metadata["surface_normal_confidence"] = float(confidence)
            primitive.metadata["surface_pixel_yx"] = [px_row, px_col]
            primitive.metadata["surface_point_2d"] = list(ip_surface.position_2d)

            # ---- 6. Hinge grounding (pivot_pull only) ----
            # The hinge prompt is built *after* the surface contact pixel is
            # known so Molmo gets explicit gripper-position context: "the gripper
            # will grip here — where is the hinge relative to that?"
            if has_pivot:
                hinge_axis: Optional[str] = primitive.parameters.get("hinge_axis") or None
                # Find the grip pixel: use the most recent move_gripper_to_pose before
                # this step that has a grounded position_2d (the handle grasp point).
                # Fall back to the surface contact pixel if none is found.
                grip_pixel_yx: Tuple[float, float] = (px_row, px_col)
                for prev in reversed(plan.primitives[:step_idx]):
                    if prev.name == "move_gripper_to_pose":
                        prev_pos2d = prev.metadata.get("position_2d")
                        if prev_pos2d and len(prev_pos2d) >= 2:
                            grip_pixel_yx = (
                                float(prev_pos2d[0]) / 1000.0 * h,
                                float(prev_pos2d[1]) / 1000.0 * w,
                            )
                        break

                hinge_prompt = build_hinge_prompt(
                    surface_label,
                    action_goal=action_goal,
                    gripper_pixel_yx=grip_pixel_yx,
                )
                primitive.metadata["hinge_prompt"] = hinge_prompt
                hinge_results = detector.get_interaction_points(
                    rgb_image=rgb,
                    depth_frame=depth,
                    camera_intrinsics=intr,
                    object_id=ref_id,
                    object_type=surface_label,
                    bounding_box_2d=bbox_2d,
                    actions={"_hinge"},
                    robot_state=artifacts.robot_state,
                    custom_prompts={"_hinge": hinge_prompt},
                    object_mask=object_mask,
                    all_masks=self._masks,
                    mark_pixel_yx=grip_pixel_yx,
                )
                ip_hinge = hinge_results.get("_hinge")
                if ip_hinge is not None and ip_hinge.input_image_bytes is not None:
                    primitive.metadata["molmo_hinge_input_image_bytes"] = ip_hinge.input_image_bytes
                if ip_hinge is None:
                    self.logger.warning(
                        "[prepare_plan] [%d] push_pull pivot: Molmo returned no hinge point — "
                        "pivot_pull will estimate pivot from TCP offset",
                        step_idx,
                    )
                else:
                    hinge_pos2d = list(ip_hinge.position_2d)  # [norm_y, norm_x] in 0-1000

                    # Snap the hinge pixel to align with the grip pixel based on
                    # hinge_axis from the LLM plan:
                    #   vertical   → side hinge, rotates around vertical axis →
                    #                hinge is left/right of grip, same height →
                    #                snap hinge row to grip row
                    #   horizontal → top/bottom hinge, rotates around horizontal axis →
                    #                hinge is above/below grip, same column →
                    #                snap hinge col to grip col
                    grip_norm_y = grip_pixel_yx[0] / h * 1000.0
                    grip_norm_x = grip_pixel_yx[1] / w * 1000.0
                    if hinge_axis == "vertical":
                        hinge_pos2d[0] = grip_norm_y
                        self.logger.info(
                            "[prepare_plan] [%d] push_pull pivot: snapped hinge row to grip row "
                            "(vertical hinge axis)",
                            step_idx,
                        )
                    elif hinge_axis == "horizontal":
                        hinge_pos2d[1] = grip_norm_x
                        self.logger.info(
                            "[prepare_plan] [%d] push_pull pivot: snapped hinge col to grip col "
                            "(horizontal hinge axis)",
                            step_idx,
                        )

                    hinge_row = float(hinge_pos2d[0]) / 1000.0 * h
                    hinge_col = float(hinge_pos2d[1]) / 1000.0 * w

                    self.logger.info(
                        "[prepare_plan] [%d] push_pull pivot: Molmo hinge at pixel "
                        "(row=%.1f, col=%.1f)",
                        step_idx, hinge_row, hinge_col,
                    )

                    # Back-project hinge pixel to camera-frame 3D.
                    # compute_3d_position expects 0-1000 normalised coords, not pixels.
                    hinge_3d = compute_3d_position(
                        list(hinge_pos2d), depth, intr
                    )
                    if hinge_3d is None:
                        self.logger.warning(
                            "[prepare_plan] [%d] push_pull pivot: hinge back-projection failed "
                            "(no depth at pixel) — pivot_pull will use TCP offset",
                            step_idx,
                        )
                    else:
                        # Rotate hinge 3D point into base frame.
                        hinge_cam = np.asarray(hinge_3d, dtype=float)
                        if cam_pose is not None:
                            hinge_base = cam_pose.rotation.apply(hinge_cam) + cam_pose.position
                        else:
                            hinge_base = hinge_cam
                            self.logger.warning(
                                "[prepare_plan] [%d] push_pull pivot: no camera pose — "
                                "hinge point stays in camera frame",
                                step_idx,
                            )

                        # Compute pivot radius: XY-plane distance from hinge to surface contact.
                        # This is the lever arm the pivot_pull arc is computed from.
                        # ip_surface.position_3d is already in base frame (transformed by
                        # molmo_point_detector._build_interaction_point via robot_state["camera"]).
                        surface_3d = ip_surface.position_3d
                        surface_base = np.asarray(surface_3d, dtype=float) if surface_3d is not None else None

                        pivot_radius_m: Optional[float] = None
                        if surface_base is not None:
                            delta_xy = surface_base[:2] - hinge_base[:2]
                            pivot_radius_m = float(np.linalg.norm(delta_xy))

                        self.logger.info(
                            "[prepare_plan] [%d] push_pull pivot: hinge_base=%s radius=%.3fm",
                            step_idx,
                            [f"{v:.3f}" for v in hinge_base],
                            pivot_radius_m if pivot_radius_m is not None else float("nan"),
                        )

                        primitive.metadata["pivot_point_base"] = hinge_base.tolist()
                        primitive.metadata["hinge_position_2d"] = list(hinge_pos2d)
                        if pivot_radius_m is not None:
                            primitive.metadata["pivot_radius_m"] = pivot_radius_m

        except Exception as exc:
            self.logger.warning(
                "[prepare_plan] [%d] push_pull surface normal / hinge grounding failed (%s) — "
                "primitives interface will use fallback direction",
                step_idx, exc,
            )
            plan.diagnostics.warnings.append(
                f"[{step_idx}] push_pull grounding failed ({exc})"
            )

    # ------------------------------------------------------------------ #
    # Molmo pointing-guidance grounding
    # ------------------------------------------------------------------ #
    def _best_viable_orientation(
        self,
        preference: str,
        clearance_profile: Optional[Any],
    ) -> Tuple[str, Optional[List[float]]]:
        """Return (preset_orientation, direction) that best matches the LLM preference
        while being clear of obstacles according to clearance_profile."""
        if clearance_profile is None:
            return preference, None

        viable = [c for c in clearance_profile.approach_corridors if c.grasp_compatible]
        if not viable:
            return preference, None

        def _z_alignment(corridor: Any) -> float:
            return float(abs(np.asarray(corridor.direction, dtype=float)[2]))

        top_down_pref = preference == "top_down"
        preferred = []
        fallback = []
        for c in viable:
            z = _z_alignment(c)
            if top_down_pref:
                (preferred if z > 0.5 else fallback).append((z, c))
            else:
                (preferred if z <= 0.5 else fallback).append((z, c))

        if preferred:
            preferred.sort(key=lambda t: (-t[0] if top_down_pref else t[0], -t[1].min_clearance))
            chosen = preferred[0][1]
        else:
            fallback.sort(key=lambda t: -t[1].min_clearance)
            chosen = fallback[0][1]

        d = np.asarray(chosen.direction, dtype=float)
        orientation = "top_down" if abs(d[2]) > 0.5 else "side"
        return orientation, d.tolist()

    def _resolve_pointing_guidance(
        self,
        primitive: Any,
        artifacts: Any,
        plan: Any,
        step_idx: int,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ground a move_gripper_to_pose primitive using pointing_guidance from metadata.

        Queries MolmoPointDetector with the guidance text using the freshest snapshot,
        then writes target_position and preset_orientation back onto the primitive.
        Falls back to point_label (registry lookup) on any error.
        """
        import io
        from PIL import Image as _PIL

        _guidance_raw: str = primitive.metadata.get("pointing_guidance", "").strip()
        # Molmo is trained on "Point to X" instructions; prepend if not already present.
        _lower = _guidance_raw.lower()
        if _guidance_raw and not (_lower.startswith("point to") or _lower.startswith("point at")):
            guidance = "Point to " + _guidance_raw
        else:
            guidance = _guidance_raw
        approach_dir: str = primitive.metadata.get("approach_direction", "from_clearance")
        ref_id = primitive.references.get("object_id")

        detector = self._molmo
        if detector is None and self.orchestrator is not None:
            tracker = getattr(self.orchestrator, "tracker", None)
            detector = getattr(tracker, "_molmo", None)
            if detector is None:
                inner = getattr(tracker, "_tracker", None)
                detector = getattr(inner, "_molmo", None)

        if detector is None:
            self.logger.warning(
                "[prepare_plan] [%d] MolmoPointDetector unavailable; keeping point_label fallback",
                step_idx,
            )
            return

        try:
            if not artifacts.color_bytes:
                raise ValueError("no RGB snapshot available")

            rgb = np.array(_PIL.open(io.BytesIO(artifacts.color_bytes)).convert("RGB"))
            depth = artifacts.depth
            intrinsics = artifacts.intrinsics
            robot_state = artifacts.robot_state

            bbox = None
            obj_type = "object"
            clearance_profile = None

            if self.orchestrator is not None and ref_id is not None:
                live_registry = getattr(
                    getattr(self.orchestrator, "tracker", None), "registry", None
                )
                if live_registry is not None:
                    live_obj = live_registry.get_object(ref_id)
                    if live_obj is not None:
                        bbox = getattr(live_obj, "bounding_box_2d", None) or getattr(live_obj, "latest_bounding_box_2d", None)
                        obj_type = getattr(live_obj, "object_type", "object") or "object"
                        clearance_profile = getattr(live_obj, "clearance_profile", None)

            if bbox is None and ref_id and world_state:
                det_map = {
                    d.get("object_id"): d
                    for d in (world_state.get("latest_detections") or [])
                }
                det = det_map.get(ref_id)
                if det:
                    bbox = det.get("bounding_box_2d")
                    if obj_type == "object":
                        obj_type = det.get("object_type") or "object"

            object_mask = self._masks.get(ref_id) if ref_id else None

            self.logger.info(
                "[prepare_plan] [%d] Running Molmo for pointing_guidance=%r on %s%s",
                step_idx, guidance, ref_id or "__guidance__",
                " (masked)" if object_mask is not None else "",
            )
            results = detector.get_interaction_points(
                rgb_image=rgb,
                depth_frame=depth,
                camera_intrinsics=intrinsics,
                object_id=ref_id or "__guidance__",
                object_type=obj_type,
                bounding_box_2d=bbox,
                actions={"_guided"},
                robot_state=robot_state,
                custom_prompts={"_guided": guidance},
                clearance_profile=clearance_profile,
                object_mask=object_mask,
                all_masks=self._masks,
            )

            ip = results.get("_guided")
            if ip is None:
                raise ValueError("Molmo returned no point for guidance prompt")

            if ip.position_3d is None:
                raise ValueError(
                    "Molmo returned a 2D point but position_3d is None "
                    "(depth back-projection failed or robot_state missing camera key)"
                )

            if ip.position_3d is not None:
                pos_3d = np.asarray(ip.position_3d, dtype=float)

                # Molmo's _transform_cam_to_world requires robot_state["camera"] to be
                # present.  If the snapshot robot_state lacked that key (e.g. dry-run),
                # position_3d is still in camera frame — apply the transform now.
                cam_key_present = bool(
                    (artifacts.robot_state or {}).get("camera")
                )
                if not cam_key_present and self.primitives is not None:
                    helper = getattr(self.primitives, "camera_pose_from_joints", None)
                    if helper is None:
                        helper = getattr(self.primitives, "get_camera_transform", None)
                    if helper is not None:
                        try:
                            snap_joints = (artifacts.robot_state or {}).get("joints")
                            self.logger.info(
                                "[prepare_plan] [%d] cam→base transform: "
                                "joints_rad=%s  joints_deg=%s",
                                step_idx,
                                [round(j, 4) for j in snap_joints] if snap_joints else None,
                                [round(np.degrees(j), 2) for j in snap_joints] if snap_joints else None,
                            )
                            cam_pos_w, cam_rot_w = helper(snap_joints)
                            if cam_pos_w is not None and cam_rot_w is not None:
                                self.logger.info(
                                    "[prepare_plan] [%d] cam→base transform: "
                                    "cam_pos=%s  cam_quat_xyzw=%s  pt_cam=%s",
                                    step_idx,
                                    [round(v, 4) for v in cam_pos_w.tolist()],
                                    [round(v, 4) for v in cam_rot_w.as_quat().tolist()],
                                    [round(v, 4) for v in pos_3d.tolist()],
                                )
                                pos_3d = cam_rot_w.apply(pos_3d) + cam_pos_w
                                self.logger.info(
                                    "[prepare_plan] [%d] cam→base result: pt_base=%s",
                                    step_idx, [round(v, 4) for v in pos_3d.tolist()],
                                )
                        except Exception as tf_exc:
                            self.logger.warning(
                                "[prepare_plan] [%d] fallback cam→base transform failed: %s",
                                step_idx, tf_exc,
                            )
                elif cam_key_present:
                    rs = artifacts.robot_state or {}
                    cam_tf = rs.get("camera", {})
                    snap_joints = rs.get("joints")
                    self.logger.info(
                        "[prepare_plan] [%d] cam→base (from snapshot robot_state): "
                        "joints_rad=%s  joints_deg=%s  cam_pos=%s  cam_quat_xyzw=%s  pt_base=%s",
                        step_idx,
                        [round(j, 4) for j in snap_joints] if snap_joints else None,
                        [round(np.degrees(j), 2) for j in snap_joints] if snap_joints else None,
                        cam_tf.get("position"),
                        cam_tf.get("quaternion_xyzw"),
                        [round(v, 4) for v in pos_3d.tolist()],
                    )

                primitive.parameters["target_position"] = pos_3d.tolist()
                primitive.parameters.pop("point_label", None)

            preference = "top_down" if approach_dir != "side" else "side"
            orientation, chosen_dir = self._best_viable_orientation(preference, clearance_profile)
            if orientation != preference:
                self.logger.warning(
                    "[prepare_plan] [%d] preferred '%s' corridor blocked — falling back to '%s' (dir=%s)",
                    step_idx, preference, orientation, chosen_dir,
                )
            primitive.parameters["preset_orientation"] = orientation
            primitive.metadata["molmo_grounded"] = True
            primitive.metadata["molmo_position_3d"] = (
                np.asarray(ip.position_3d).tolist() if ip.position_3d is not None else None
            )
            # Store 2D pixel location for run output annotation — always, even if 3D failed
            if ip.position_2d is not None:
                primitive.metadata["position_2d"] = list(ip.position_2d)
            self.logger.info(
                "[prepare_plan] [%d] Molmo grounded target_position=%s orientation=%s",
                step_idx, primitive.parameters.get("target_position"), orientation,
            )

        except Exception as exc:
            self.logger.warning(
                "[prepare_plan] [%d] pointing_guidance grounding failed (%s); keeping point_label fallback",
                step_idx, exc,
            )
            plan.diagnostics.warnings.append(
                f"[{step_idx}] pointing_guidance grounding failed ({exc}); using point_label fallback"
            )

    # ------------------------------------------------------------------ #
    # Result normalization
    # ------------------------------------------------------------------ #
    def _json_safe(self, value: Any) -> Any:
        """Best-effort conversion of planner return values into JSON-safe objects."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
        except Exception:
            pass
        try:
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass
        if is_dataclass(value):
            return {k: self._json_safe(v) for k, v in asdict(value).items()}
        return str(value)
