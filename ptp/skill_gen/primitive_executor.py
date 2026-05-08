"""
Primitive executor that translates Gemini-friendly parameters before calling CuRobo primitives.

LLM outputs reference image-grounded cues (pixel [y, x] pointers, normals, standoffs).
This executor back-projects those cues into metric coordinates using the latest snapshot depth and
camera intrinsics, validates the plan, and optionally drives the configured motion planner.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import time

from src.perception.utils.coordinates import compute_3d_position
from src.primitives.skill_plan_types import PRIMITIVE_LIBRARY, PrimitiveCall, SkillPlan
from src.planning.utils.snapshot_utils import SnapshotArtifacts, SnapshotCache, load_snapshot_artifacts
from src.utils.logging_utils import get_structured_logger


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


class PrimitiveExecutor:
    """Translate and execute primitive plans against the configured primitives interface."""

    def __init__(
        self,
        primitives: Optional[Any],
        perception_pool_dir: Path,
        logger: Optional[logging.Logger] = None,
        orchestrator: Optional[Any] = None,
    ):
        self.primitives = primitives
        self.perception_pool_dir = Path(perception_pool_dir)
        self._snapshot_cache = SnapshotCache()
        self.logger = logger or get_structured_logger("PrimitiveExecutor")
        self.orchestrator = orchestrator

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
        dry_run: bool = False,
    ) -> PrimitiveExecutionResult:
        """
        Translate (and optionally execute) a primitive plan.
        """
        self.logger.info("Executing plan: %s", plan)
        translated_plan = self.prepare_plan(plan, world_state)
        if dry_run:
            self.logger.info("Dry run requested; execution skipped.")
            return PrimitiveExecutionResult(executed=False)
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
            if (
                schema is not None
                and "execute" in schema.optional_params
            ):
                primitive.parameters.setdefault("execute", True)
            method = getattr(self.primitives, primitive.name, None)
            if not callable(method):
                raise AttributeError(f"Primitives interface missing primitive '{primitive.name}'")
            # Inject references.object_id as a kwarg so move_gripper_to_pose
            # can build ignore_labels and mask the target object out of collision
            # checking.  It lands in **_kwargs so primitives that don't use it
            # are unaffected.
            call_params = dict(primitive.parameters)
            ref_obj_id = primitive.references.get("object_id")
            if ref_obj_id:
                call_params.setdefault("object_id", ref_obj_id)
            self.logger.debug("Calling primitive method '%s' with parameters: %s", primitive.name, call_params)
            success = True
            try:
                raw_result = method(**call_params)
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

            # Print running executed-primitives log after each step.
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
                time.sleep(0.5)  # Small delay to avoid overwhelming the primitives interface
            else:
                break

        return PrimitiveExecutionResult(
            executed=True,
            primitive_results=primitive_results,
            executed_primitives=executed_primitives,
        )

    def prepare_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
    ) -> SkillPlan:
        """
        Translate parameters and validate the plan without executing it.
        """
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
        self.logger.debug("[prepare_plan] Robot joints from snapshot: %s", joints)

        if self.primitives:
            helper = getattr(self.primitives, "camera_pose_from_joints", None)
            if helper:
                self.logger.debug("[prepare_plan] Getting camera pose from joints...")
                pos, rot = helper(joints)
                cam_pose = SnapshotCameraPose(position=np.asarray(pos, dtype=float), rotation=rot)
                self.logger.debug("[prepare_plan] Camera pose: pos=%s, rot=%s", cam_pose.position, cam_pose.rotation)
            else:
                self.logger.info("[prepare_plan] Primitives interface missing camera_pose_from_joints; skipping base-frame transform")
        else:
            self.logger.info("[prepare_plan] No primitives interface; skipping base-frame transform")

        # Ground any deferred pointing_guidance via Molmo before coordinate translation.
        for idx, primitive in enumerate(plan.primitives):
            if primitive.name == "move_gripper_to_pose" and primitive.metadata.get("pointing_guidance"):
                self._resolve_pointing_guidance(primitive, artifacts, plan, idx)

        # Translate each primitive
        # (antipodal grasp sampling runs after translation, once target_position is in world frame)
        for idx, primitive in enumerate(plan.primitives):
            self.logger.debug(
                "[prepare_plan] [%d/%d] %s parameters: %s references: %s",
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
                        "%s: cannot back-project pixel %s (missing depth/intrinsics) — "
                        "falling back to registry position via references.object_id",
                        primitive.name, pixel,
                    )
                else:
                    coords = [float(pixel[0]), float(pixel[1])]
                    point = compute_3d_position(coords, artifacts.depth, artifacts.intrinsics)
                    if point is None:
                        self.logger.warning(
                            "%s: back-projection returned no point for %s — "
                            "falling back to registry position via references.object_id",
                            primitive.name, pixel,
                        )
                    else:
                        depth_offset = float(primitive.parameters.get("depth_offset_m", 0.0) or 0.0)
                        if depth_offset:
                            point = [point[0], point[1], point[2] + depth_offset]
                        primitive.parameters["target_position"] = point
                        primitive.parameters.pop("depth_offset_m", None)

            # If no target_position yet, inject point_label from references so the
            # primitives implementation can resolve the position from the registry.
            if "target_position" not in primitive.parameters:
                obj_id = primitive.references.get("object_id")
                if obj_id and "point_label" not in primitive.parameters:
                    primitive.parameters["point_label"] = obj_id
                    self.logger.debug(
                        "%s: injected point_label=%r from references.object_id",
                        primitive.name, obj_id,
                    )

            # Transform coordinates from camera frame to base frame.
            # Skip primitives that are already in base frame — Molmo-grounded
            # positions are back-projected to world frame above and must not
            # be transformed a second time.
            already_in_base = (
                primitive.frame == "base"
                or primitive.metadata.get("molmo_grounded")
            )
            if cam_pose and not already_in_base:
                for key in ("target_position", "pivot_point"):
                    if key not in primitive.parameters:
                        continue
                    pos = primitive.parameters[key]
                    base_pos = cam_pose.rotation.apply(pos) + cam_pose.position
                    primitive.parameters[key] = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                    self.logger.debug("Transformed %s: %s -> %s", key, pos, primitive.parameters[key])

        # Antipodal grasp sampling — refine target_position and set target_orientation
        # for any move_gripper_to_pose that is a grasp (not a place).
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
            use_side = preset == "side"
            seed_quat = np.array([0.0, 0.707, 0.0, 0.707] if use_side else [0.0, 1.0, 0.0, 0.0])

            ref_id = primitive.references.get("object_id")
            obj_pts = self._get_object_point_cloud(ref_id, artifacts)

            # Collider from primitives interface — check direct attachment first,
            # then via ._planner (XArmPybulletPlannedPrimitives) or ._robot
            # (XArmPybulletPrimitives sim interface).
            _prim_collider = getattr(self.primitives, "_collider", None)
            _plan_collider = getattr(getattr(self.primitives, "_planner", None), "_collider", None)
            _robot_collider = getattr(getattr(self.primitives, "_robot", None), "_collider", None)
            collider = _prim_collider or _plan_collider or _robot_collider

            if obj_pts is not None and len(obj_pts) >= 10:
                grasp_result = self._sample_antipodal_grasp(
                    contact_point=contact_point,
                    object_points=obj_pts,
                    seed_quaternion=seed_quat,
                    floor_z=0.0,  # PyBullet plane.urdf at robot base origin
                    collider=collider,
                    ignore_object_id=ref_id,
                )
                if grasp_result is not None:
                    grasp_pos, grasp_quat = grasp_result
                    primitive.parameters["target_position"] = grasp_pos.tolist()
                    primitive.parameters["target_orientation"] = grasp_quat.tolist()
                    primitive.parameters.pop("preset_orientation", None)
                    primitive.metadata["antipodal_grounded"] = True
                    self.logger.info(
                        "[prepare_plan] [%d] antipodal grasp: pos=%s quat=%s (ref=%s, %d pts)",
                        idx,
                        [f"{v:.3f}" for v in grasp_pos],
                        [f"{v:.3f}" for v in grasp_quat],
                        ref_id,
                        len(obj_pts),
                    )
                else:
                    self.logger.warning(
                        "[prepare_plan] [%d] antipodal grasp failed — all candidates in collision "
                        "or unreachable for %r; keeping Molmo-grounded pose",
                        idx, ref_id,
                    )
            else:
                self.logger.warning(
                    "[prepare_plan] [%d] antipodal grasp skipped — no point cloud for %r (%s pts)",
                    idx, ref_id, len(obj_pts) if obj_pts is not None else 0,
                )

        # Auto-strip unknown parameters (LLM cross-contamination fallback).
        # Unknown parameters are logged as warnings and removed rather than
        # raising hard failures — the remaining params may still be valid.
        for idx, primitive in enumerate(plan.primitives):
            schema = PRIMITIVE_LIBRARY.get(primitive.name)
            if schema is None:
                continue
            allowed = set(schema.required_params) | set(schema.optional_params)
            unknown = [k for k in list(primitive.parameters) if k not in allowed]
            for k in unknown:
                self.logger.warning(
                    "[prepare_plan] [%d] stripping unknown param '%s' from %s",
                    idx, k, primitive.name,
                )
                primitive.parameters.pop(k)
                plan.diagnostics.warnings.append(
                    f"[{idx}] stripped unknown parameter '{k}' from {primitive.name}"
                )

        validation_errors = plan.validate(PRIMITIVE_LIBRARY)
        if validation_errors:
            raise ValueError(f"Plan validation failed: {validation_errors}")
        self.logger.info("[prepare_plan] Validation passed")

        return plan

    # ------------------------------------------------------------------ #
    # Antipodal grasp sampling
    # ------------------------------------------------------------------ #
    def _sample_antipodal_grasp(
        self,
        contact_point: np.ndarray,
        object_points: np.ndarray,
        seed_quaternion: np.ndarray,
        gripper_width_m: float = 0.085,
        finger_thickness_m: float = 0.012,
        n_rotations: int = 36,
        standoff_m: float = 0.0,
        floor_z: float = 0.0,
        collider: Optional[Any] = None,
        ignore_object_id: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Sample antipodal grasps around contact_point and return the best (position, quaternion).

        Strategy:
          1. Any candidate whose jaws would be closer than finger_thickness_m above floor_z
             is rejected to avoid driving the gripper into the ground plane.
          2. Project all object points onto the plane perpendicular to the seed approach axis.
          3. Rotate candidate grasp axes around the approach axis in n_rotations steps.
          4. For each candidate axis, find the antipodal pair along that axis inside the
             gripper-width budget.  Score by centering quality (contact_point midway between
             jaws) plus secondary bonuses for jaw spread width and floor clearance margin.
          5. If a collider is provided, plan and collision-check each candidate trajectory —
             only reachable, collision-free candidates are scored.
          6. Return the (position, quaternion) of the highest-scoring floor-safe candidate.
             Falls back to (contact_point, seed_quaternion) if no valid pair is found.

        Args:
            contact_point: 3-D world-frame interaction point (from Molmo / point_label).
            object_points: (N, 3) world-frame point cloud for the target object.
            seed_quaternion: Starting approach quaternion [x, y, z, w] (from preset_orientation).
            gripper_width_m: Maximum jaw spread in metres (xArm default 0.085 m).
            finger_thickness_m: Finger body height used as collision margin above the floor.
            n_rotations: Number of in-plane rotation candidates to try.
            standoff_m: Extra approach standoff added to the returned position along the
                        approach axis (useful for pre-grasp waypoints).
            floor_z: Support-surface Z in world frame.  Defaults to 0.0 (PyBullet plane.urdf
                     at the robot base origin).
            collider: Optional DepthEnvironmentCollider for trajectory collision checking.
            ignore_object_id: Object label to exclude from collision checks (the grasp target).

        Returns:
            (position, quaternion) — best grasp pose in world frame, or None if
            no valid candidate was found (all are in collision or unreachable).
        """
        seed_rot = Rotation.from_quat(seed_quaternion)
        # Approach axis is the local Z of the gripper (pointing toward the object).
        approach = seed_rot.apply(np.array([0.0, 0.0, 1.0]))
        approach = approach / (np.linalg.norm(approach) + 1e-9)

        # Minimum Z the lower jaw edge must clear.
        floor_clearance_z = floor_z + finger_thickness_m

        # Build a stable orthonormal frame in the plane perpendicular to approach.
        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(approach, up_hint)) > 0.9:
            up_hint = np.array([1.0, 0.0, 0.0])
        u = np.cross(approach, up_hint)
        u /= np.linalg.norm(u) + 1e-9
        v = np.cross(approach, u)
        v /= np.linalg.norm(v) + 1e-9

        # Relative vectors from contact_point to each object point.
        relative = object_points - contact_point

        best_score = -np.inf
        best_pos: Optional[np.ndarray] = None
        best_quat: Optional[np.ndarray] = None

        half_width = gripper_width_m / 2.0
        angles = np.linspace(0.0, np.pi, n_rotations, endpoint=False)

        # Ignore set for collision checker: skip the grasp target mesh + background
        # (we expect the gripper to be near the object during grasping).
        ignore: Optional[set] = None
        if collider is not None:
            ignore = {ignore_object_id} if ignore_object_id else set()

        for angle in angles:
            # Grasp axis in the plane: rotate u by angle about approach.
            grasp_axis = np.cos(angle) * u + np.sin(angle) * v
            grasp_axis /= np.linalg.norm(grasp_axis) + 1e-9

            # Project points onto grasp axis.
            proj = relative @ grasp_axis  # (N,)

            # Find points within gripper-width budget on each side.
            pos_mask = (proj >= 0) & (proj <= half_width)
            neg_mask = (proj <= 0) & (proj >= -half_width)
            if not (np.any(pos_mask) and np.any(neg_mask)):
                continue

            # Antipodal contacts: outermost reachable points on each side.
            jaw_pos = float(np.max(proj[pos_mask]))
            jaw_neg = float(np.min(proj[neg_mask]))

            # Centering: contact_point should be midway between the two jaw contacts.
            midpoint = (jaw_pos + jaw_neg) / 2.0
            grasp_center = contact_point + midpoint * grasp_axis

            # Floor collision check: both jaw tips must clear the support surface.
            # The jaw tips sit at ±half_width along grasp_axis from grasp_center.
            jaw_tip_a = grasp_center + half_width * grasp_axis
            jaw_tip_b = grasp_center - half_width * grasp_axis
            lower_jaw_z = min(jaw_tip_a[2], jaw_tip_b[2])
            if lower_jaw_z < floor_clearance_z:
                continue

            grasp_pos = grasp_center - standoff_m * approach

            # Build rotation: gripper X-axis aligns with grasp_axis,
            # Z-axis remains the approach direction.
            x_axis = grasp_axis
            z_axis = approach
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-9
            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            cand_quat = Rotation.from_matrix(R).as_quat()

            # Trajectory collision check: reject candidates that can't be reached
            # without hitting the environment or the floor.
            if collider is not None:
                planner = collider._planner
                traj = planner.plan_joint_trajectory_to_pose(
                    target_position=grasp_pos.tolist(),
                    target_orientation=cand_quat.tolist(),
                )
                if traj is None:
                    continue
                if collider.check_trajectory(traj, ignore=ignore) is not None:
                    continue

            centering_score = -abs(midpoint)
            width_score = (jaw_pos - jaw_neg) / gripper_width_m
            # Extra reward for keeping jaws well above the floor.
            floor_margin = lower_jaw_z - floor_clearance_z
            floor_score = min(floor_margin / 0.05, 1.0)  # saturates at 5 cm margin

            score = centering_score + 0.3 * width_score + 0.2 * floor_score

            if score > best_score:
                best_score = score
                best_quat = cand_quat
                best_pos = grasp_pos

        return (best_pos, best_quat) if best_pos is not None else None

    def _get_object_point_cloud(
        self,
        object_id: Optional[str],
        artifacts: Any,
    ) -> Optional[np.ndarray]:
        """Return world-frame (N, 3) point cloud for object_id using its GSAM2 mask + depth.

        Returns None if the mask, depth, or intrinsics are unavailable.
        """
        if object_id is None or artifacts.depth is None or artifacts.intrinsics is None:
            return None

        # Prefer masks cached on the tracker (_last_masks).
        masks: Dict[str, Any] = {}
        if self.orchestrator is not None:
            tracker = getattr(self.orchestrator, "tracker", None)
            raw = getattr(tracker, "_last_masks", None) or {}
            # GSAM2ContinuousObjectTracker wraps an inner _tracker
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

        # Transform to world frame using camera pose from the snapshot.
        cam_pos: Optional[np.ndarray] = None
        cam_rot: Optional[Rotation] = None
        if self.orchestrator is not None:
            robot = getattr(getattr(self.orchestrator, "config", None), "robot", None)
            if robot is not None:
                try:
                    cam_pos, cam_rot = robot.get_camera_transform()
                except Exception:
                    pass

        if cam_pos is None or cam_rot is None:
            # Fall back to snapshot robot_state joints if available.
            joints = (artifacts.robot_state or {}).get("joints")
            if self.primitives and joints is not None:
                helper = getattr(self.primitives, "camera_pose_from_joints", None)
                if helper:
                    try:
                        cam_pos, cam_rot = helper(joints)
                    except Exception:
                        pass

        if cam_pos is None or cam_rot is None:
            return pts_cam.astype(np.float32)  # return camera-frame points as fallback

        pts_world = cam_rot.apply(pts_cam) + cam_pos
        return pts_world.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Molmo pointing-guidance grounding
    # ------------------------------------------------------------------ #
    def _best_viable_orientation(
        self,
        preference: str,
        clearance_profile: Optional[Any],
    ) -> Tuple[str, Optional[List[float]]]:
        """
        Return (preset_orientation, chosen_direction) that best matches the LLM's preference
        while being clear of obstacles.

        Strategy:
          1. Filter approach_corridors to grasp_compatible ones.
          2. Score each by alignment with the preferred direction type:
               top_down → want high |Z| component (near-vertical approach from above)
               side     → want low |Z| component (near-horizontal approach)
          3. Sort preferred-aligned corridors by alignment score desc, then min_clearance desc.
          4. If no preferred-aligned corridor is viable, fall back to the best remaining
             grasp-compatible corridor regardless of direction.
          5. If no grasp-compatible corridor exists at all, honour the raw LLM preference.

        Returns the orientation string and the chosen direction vector (for logging).
        """
        if clearance_profile is None:
            return preference, None

        viable = [c for c in clearance_profile.approach_corridors if c.grasp_compatible]
        if not viable:
            return preference, None

        # For top_down: prefer corridors where the approach comes from above (negative Z
        # component when approaching downward, i.e. direction[2] < 0 and |Z| is large).
        # We score by |Z| for both cases — top_down wants high |Z|, side wants low |Z|.
        def _z_alignment(corridor: Any) -> float:
            d = np.asarray(corridor.direction, dtype=float)
            return float(abs(d[2]))

        top_down_pref = preference == "top_down"

        # Split into preferred and fallback groups
        preferred = []
        fallback = []
        for c in viable:
            z = _z_alignment(c)
            # top_down: |Z| > 0.5 (direction has meaningful vertical component)
            # side: |Z| <= 0.5 (direction is more horizontal)
            if top_down_pref:
                (preferred if z > 0.5 else fallback).append((z, c))
            else:
                (preferred if z <= 0.5 else fallback).append((z, c))

        if preferred:
            # Sort by alignment (best first), then by clearance margin
            preferred.sort(key=lambda t: (
                -t[0] if top_down_pref else t[0],
                -t[1].min_clearance,
            ))
            chosen = preferred[0][1]
        else:
            # Preferred direction is fully blocked — take best available
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
    ) -> None:
        """
        Ground a move_gripper_to_pose primitive using pointing_guidance from metadata.

        Queries MolmoPointDetector with the guidance text as a custom prompt using
        the freshest snapshot available at execution time, then writes target_position
        and preset_orientation back onto the primitive.
        Falls back to point_label (registry lookup) on any error.
        """
        import io
        import numpy as np
        from PIL import Image as _PIL

        guidance: str = primitive.metadata.get("pointing_guidance", "")
        approach_dir: str = primitive.metadata.get("approach_direction", "from_clearance")
        ref_id = primitive.references.get("object_id")

        # Locate detector: GSAM2ContinuousObjectTracker wraps inner tracker as ._tracker
        detector = None
        if self.orchestrator is not None:
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

            self.logger.info(
                "[prepare_plan] [%d] Running Molmo for pointing_guidance=%r on %s",
                step_idx, guidance, ref_id or "__guidance__",
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
            )

            ip = results.get("_guided")
            if ip is None:
                raise ValueError("Molmo returned no point for guidance prompt")

            if ip.position_3d is not None:
                primitive.parameters["target_position"] = np.asarray(
                    ip.position_3d, dtype=float
                ).tolist()
                primitive.parameters.pop("point_label", None)

            preference = "top_down" if approach_dir != "side" else "side"
            orientation, chosen_dir = self._best_viable_orientation(preference, clearance_profile)
            if orientation != preference:
                self.logger.warning(
                    "[prepare_plan] [%d] preferred '%s' corridor blocked — falling back to '%s' "
                    "(dir=%s)",
                    step_idx, preference, orientation, chosen_dir,
                )
            primitive.parameters["preset_orientation"] = orientation

            primitive.metadata["molmo_grounded"] = True
            primitive.metadata["molmo_position_3d"] = (
                np.asarray(ip.position_3d).tolist() if ip.position_3d is not None else None
            )
            self.logger.info(
                "[prepare_plan] [%d] Molmo grounded target_position=%s orientation=%s "
                "(preferred=%s chosen_dir=%s)",
                step_idx, primitive.parameters.get("target_position"), orientation,
                preference, chosen_dir,
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
        """
        Best-effort conversion of planner return values into JSON-safe objects.
        Keeps executor behavior unchanged while avoiding serialization failures.
        """
        # Primitive scalars / passthrough
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # Containers
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        # Torch tensors
        try:  # Lazy import to avoid hard dependency when unused
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
        except Exception:
            pass

        # NumPy arrays
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        # Generic dataclasses
        if is_dataclass(value):
            return {k: self._json_safe(v) for k, v in asdict(value).items()}

        # Fallback to string representation
        return str(value)
