"""run_ptp.py — Prompt-to-Primitives entry point (no PDDL/orchestrator).

Pipeline:
    1. Connect to xArm + initialise PyBullet FK/IK planner
    2. Capture RealSense RGB-D frame
    3. Save frame as a perception-pool snapshot so executor can load it
    4. GSAM2: detect and segment objects → DetectedObjectRegistry
    5. Molmo: loaded once, shared between detect_objects and PrimitiveExecutor
    6. User provides a task string
    7. SkillDecomposer: task → SkillPlan (LLM-backed primitive decomposition)
    8. PrimitiveExecutor: translate pixel/camera coords → base frame, then execute

Run without real hardware:
    python scripts/run_ptp.py --no-execute --task "pick up the cup"

Run with a real xArm + RealSense:
    python scripts/run_ptp.py --task "push the red block to the left"
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on the path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

from ptp.utils.logging_utils import configure_logging, get_structured_logger, RunTimer

logger = get_structured_logger("run_ptp")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt-to-Primitives pipeline runner")
    p.add_argument("--task", required=True, help="Natural language task description")
    p.add_argument("--no-execute", action="store_true", help="Plan but do not send commands to robot")
    p.add_argument("--robot-ip", default="192.168.1.224", help="xArm IP address")
    p.add_argument("--llm-model", default="gpt-4o", help="OpenAI model for decomposition")
    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--perception-pool",
                   default=str(_REPO_ROOT / "outputs" / "perception_pool"),
                   help="Path to perception pool directory for snapshots")
    p.add_argument("--gsam2-model", default="IDEA-Research/grounding-dino-tiny",
                   help="GroundingDINO model name for GSAM2")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="LLM sampling temperature for decomposition")
    p.add_argument("--output-plan", default=None,
                   help="Path to write the generated SkillPlan JSON")
    p.add_argument("--pybullet-gui", action="store_true",
                   help="Open the PyBullet GUI viewer during planning and execution")
    p.add_argument("--molmo-server", default=None, metavar="URL",
                   help="Use a running molmo_server instead of loading the model locally "
                        "(e.g. http://127.0.0.1:8765). Start the server with: "
                        "python scripts/molmo_server.py")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_primitives_interface(robot_ip: str, dry_run: bool, use_gui: bool = False) -> Optional[object]:
    """Connect to xArm and build the planned-primitives interface."""
    if dry_run:
        logger.info("Dry-run mode: no robot interface created")
        return None
    try:
        from ptp.kinematics.xarm_robot_adapter import XArmRobotAdapter
        from ptp.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
        logger.info("Connecting to xArm at %s …", robot_ip)
        robot = XArmRobotAdapter(robot_ip)
        primitives = XArmPybulletPlannedPrimitives(robot=robot, use_gui=use_gui)
        logger.info("xArm connected")
        return primitives
    except Exception as exc:
        logger.error("Failed to connect to robot: %s", exc)
        raise


def capture_frame() -> Tuple[Any, Any, Any]:
    """Capture an RGB-D frame from the wrist-mounted RealSense camera."""
    import numpy as np
    from ptp.camera import REALSENSE_AVAILABLE, CameraIntrinsics

    if REALSENSE_AVAILABLE:
        from ptp.camera.realsense_camera import RealSenseCamera
        with RealSenseCamera(width=640, height=480, fps=30) as cam:
            color, depth = cam.get_aligned_frames()
            intrinsics = cam.get_camera_intrinsics()
        logger.info("Captured RealSense frame: %dx%d", color.shape[1], color.shape[0])
    else:
        logger.warning("pyrealsense2 not available — using synthetic blank frame")
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)

    return color, depth, intrinsics


def save_snapshot(
    color,
    depth,
    intrinsics,
    perception_pool: Path,
    snapshot_id: str,
    robot_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the current frame to the perception pool so the executor can load it."""
    import numpy as np
    from PIL import Image as _PIL

    snap_dir = perception_pool / "snapshots" / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Color image
    color_path = snap_dir / "color.png"
    _PIL.fromarray(color.astype("uint8")).save(color_path)

    # Depth
    depth_path = snap_dir / "depth.npz"
    np.savez_compressed(depth_path, depth_m=depth)

    # Camera intrinsics
    intr_path = snap_dir / "intrinsics.json"
    intr_path.write_text(json.dumps({
        "fx": intrinsics.fx, "fy": intrinsics.fy,
        "cx": intrinsics.cx, "cy": intrinsics.cy,
        "width": getattr(intrinsics, "width", color.shape[1]),
        "height": getattr(intrinsics, "height", color.shape[0]),
    }))

    # Robot state (joints + camera tf) — used for camera→base transform at execution time
    robot_state_file = None
    if robot_state:
        rs_path = snap_dir / "robot_state.json"
        rs_path.write_text(json.dumps(robot_state))
        robot_state_file = f"snapshots/{snapshot_id}/robot_state.json"

    # Update pool index
    index_path = perception_pool / "index.json"
    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            pass
    files: Dict[str, Any] = {
        "color": f"snapshots/{snapshot_id}/color.png",
        "depth_npz": f"snapshots/{snapshot_id}/depth.npz",
        "intrinsics": f"snapshots/{snapshot_id}/intrinsics.json",
    }
    if robot_state_file:
        files["robot_state"] = robot_state_file
    index.setdefault("snapshots", {})[snapshot_id] = {
        "captured_at": datetime.datetime.now().isoformat(),
        "files": files,
    }
    index_path.write_text(json.dumps(index, indent=2))
    logger.info("Snapshot '%s' saved to %s", snapshot_id, snap_dir)


def build_molmo(server_url: Optional[str] = None) -> Any:
    """Return a Molmo detector — either a local model or a server client."""
    if server_url:
        from ptp.perception.molmo_client import MolmoClient
        logger.info("Using Molmo server at %s", server_url)
        detector = MolmoClient(server_url=server_url)
        detector.load()  # health-checks the server
        return detector
    from ptp.perception.molmo_point_detector import MolmoPointDetector
    logger.info("Loading Molmo2-4B locally…")
    detector = MolmoPointDetector()
    detector.load()
    return detector


def detect_objects(color, depth, intrinsics, gsam2_model: str) -> Tuple[Any, Dict[str, Any], Any]:
    """Run GSAM2 segmentation for one frame; return (registry, masks, tracker).

    tracker is returned so the caller can extract load/detect timings.
    """
    import asyncio
    from ptp.perception.gsam2.gsam2_tracker import GSAM2ObjectTracker

    logger.info("Running GSAM2 detection…")
    tracker = GSAM2ObjectTracker(
        grounding_model_id=gsam2_model,
        compute_clearances=False,
        compute_contacts=False,
        compute_occlusion=False,
    )
    asyncio.get_event_loop().run_until_complete(
        tracker.detect_objects(color, depth, intrinsics)
    )
    detected = tracker.registry.get_all_objects()
    masks: Dict[str, Any] = dict(tracker._last_masks)
    logger.info("GSAM2 detected %d objects: %s",
                len(detected), [o.object_id for o in detected])
    logger.info("GSAM2 masks available for: %s", list(masks.keys()))
    return tracker.registry, masks, tracker


def save_detections_to_snapshot(
    registry: Any,
    perception_pool: Path,
    snapshot_id: str,
) -> None:
    """Write detections.json so the decomposer can draw bounding boxes on the scene image."""
    import time as _time

    snap_dir = perception_pool / "snapshots" / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    objects = registry.get_all_objects() if registry else []
    payload = {
        "stamp": _time.time(),
        "objects": [
            {
                "object_id": o.object_id,
                "object_type": getattr(o, "object_type", None),
                "bounding_box_2d": getattr(o, "bounding_box_2d", None),
                "position_2d": getattr(o, "position_2d", None),
                "position_3d": (
                    o.position_3d.tolist()
                    if getattr(o, "position_3d", None) is not None
                    else None
                ),
                "snapshot_id": snapshot_id,
            }
            for o in objects
        ],
    }
    det_path = snap_dir / "detections.json"
    det_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved detections.json (%d objects) → %s", len(objects), det_path)


def build_world_state(registry, snapshot_id: str) -> Dict[str, Any]:
    """Assemble the world_state dict consumed by the decomposer and executor."""
    objects = registry.get_all_objects() if registry else []
    latest_detections = [
        {
            "object_id": o.object_id,
            "object_type": getattr(o, "object_type", None),
            "bounding_box_2d": getattr(o, "bounding_box_2d", None),
            "position_2d": getattr(o, "position_2d", None),
            "position_3d": getattr(o, "position_3d", None),
        }
        for o in objects
    ]
    return {
        "last_snapshot_id": snapshot_id,
        "registry": registry.to_dict() if registry else {"objects": []},
        "latest_detections": latest_detections,
    }


def build_scene_labeled_image(color: Any, registry: Any) -> Optional[bytes]:
    """Build the labeled scene image (bbox + object ID overlays) used by both the
    task parser and the primitive decomposer."""
    import io as _io
    from PIL import Image as _PIL
    from ptp.primitives.decomposer import build_labeled_image

    if color is None:
        return None
    buf = _io.BytesIO()
    _PIL.fromarray(color.astype("uint8")).save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    objects = registry.get_all_objects() if registry else []
    detections = [
        {
            "object_id": o.object_id,
            "bounding_box_2d": getattr(o, "bounding_box_2d", None),
        }
        for o in objects
        if getattr(o, "bounding_box_2d", None) is not None
    ]
    return build_labeled_image(raw_bytes, detections) if detections else raw_bytes


def parse_task(
    task: str,
    registry: Any,
    labeled_image_bytes: Optional[bytes],
    llm_model: str,
    api_key: Optional[str],
    temperature: float,
) -> Any:
    """Translate a natural language task into grounded (action, object_id) pairs."""
    from ptp.primitives.task_parser import TaskParser
    from ptp.llm_interface.openai_client import OpenAIClient

    llm = OpenAIClient(model=llm_model, api_key=api_key or None)
    parser = TaskParser(llm_client=llm)
    logger.info("Parsing task: %r", task)
    result = parser.parse(task=task, registry=registry, image_bytes=labeled_image_bytes, temperature=temperature)
    logger.info("Task parser rationale: %s", result.rationale)
    for i, a in enumerate(result.actions):
        logger.info("  [%d] %s → %s  (%s)", i, a.action, a.object_id, a.description)
    return result


def decompose_task(
    parsed_action: Any,
    world_state: Dict[str, Any],
    llm_model: str,
    api_key: Optional[str],
    temperature: float,
    perception_pool: Optional[Path] = None,
    labeled_image_bytes: Optional[bytes] = None,
) -> Any:
    """Run the LLM decomposer on a single ParsedAction to produce a SkillPlan."""
    from ptp.primitives.decomposer import SkillDecomposer
    from ptp.llm_interface.openai_client import OpenAIClient

    llm = OpenAIClient(model=llm_model, api_key=api_key or None)
    decomposer = SkillDecomposer(llm_client=llm, perception_pool_dir=perception_pool)
    parameters = {"object_id": parsed_action.object_id} if parsed_action.object_id else {}
    logger.info("Decomposing action: %r → object=%s", parsed_action.action, parsed_action.object_id)
    plan = decomposer.plan(
        action_name=parsed_action.description,
        parameters=parameters,
        world_hint=world_state,
        temperature=temperature,
        high_level_action=parsed_action.action,
        target_object_id=parsed_action.object_id,
        labeled_image_bytes=labeled_image_bytes,
    )
    logger.info("Plan has %d primitives", len(plan.primitives))
    for i, p in enumerate(plan.primitives):
        logger.info("  [%d] %s %s", i, p.name, p.parameters)
    return plan


def execute_plan(plan, primitives, molmo, world_state: Dict[str, Any],
                 perception_pool: Path, dry_run: bool,
                 masks: Optional[Dict[str, Any]] = None) -> Any:
    """Translate and (optionally) execute the plan."""
    from ptp.primitives.executor import PrimitiveExecutor

    executor = PrimitiveExecutor(
        primitives=primitives,
        perception_pool_dir=perception_pool,
        molmo=molmo,
        masks=masks,
    )
    result = executor.execute_plan(plan, world_state, dry_run=dry_run)
    if result.executed:
        ok = sum(1 for r in result.executed_primitives if r.success)
        logger.info("Execution complete: %d/%d primitives succeeded",
                    ok, len(result.executed_primitives))
    else:
        logger.info("Plan translated (dry-run); no execution")
    return result


# ---------------------------------------------------------------------------
# Run output saving
# ---------------------------------------------------------------------------

_PALETTE = [
    (255, 80,  80),  (80,  200,  80), (80,  160, 255), (255, 200,  40),
    (200, 80,  255), (40,  220, 220), (255, 140,   0), (180, 255,  80),
]


def _obj_colour(obj_id: str) -> tuple:
    import hashlib
    idx = int(hashlib.md5(obj_id.encode()).hexdigest(), 16) % len(_PALETTE)
    return _PALETTE[idx]


def save_run_outputs(
    run_dir: Path,
    color: Any,
    task: str,
    snapshot_id: str,
    registry: Any,
    plan: Any,
    execution_result: Any,
    timings: Optional[Dict[str, Any]] = None,
) -> None:
    """Write annotated images, plan JSON, and run summary to run_dir."""
    import io as _io
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    run_dir.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(color.astype("uint8")).convert("RGB")
    iw, ih = img.size

    def _norm_to_px(ny, nx):
        return int(nx * iw / 1000), int(ny * ih / 1000)

    def _load_font(size: int):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    fsz = max(14, iw // 55)
    font = _load_font(fsz)

    # ── 1. detections.png ────────────────────────────────────────────────────
    det_img = img.copy()
    det_draw = ImageDraw.Draw(det_img, "RGBA")
    objects = registry.get_all_objects() if registry else []
    for obj in objects:
        bbox = getattr(obj, "bounding_box_2d", None)
        if not bbox or len(bbox) < 4:
            continue
        ny1, nx1, ny2, nx2 = bbox
        x1, y1 = _norm_to_px(ny1, nx1)
        x2, y2 = _norm_to_px(ny2, nx2)
        c = _obj_colour(obj.object_id)
        det_draw.rectangle([x1, y1, x2, y2], outline=c + (220,), width=3, fill=c + (35,))
        label = f"{obj.object_id} ({getattr(obj, 'object_type', '')})"
        try:
            tb = font.getbbox(label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * fsz // 2, fsz
        pad = 4
        lx, ly = x1, max(0, y1 - th - pad * 2)
        det_draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=c + (210,))
        det_draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255), font=font)
    det_img.save(run_dir / "detections.png")

    # ── 2. interaction_points.png ─────────────────────────────────────────────
    # Collect grounded positions_2d from plan primitives (set by executor).
    ip_img = det_img.copy()
    ip_draw = ImageDraw.Draw(ip_img, "RGBA")
    if plan is not None:
        for i, prim in enumerate(plan.primitives):
            pos2d = prim.metadata.get("position_2d") or prim.parameters.get("target_pixel_yx")
            ref_id = prim.references.get("object_id") or f"step{i}"
            c = _obj_colour(ref_id)
            r = max(8, iw // 80)

            # Gripper destination dot — only for move_gripper_to_pose.
            if pos2d is not None and len(pos2d) >= 2 and prim.name == "move_gripper_to_pose":
                # position_2d is [norm_y, norm_x] in 0-1000 scale
                px, py = _norm_to_px(pos2d[0], pos2d[1])
                ip_draw.ellipse([px - r, py - r, px + r, py + r], fill=c + (230,), outline=(255,255,255,220), width=2)
                ip_draw.text((px + r + 3, py - fsz // 2), f"[{i}] {prim.name}", fill=c, font=font)

            # Surface normal sampling point for push/pull — small square marker.
            surf2d = prim.metadata.get("surface_point_2d")
            if surf2d is not None and len(surf2d) >= 2 and prim.name in ("push", "pull"):
                sx, sy = _norm_to_px(surf2d[0], surf2d[1])
                sr = max(4, r // 2)
                ip_draw.rectangle([sx - sr, sy - sr, sx + sr, sy + sr], fill=c + (180,), outline=(255,255,255,180), width=1)
                ip_draw.text((sx + sr + 3, sy - fsz // 2), f"[{i}] {prim.name} surface", fill=c, font=font)

            # Draw hinge/pivot point as a diamond if present
            hinge_2d = prim.metadata.get("hinge_position_2d")
            if hinge_2d is not None and len(hinge_2d) >= 2:
                hx, hy = _norm_to_px(hinge_2d[0], hinge_2d[1])
                ip_draw.polygon([(hx, hy - r), (hx + r, hy), (hx, hy + r), (hx - r, hy)],
                                fill=(255, 255, 80, 220), outline=(255, 255, 255, 220))
                ip_draw.text((hx + r + 3, hy - fsz // 2), f"[{i}] hinge", fill=(255, 255, 80), font=font)

            # For primitives with no dedicated marker and no position_2d, draw a dot
            # at the bbox centroid as a fallback label anchor.
            elif pos2d is None and prim.name not in ("push", "pull") and prim.metadata.get("molmo_position_3d") is not None:
                # Find the object bbox centroid to anchor the label
                ref_obj = next((o for o in objects if o.object_id == ref_id), None)
                if ref_obj is not None:
                    bbox = getattr(ref_obj, "bounding_box_2d", None)
                    if bbox and len(bbox) == 4:
                        ny1, nx1, ny2, nx2 = bbox
                        cx = int((nx1 + nx2) / 2 * iw / 1000)
                        cy = int((ny1 + ny2) / 2 * ih / 1000)
                        ip_draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=c+(200,), outline=(255,255,255,180), width=2)
                        ip_draw.text((cx+r+3, cy-fsz//2), f"[{i}] {prim.name}", fill=c, font=font)
    ip_img.save(run_dir / "interaction_points.png")

    # ── 3. plan.json ──────────────────────────────────────────────────────────
    plan_dict = plan.to_dict() if plan is not None else {}
    (run_dir / "plan.json").write_text(json.dumps(plan_dict, indent=2))

    # ── 4. run_summary.json ───────────────────────────────────────────────────
    exec_summary = []
    if execution_result is not None and execution_result.executed_primitives:
        for ep in execution_result.executed_primitives:
            exec_summary.append({
                "index": ep.index,
                "name": ep.name,
                "success": ep.success,
                "parameters": ep.parameters,
                "references": ep.references,
            })

    summary = {
        "task": task,
        "snapshot_id": snapshot_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "detections": [
            {
                "object_id": o.object_id,
                "object_type": getattr(o, "object_type", None),
                "bounding_box_2d": getattr(o, "bounding_box_2d", None),
                "position_3d": (
                    getattr(o, "position_3d", None).tolist()
                    if getattr(o, "position_3d", None) is not None else None
                ),
            }
            for o in objects
        ],
        "plan": plan_dict,
        "execution": exec_summary,
        "dry_run": execution_result is not None and not execution_result.executed,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # ── 5. timings.json ───────────────────────────────────────────────────────
    if timings:
        (run_dir / "timings.json").write_text(json.dumps(timings, indent=2))

    # ── 6. llm_response.json — raw LLM output with rationale/diagnostics ─────
    if plan is not None and plan.raw_llm_response is not None:
        try:
            parsed = json.loads(plan.raw_llm_response)
            (run_dir / "llm_response.json").write_text(json.dumps(parsed, indent=2))
        except Exception:
            (run_dir / "llm_response.json").write_text(plan.raw_llm_response)

    # ── 8. molmo_input_<step>.png — image sent to Molmo per push/pull primitive ─
    if plan is not None:
        for i, prim in enumerate(plan.primitives):
            surface_bytes = prim.metadata.get("molmo_input_image_bytes")
            if surface_bytes:
                (run_dir / f"molmo_input_{i}_{prim.name}.png").write_bytes(surface_bytes)
            hinge_bytes = prim.metadata.get("molmo_hinge_input_image_bytes")
            if hinge_bytes:
                (run_dir / f"molmo_input_{i}_{prim.name}_hinge.png").write_bytes(hinge_bytes)

    # ── 9. depth_debug.png — overlay interaction points if file already exists ─
    depth_debug_path = run_dir / "depth_debug.png"
    if depth_debug_path.exists() and plan is not None:
        try:
            import cv2
            dd = cv2.imread(str(depth_debug_path))
            if dd is not None:
                dh, dw = dd.shape[:2]
                _POINT_COLOURS_BGR = [
                    (80,  200,  80),   # green
                    (80,   80, 220),   # red
                    (220, 160,  40),   # blue
                    (40,  220, 220),   # yellow
                    (200,  80, 200),   # magenta
                    (80,  200, 200),   # cyan
                ]
                obj_idx = 0
                for i, prim in enumerate(plan.primitives):
                    c = _POINT_COLOURS_BGR[obj_idx % len(_POINT_COLOURS_BGR)]
                    drew_any = False

                    for meta_key, shape, label_suffix in [
                        ("position_2d",       "circle",   prim.name),
                        ("surface_point_2d",  "square",   f"{prim.name} surface"),
                        ("hinge_position_2d", "diamond",  "hinge"),
                    ]:
                        pos2d = prim.metadata.get(meta_key)
                        if pos2d is None or len(pos2d) < 2:
                            continue
                        # position_2d is [norm_y, norm_x] in 0-1000
                        px = int(pos2d[1] / 1000 * dw)
                        py = int(pos2d[0] / 1000 * dh)
                        r = max(8, dw // 80)
                        if shape == "circle":
                            cv2.circle(dd, (px, py), r, c, -1)
                            cv2.circle(dd, (px, py), r, (255, 255, 255), 2)
                        elif shape == "square":
                            cv2.rectangle(dd, (px - r, py - r), (px + r, py + r), c, -1)
                            cv2.rectangle(dd, (px - r, py - r), (px + r, py + r), (255, 255, 255), 2)
                        elif shape == "diamond":
                            pts = np.array([(px, py-r),(px+r, py),(px, py+r),(px-r, py)], np.int32)
                            cv2.fillPoly(dd, [pts], (40, 220, 220))
                            cv2.polylines(dd, [pts], True, (255, 255, 255), 2)
                        fsz_cv = max(0.4, dw / 1200)
                        cv2.putText(dd, f"[{i}] {label_suffix}", (px + r + 4, py + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, fsz_cv, (255, 255, 255), 1, cv2.LINE_AA)
                        drew_any = True

                    if drew_any:
                        obj_idx += 1

                cv2.imwrite(str(depth_debug_path), dd)
        except Exception as exc:
            logger.debug("depth_debug interaction-point overlay failed: %s", exc)

    logger.info(
        "Run outputs saved to %s  "
        "(detections.png, interaction_points.png, planner_input.png, plan.json, "
        "llm_response.json, timings.json, run_summary.json, molmo_input_*.png, "
        "depth_debug.png, depth_debug_masks.png)",
        run_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    configure_logging()

    perception_pool = Path(args.perception_pool)
    perception_pool.mkdir(parents=True, exist_ok=True)

    snapshot_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _REPO_ROOT / "outputs" / "run_ptp" / snapshot_id

    timer = RunTimer()

    # 1. Robot interface
    with timer.measure("robot_init"):
        primitives = build_primitives_interface(args.robot_ip, dry_run=args.no_execute,
                                                use_gui=args.pybullet_gui)

    # 2. Capture robot state (joints + camera transform) at the moment of capture.
    robot_state: Optional[Dict[str, Any]] = None
    if primitives is not None:
        try:
            robot_state = primitives.get_robot_state()
            logger.info(
                "Robot state captured: joints=%s camera_pos=%s",
                robot_state.get("joints"),
                (robot_state.get("camera") or {}).get("position"),
            )
        except Exception as exc:
            logger.warning("Could not capture robot state: %s", exc)

    # 3. Capture frame
    with timer.measure("frame_capture"):
        color, depth, intrinsics = capture_frame()

    # 4. Save snapshot
    save_snapshot(color, depth, intrinsics, perception_pool, snapshot_id,
                  robot_state=robot_state)

    # 5. Load Molmo (or connect to server)
    with timer.measure("model_load.molmo"):
        molmo = build_molmo(server_url=args.molmo_server)
    if molmo is not None:
        timer.record("model_load.molmo_detail", getattr(molmo, "load_time_s", 0.0))

    # 6. Detect objects (GSAM2 model load happens inside; load_time_s pulled out after)
    with timer.measure("perception.total"):
        registry, masks, gsam2_tracker = detect_objects(color, depth, intrinsics, args.gsam2_model)
    # Pull sub-timings from tracker
    gsam2_load_s = getattr(getattr(gsam2_tracker, "_gsam2", None), "load_time_s",
                           getattr(gsam2_tracker, "load_time_s", 0.0))
    timer.record("model_load.gsam2", gsam2_load_s)
    if hasattr(gsam2_tracker, "last_detect_timings"):
        for k, v in gsam2_tracker.last_detect_timings.items():
            timer.record(f"perception.{k}", v)

    # 6a. Save detections.json so decomposer can annotate the scene image
    save_detections_to_snapshot(registry, perception_pool, snapshot_id)

    # 6b. Build collision meshes from the already-captured depth frame
    with timer.measure("perception.collision_mesh_build"):
        if primitives is not None and masks:
            try:
                from ptp.kinematics.depth_environment_collider import DepthEnvironmentCollider
                collider = DepthEnvironmentCollider(primitives._planner)
                snap_joints = (robot_state or {}).get("joints")
                if snap_joints is not None:
                    primitives._planner.set_current_joint_state(snap_joints)
                run_dir.mkdir(parents=True, exist_ok=True)
                result = collider.update_from_depth(
                    depth_m=depth.astype("float32"),
                    intrinsics=intrinsics,
                    masks=masks,
                    debug_dir=str(run_dir),
                )
                logger.info("Collision meshes built: %s", result)
                primitives._depth_collider = collider
                primitives._planner.attach_collider(collider)
            except Exception as exc:
                logger.warning("Could not build collision meshes: %s", exc)

    # 7. Build world state
    world_state = build_world_state(registry, snapshot_id)

    input("\nPerception complete — press Enter to continue to task parsing…\n")

    # 8. Build labeled scene image once — shared by task parser and decomposer.
    labeled_image_bytes = build_scene_labeled_image(color, registry)
    if labeled_image_bytes is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "planner_input.png").write_bytes(labeled_image_bytes)

    # 9. Parse task → grounded (action, object_id) sequence
    with timer.measure("task_parsing"):
        parse_result = parse_task(
            task=args.task,
            registry=registry,
            labeled_image_bytes=labeled_image_bytes,
            llm_model=args.llm_model,
            api_key=args.api_key,
            temperature=args.temperature,
        )
    (run_dir / "task_parse.json").parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_parse.json").write_text(
        json.dumps({
            "task": args.task,
            "rationale": parse_result.rationale,
            "actions": [
                {"action": a.action, "object_id": a.object_id, "description": a.description}
                for a in parse_result.actions
            ],
        }, indent=2)
    )

    # 9. Decompose each parsed action into primitives
    from ptp.primitives.types import SkillPlan
    plans: List[Any] = []
    with timer.measure("decomposition"):
        for i, parsed_action in enumerate(parse_result.actions):
            with timer.measure(f"decomposition.action[{i}]"):
                sub_plan = decompose_task(
                    parsed_action=parsed_action,
                    world_state=world_state,
                    llm_model=args.llm_model,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    perception_pool=perception_pool,
                    labeled_image_bytes=labeled_image_bytes,
                )
            plans.append(sub_plan)

    # Merge all sub-plans into a single plan for execution and output.
    plan = plans[0] if plans else SkillPlan(action_name=args.task)
    for sub in plans[1:]:
        plan.primitives.extend(sub.primitives)
        plan.diagnostics.warnings.extend(sub.diagnostics.warnings)
        plan.diagnostics.assumptions.extend(sub.diagnostics.assumptions)
        plan.diagnostics.freshness_notes.extend(sub.diagnostics.freshness_notes)

    # 10. Execute
    with timer.measure("execution.total"):
        execution_result = execute_plan(plan, primitives, molmo, world_state, perception_pool,
                                        dry_run=args.no_execute, masks=masks)
    # Merge per-primitive timings from executor
    for k, v in (execution_result.timings or {}).items():
        timer.record(f"execution.{k}", v)

    timer.log_summary(logger)

    # 11. Save run outputs
    save_run_outputs(
        run_dir=run_dir,
        color=color,
        task=args.task,
        snapshot_id=snapshot_id,
        registry=registry,
        plan=plan,
        execution_result=execution_result,
        timings=timer.to_dict(),
    )

    # Legacy --output-plan path
    if args.output_plan:
        Path(args.output_plan).write_text(json.dumps(plan.to_dict(), indent=2))
        logger.info("Plan also written to %s", args.output_plan)


if __name__ == "__main__":
    main()
