"""run_ptp.py — Prompt-to-Primitives entry point (no PDDL/orchestrator).

Pipeline:
    1. Connect to xArm + initialise PyBullet FK/IK planner
    2. Capture RealSense RGB-D frame
    3. GSAM2: detect and segment objects → DetectedObjectRegistry
    4. Molmo: get interaction points for each detected object
    5. User provides a task string
    6. SkillDecomposer: task → SkillPlan (LLM-backed primitive decomposition)
    7. PrimitiveExecutor: translate pixel/camera coords → base frame, then execute

Run without real hardware:
    python scripts/run_ptp.py --no-execute --task "pick up the cup"

Run with a real xArm + RealSense:
    python scripts/run_ptp.py --task "push the red block to the left"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on the path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptp.utils.logging_utils import get_structured_logger

logger = get_structured_logger("run_ptp")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt-to-Primitives pipeline runner")
    p.add_argument("--task", required=True, help="Natural language task description")
    p.add_argument("--no-execute", action="store_true", help="Plan but do not send commands to robot")
    p.add_argument("--robot-ip", default="192.168.1.224", help="xArm IP address")
    p.add_argument("--llm-model", default="gemini-2.0-flash", help="LLM model for decomposition")
    p.add_argument("--api-key", default=None, help="API key for the LLM provider (or set env var)")
    p.add_argument("--perception-pool", default="/tmp/ptp_perception_pool",
                   help="Path to perception pool directory for snapshots")
    p.add_argument("--gsam2-model", default="IDEA-Research/grounding-dino-tiny",
                   help="GroundingDINO model name for GSAM2")
    p.add_argument("--object-classes", nargs="+", default=["cup", "block", "bottle", "bowl"],
                   help="Object classes to detect")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="LLM sampling temperature for decomposition")
    p.add_argument("--output-plan", default=None,
                   help="Path to write the generated SkillPlan JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_primitives_interface(robot_ip: str, dry_run: bool) -> Optional[object]:
    """Connect to xArm and build the planned-primitives interface."""
    if dry_run:
        logger.info("Dry-run mode: no robot interface created")
        return None
    try:
        from ptp.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
        logger.info("Connecting to xArm at %s …", robot_ip)
        primitives = XArmPybulletPlannedPrimitives(robot_ip=robot_ip)
        logger.info("xArm connected")
        return primitives
    except Exception as exc:
        logger.error("Failed to connect to robot: %s", exc)
        raise


def capture_frame(primitives: Optional[object]):
    """Capture an RGB-D frame from the wrist-mounted RealSense camera."""
    import numpy as np
    from ptp.camera import REALSENSE_AVAILABLE, CameraIntrinsics

    if REALSENSE_AVAILABLE:
        from ptp.camera.realsense_camera import RealSenseCamera
        with RealSenseCamera(width=640, height=480, fps=30) as cam:
            color, depth = cam.get_aligned_frames()
            intrinsics = cam.get_camera_intrinsics()
        logger.info("Captured RealSense frame: %dx%d", color.shape[1], color.shape[0])
        return color, depth, intrinsics
    else:
        logger.warning("pyrealsense2 not available — using synthetic blank frame")
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
        return color, depth, intrinsics


def detect_objects(color, depth, intrinsics, object_classes, gsam2_model: str):
    """Run GSAM2 segmentation and Molmo interaction-point grounding."""
    from ptp.perception.object_registry import DetectedObjectRegistry, DetectedObject, InteractionPoint
    from ptp.perception.molmo_point_detector import MolmoPointDetector

    registry = DetectedObjectRegistry()

    try:
        from ptp.perception.gsam2.gsam2_tracker import GSAM2ContinuousObjectTracker
        logger.info("Running GSAM2 detection for classes: %s", object_classes)
        tracker = GSAM2ContinuousObjectTracker(
            classes=object_classes,
            grounding_model=gsam2_model,
        )
        detections = tracker.detect_frame(color)
        logger.info("GSAM2 detected %d objects", len(detections))

        molmo = MolmoPointDetector()
        for det in detections:
            obj_id = registry.generate_unique_id(det.label)
            ip_results = molmo.get_interaction_points(
                rgb_image=color,
                depth_frame=depth,
                camera_intrinsics=intrinsics,
                object_id=obj_id,
                object_type=det.label,
                bounding_box_2d=det.bbox,
                actions={"grasp", "push"},
            )
            obj = DetectedObject(
                object_type=det.label,
                object_id=obj_id,
                interaction_points=ip_results,
                bounding_box_2d=det.bbox,
            )
            registry.add_object(obj)
            logger.info("Registered %s (%s) with %d interaction points", det.label, obj_id, len(ip_results))

    except Exception as exc:
        logger.warning("Object detection failed (%s) — using empty registry", exc)

    return registry


def build_world_state(registry, depth, intrinsics, snapshot_id: str = "run0") -> dict:
    """Assemble the world_state dict consumed by the decomposer and executor."""
    import io, numpy as np
    from PIL import Image

    world_state: dict = {
        "last_snapshot_id": snapshot_id,
        "object_registry": registry.to_dict() if registry else {},
    }
    return world_state


def decompose_task(task: str, world_state: dict, llm_model: str, api_key: Optional[str], temperature: float):
    """Run the LLM decomposer to produce a SkillPlan."""
    from ptp.primitives.decomposer import SkillDecomposer
    from ptp.llm_interface.google_genai import GoogleGenAIClient

    llm = GoogleGenAIClient(model=llm_model, api_key=api_key)
    decomposer = SkillDecomposer(llm_client=llm)
    logger.info("Decomposing task: %r", task)
    plan = decomposer.plan(
        action_name=task,
        parameters={},
        world_hint=world_state,
        temperature=temperature,
    )
    logger.info("Plan has %d primitives", len(plan.primitives))
    for i, p in enumerate(plan.primitives):
        logger.info("  [%d] %s %s", i, p.name, p.parameters)
    return plan


def execute_plan(plan, primitives, world_state: dict, perception_pool: Path, dry_run: bool):
    """Translate and (optionally) execute the plan."""
    from ptp.primitives.executor import PrimitiveExecutor

    executor = PrimitiveExecutor(
        primitives=primitives,
        perception_pool_dir=perception_pool,
    )
    result = executor.execute_plan(plan, world_state, dry_run=dry_run)
    if result.executed:
        ok = sum(1 for r in result.executed_primitives if r.success)
        logger.info("Execution complete: %d/%d primitives succeeded", ok, len(result.executed_primitives))
    else:
        logger.info("Plan translated (dry-run); no execution")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    perception_pool = Path(args.perception_pool)
    perception_pool.mkdir(parents=True, exist_ok=True)

    # 1. Robot interface
    primitives = build_primitives_interface(args.robot_ip, dry_run=args.no_execute)

    # 2. Capture frame
    color, depth, intrinsics = capture_frame(primitives)

    # 3. Detect objects + get interaction points
    registry = detect_objects(color, depth, intrinsics, args.object_classes, args.gsam2_model)

    # 4. Build world state
    world_state = build_world_state(registry, depth, intrinsics)

    # 5. Decompose task
    plan = decompose_task(
        task=args.task,
        world_state=world_state,
        llm_model=args.llm_model,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    # 6. Optionally save plan
    if args.output_plan:
        Path(args.output_plan).write_text(json.dumps(plan.to_dict(), indent=2))
        logger.info("Plan written to %s", args.output_plan)

    # 7. Execute
    execute_plan(plan, primitives, world_state, perception_pool, dry_run=args.no_execute)


if __name__ == "__main__":
    main()
