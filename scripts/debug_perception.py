"""debug_perception.py — Run the vision pipeline and print object locations in robot base frame.

Captures a single RGB-D frame, runs GSAM2 detection, back-projects each object
centroid through the camera intrinsics and FK camera-to-base transform, and prints
a table of object positions in the robot base frame.  Saves an annotated image and
a JSON summary to outputs/debug_perception/.

Usage:
    # With real hardware:
    uv run python scripts/debug_perception.py

    # Dry-run (synthetic blank frame, no robot):
    uv run python scripts/debug_perception.py --no-execute

    # Override model / API key:
    uv run python scripts/debug_perception.py --gsam2-model IDEA-Research/grounding-dino-tiny
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision pipeline debug: object locations in robot frame")
    p.add_argument("--no-execute", action="store_true",
                   help="Skip real robot and camera — use synthetic blank frame")
    p.add_argument("--robot-ip", default="192.168.1.224")
    p.add_argument("--gsam2-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--api-key", default=None)
    p.add_argument("--task", default=None,
                   help="Optional task hint to inject object noun phrases into GSAM2")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "debug_perception"))
    p.add_argument("--no-pybullet", action="store_true",
                   help="Skip PyBullet GUI (collision mesh visualisation)")
    p.add_argument("--sam3d-server", default=None, metavar="URL",
                   help="SAM3D server URL for mesh reconstruction (e.g. http://192.168.0.88:8766)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Camera + robot
# ---------------------------------------------------------------------------

def open_camera():
    from ptp.camera import REALSENSE_AVAILABLE
    if not REALSENSE_AVAILABLE:
        return None
    from ptp.camera.realsense_camera import RealSenseCamera
    cam = RealSenseCamera(width=640, height=480, fps=30)
    return cam


def capture_frame(camera):
    from ptp.camera import CameraIntrinsics
    if camera is not None:
        color, depth = camera.get_aligned_frames()
        intrinsics = camera.get_camera_intrinsics()
    else:
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    return color, depth, intrinsics


def build_robot(robot_ip: str, use_gui: bool = False):
    from ptp.kinematics.xarm_robot_adapter import XArmRobotAdapter
    from ptp.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
    robot = XArmRobotAdapter(robot_ip)
    return XArmPybulletPlannedPrimitives(robot=robot, use_gui=use_gui)


def get_camera_to_base(primitives) -> Optional[np.ndarray]:
    """Return the 4×4 camera-to-base transform from FK, or None."""
    cam_pos, cam_rot = primitives._planner.get_camera_transform()
    if cam_pos is None or cam_rot is None:
        return None
    T = np.eye(4)
    T[:3, :3] = cam_rot.as_matrix()
    T[:3, 3] = cam_pos
    return T


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(color, depth, intrinsics, gsam2_model: str, llm_client, task: Optional[str]):
    from ptp.perception.gsam2.gsam2_tracker import GSAM2ObjectTracker, _extract_noun_phrases

    tracker = GSAM2ObjectTracker(
        grounding_model_id=gsam2_model,
        llm_client=llm_client,
        compute_clearances=False,
        compute_contacts=False,
        compute_occlusion=False,
    )
    if task:
        tracker.set_task_description(task)
        hints = _extract_noun_phrases(task)
        if hints:
            tracker.set_extra_tags(hints)
    asyncio.get_event_loop().run_until_complete(
        tracker.detect_objects(color, depth, intrinsics)
    )
    return tracker.registry, dict(tracker._last_masks)


# ---------------------------------------------------------------------------
# 3-D localisation
# ---------------------------------------------------------------------------

def localise_objects(registry, masks, depth, intrinsics, T_base_cam: Optional[np.ndarray]):
    """Back-project each object's mask centroid to 3D and optionally transform to base frame."""
    from ptp.perception.utils.coordinates import compute_3d_position_masked

    results = []
    for obj in registry.get_all_objects():
        mask = masks.get(obj.object_id)
        pos_cam = None
        pos_base = None

        if obj.position_2d is not None and mask is not None:
            pos_cam = compute_3d_position_masked(
                obj.position_2d, depth, intrinsics, object_mask=mask
            )

        if pos_cam is not None and T_base_cam is not None:
            p_h = np.array([*pos_cam, 1.0])
            pos_base = (T_base_cam @ p_h)[:3]

        results.append({
            "object_id": obj.object_id,
            "object_type": obj.object_type,
            "position_2d": obj.position_2d,
            "position_cam_xyz_m": pos_cam.tolist() if pos_cam is not None else None,
            "position_base_xyz_m": pos_base.tolist() if pos_base is not None else None,
        })
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def annotate_image(color, registry, masks, locations):
    """Draw bounding boxes, centroids, and base-frame XYZ on the image."""
    try:
        import cv2
    except ImportError:
        return color

    img = color.copy()
    h, w = img.shape[:2]

    loc_by_id = {r["object_id"]: r for r in locations}

    colours = [
        (80, 200, 80), (80, 80, 220), (220, 160, 40),
        (40, 220, 220), (200, 80, 200), (80, 200, 200),
    ]

    for idx, obj in enumerate(registry.get_all_objects()):
        colour = colours[idx % len(colours)]
        loc = loc_by_id.get(obj.object_id, {})

        # Bounding box
        bb = obj.bounding_box_2d
        if bb is not None:
            # bb is [y1, x1, y2, x2] in 0-1000 scale
            y1 = int(bb[0] / 1000 * h)
            x1 = int(bb[1] / 1000 * w)
            y2 = int(bb[2] / 1000 * h)
            x2 = int(bb[3] / 1000 * w)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # Mask overlay
        mask = masks.get(obj.object_id)
        if mask is not None and mask.shape == (h, w):
            tint = np.zeros_like(img)
            tint[mask.astype(bool)] = colour
            img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)

        # Label + base-frame position
        pos_base = loc.get("position_base_xyz_m")
        pos_cam = loc.get("position_cam_xyz_m")
        if bb is not None:
            label_x = int(bb[1] / 1000 * w)
            label_y = max(0, int(bb[0] / 1000 * h) - 6)
        elif obj.position_2d is not None:
            label_x = int(obj.position_2d[1] / 1000 * w)
            label_y = max(0, int(obj.position_2d[0] / 1000 * h) - 6)
        else:
            continue

        text = obj.object_id
        if pos_base is not None:
            text += f"  base=({pos_base[0]:.2f},{pos_base[1]:.2f},{pos_base[2]:.2f})"
        elif pos_cam is not None:
            text += f"  cam=({pos_cam[0]:.2f},{pos_cam[1]:.2f},{pos_cam[2]:.2f})"

        font_scale = max(0.35, w / 1600)
        cv2.putText(img, text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    from ptp.llm_interface.openai_client import OpenAIClient
    llm_client = OpenAIClient(model=args.llm_model, api_key=args.api_key or None)

    # --- Robot / camera --------------------------------------------------
    primitives = None
    camera = None
    T_base_cam = None

    robot_state = None
    if not args.no_execute:
        print("Connecting to robot…")
        primitives = build_robot(args.robot_ip, use_gui=not args.no_pybullet)

        # Read joints directly from hardware first so we can see what the adapter returns.
        raw_joints = primitives._robot.get_robot_joint_state()
        print(f"Raw joints from hardware adapter: {raw_joints}")

        # sync_planner_to_real_robot is called inside get_robot_state — this also
        # sets the joints in the planner.
        robot_state = primitives.get_robot_state()
        joints = robot_state.get("joints")
        print(f"Robot state joints (from planner after sync): {joints}")

        if joints is not None:
            primitives._planner.set_current_joint_state(joints)
            # _apply_joints_to_sim must be called explicitly in DIRECT mode since
            # set_current_joint_state only does it when use_gui=True.
            primitives._planner._apply_joints_to_sim()
            print("Joint state applied to sim.")
        else:
            print("WARNING: no joint state returned — check robot connection")

        T_base_cam = get_camera_to_base(primitives)
        if T_base_cam is not None:
            print(f"Camera position in base frame: {T_base_cam[:3, 3]}")
        else:
            print("WARNING: could not read camera-to-base transform from FK")

        print("Opening camera…")
        camera = open_camera()

    print("Capturing frame…")
    color, depth, intrinsics = capture_frame(camera)
    print(f"Frame: {color.shape[1]}x{color.shape[0]}  "
          f"depth range [{depth[depth > 0].min():.3f}, {depth.max():.3f}] m"
          if depth.any() else f"Frame: {color.shape[1]}x{color.shape[0]}  (blank)")

    # --- Detection -------------------------------------------------------
    print("Running GSAM2 detection…")
    registry, masks = detect(color, depth, intrinsics, args.gsam2_model, llm_client, args.task)
    detected = registry.get_all_objects()
    print(f"Detected {len(detected)} objects: {[o.object_id for o in detected]}")

    # --- Localise --------------------------------------------------------
    locations = localise_objects(registry, masks, depth, intrinsics, T_base_cam)

    # --- Print table -----------------------------------------------------
    print()
    print(f"{'Object ID':<30} {'2D [y,x]':<16} {'Camera XYZ (m)':<30} {'Base XYZ (m)':<30}")
    print("-" * 106)
    for loc in locations:
        p2 = str(loc["position_2d"]) if loc["position_2d"] else "—"
        pc = (f"[{loc['position_cam_xyz_m'][0]:.3f}, "
              f"{loc['position_cam_xyz_m'][1]:.3f}, "
              f"{loc['position_cam_xyz_m'][2]:.3f}]"
              if loc["position_cam_xyz_m"] else "—")
        pb = (f"[{loc['position_base_xyz_m'][0]:.3f}, "
              f"{loc['position_base_xyz_m'][1]:.3f}, "
              f"{loc['position_base_xyz_m'][2]:.3f}]"
              if loc["position_base_xyz_m"] else "—  (no FK transform)")
        print(f"{loc['object_id']:<30} {p2:<16} {pc:<30} {pb:<30}")
    print()

    # --- Annotate + display image ----------------------------------------
    ann = annotate_image(color, registry, masks, locations)

    try:
        import cv2
        ann_bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
        cv2.imshow("debug_perception — press any key to continue", ann_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"cv2 display unavailable: {exc}")

    # --- PyBullet collision mesh visualisation ---------------------------
    if primitives is not None and masks and not args.no_pybullet:
        print("Building collision meshes in PyBullet GUI…")
        try:
            from ptp.kinematics.depth_environment_collider import DepthEnvironmentCollider
            from ptp.perception.sam3d_meshifier import Sam3DMeshifier

            sam3d = None
            if args.sam3d_server:
                sam3d = Sam3DMeshifier(server_url=args.sam3d_server)
                sam3d.load()
                print(f"Using SAM3D server at {args.sam3d_server}")

            collider = DepthEnvironmentCollider(primitives._planner, sam3d=sam3d)
            result = collider.update_from_depth(
                depth_m=depth.astype("float32"),
                intrinsics=intrinsics,
                masks=masks,
                color_image=color,
            )
            print(f"Collision meshes built: {result}")

            # Draw a small sphere at each object's base-frame position.
            try:
                import pybullet as p
                client = primitives._planner._physics_client
                sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.015,
                                                      physicsClientId=client)
                colours = [
                    [1.0, 0.3, 0.3, 1.0], [0.3, 1.0, 0.3, 1.0], [0.3, 0.6, 1.0, 1.0],
                    [1.0, 0.8, 0.2, 1.0], [0.8, 0.3, 1.0, 1.0], [0.2, 0.9, 0.9, 1.0],
                ]
                for idx, loc in enumerate(locations):
                    pb = loc.get("position_base_xyz_m")
                    if pb is None:
                        continue
                    rgba = colours[idx % len(colours)]
                    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.015,
                                             rgbaColor=rgba, physicsClientId=client)
                    body = p.createMultiBody(baseMass=0,
                                             baseCollisionShapeIndex=sphere_shape,
                                             baseVisualShapeIndex=vis,
                                             basePosition=pb,
                                             physicsClientId=client)
                    # Label via a text annotation (PyBullet addUserDebugText)
                    p.addUserDebugText(
                        loc["object_id"],
                        [pb[0], pb[1], pb[2] + 0.04],
                        textColorRGB=[1, 1, 1],
                        textSize=1.2,
                        physicsClientId=client,
                    )
            except Exception as exc:
                print(f"Could not draw position spheres: {exc}")

            print("PyBullet GUI open — close the window or press Ctrl-C to exit.")
            try:
                import pybullet as p
                client = primitives._planner._physics_client
                while True:
                    p.stepSimulation(physicsClientId=client)
            except KeyboardInterrupt:
                pass
        except Exception as exc:
            print(f"PyBullet visualisation failed: {exc}")

    # --- Save outputs ----------------------------------------------------
    summary = {
        "timestamp": ts,
        "T_base_cam": T_base_cam.tolist() if T_base_cam is not None else None,
        "objects": locations,
    }
    json_path = out_dir / f"perception_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"JSON saved → {json_path}")

    try:
        from PIL import Image as _PIL
        img_path = out_dir / f"perception_{ts}.png"
        _PIL.fromarray(ann).save(img_path)
        print(f"Annotated image saved → {img_path}")
    except Exception as exc:
        print(f"Could not save annotated image: {exc}")

    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
