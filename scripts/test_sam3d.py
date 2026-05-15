"""test_sam3d.py — End-to-end SAM3D test: detect → reconstruct → PyBullet.

Captures a RealSense frame, runs full GSAM2 detection, sends each object to
the SAM3D server for mesh reconstruction, scales each mesh to match the
depth-derived real-world size, positions it at the depth-derived centroid,
and loads everything into a PyBullet GUI scene.

Usage:
    uv run python scripts/test_sam3d.py
    uv run python scripts/test_sam3d.py --server http://192.168.0.88:8766
    uv run python scripts/test_sam3d.py --no-viz     # skip PyBullet GUI
    uv run python scripts/test_sam3d.py --synthetic  # no camera needed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("test_sam3d")

_PALETTE = [
    (1.0, 0.3, 0.3, 0.7),
    (0.3, 1.0, 0.3, 0.7),
    (0.3, 0.5, 1.0, 0.7),
    (1.0, 0.8, 0.2, 0.7),
    (0.8, 0.3, 1.0, 0.7),
    (0.2, 0.9, 0.9, 0.7),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test SAM3D end-to-end with PyBullet")
    p.add_argument("--server",
                   default=os.environ.get("SAM3D_SERVER", "http://192.168.0.88:8766"),
                   help="SAM3D server URL (default: $SAM3D_SERVER)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use a synthetic image + centre-crop mask (no camera needed)")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip PyBullet GUI and matplotlib display")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def capture_realsense():
    from ptp.camera.realsense_camera import RealSenseCamera

    logger.info("Connecting to RealSense …")
    with RealSenseCamera(width=640, height=480, fps=30) as cam:
        for _ in range(30):
            cam.get_aligned_frames()
        color, depth = cam.get_aligned_frames()
        intrinsics = cam.get_camera_intrinsics()
    logger.info("Captured frame: %s", color.shape)
    return color, depth, intrinsics


def make_synthetic():
    import numpy as np
    from collections import namedtuple

    logger.info("Generating synthetic image …")
    h, w = 480, 640
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)
    color[:, :, 1] = np.linspace(80, 150, h, dtype=np.uint8).reshape(-1, 1)
    color[:, :, 2] = 120

    depth = np.zeros((h, w), dtype=np.float32)
    depth[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 0.6  # 0.6 m away

    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True

    CameraIntrinsics = namedtuple("CameraIntrinsics", ["fx", "fy", "cx", "cy", "width", "height"])
    intrinsics = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=w, height=h)

    return color, depth, intrinsics, {"synthetic_object": mask}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_objects(color, depth, intrinsics):
    from ptp.perception.gsam2.gsam2_tracker import GSAM2ObjectTracker
    from ptp.llm_interface.openai_client import OpenAIClient

    llm_client = OpenAIClient(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    tracker = GSAM2ObjectTracker(
        grounding_model_id=os.environ.get("DINO_CKPT", "IDEA-Research/grounding-dino-base"),
        llm_client=llm_client,
        compute_clearances=False,
        compute_contacts=True,
        compute_occlusion=False,
    )

    logger.info("Running GSAM2 detection …")
    asyncio.run(tracker.detect_objects(color, depth, intrinsics))

    objects = tracker.registry.get_all_objects()
    masks = {obj.object_id: tracker._last_masks[obj.object_id]
             for obj in objects if obj.object_id in tracker._last_masks}
    logger.info("Detected %d objects: %s", len(objects), [o.object_id for o in objects])
    return objects, masks


# ---------------------------------------------------------------------------
# Annotated image display
# ---------------------------------------------------------------------------

def show_annotated(color, objects):
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    img = Image.fromarray(color)
    h, w = color.shape[:2]
    draw = ImageDraw.Draw(img, "RGBA")
    for i, obj in enumerate(objects):
        bbox = getattr(obj, "bounding_box_2d", None)
        if not bbox or len(bbox) < 4:
            continue
        ny1, nx1, ny2, nx2 = bbox
        x1, y1 = int(nx1 * w / 1000), int(ny1 * h / 1000)
        x2, y2 = int(nx2 * w / 1000), int(ny2 * h / 1000)
        r, g, b, _ = _PALETTE[i % len(_PALETTE)]
        colour = (int(r * 255), int(g * 255), int(b * 255))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2, fill=(*colour, 60))
        draw.text((x1 + 4, y1 + 4), obj.object_id, fill=(255, 255, 255))

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title("RealSense — detected objects")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# ---------------------------------------------------------------------------
# Depth back-projection → centroid + scale
# ---------------------------------------------------------------------------

def depth_centroid_and_extent(depth, mask, intrinsics):
    """Return (centroid_xyz, extent_xyz) in camera frame for the masked region."""
    import numpy as np

    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    ys, xs = np.where(mask & (depth > 0.05) & (depth < 5.0))
    if len(ys) == 0:
        return None, None

    d = depth[ys, xs].astype(float)
    x = (xs - cx) * d / fx
    y = (ys - cy) * d / fy
    pts = np.stack([x, y, d], axis=1)

    centroid = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return centroid, extent


# ---------------------------------------------------------------------------
# Scale + position mesh
# ---------------------------------------------------------------------------

def fit_mesh_to_depth(mesh, centroid, extent):
    """Scale the SAM3D mesh to match the depth extent, centre at depth centroid."""
    import numpy as np

    verts = np.asarray(mesh.vertices)
    mesh_extent = verts.max(axis=0) - verts.min(axis=0)

    # Avoid division by zero
    safe_mesh = np.where(mesh_extent > 1e-6, mesh_extent, 1.0)
    safe_depth = np.where(extent > 1e-6, extent, safe_mesh)

    # Use the median axis scale to avoid distortion from noisy single axes
    per_axis = safe_depth / safe_mesh
    scale = float(np.median(per_axis))
    scale = max(0.01, min(scale, 2.0))  # clamp to sane range

    # Centre mesh at origin, scale, translate to centroid
    mesh_centre = (verts.max(axis=0) + verts.min(axis=0)) / 2
    mesh.translate(-mesh_centre)
    mesh.scale(scale, center=[0, 0, 0])
    mesh.translate(centroid)
    return mesh


# ---------------------------------------------------------------------------
# PyBullet loading
# ---------------------------------------------------------------------------

def load_into_pybullet(meshes_positioned: dict) -> None:
    import pybullet as p
    import pybullet_data
    import numpy as np

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    for i, (obj_id, mesh) in enumerate(meshes_positioned.items()):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        if len(verts) == 0 or len(tris) == 0:
            continue

        r, g, b, a = _PALETTE[i % len(_PALETTE)]
        col = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=verts.tolist(),
            indices=tris.flatten().tolist(),
            physicsClientId=client,
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            vertices=verts.tolist(),
            indices=tris.flatten().tolist(),
            rgbaColor=[r, g, b, a],
            physicsClientId=client,
        )
        centroid = verts.mean(axis=0).tolist()
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=centroid,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=client,
        )
        logger.info("PyBullet: loaded %s at %.3f,%.3f,%.3f  (%d tris)",
                    obj_id, *centroid, len(tris))

    logger.info("PyBullet GUI open — press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=client)
    except KeyboardInterrupt:
        pass
    p.disconnect(client)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    import numpy as np
    from ptp.perception.sam3d_meshifier import Sam3DMeshifier

    meshifier = Sam3DMeshifier(server_url=args.server)
    logger.info("Health-checking %s …", args.server)
    try:
        meshifier.load()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
    logger.info("Server is up.")

    if args.synthetic:
        color, depth, intrinsics, masks = make_synthetic()
        objects = []
    else:
        color, depth, intrinsics = capture_realsense()
        objects, masks = detect_objects(color, depth, intrinsics)
        if not args.no_viz:
            show_annotated(color, objects)

    if not masks:
        logger.error("No objects detected.")
        sys.exit(1)

    # Compute depth-derived centroid + extent per object
    centroids, extents = {}, {}
    for obj_id, mask in masks.items():
        c, e = depth_centroid_and_extent(depth, mask, intrinsics)
        if c is not None:
            centroids[obj_id] = c
            extents[obj_id] = e
        else:
            logger.warning("No valid depth for %s — skipping", obj_id)

    valid_masks = {k: v for k, v in masks.items() if k in centroids}
    if not valid_masks:
        logger.error("No objects with valid depth.")
        sys.exit(1)

    T_base_cam = np.eye(4)
    logger.info("Reconstructing %d object(s) via SAM3D …", len(valid_masks))
    meshes = meshifier.reconstruct_all(color, valid_masks, T_base_cam)

    # Scale and position each mesh
    meshes_positioned = {}
    for obj_id, mesh in meshes.items():
        if mesh is None:
            logger.warning("SAM3D failed for %s", obj_id)
            continue
        mesh.compute_vertex_normals()
        fit_mesh_to_depth(mesh, centroids[obj_id], extents[obj_id])
        meshes_positioned[obj_id] = mesh
        logger.info("  %s: centroid=%.3f,%.3f,%.3f  extent=%.3f,%.3f,%.3f",
                    obj_id, *centroids[obj_id], *extents[obj_id])

    if not meshes_positioned:
        logger.error("No meshes to load.")
        sys.exit(1)

    if not args.no_viz:
        load_into_pybullet(meshes_positioned)
    else:
        logger.info("Skipping PyBullet GUI (--no-viz).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
