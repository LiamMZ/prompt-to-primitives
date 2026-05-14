"""test_sam3d.py — Test the SAM3D server with a live RealSense frame or saved images.

Sends an RGB image + segmentation mask to the SAM3D server and visualises the
reconstructed mesh using Open3D.

Usage — live RealSense (captures one frame, segments the central region):
    python scripts/test_sam3d.py --server http://192.168.0.88:8766

Usage — saved images:
    python scripts/test_sam3d.py --server http://192.168.0.88:8766 \\
        --rgb-path /tmp/rgb.png --mask-path /tmp/mask.png

Usage — synthetic test (no camera or images needed):
    python scripts/test_sam3d.py --server http://192.168.0.88:8766 --synthetic
"""

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("test_sam3d")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test SAM3D server")
    p.add_argument("--server", default="http://192.168.0.88:8766",
                   help="SAM3D server URL (default: http://192.168.0.88:8766)")
    p.add_argument("--rgb-path", help="Path to RGB image (PNG/JPG)")
    p.add_argument("--mask-path", help="Path to binary mask image (white=object)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use a synthetic RGB image and centre-crop mask (no camera needed)")
    p.add_argument("--no-viz", action="store_true", help="Skip Open3D visualisation")
    return p.parse_args()


def load_realsense():
    import pyrealsense2 as rs
    import numpy as np
    from PIL import Image

    logger.info("Connecting to RealSense …")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipe.start(cfg)
    try:
        for _ in range(30):  # warm up
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        color = np.asarray(color_frame.get_data())
    finally:
        pipe.stop()

    logger.info("Captured frame: %s", color.shape)
    h, w = color.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True  # centre crop
    return color, mask


def load_from_files(rgb_path: str, mask_path: str):
    import numpy as np
    from PIL import Image

    color = np.array(Image.open(rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    logger.info("Loaded rgb=%s mask=%s", color.shape, mask.shape)
    return color, mask


def make_synthetic():
    import numpy as np

    logger.info("Generating synthetic test image …")
    h, w = 480, 640
    color = np.zeros((h, w, 3), dtype=np.uint8)
    # Simple gradient so it's not entirely uniform
    color[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)
    color[:, :, 1] = np.linspace(80, 150, h, dtype=np.uint8).reshape(-1, 1)
    color[:, :, 2] = 120
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return color, mask


def main() -> None:
    args = parse_args()

    # --- health check ---
    from ptp.perception.sam3d_meshifier import Sam3DMeshifier
    import numpy as np

    meshifier = Sam3DMeshifier(server_url=args.server)
    logger.info("Health-checking %s …", args.server)
    try:
        meshifier.load()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
    logger.info("Server is up.")

    # --- acquire image + mask ---
    if args.synthetic:
        color, mask = make_synthetic()
    elif args.rgb_path and args.mask_path:
        color, mask = load_from_files(args.rgb_path, args.mask_path)
    else:
        try:
            color, mask = load_realsense()
        except Exception as e:
            logger.error("RealSense failed (%s). Use --synthetic or --rgb-path/--mask-path.", e)
            sys.exit(1)

    logger.info("Mask covers %d px (%.1f%% of image)",
                mask.sum(), 100 * mask.mean())

    # --- reconstruct ---
    T_base_cam = np.eye(4)  # identity — mesh will be in camera frame
    logger.info("Sending reconstruct request …")
    mesh = meshifier.reconstruct(color, mask, T_base_cam)

    if mesh is None:
        logger.error("Reconstruction returned None — check server logs.")
        sys.exit(1)

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    logger.info("Mesh received: %d vertices, %d triangles", n_verts, n_tris)

    if not args.no_viz:
        import open3d as o3d
        logger.info("Launching Open3D viewer (close window to exit) …")
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="SAM3D reconstruction",
            width=1024,
            height=768,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
