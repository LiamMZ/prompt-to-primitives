"""Intel RealSense D4xx camera implementation.

Provides hardware-aligned RGB-D capture with the full RealSense filter chain
(spatial, temporal, hole-filling) and robust startup retry/reset logic to
handle stale USB states that occasionally affect D435/D455 devices.

Usage:
    from ptp.camera.realsense_camera import RealSenseCamera

    with RealSenseCamera() as cam:
        color, depth = cam.get_aligned_frames()
        intrinsics   = cam.get_camera_intrinsics()
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None  # type: ignore

from ptp.camera.base_camera import BaseCamera, CameraFrame, CameraIntrinsics
from ptp.utils.logging_utils import get_structured_logger


class RealSenseCamera(BaseCamera):
    """Intel RealSense D4xx camera with RGB-D support.

    Applies a spatial → temporal → hole-filling depth filter chain before
    returning frames from get_aligned_frames().  Raw unfiltered depth is
    returned by get_depth() for cases where the filter chain is not wanted.

    Args:
        width: Frame width in pixels (default 640).
        height: Frame height in pixels (default 480).
        fps: Target capture frame rate (default 30).
        enable_depth: Enable depth stream (default True).
        auto_start: Connect and start streaming on construction (default True).
        logger: Logger instance; creates a structured logger if None.

    Raises:
        ImportError: If pyrealsense2 is not installed.
        RuntimeError: If the camera fails to start after all retries.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        auto_start: bool = True,
        logger: Optional[logging.Logger] = None,
        color_exposure: int = 250,
        depth_exposure: int = -1,
    ) -> None:
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2 is not installed. Install with: pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.color_exposure = color_exposure
        self.depth_exposure = depth_exposure
        self.logger = logger or get_structured_logger("RealSenseCamera")

        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.frame_count: int = 0
        self._start_time: Optional[float] = None

        # Depth filter chain — built in start()
        self.spatial: Optional[rs.spatial_filter] = None
        self.temporal: Optional[rs.temporal_filter] = None
        self.threshold: Optional[rs.threshold_filter] = None
        self.hole_filling: Optional[rs.hole_filling_filter] = None
        self.depth_to_disparity: Optional[rs.disparity_transform] = None
        self.disparity_to_depth: Optional[rs.disparity_transform] = None

        if auto_start:
            self.start()

    # ------------------------------------------------------------------
    # BaseCamera interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the RealSense pipeline and wait for first valid frame."""
        if self.pipeline is not None:
            self.logger.debug("Camera already started — skipping start()")
            return

        self.logger.info(
            "Starting RealSense camera: %dx%d @ %d FPS", self.width, self.height, self.fps
        )

        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.align = rs.align(rs.stream.color)

        self._build_filter_chain()

        _MAX_CYCLES = 3
        last_error: Optional[RuntimeError] = None
        reset_attempted = False

        for cycle in range(1, _MAX_CYCLES + 1):
            device_for_reset = None
            try:
                self.pipeline = rs.pipeline()
                self.profile = self.pipeline.start(self.config)
                self._start_time = time.time()
                self.logger.info("RealSense pipeline started (cycle %d/%d)", cycle, _MAX_CYCLES)
                self._set_fixed_exposure(self.color_exposure, self.depth_exposure)
                attempt = self._wait_for_initial_frames(max_retries=30, retry_delay=0.1, timeout_ms=1000)
                self.logger.info(
                    "Camera ready (first valid frame on attempt %d)", attempt
                )
                return
            except RuntimeError as exc:
                last_error = exc
                if self.profile is not None:
                    try:
                        device_for_reset = self.profile.get_device()
                    except Exception:
                        pass
                self.logger.warning("Startup cycle %d/%d failed: %s", cycle, _MAX_CYCLES, exc)
                self.stop()
                if cycle < _MAX_CYCLES:
                    if cycle == (_MAX_CYCLES - 1) and not reset_attempted:
                        reset_attempted = self._attempt_hardware_reset(device_for_reset)
                    else:
                        time.sleep(1.0)

        self.stop()
        raise RuntimeError(f"Failed to start RealSense camera: {last_error}")

    def stop(self) -> None:
        """Stop the pipeline and release all resources."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.profile = None
            self.logger.info("RealSense camera stopped")

    def capture_frame(self) -> np.ndarray:
        """Return a single RGB frame as (H, W, 3) uint8."""
        frames = self._wait_frames()
        if self.align and self.enable_depth:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame")
        self.frame_count += 1
        return np.asanyarray(color_frame.get_data())

    def get_depth(self) -> np.ndarray:
        """Return a raw (unfiltered) depth frame as (H, W) float32 metres."""
        if not self.enable_depth:
            raise NotImplementedError("Depth stream not enabled")
        frames = self._wait_frames()
        if self.align:
            frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("Failed to capture depth frame")
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (color, depth_metres) with the full depth filter chain applied.

        The filter pipeline is: threshold → disparity → spatial → temporal →
        disparity_inverse → hole_filling, which significantly reduces noise and
        fills small holes in the depth map.
        """
        if not self.enable_depth:
            raise RuntimeError("Depth stream not enabled")
        frames = self._wait_frames()
        if self.align:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        # Apply filter chain.
        depth_frame = self.threshold.process(depth_frame)
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        self.frame_count += 1
        return color, depth

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """Return color-stream camera intrinsics from the live device."""
        if self.profile is None:
            raise RuntimeError("Camera not started")
        color_stream = self.profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        distortion = np.array(intr.coeffs, dtype=np.float32) if intr.coeffs else None
        self.logger.debug(
            "Camera intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            intr.fx, intr.fy, intr.ppx, intr.ppy,
        )
        return CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
            distortion=distortion,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_fps(self) -> float:
        """Return actual capture FPS since start()."""
        if self._start_time is None or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    def is_depth_available(self) -> bool:
        return self.enable_depth

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_frames(self, timeout_ms: int = 5000) -> rs.composite_frame:
        if self.pipeline is None:
            raise RuntimeError("Camera not started")
        return self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

    def _build_filter_chain(self) -> None:
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 5)

        self.threshold = rs.threshold_filter(min_dist=0.2, max_dist=2.0)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.3)
        self.temporal.set_option(rs.option.filter_smooth_delta, 30)
        self.temporal.set_option(rs.option.holes_fill, 3)

        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)

    def _set_fixed_exposure(self, color_exposure: int, depth_exposure: int) -> None:
        """Set exposure for both sensors. A value of -1 leaves auto-exposure enabled."""
        for sensor_getter, label, value in [
            ("first_color_sensor", "color", color_exposure),
            ("first_depth_sensor", "depth", depth_exposure),
        ]:
            try:
                sensor = getattr(self.profile.get_device(), sensor_getter)()
                if value == -1:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    self.logger.info("%s sensor exposure set to auto", label)
                else:
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    sensor.set_option(rs.option.exposure, float(value))
                    self.logger.info("%s sensor exposure set to %d (auto-exposure disabled)", label, value)
            except Exception as exc:
                self.logger.warning("Could not set exposure for %s sensor: %s", label, exc)

    def _wait_for_initial_frames(
        self,
        *,
        max_retries: int,
        retry_delay: float,
        timeout_ms: int,
    ) -> int:
        """Poll until both color and depth frames are valid. Returns 1-indexed attempt number."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not started")
        for attempt in range(1, max_retries + 1):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                color_ok = bool(frames.get_color_frame())
                depth_ok = bool(frames.get_depth_frame()) if self.enable_depth else True
                if color_ok and depth_ok:
                    return attempt
            except RuntimeError:
                pass
            time.sleep(retry_delay)
        raise RuntimeError(f"Failed to get valid frames after {max_retries} attempts")

    def _attempt_hardware_reset(self, device: Optional[object] = None) -> bool:
        """Issue a hardware reset and wait for the device to re-enumerate."""
        target = device or self._get_first_visible_device()
        if target is None:
            self.logger.warning("No RealSense device found for hardware reset")
            time.sleep(1.0)
            return False
        try:
            serial = None
            try:
                serial = target.get_info(rs.camera_info.serial_number)
            except Exception:
                pass
            self.logger.warning("Hardware reset (serial=%s)", serial)
            target.hardware_reset()
            return self._wait_for_device_reconnect(serial, timeout_seconds=20.0)
        except Exception as exc:
            self.logger.warning("Hardware reset failed: %s", exc)
            time.sleep(1.0)
            return False

    def _get_first_visible_device(self) -> Optional[object]:
        try:
            devices = rs.context().query_devices()
            for i in range(len(devices)):
                try:
                    return devices[i]
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def _wait_for_device_reconnect(self, serial: Optional[str], timeout_seconds: float) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            device = self._get_first_visible_device()
            if device is None:
                time.sleep(0.5)
                continue
            if serial is None:
                return True
            try:
                if device.get_info(rs.camera_info.serial_number) == serial:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        self.logger.warning("Timed out waiting for device reconnect (serial=%s)", serial)
        return False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
