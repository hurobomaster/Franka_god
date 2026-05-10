#!/usr/bin/env python3
"""Validate camera devices and preview the actual input frames.

This script focuses on RealSense cameras and shows the processed image that
would be fed into downstream pipelines (resolution/fps/crop applied).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# =========================
# User-editable parameters
# =========================
# Camera runtime settings (applied to both cameras unless overridden per camera).
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

# Crop settings for the "input" preview.
# If ENABLE_CROP=False, input preview equals full frame.
ENABLE_CROP = True
CROP_X = 120
CROP_Y = 60
CROP_W = 400
CROP_H = 300

# Visualization options.
SHOW_RAW_PREVIEW = True
SHOW_DEPTH_PREVIEW = False
DEPTH_COLORMAP_MAX_METERS = 2.0

# Timeout for waiting camera frames.
WAIT_FRAMES_TIMEOUT_MS = 1200
WARN_INTERVAL_SEC = 2.0

# Camera serials (update here if needed).
CAMERA_SERIALS = {
    "d455": "327522300259",
    "d435": "153222071977",
}


try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("opencv-python is required. Install with: pip install opencv-python") from exc


def _extend_local_repo_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    for p in (root / "third_repo" / "franky", root / "third_repo" / "PySpaceMouse"):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.append(p_str)


try:
    import pyrealsense2 as rs
except Exception:
    _extend_local_repo_paths()
    try:
        import pyrealsense2 as rs
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyrealsense2 is required. Install Intel RealSense SDK Python bindings.") from exc


@dataclass
class CameraRuntime:
    name: str
    serial: str
    pipeline: rs.pipeline
    profile: rs.pipeline_profile
    frame_count: int = 0
    started_at: float = 0.0
    last_ok_t: float = 0.0
    last_warn_t: float = 0.0
    latest_color: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    latest_error: Optional[str] = None
    lock: threading.Lock = threading.Lock()


def _clamp_crop(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _apply_crop(color: np.ndarray) -> np.ndarray:
    if not ENABLE_CROP:
        return color
    img_h, img_w = color.shape[:2]
    x, y, w, h = _clamp_crop(CROP_X, CROP_Y, CROP_W, CROP_H, img_w, img_h)
    return color[y : y + h, x : x + w]


def _start_camera(name: str, serial: str) -> CameraRuntime:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
    if SHOW_DEPTH_PREVIEW:
        config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FRAME_FPS)

    profile = pipeline.start(config)

    runtime = CameraRuntime(
        name=name,
        serial=serial,
        pipeline=pipeline,
        profile=profile,
        started_at=time.perf_counter(),
        last_ok_t=time.perf_counter(),
        last_warn_t=0.0,
        latest_color=None,
        latest_depth=None,
        latest_error=None,
        lock=threading.Lock(),
    )
    return runtime


def _stop_camera(runtime: CameraRuntime) -> None:
    try:
        runtime.pipeline.stop()
    except Exception:
        pass


def _draw_overlay(img: np.ndarray, text_lines: Tuple[str, ...]) -> np.ndarray:
    out = img.copy()
    y = 24
    for line in text_lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
        y += 24
    return out


def _depth_to_colormap(depth_frame) -> np.ndarray:
    depth = np.asanyarray(depth_frame).astype(np.float32)
    max_m = max(0.1, float(DEPTH_COLORMAP_MAX_METERS))
    depth_m = depth / 1000.0
    depth_m = np.clip(depth_m / max_m, 0.0, 1.0)
    depth_u8 = (depth_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def _camera_worker(rt: CameraRuntime, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            frames = rt.pipeline.wait_for_frames(timeout_ms=WAIT_FRAMES_TIMEOUT_MS)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = None
            if SHOW_DEPTH_PREVIEW:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data())

            with rt.lock:
                rt.latest_color = color
                rt.latest_depth = depth
                rt.frame_count += 1
                rt.last_ok_t = time.perf_counter()
                rt.latest_error = None
        except Exception as exc:
            now = time.perf_counter()
            with rt.lock:
                rt.latest_error = str(exc)
                if now - rt.last_warn_t >= WARN_INTERVAL_SEC:
                    print(f"[WARN] {rt.name} frame read error: {exc}")
                    rt.last_warn_t = now


def run_validation() -> int:
    runtimes: Dict[str, CameraRuntime] = {}
    stop_event = threading.Event()
    workers: Dict[str, threading.Thread] = {}

    print("[VALIDATION] Starting camera validation")
    print(f"[CONFIG] size={FRAME_WIDTH}x{FRAME_HEIGHT}, fps={FRAME_FPS}")
    print(f"[CONFIG] crop={'ON' if ENABLE_CROP else 'OFF'} x={CROP_X}, y={CROP_Y}, w={CROP_W}, h={CROP_H}")

    try:
        for name, serial in CAMERA_SERIALS.items():
            try:
                rt = _start_camera(name, serial)
                runtimes[name] = rt
                print(f"[OK] {name} started (serial={serial})")
            except Exception as exc:
                print(f"[FAIL] {name} failed to start (serial={serial}): {exc}")

        if not runtimes:
            print("[ERROR] No cameras started. Check serial numbers and USB connection.")
            return 2

        for name, rt in runtimes.items():
            worker = threading.Thread(target=_camera_worker, args=(rt, stop_event), name=f"{name}-worker", daemon=True)
            workers[name] = worker
            worker.start()

        print("[RUN] Press 'q' to quit")

        while True:
            any_frame = False
            for name, rt in runtimes.items():
                with rt.lock:
                    color = None if rt.latest_color is None else rt.latest_color.copy()
                    depth = None if rt.latest_depth is None else rt.latest_depth.copy()
                    frame_count = rt.frame_count
                    last_ok_t = rt.last_ok_t
                    latest_error = rt.latest_error

                if color is None:
                    if latest_error:
                        now = time.perf_counter()
                        if now - rt.last_warn_t >= WARN_INTERVAL_SEC:
                            print(f"[WARN] {name} waiting frames: {latest_error}")
                            rt.last_warn_t = now
                    continue

                input_img = _apply_crop(color)
                elapsed = max(1e-6, last_ok_t - rt.started_at)
                fps_est = frame_count / elapsed

                overlay_lines = (
                    f"{name} serial={rt.serial}",
                    f"input={input_img.shape[1]}x{input_img.shape[0]}  source={FRAME_WIDTH}x{FRAME_HEIGHT}@{FRAME_FPS}",
                    f"fps_est={fps_est:.1f}",
                )
                preview_input = _draw_overlay(input_img, overlay_lines)
                cv2.imshow(f"{name}-input", preview_input)

                if SHOW_RAW_PREVIEW:
                    cv2.imshow(f"{name}-raw", color)

                if SHOW_DEPTH_PREVIEW and depth is not None:
                    depth_col = _depth_to_colormap(depth)
                    cv2.imshow(f"{name}-depth", depth_col)

                any_frame = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if not any_frame:
                time.sleep(0.01)

    finally:
        stop_event.set()
        for worker in workers.values():
            worker.join(timeout=1.0)
        for rt in runtimes.values():
            _stop_camera(rt)
        cv2.destroyAllWindows()

    print("[DONE] Camera validation finished")
    return 0


def main() -> int:
    return run_validation()


if __name__ == "__main__":
    raise SystemExit(main())
