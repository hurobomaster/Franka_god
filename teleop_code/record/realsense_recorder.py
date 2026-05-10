from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - optional dependency
    rs = None


@dataclass
class CameraSpec:
    name: str
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_rgb: bool = True
    enable_depth: bool = True
    depth_every_n: int = 1
    color_format: str = "bgr8"
    auto_exposure: bool = False
    auto_white_balance: bool = False
    exposure: int = 120
    gain: int = 64


class _CameraWorker:
    def __init__(self, spec: CameraSpec, out_dir: Path) -> None:
        self.spec = spec
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"rs-{spec.name}", daemon=True)

        self._frame_count = 0
        self._depth_count = 0
        self._started = False
        self._error: Optional[str] = None
        self._last_host_t = 0.0

        self._depth_frames: List[np.ndarray] = []
        self._depth_ts: List[float] = []
        self._latest_rgb: Optional[np.ndarray] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.spec.name,
                "serial": self.spec.serial,
                "started": self._started,
                "frame_count": self._frame_count,
                "depth_count": self._depth_count,
                "last_host_t": self._last_host_t,
                "error": self._error,
                "has_rgb": self._latest_rgb is not None,
            }

    def latest_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy()

    def _run(self) -> None:
        writer = None
        pipeline = None
        profile = None

        try:
            if rs is None:
                raise RuntimeError("pyrealsense2 is not installed")
            if cv2 is None:
                raise RuntimeError("opencv-python (cv2) is not installed")

            pipeline = rs.pipeline()
            config = rs.config()
            if self.spec.serial:
                config.enable_device(self.spec.serial)
            if self.spec.enable_depth:
                config.enable_stream(rs.stream.depth, self.spec.width, self.spec.height, rs.format.z16, self.spec.fps)
            if self.spec.enable_rgb:
                if self.spec.color_format != "bgr8":
                    raise ValueError(f"unsupported color_format '{self.spec.color_format}', only 'bgr8' is supported")
                config.enable_stream(rs.stream.color, self.spec.width, self.spec.height, rs.format.bgr8, self.spec.fps)

            profile = pipeline.start(config)
            self._apply_manual_color_controls(profile)
            self._started = True

            intrinsics = {}
            try:
                if self.spec.enable_rgb:
                    cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
                    ci = cp.get_intrinsics()
                    intrinsics["color"] = {
                        "width": ci.width,
                        "height": ci.height,
                        "fx": ci.fx,
                        "fy": ci.fy,
                        "ppx": ci.ppx,
                        "ppy": ci.ppy,
                        "model": str(ci.model),
                        "coeffs": list(ci.coeffs),
                        "format": self.spec.color_format,
                    }
                if self.spec.enable_depth:
                    dp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                    di = dp.get_intrinsics()
                    intrinsics["depth"] = {
                        "width": di.width,
                        "height": di.height,
                        "fx": di.fx,
                        "fy": di.fy,
                        "ppx": di.ppx,
                        "ppy": di.ppy,
                        "model": str(di.model),
                        "coeffs": list(di.coeffs),
                    }
            except Exception:
                pass

            (self.out_dir / "intrinsics.json").write_text(
                json.dumps({"camera": self.spec.name, "serial": self.spec.serial, "intrinsics": intrinsics}, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

            if self.spec.enable_rgb:
                rgb_path = self.out_dir / "rgb.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(rgb_path), fourcc, float(self.spec.fps), (self.spec.width, self.spec.height))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open VideoWriter for {rgb_path}")

            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                host_t = time.perf_counter()

                color_frame = frames.get_color_frame() if self.spec.enable_rgb else None
                depth_frame = frames.get_depth_frame() if self.spec.enable_depth else None

                if self.spec.enable_rgb and color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    if writer is not None:
                        writer.write(color_image)
                    with self._lock:
                        self._latest_rgb = color_image

                if self.spec.enable_depth and depth_frame:
                    frame_idx = self._frame_count
                    if self.spec.depth_every_n <= 1 or (frame_idx % self.spec.depth_every_n == 0):
                        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                        self._depth_frames.append(depth)
                        self._depth_ts.append(host_t)
                        self._depth_count += 1

                with self._lock:
                    self._frame_count += 1
                    self._last_host_t = host_t

        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            try:
                if pipeline is not None:
                    pipeline.stop()
            except Exception:
                pass

            try:
                depth_path = self.out_dir / "depth.npz"
                if self._depth_frames:
                    depth_stack = np.stack(self._depth_frames, axis=0)
                    np.savez_compressed(depth_path, depth=depth_stack, host_t=np.asarray(self._depth_ts, dtype=float))
                else:
                    np.savez_compressed(depth_path, depth=np.empty((0,), dtype=np.uint16), host_t=np.empty((0,), dtype=float))
            except Exception as exc:
                with self._lock:
                    self._error = str(exc) if self._error is None else self._error

            summary = self.status()
            (self.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    def _apply_manual_color_controls(self, profile) -> None:
        if rs is None:
            return
        if not self.spec.enable_rgb:
            return
        device = profile.get_device()
        sensors = device.query_sensors()
        color_sensor = None
        for s in sensors:
            try:
                sname = s.get_info(rs.camera_info.name).lower()
            except Exception:
                continue
            if "rgb" in sname or "color" in sname:
                color_sensor = s
                break
        if color_sensor is None:
            return

        try:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if self.spec.auto_exposure else 0.0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if self.spec.auto_white_balance else 0.0)

            if not self.spec.auto_exposure and color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, float(self.spec.exposure))
            if color_sensor.supports(rs.option.gain):
                color_sensor.set_option(rs.option.gain, float(self.spec.gain))
        except Exception as exc:
            with self._lock:
                self._error = str(exc)


class DualRealSenseRecorder:
    def __init__(self, output_dir: Path, config_path: Path) -> None:
        self.output_dir = output_dir
        self.config_path = config_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._workers: List[_CameraWorker] = []
        self._load_config()

    def _load_config(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        cameras = raw.get("cameras", [])
        if not isinstance(cameras, list) or not cameras:
            raise ValueError(f"invalid realsense config: {self.config_path}")

        self._workers = []
        for item in cameras:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("enabled", True)):
                continue
            spec = CameraSpec(
                name=str(item.get("name", "camera")),
                serial=str(item.get("serial", "")),
                width=int(item.get("width", 640)),
                height=int(item.get("height", 480)),
                fps=int(item.get("fps", 30)),
                enable_rgb=bool(item.get("enable_rgb", True)),
                enable_depth=bool(item.get("enable_depth", True)),
                depth_every_n=max(1, int(item.get("depth_every_n", 1))),
                color_format=str(item.get("color_format", "bgr8")),
                auto_exposure=bool(item.get("auto_exposure", False)),
                auto_white_balance=bool(item.get("auto_white_balance", False)),
                exposure=int(item.get("exposure", 120)),
                gain=int(item.get("gain", 64)),
            )
            self._workers.append(_CameraWorker(spec=spec, out_dir=self.output_dir / spec.name))

        if not self._workers:
            raise ValueError(f"no enabled cameras in config: {self.config_path}")

        self._validate_consistent_streams()

    def _validate_consistent_streams(self) -> None:
        if len(self._workers) <= 1:
            return
        first = self._workers[0].spec
        for w in self._workers[1:]:
            s = w.spec
            if s.width != first.width or s.height != first.height:
                raise ValueError("all cameras must have identical width/height")
            if s.fps != first.fps:
                raise ValueError("all cameras must have identical fps")
            if s.color_format != first.color_format:
                raise ValueError("all cameras must have identical color_format")

    def start(self) -> None:
        for w in self._workers:
            w.start()

    def snapshot(self) -> Dict[str, Any]:
        return {w.spec.name: w.status() for w in self._workers}

    def latest_rgb_frames(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for w in self._workers:
            rgb = w.latest_rgb()
            if rgb is not None:
                out[w.spec.name] = rgb
        return out

    def run_static_frame_check(self, duration_sec: float = 2.0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        start = self.snapshot()
        time.sleep(max(0.2, duration_sec))
        end = self.snapshot()

        per_camera: Dict[str, Any] = {}
        for name, st in end.items():
            s0 = start.get(name, {})
            delta_frames = int(st.get("frame_count", 0)) - int(s0.get("frame_count", 0))
            fps_obs = delta_frames / max(1e-6, float(duration_sec))
            per_camera[name] = {
                "delta_frames": delta_frames,
                "fps_observed": fps_obs,
                "error": st.get("error"),
                "ok": (delta_frames >= max(1, int(0.8 * self._workers[0].spec.fps * duration_sec))) and (st.get("error") is None),
            }

        host_ts = [float(st.get("last_host_t", 0.0)) for st in end.values() if float(st.get("last_host_t", 0.0)) > 0.0]
        skew_sec = max(host_ts) - min(host_ts) if len(host_ts) >= 2 else 0.0
        all_ok = all(v.get("ok", False) for v in per_camera.values()) and skew_sec <= 0.08
        return {
            "duration_sec": float(duration_sec),
            "skew_sec": float(skew_sec),
            "all_ok": bool(all_ok),
            "cameras": per_camera,
        }

    def close(self) -> Dict[str, Any]:
        for w in self._workers:
            w.stop()
        summary = self.snapshot()
        (self.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        return summary
