from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _prepare_lerobot_import() -> None:
    """Allow importing local lerobot source without global installation."""
    candidates = [
        Path(__file__).resolve().parents[2] / "lerobot",
        Path(__file__).resolve().parents[3] / "lerobot",
    ]
    for repo_root in candidates:
        src_dir = repo_root / "src"
        if src_dir.exists():
            src_dir_str = str(src_dir)
            if src_dir_str not in sys.path:
                sys.path.insert(0, src_dir_str)
            return


_prepare_lerobot_import()

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class EpisodeInfo:
    episode_dir: Path
    started_at: float


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


class TeleopRecorder:
    """LeRobot ACT-compatible dataset writer.

    Writes frames through LeRobotDataset.add_frame()/save_episode()/finalize().
    """

    def __init__(
        self,
        root: Path,
        run_config: Dict[str, Any],
        episode_name: Optional[str] = None,
        fps: int = 30,
        repo_id: str = "local/franka_act_teleop",
        task_name: str = "franka_teleop",
        image_height: int = 480,
        image_width: int = 640,
        camera_names: tuple[str, str] = ("front", "wrist"),
    ) -> None:
        if episode_name:
            dataset_root = root / episode_name
            dataset_root.mkdir(parents=True, exist_ok=True)
        else:
            dataset_root = root
            dataset_root.mkdir(parents=True, exist_ok=True)

        self.info = EpisodeInfo(episode_dir=dataset_root, started_at=time.time())

        self._closed = False
        self._samples = 0
        self._start_mono = time.perf_counter()
        self._task_name = task_name
        self._image_height = int(image_height)
        self._image_width = int(image_width)
        self._camera_names = tuple(camera_names)
        self._last_images: Dict[str, np.ndarray] = {}
        self._current_episode_frames = 0
        self._saved_episodes = 0
        self._discarded_episodes = 0

        features = self._build_features(self._camera_names, self._image_height, self._image_width)

        has_meta = (dataset_root / "meta" / "info.json").exists()
        should_resume = episode_name is None and has_meta
        if should_resume:
            self._dataset = LeRobotDataset.resume(repo_id=repo_id, root=dataset_root)
        else:
            # 单条 episode 目录采用“始终新建”的策略，避免离线环境下
            # 对半成品目录执行 resume 时触发 Hugging Face 版本检查。
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=dataset_root,
                fps=int(fps),
                features=features,
                use_videos=True,
                streaming_encoding=True,
            )

        self._metadata_path = self.info.episode_dir / "metadata.json"

        metadata = {
            "started_at_unix": self.info.started_at,
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.info.started_at)),
            "schema_version": 2,
            "format": "lerobot",
            "run_config": _to_builtin(run_config),
            "task": self._task_name,
            "camera_names": list(self._camera_names),
        }
        self._metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _build_features(camera_names: tuple[str, ...], image_h: int, image_w: int) -> Dict[str, Any]:
        features: Dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "joint_7",
                    "gripper_width",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "joint_7",
                    "gripper_cmd",
                ],
            },
        }
        for cam in camera_names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [image_h, image_w, 3],
                "names": ["height", "width", "channels"],
            }
        return features

    @staticmethod
    def _to_state_vector(robot_state: Dict[str, Any]) -> np.ndarray:
        q = np.asarray(robot_state.get("q", []), dtype=np.float32).reshape(-1)
        if q.size < 7:
            q_pad = np.zeros((7,), dtype=np.float32)
            if q.size > 0:
                q_pad[: min(7, q.size)] = q[:7]
            q = q_pad
        else:
            q = q[:7]

        gripper_width = float(robot_state.get("gripper_width", 0.0))
        return np.concatenate([q, np.asarray([gripper_width], dtype=np.float32)], axis=0).astype(np.float32)

    @staticmethod
    def _to_action_vector(robot_state: Dict[str, Any], command_state: Dict[str, Any]) -> np.ndarray:
        q = np.asarray(robot_state.get("q", []), dtype=np.float32).reshape(-1)
        if q.size < 7:
            q_pad = np.zeros((7,), dtype=np.float32)
            if q.size > 0:
                q_pad[: min(7, q.size)] = q[:7]
            q = q_pad
        else:
            q = q[:7]
        g = float(command_state.get("gripper", 0.0))
        return np.concatenate([q, np.asarray([g], dtype=np.float32)], axis=0).astype(np.float32)

    def _resolve_image(self, camera_name: str, camera_frames: Dict[str, Any]) -> np.ndarray:
        frame = camera_frames.get(camera_name)
        if frame is None:
            raise AssertionError(f"missing required image key: observation.images.{camera_name}")
        arr = np.asarray(frame)
        if arr.shape != (self._image_height, self._image_width, 3):
            raise AssertionError(
                f"invalid shape for observation.images.{camera_name}: {arr.shape}, expected {(self._image_height, self._image_width, 3)}"
            )
        if arr.dtype != np.uint8:
            raise AssertionError(
                f"invalid dtype for observation.images.{camera_name}: {arr.dtype}, expected uint8"
            )
        self._last_images[camera_name] = arr
        return arr

    def _assert_vector(self, name: str, value: np.ndarray, shape0: int, dtype: np.dtype) -> None:
        if value.shape != (shape0,):
            raise AssertionError(f"invalid shape for {name}: {value.shape}, expected ({shape0},)")
        if value.dtype != dtype:
            raise AssertionError(f"invalid dtype for {name}: {value.dtype}, expected {dtype}")

    @property
    def current_episode_frames(self) -> int:
        return int(self._current_episode_frames)

    @property
    def saved_episodes(self) -> int:
        return int(self._saved_episodes)

    @property
    def discarded_episodes(self) -> int:
        return int(self._discarded_episodes)

    @property
    def episode_dir(self) -> Path:
        return self.info.episode_dir

    def log_sample(
        self,
        robot_state: Dict[str, Any],
        input_state: Dict[str, Any],
        command_state: Dict[str, Any],
        camera_state: Optional[Dict[str, Any]] = None,
        camera_frames: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._closed:
            return

        _ = input_state
        _ = camera_state

        frame: Dict[str, Any] = {
            "observation.state": self._to_state_vector(robot_state),
            "action": self._to_action_vector(robot_state, command_state),
            "task": self._task_name,
        }

        self._assert_vector("observation.state", frame["observation.state"], 8, np.dtype(np.float32))
        self._assert_vector("action", frame["action"], 8, np.dtype(np.float32))

        cf = camera_frames or {}
        for cam in self._camera_names:
            frame[f"observation.images.{cam}"] = self._resolve_image(cam, cf)

        self._dataset.add_frame(frame)

        self._samples += 1
        self._current_episode_frames += 1

    def save_current_episode(self) -> bool:
        if self._closed:
            return False
        if self._current_episode_frames <= 0:
            return False
        self._dataset.save_episode()
        self._saved_episodes += 1
        self._current_episode_frames = 0
        return True

    def discard_current_episode(self, reason: str = "discarded") -> bool:
        if self._closed:
            return False
        if self._current_episode_frames <= 0:
            return False
        discarded_frames = int(self._current_episode_frames)
        self._dataset.clear_episode_buffer(delete_images=True)
        self._discarded_episodes += 1
        self._current_episode_frames = 0
        current = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        history = current.get("discard_history", [])
        history.append(
            {
                "reason": str(reason),
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "frames": discarded_frames,
            }
        )
        current["discard_history"] = history
        self._metadata_path.write_text(json.dumps(current, ensure_ascii=True, indent=2), encoding="utf-8")
        return True

    def update_metadata(self, patch: Dict[str, Any]) -> None:
        if self._closed:
            return
        current = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        for k, v in patch.items():
            current[k] = _to_builtin(v)
        self._metadata_path.write_text(json.dumps(current, ensure_ascii=True, indent=2), encoding="utf-8")

    def close(self, status: str = "completed") -> None:
        if self._closed:
            return
        self._closed = True

        duration = max(0.0, time.perf_counter() - self._start_mono)

        if self._current_episode_frames > 0:
            self._dataset.clear_episode_buffer(delete_images=True)
            self._discarded_episodes += 1
            self._current_episode_frames = 0
        self._dataset.finalize()

        final_meta = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        final_meta["final"] = {
            "samples": self._samples,
            "duration_sec": duration,
            "status": status,
            "saved_episodes": self._saved_episodes,
            "discarded_episodes": self._discarded_episodes,
        }
        self._metadata_path.write_text(json.dumps(final_meta, ensure_ascii=True, indent=2), encoding="utf-8")
