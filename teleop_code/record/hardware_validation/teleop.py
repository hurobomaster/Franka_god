#!/usr/bin/env python3
"""SpaceMouse teleoperation for Franka arm (3-DOF translation).

Single-file edition for solo development:
- Keep all runtime logic in one file to reduce maintenance overhead.
- Preserve current stable teleop and gripper behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

TELEOP_CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(TELEOP_CODE_ROOT) not in sys.path:
    sys.path.append(str(TELEOP_CODE_ROOT))

from input.keyboard_monitor import KeyboardMonitor
from input.spacemouse_reader import SpaceMouseReader

if TYPE_CHECKING:
    from record.realsense_recorder import DualRealSenseRecorder
    from record.recorder import TeleopRecorder


def _extend_local_repo_paths() -> None:
    """Fallback only: add local repos if packages are not installed."""
    root = PROJECT_ROOT
    franky_repo = root / "third_repo" / "franky"
    spacemouse_repo = root / "third_repo" / "PySpaceMouse"
    for p in (franky_repo, spacemouse_repo):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.append(p_str)


pyspacemouse = None
CartesianVelocityMotion = None
Duration = None
Gripper = None
JointMotion = None
RelativeDynamicsFactor = None
Robot = None
Twist = None
ControlException = None


def _load_runtime_deps() -> None:
    global pyspacemouse
    global CartesianVelocityMotion, Duration, Gripper, JointMotion, RelativeDynamicsFactor, Robot, Twist, ControlException

    if pyspacemouse is not None and Robot is not None:
        return

    try:
        import pyspacemouse as _pyspacemouse
        from franky import (
            CartesianVelocityMotion as _CartesianVelocityMotion,
            Duration as _Duration,
            Gripper as _Gripper,
            JointMotion as _JointMotion,
            RelativeDynamicsFactor as _RelativeDynamicsFactor,
            Robot as _Robot,
            Twist as _Twist,
        )
        from franky._franky import ControlException as _ControlException
    except Exception:
        _extend_local_repo_paths()
        import pyspacemouse as _pyspacemouse
        from franky import (
            CartesianVelocityMotion as _CartesianVelocityMotion,
            Duration as _Duration,
            Gripper as _Gripper,
            JointMotion as _JointMotion,
            RelativeDynamicsFactor as _RelativeDynamicsFactor,
            Robot as _Robot,
            Twist as _Twist,
        )
        from franky._franky import ControlException as _ControlException

    pyspacemouse = _pyspacemouse
    CartesianVelocityMotion = _CartesianVelocityMotion
    Duration = _Duration
    Gripper = _Gripper
    JointMotion = _JointMotion
    RelativeDynamicsFactor = _RelativeDynamicsFactor
    Robot = _Robot
    Twist = _Twist
    ControlException = _ControlException


DEFAULT_HOME_JOINTS = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]


def parse_joint_list(raw: str) -> List[float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("home joints must contain exactly 7 values")
    return values


def clip_with_deadzone(v: float, dead_zone: float) -> float:
    return 0.0 if abs(v) < dead_zone else float(v)


def load_home_from_file(path: Path) -> Optional[List[float]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    joints = data.get("home_joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise ValueError(f"invalid home file format: {path}")
    return [float(v) for v in joints]


def save_home_to_file(path: Path, joints: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "home_joints": joints,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def resolve_home_joints(robot: Robot, args: argparse.Namespace) -> Optional[List[float]]:
    strategy = args.home_strategy
    home_file = Path(args.home_file)

    if strategy == "current":
        return None

    if strategy == "capture":
        q = [float(v) for v in robot.state.q]
        save_home_to_file(home_file, q)
        print(f"[HOME] 已采集当前关节角并保存到: {home_file}")
        return q

    if strategy == "file":
        file_joints = load_home_from_file(home_file)
        if file_joints is not None:
            print(f"[HOME] 已从文件加载关节角: {home_file}")
            return file_joints
        print("[HOME] 未找到 home 文件，回退到固定关节角。")

    return list(args.fixed_home_joints)


def map_velocity(
    state,
    max_speed: float,
    dead_zone: float,
    axis_sign: List[float],
) -> np.ndarray:
    raw = np.array([state.x, state.y, state.z], dtype=float)
    raw = np.array([clip_with_deadzone(v, dead_zone) for v in raw], dtype=float)
    raw = raw * np.array(axis_sign, dtype=float)
    return raw * max_speed


def limit_velocity_step(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
    max_accel: float,
    release_accel: float,
) -> np.ndarray:
    delta = target - current
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < 1e-9:
        return target

    accel_limit = release_accel if float(np.linalg.norm(target)) < 1e-6 else max_accel
    max_step = max(0.0, accel_limit * dt)
    if delta_norm <= max_step:
        return target
    return current + delta * (max_step / delta_norm)


def handle_gripper_button_presses(
    pressed_buttons: List[int],
    gripper: Gripper,
    open_speed: float,
    close_width: float,
    close_speed: float,
    close_force: float,
    epsilon_inner: float,
    epsilon_outer: float,
) -> None:
    if 0 in pressed_buttons:
        print("[GRIPPER] OPEN")
        gripper.open_async(open_speed)

    if 1 in pressed_buttons:
        print("[GRIPPER] CLOSE")
        gripper.grasp_async(
            width=close_width,
            speed=close_speed,
            force=close_force,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceMouse teleop for Franka robot")
    parser.add_argument("--robot-ip", default="10.19.131.202", help="Franka FCI IP")
    parser.add_argument("--loop-hz", type=float, default=30.0, help="control/data loop frequency")
    parser.add_argument("--mouse-hz", type=float, default=250.0, help="SpaceMouse polling frequency")
    parser.add_argument(
        "--cmd-duration-ms",
        type=int,
        default=80,
        help="duration of each velocity command; keep above loop jitter to avoid command dropouts",
    )
    parser.add_argument("--dead-zone", type=float, default=0.02, help="SpaceMouse dead-zone")
    parser.add_argument(
        "--normal-max-speed",
        type=float,
        default=0.08,
        help="max translation speed in BASE mode (m/s)",
    )
    parser.add_argument(
        "--axis-sign",
        type=parse_joint_list,
        default=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        help="use first three values as x,y,z sign (comma-separated), e.g. 1,-1,1,0,0,0,0",
    )
    parser.add_argument(
        "--home-strategy",
        choices=["fixed", "file", "current", "capture"],
        default="file",
        help="fixed: use --fixed-home-joints; file: load --home-file; current: do not move; capture: save current as home",
    )
    parser.add_argument(
        "--home-file",
        default=str(PROJECT_ROOT / "info" / "franka_home_joints.json"),
        help="json file for persisted home joints",
    )
    parser.add_argument(
        "--fixed-home-joints",
        type=parse_joint_list,
        default=DEFAULT_HOME_JOINTS,
        help="7 joint values (comma-separated)",
    )
    parser.add_argument(
        "--relative-dynamics",
        type=float,
        default=0.12,
        help="franky relative dynamics factor for teleop",
    )
    parser.add_argument(
        "--max-command-accel",
        type=float,
        default=0.5,
        help="max Cartesian velocity command change rate while moving (m/s^2)",
    )
    parser.add_argument(
        "--release-command-accel",
        type=float,
        default=0.6,
        help="max Cartesian velocity command change rate when SpaceMouse is released (m/s^2)",
    )
    parser.add_argument(
        "--home-relative-dynamics",
        type=float,
        default=0.06,
        help="slower relative dynamics factor used only for moving to home",
    )
    parser.add_argument(
        "--gripper-open-speed",
        type=float,
        default=0.08,
        help="Franka native gripper open speed",
    )
    parser.add_argument(
        "--gripper-close-width",
        type=float,
        default=0.010,
        help="target width for close grasp (m); acts as minimum acceptable object thickness",
    )
    parser.add_argument(
        "--gripper-close-speed",
        type=float,
        default=0.05,
        help="Franka native gripper close speed",
    )
    parser.add_argument(
        "--gripper-close-force",
        type=float,
        default=60.0,
        help="Franka native gripper close force (N), max 70",
    )
    parser.add_argument(
        "--gripper-epsilon-inner",
        type=float,
        default=0.001,
        help="grasp success tolerance: inner (m); object thinner than close_width-epsilon_inner is rejected",
    )
    parser.add_argument(
        "--gripper-epsilon-outer",
        type=float,
        default=0.08,
        help="grasp success tolerance: outer (m); set large to succeed regardless of object width",
    )
    parser.add_argument(
        "--gripper-homing",
        action="store_true",
        help="run gripper homing at startup",
    )
    parser.add_argument(
        "--dry-run-home-only",
        action="store_true",
        help="connect and move to home, then exit without entering teleop loop",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="enable LeRobot ACT dataset recording",
    )
    parser.add_argument(
        "--record-root",
        default=str(PROJECT_ROOT / "data"),
        help="root directory for LeRobot dataset",
    )
    parser.add_argument(
        "--episode-name",
        default="",
        help="optional custom dataset folder name under record root",
    )
    parser.add_argument(
        "--record-realsense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record dual RealSense and feed observation.images.* (default: enabled in --record mode; use --no-record-realsense to disable)",
    )
    parser.add_argument(
        "--realsense-config",
        default=str(PROJECT_ROOT / "info" / "realsense_config.json"),
        help="json config path for dual RealSense cameras",
    )
    parser.add_argument(
        "--task-name",
        default="franka_teleop",
        help="task string written into LeRobot frame['task']",
    )
    parser.add_argument(
        "--lerobot-repo-id",
        default="local/franka_act_teleop",
        help="LeRobot dataset repo_id metadata",
    )
    parser.add_argument(
        "--drift-alert-ratio",
        type=float,
        default=0.05,
        help="print alert when loop overtime ratio exceeds this threshold (does not pause recording)",
    )
    parser.add_argument(
        "--static-check-sec",
        type=float,
        default=2.0,
        help="seconds used for pre-record static camera check",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=None,
        help="override recording frame rate (default: use loop-hz); only affects LeRobot dataset, not camera capture",
    )
    return parser


def move_to_reset_pose(robot: Robot, home_joints: Optional[List[float]], home_relative_dynamics: float) -> None:
    if home_joints is None:
        return
    robot.relative_dynamics_factor = RelativeDynamicsFactor(home_relative_dynamics)
    robot.move(JointMotion(home_joints))


def main() -> int:
    args = create_parser().parse_args()
    _load_runtime_deps()

    axis_sign = list(args.axis_sign[:3])
    loop_hz = max(1.0, float(args.loop_hz))
    mouse_hz = max(loop_hz, float(args.mouse_hz))
    dt = 1.0 / loop_hz
    cmd_duration_ms = max(20, int(args.cmd_duration_ms))
    teleop_relative_dynamics = float(args.relative_dynamics)
    home_relative_dynamics = float(args.home_relative_dynamics)

    max_speed = float(args.normal_max_speed)

    print("[INIT] 正在连接机器人...")
    robot = Robot(args.robot_ip)
    robot.recover_from_errors()

    print("[INIT] 正在连接 Franka 原生夹爪...")
    gripper = Gripper(args.robot_ip)
    if args.gripper_homing:
        print("[GRIPPER] 正在执行 homing...")
        gripper.homing()

    home_joints = resolve_home_joints(robot, args)
    if home_joints is not None:
        robot.relative_dynamics_factor = RelativeDynamicsFactor(home_relative_dynamics)
        print(f"[HOME] 使用较慢动态系数: {home_relative_dynamics}")
        print("[HOME] 正在张开夹爪...")
        gripper.open(float(args.gripper_open_speed))
        print(f"[HOME] 正在移动到 home 关节角: {home_joints}")
        robot.move(JointMotion(home_joints))
    else:
        print("[HOME] home_strategy=current，跳过回零运动。")

    robot.relative_dynamics_factor = RelativeDynamicsFactor(teleop_relative_dynamics)

    if args.dry_run_home_only:
        print("[DRY-RUN] 回零完成，按 dry-run 设置退出，不进入遥操作循环。")
        return 0

    print("[INIT] 正在打开 SpaceMouse...")
    device = pyspacemouse.open(nonblocking=True)
    mouse_reader = SpaceMouseReader(device=device, poll_hz=mouse_hz)
    mouse_reader.start()

    print("\n[READY] 遥操作已启动")
    print("[MODE] BASE_WORLD")
    print("[KEY] 按 'Q' 退出")
    print("[KEY] 按 'S' 丢弃当前 episode 并复位")
    print("[KEY] 按 'L' 保存当前 episode 并复位")
    print("[KEY] 按 'R' 在复位后开始录制")
    print("[GRIPPER] SpaceMouse 按键 0 -> 张开, 按键 1 -> 闭合")
    print(f"[CONTROL] 控制频率 {loop_hz:.1f} Hz, 指令时长 {cmd_duration_ms} ms")
    print(f"[INPUT] SpaceMouse 轮询频率 {mouse_hz:.1f} Hz")

    commanded_v = np.zeros(3, dtype=float)
    gripper_cmd = 1.0
    last_control_time = time.perf_counter()
    recorder = None
    rs_recorder = None
    rs_summary: Optional[dict] = None
    record_status = "completed"
    preview_window_name = "RealSense 预览 (front | wrist)"
    preview_error_reported = False
    reflex_warned_once = False

    if args.record:
        from record.recorder import TeleopRecorder
        from record.realsense_recorder import DualRealSenseRecorder

        if abs(loop_hz - 30.0) > 1e-6:
            raise ValueError("record mode requires --loop-hz 30 for strict sync with 30fps cameras")

        run_config = {
            "robot_ip": args.robot_ip,
            "loop_hz": loop_hz,
            "mouse_hz": mouse_hz,
            "cmd_duration_ms": cmd_duration_ms,
            "dead_zone": float(args.dead_zone),
            "normal_max_speed": max_speed,
            "axis_sign": axis_sign,
            "relative_dynamics": teleop_relative_dynamics,
            "home_relative_dynamics": home_relative_dynamics,
            "max_command_accel": float(args.max_command_accel),
            "release_command_accel": float(args.release_command_accel),
            "gripper": {
                "open_speed": float(args.gripper_open_speed),
                "close_width": float(args.gripper_close_width),
                "close_speed": float(args.gripper_close_speed),
                "close_force": float(args.gripper_close_force),
                "epsilon_inner": float(args.gripper_epsilon_inner),
                "epsilon_outer": float(args.gripper_epsilon_outer),
            },
            "task_name": str(args.task_name),
            "lerobot_repo_id": str(args.lerobot_repo_id),
            "drift_alert_ratio": float(args.drift_alert_ratio),
            "static_check_sec": float(args.static_check_sec),
        }
        recorder = TeleopRecorder(
            root=Path(args.record_root),
            run_config=run_config,
            episode_name=args.episode_name or None,
            fps=args.record_fps if args.record_fps is not None else int(round(loop_hz)),
            repo_id=str(args.lerobot_repo_id),
            task_name=str(args.task_name),
            image_height=480,
            image_width=640,
            camera_names=("front", "wrist"),
        )
        print(f"[RECORD] 录制已开启 -> {recorder.episode_dir}")

        if args.record_realsense:
            rs_out = recorder.episode_dir / "realsense"
            rs_recorder = DualRealSenseRecorder(
                output_dir=rs_out,
                config_path=Path(args.realsense_config),
            )
            rs_recorder.start()
            print(f"[RECORD] RealSense 录制已开启 -> {rs_out}")
            print("[CHECK] 等待相机 warm-up 2s...")
            time.sleep(2.0)
            static_result = rs_recorder.run_static_frame_check(duration_sec=float(args.static_check_sec))
            print(f"[CHECK] 相机静态检查: {json.dumps(static_result, ensure_ascii=True)}")
            if not bool(static_result.get("all_ok", False)):
                print("[WARN] 相机静态检查未通过（帧率偏低或时钟偏差过大），5秒后自动继续...")
                print("[WARN] 如需中止请按 Ctrl+C")
                time.sleep(5.0)
            if cv2 is not None:
                print("[VIEW] 已开启实时相机预览窗口。")
                try:
                    cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(preview_window_name, 1280, 480)
                except Exception as preview_init_exc:
                    print(f"[WARN] 预览窗口初始化失败: {preview_init_exc}")
            else:
                print("[WARN] 未安装 cv2，无法显示实时相机预览窗口。")
        else:
            print("[RECORD] RealSense 已关闭（由 --no-record-realsense 指定）")

    episode_index = 1
    waiting_for_record_start = recorder is not None
    total_loops = 0
    overtime_loops = 0
    episode_loops = 0
    episode_overtime_loops = 0
    control_dt_history = []  # 用于计算真实的控制频率
    profile_interval_sec = 2.0
    profile_last_print = time.perf_counter()
    profile_loops = 0
    profile_acc_ms = {
        "loop": 0.0,
        "input": 0.0,
        "control": 0.0,
        "preview": 0.0,
        "sample": 0.0,
        "log": 0.0,
    }

    try:
        with KeyboardMonitor() as keys:
            if recorder is not None:
                print(f"[EPISODE] 复位完成，按下R开始第{episode_index}条录制。")
            while True:
                start = time.perf_counter()
                control_dt = max(1e-3, start - last_control_time)
                last_control_time = start

                key = keys.read_key()
                key_upper = key.upper() if isinstance(key, str) else None

                if key_upper in ("Q", "S", "L", "R"):
                    print(f"[KEY] {key_upper} pressed!")

                if key_upper in ("R", "S", "L") and recorder is None:
                    print("[RECORD] 当前未启用录制。请使用: python teleop.py --record --episode-name <name>")

                if key in ("q", "Q"):
                    print("[EXIT] 收到退出按键。")
                    break

                if key in ("r", "R") and recorder is not None and waiting_for_record_start:
                    waiting_for_record_start = False
                    total_loops = 0
                    overtime_loops = 0
                    episode_loops = 0
                    episode_overtime_loops = 0
                    control_dt_history = []
                    profile_last_print = time.perf_counter()
                    profile_loops = 0
                    for _k in profile_acc_ms:
                        profile_acc_ms[_k] = 0.0
                    print(f"[EPISODE] 第{episode_index}条录制开始。")
                    continue

                if key in ("s", "S") and recorder is not None:
                    dropped = recorder.discard_current_episode(reason="manual_discard_key")
                    if dropped:
                        print(f"[EPISODE] 第{episode_index}条已丢弃，正在复位...")
                        episode_index += 1
                    else:
                        print("[EPISODE] 当前无可丢弃帧，正在复位...")
                    move_to_reset_pose(robot, home_joints, home_relative_dynamics)
                    robot.relative_dynamics_factor = RelativeDynamicsFactor(teleop_relative_dynamics)
                    waiting_for_record_start = True
                    total_loops = 0
                    overtime_loops = 0
                    episode_loops = 0
                    episode_overtime_loops = 0
                    control_dt_history = []
                    profile_last_print = time.perf_counter()
                    profile_loops = 0
                    for _k in profile_acc_ms:
                        profile_acc_ms[_k] = 0.0
                    print(f"[EPISODE] 复位完成，按下R开始第{episode_index}条录制。")
                    continue

                if key in ("l", "L") and recorder is not None:
                    saved = recorder.save_current_episode()
                    if saved:
                        print(f"[EPISODE] 第{episode_index}条已保存，正在复位...")
                        episode_index += 1
                    else:
                        print("[EPISODE] 当前无可保存帧，正在复位...")
                    move_to_reset_pose(robot, home_joints, home_relative_dynamics)
                    robot.relative_dynamics_factor = RelativeDynamicsFactor(teleop_relative_dynamics)
                    waiting_for_record_start = True
                    total_loops = 0
                    overtime_loops = 0
                    episode_loops = 0
                    episode_overtime_loops = 0
                    control_dt_history = []
                    profile_last_print = time.perf_counter()
                    profile_loops = 0
                    for _k in profile_acc_ms:
                        profile_acc_ms[_k] = 0.0
                    print(f"[EPISODE] 复位完成，按下R开始第{episode_index}条录制。")
                    continue

                is_recording_active = recorder is not None and (not waiting_for_record_start)
                input_ms = 0.0
                control_ms = 0.0
                preview_ms = 0.0
                sample_ms = 0.0
                log_ms = 0.0

                t_input0 = time.perf_counter()
                state, pressed_buttons, mouse_error = mouse_reader.latest()
                if mouse_error is not None:
                    raise RuntimeError("SpaceMouse reader failed") from mouse_error

                handle_gripper_button_presses(
                    pressed_buttons=pressed_buttons,
                    gripper=gripper,
                    open_speed=float(args.gripper_open_speed),
                    close_width=float(args.gripper_close_width),
                    close_speed=float(args.gripper_close_speed),
                    close_force=float(args.gripper_close_force),
                    epsilon_inner=float(args.gripper_epsilon_inner),
                    epsilon_outer=float(args.gripper_epsilon_outer),
                )

                if 0 in pressed_buttons:
                    gripper_cmd = 1.0
                elif 1 in pressed_buttons:
                    gripper_cmd = 0.0
                t_input1 = time.perf_counter()
                input_ms = (t_input1 - t_input0) * 1000.0

                t_control0 = time.perf_counter()
                desired_v = map_velocity(
                    state=state,
                    max_speed=max_speed,
                    dead_zone=float(args.dead_zone),
                    axis_sign=axis_sign,
                )
                commanded_v = limit_velocity_step(
                    current=commanded_v,
                    target=desired_v,
                    dt=control_dt,
                    max_accel=float(args.max_command_accel),
                    release_accel=float(args.release_command_accel),
                )
                if float(np.linalg.norm(desired_v)) < 1e-6 and float(np.linalg.norm(commanded_v)) < 1e-4:
                    commanded_v = np.zeros(3, dtype=float)

                motion = CartesianVelocityMotion(
                    Twist(
                        linear_velocity=np.asarray(commanded_v, dtype=float),  # type: ignore[arg-type]
                        angular_velocity=np.asarray([0.0, 0.0, 0.0], dtype=float),  # type: ignore[arg-type]
                    ),
                    duration=Duration(cmd_duration_ms),
                )
                try:
                    robot.move(motion, asynchronous=True)
                except ControlException as e:
                    if not reflex_warned_once:
                        print("[WARN] 检测到 Reflex，已自动恢复（后续同类提示省略）。")
                        reflex_warned_once = True
                    robot.recover_from_errors()
                t_control1 = time.perf_counter()
                control_ms = (t_control1 - t_control0) * 1000.0

                t_preview0 = time.perf_counter()
                if rs_recorder is not None and cv2 is not None:
                    raw_frames = rs_recorder.latest_rgb_frames()
                    front = raw_frames.get("front")
                    wrist = raw_frames.get("wrist")
                    cam_keys = list(raw_frames.keys())
                    if front is None and len(cam_keys) >= 1:
                        front = raw_frames.get(cam_keys[0])
                    if wrist is None and len(cam_keys) >= 2:
                        wrist = raw_frames.get(cam_keys[1])

                    if front is not None or wrist is not None:
                        if front is None and wrist is not None:
                            front = np.zeros_like(wrist)
                        if wrist is None and front is not None:
                            wrist = np.zeros_like(front)
                        if front is not None and wrist is not None:
                            try:
                                preview = np.hstack([front, wrist])
                                cv2.imshow(preview_window_name, preview)
                                cv2.waitKey(1)
                            except Exception as preview_exc:
                                if not preview_error_reported:
                                    preview_error_reported = True
                                    print(f"[WARN] 相机预览显示失败: {preview_exc}")
                                    print("[WARN] 若在无图形界面环境，请设置 DISPLAY 或改用带GUI支持的 opencv-python。")
                t_preview1 = time.perf_counter()
                preview_ms = (t_preview1 - t_preview0) * 1000.0

                if is_recording_active:
                    t_sample0 = time.perf_counter()
                    try:
                        rs = robot.state
                        robot_sample = {
                            "q": np.asarray(rs.q, dtype=float),
                            "dq": np.asarray(rs.dq, dtype=float),
                            "tau_J": np.asarray(rs.tau_J, dtype=float),
                            "O_T_EE": np.asarray(rs.O_T_EE, dtype=float),
                            "gripper_width": float(
                                getattr(
                                    rs,
                                    "gripper_width",
                                    getattr(rs, "gripper_opening_width", 0.0),
                                )
                            ),
                        }
                    except Exception as sample_exc:
                        robot_sample = {"state_error": str(sample_exc)}

                    input_sample = {
                        "x": float(state.x),
                        "y": float(state.y),
                        "z": float(state.z),
                        "buttons": list(state.buttons),
                        "timestamp": float(state.timestamp),
                        "pressed_buttons": list(pressed_buttons),
                    }
                    cmd_sample = {
                        "desired_v": np.asarray(desired_v, dtype=float),
                        "commanded_v": np.asarray(commanded_v, dtype=float),
                        "gripper": float(gripper_cmd),
                    }

                    camera_sample = rs_recorder.snapshot() if rs_recorder is not None else None
                    camera_frames = None
                    if rs_recorder is not None:
                        def _to_rgb(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
                            if arr is None:
                                return None
                            # RealSense color_format 为 bgr8；LeRobot 期望 RGB。
                            return np.ascontiguousarray(arr[:, :, ::-1])

                        raw_frames = rs_recorder.latest_rgb_frames()
                        cam_keys = list(raw_frames.keys())
                        front = raw_frames.get("front")
                        wrist = raw_frames.get("wrist")
                        if front is None and len(cam_keys) >= 1:
                            front = raw_frames.get(cam_keys[0])
                        if wrist is None and len(cam_keys) >= 2:
                            wrist = raw_frames.get(cam_keys[1])
                        camera_frames = {
                            "front": _to_rgb(front),
                            "wrist": _to_rgb(wrist),
                        }
                    t_sample1 = time.perf_counter()
                    sample_ms = (t_sample1 - t_sample0) * 1000.0
                    try:
                        t_log0 = time.perf_counter()
                        recorder.log_sample(
                            robot_state=robot_sample,
                            input_state=input_sample,
                            command_state=cmd_sample,
                            camera_state=camera_sample,
                            camera_frames=camera_frames,
                        )
                        t_log1 = time.perf_counter()
                        log_ms = (t_log1 - t_log0) * 1000.0
                    except AssertionError as frame_assert_exc:
                        recorder.discard_current_episode(reason=f"frame_assert:{frame_assert_exc}")
                        print(f"[EPISODE] 第{episode_index}条因帧断言失败被丢弃: {frame_assert_exc}")
                        move_to_reset_pose(robot, home_joints, home_relative_dynamics)
                        robot.relative_dynamics_factor = RelativeDynamicsFactor(teleop_relative_dynamics)
                        episode_index += 1
                        waiting_for_record_start = True
                        total_loops = 0
                        overtime_loops = 0
                        episode_loops = 0
                        episode_overtime_loops = 0
                        control_dt_history = []
                        profile_last_print = time.perf_counter()
                        profile_loops = 0
                        for _k in profile_acc_ms:
                            profile_acc_ms[_k] = 0.0
                        print(f"[EPISODE] 复位完成，按下R开始第{episode_index}条录制。")
                        continue

                elapsed = time.perf_counter() - start
                if is_recording_active:
                    episode_loops += 1
                    total_loops += 1
                    # 记录真实的 control_dt（控制循环实际间隔）
                    control_dt_history.append(control_dt)
                    if len(control_dt_history) > 300:  # 保持最近 300 个数据点
                        control_dt_history.pop(0)

                    profile_loops += 1
                    profile_acc_ms["loop"] += elapsed * 1000.0
                    profile_acc_ms["input"] += input_ms
                    profile_acc_ms["control"] += control_ms
                    profile_acc_ms["preview"] += preview_ms
                    profile_acc_ms["sample"] += sample_ms
                    profile_acc_ms["log"] += log_ms

                    if (start - profile_last_print) >= profile_interval_sec and profile_loops > 0:
                        avg_loop_ms = profile_acc_ms["loop"] / profile_loops
                        avg_loop_hz = 1000.0 / max(1e-6, avg_loop_ms)
                        base = max(1e-6, avg_loop_ms)
                        print(
                            "[PROFILE] "
                            f"N={profile_loops} "
                            f"loop={avg_loop_ms:.2f}ms({avg_loop_hz:.1f}Hz) "
                            f"input={profile_acc_ms['input'] / profile_loops:.2f}ms({profile_acc_ms['input'] / profile_loops / base * 100.0:.1f}%) "
                            f"control={profile_acc_ms['control'] / profile_loops:.2f}ms({profile_acc_ms['control'] / profile_loops / base * 100.0:.1f}%) "
                            f"preview={profile_acc_ms['preview'] / profile_loops:.2f}ms({profile_acc_ms['preview'] / profile_loops / base * 100.0:.1f}%) "
                            f"sample={profile_acc_ms['sample'] / profile_loops:.2f}ms({profile_acc_ms['sample'] / profile_loops / base * 100.0:.1f}%) "
                            f"log={profile_acc_ms['log'] / profile_loops:.2f}ms({profile_acc_ms['log'] / profile_loops / base * 100.0:.1f}%)"
                        )
                        profile_last_print = start
                        profile_loops = 0
                        for _k in profile_acc_ms:
                            profile_acc_ms[_k] = 0.0
                    
                    if elapsed > dt:
                        overtime_loops += 1
                        episode_overtime_loops += 1

                if is_recording_active and episode_loops >= 120:
                    drift_ratio = episode_overtime_loops / max(1, episode_loops)
                    if drift_ratio > float(args.drift_alert_ratio):
                        # 计算真实的平均控制频率
                        avg_control_dt = sum(control_dt_history) / len(control_dt_history) if control_dt_history else dt
                        real_hz = 1.0 / avg_control_dt if avg_control_dt > 0 else 0
                        print(
                            "[ALERT] 循环超时比例 "
                            f"{drift_ratio * 100.0:.2f}% > {float(args.drift_alert_ratio) * 100.0:.2f}%. "
                            f"真实控制频率 {real_hz:.1f} Hz (目标 {loop_hz:.1f} Hz). "
                            "继续录制中（仅告警，不暂停）。"
                        )

                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[EXIT] 收到 KeyboardInterrupt")
        record_status = "keyboard_interrupt"
    except Exception:
        record_status = "error"
        raise
    finally:
        mouse_reader.stop()

        try:
            robot.recover_from_errors()
            stop_motion = CartesianVelocityMotion(
                Twist(
                    linear_velocity=np.asarray([0.0, 0.0, 0.0], dtype=float),  # type: ignore[arg-type]
                    angular_velocity=np.asarray([0.0, 0.0, 0.0], dtype=float),  # type: ignore[arg-type]
                ),
                duration=Duration(120),
            )
            robot.move(stop_motion, asynchronous=True)
            robot.join_motion(0.3)
        except Exception as exc:
            print(f"[WARN] 停机序列异常: {exc}")

        try:
            device.close()
        except Exception:
            pass

        if rs_recorder is not None:
            try:
                rs_summary = rs_recorder.close()
            except Exception as rs_exc:
                print(f"[WARN] RealSense 关闭失败: {rs_exc}")

        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if recorder is not None:
            try:
                if rs_summary is not None:
                    recorder.update_metadata({"realsense": rs_summary})
                recorder.close(status=record_status)
                print(
                    f"[RECORD] 已关闭({record_status}): {recorder.episode_dir}, "
                    f"已保存={recorder.saved_episodes}, 已丢弃={recorder.discarded_episodes}"
                )
            except Exception as rec_exc:
                print(f"[WARN] 录制器关闭失败: {rec_exc}")

        print("[DONE] 遥操作已停止。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
