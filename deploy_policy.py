#!/usr/bin/env python3
"""
Franka ACT Policy Deployment Script

Deploys a trained ACT policy to control the Franka robot in real-time.
Combines lerobot's policy inference with franky's hardware control.

Usage:
    python run_policy.py \
        --policy-path /path/to/pretrained_model \
        --robot-ip 10.19.131.202 \
        --fps 30 \
        --duration 60 \
        [--dry-run] [--record-dataset local/eval_franka_act]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Setup path for lerobot import
LEROBOT_ROOT = Path(__file__).resolve().parents[2] / "lerobot"
if str(LEROBOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.common.control_utils import predict_action


DEFAULT_HOME_JOINTS = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]
DEFAULT_HOME_FILE = Path(__file__).resolve().parent / "info" / "franka_home_joints.json"


def parse_joint_list(raw: str) -> list[float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("home joints must contain exactly 7 values")
    return values


def load_home_joints_from_file(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    joints = data.get("home_joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise ValueError(f"Invalid home joints file format: {path}")
    return [float(v) for v in joints]


def resolve_home_joints(args: argparse.Namespace) -> list[float]:
    if args.home_joints is not None:
        return list(args.home_joints)

    home_from_file = load_home_joints_from_file(Path(args.home_file))
    if home_from_file is not None:
        print(f"[RESET] Using home joints from file: {args.home_file}")
        return home_from_file

    print(f"[WARN] Home file not found: {args.home_file}")
    print(f"[WARN] Falling back to built-in default home joints: {DEFAULT_HOME_JOINTS}")
    return list(DEFAULT_HOME_JOINTS)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy to Franka robot"
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        type=str,
        help="Path to pretrained ACT model",
    )
    parser.add_argument(
        "--robot-ip",
        default="10.19.131.202",
        type=str,
        help="Franka FCI IP address",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Deployment duration (seconds)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for policy inference (cuda:0, cpu, etc)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load policy and run one inference without connecting robot",
    )
    parser.add_argument(
        "--record-dataset",
        type=str,
        default=None,
        help="Record rollout as LeRobot dataset (e.g., local/eval_franka_act)",
    )
    parser.add_argument(
        "--record-root",
        type=str,
        default=None,
        help="Root directory for dataset recording",
    )
    parser.add_argument(
        "--gripper-move-speed",
        type=float,
        default=0.08,
        help="Franka gripper move speed (m/s)",
    )
    parser.add_argument(
        "--front-camera-id",
        type=str,
        default="327522300259",
        help="Front RealSense camera serial or name (D455)",
    )
    parser.add_argument(
        "--wrist-camera-id",
        type=str,
        default="153222071977",
        help="Wrist RealSense camera serial or name (D435)",
    )
    parser.add_argument(
        "--disable-gripper",
        action="store_true",
        help="Disable gripper control for stability testing",
    )
    parser.add_argument(
        "--reset-before-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset robot to home pose before each non-dry-run deployment",
    )
    parser.add_argument(
        "--home-joints",
        type=parse_joint_list,
        default=None,
        help="Optional override: home joints as 7 comma-separated values",
    )
    parser.add_argument(
        "--home-file",
        type=str,
        default=str(DEFAULT_HOME_FILE),
        help="Path to JSON file containing {\"home_joints\": [7 values]}",
    )
    parser.add_argument(
        "--reset-wait-sec",
        type=float,
        default=3.0,
        help="Seconds to wait after reset before inference loop",
    )
    parser.add_argument(
        "--joint-vel-limit",
        type=float,
        default=0.25,
        help="Clamp abs joint velocity command (rad/s)",
    )
    parser.add_argument(
        "--joint-accel-limit",
        type=float,
        default=1.2,
        help="Clamp joint velocity change rate (rad/s^2)",
    )
    parser.add_argument(
        "--action-ema-alpha",
        type=float,
        default=0.85,
        help="EMA smoothing factor for joint velocity commands in [0,1]",
    )
    parser.add_argument(
        "--gripper-rate-limit",
        type=float,
        default=0.0,
        help="Clamp gripper target width change rate (m/s)",
    )
    parser.add_argument(
        "--gripper-close-delay-steps",
        type=int,
        default=0,
        help="Prevent gripper closing during first N control steps",
    )
    parser.add_argument(
        "--debug-print-every",
        type=int,
        default=30,
        help="Print detailed debug metrics every N steps (<=0 disables)",
    )
    parser.add_argument(
        "--debug-print-first-n",
        type=int,
        default=10,
        help="Always print detailed debug for first N steps",
    )
    parser.add_argument(
        "--adaptive-joint-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt joint velocity command duration to observed loop period",
    )
    parser.add_argument(
        "--joint-duration-min-ms",
        type=int,
        default=45,
        help="Minimum adaptive joint command duration in ms",
    )
    parser.add_argument(
        "--joint-duration-max-ms",
        type=int,
        default=110,
        help="Maximum adaptive joint command duration in ms",
    )
    parser.add_argument(
        "--vscale-min",
        type=float,
        default=0.25,
        help="Minimum velocity time-scale compensation factor in [0,1]",
    )
    parser.add_argument(
        "--obs-timeout-ms",
        type=float,
        default=220.0,
        help="If observation read exceeds this, reuse last observation frame when available",
    )
    parser.add_argument(
        "--reuse-last-observation-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse last valid observation when observation read spikes over timeout",
    )
    parser.add_argument(
        "--vision-update-interval",
        type=int,
        default=2,
        help="Refresh camera frames every N control steps; intermediate steps reuse images with fresh robot state",
    )
    parser.add_argument(
        "--stop-on-user-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop deployment loop immediately when User Stop is pressed",
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.062,
        help="Open threshold for gripper hysteresis (m)",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.048,
        help="Close threshold for gripper hysteresis (m)",
    )
    parser.add_argument(
        "--gripper-sustain-steps",
        type=int,
        default=2,
        help="Required consecutive steps before gripper state switch",
    )
    parser.add_argument(
        "--gripper-toggle-min-steps",
        type=int,
        default=6,
        help="Minimum steps between gripper open/close toggles",
    )
    parser.add_argument(
        "--startup-ramp-steps",
        type=int,
        default=0,
        help="Linearly ramp joint velocity limit for first N steps",
    )
    parser.add_argument(
        "--recover-cooldown-steps",
        type=int,
        default=8,
        help="After reflex recover, force zero joint velocity for N steps",
    )
    parser.add_argument(
        "--z-safety-floor",
        type=float,
        default=0.105,
        help="Minimum Z height (m) in robot base frame; joint velocity zeroed below this",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Enable temporal ensembling with given coefficient (e.g. 0.01). Smooths action chunks across replan boundaries with no retraining needed.",
    )
    parser.add_argument(
        "--num-z-samples",
        type=int,
        default=0,
        help="Number of latent z vectors to sample per replan (0 = disabled, standard inference). "
        "Samples multiple trajectories from the VAE prior and selects the best one.",
    )
    parser.add_argument(
        "--z-selection",
        type=str,
        default="smoothness",
        choices=["random", "smoothness", "confidence", "combined"],
        help="Criterion for selecting the best z sample. "
        "'smoothness': minimize discontinuity with previous action. "
        "'confidence': minimize within-chunk variance (most decisive). "
        "'combined': weighted combination of smoothness and confidence.",
    )
    return parser


def filter_action_command(
    action: np.ndarray,
    obs: dict,
    dt: float,
    step: int,
    prev_action: np.ndarray | None,
    gripper_state: dict[str, float | int],
    recovery_cooldown_remaining: int,
    velocity_time_scale: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply runtime safety/stability filters to reduce control drift."""
    action = np.asarray(action, dtype=np.float32).copy()
    raw_joint = action[:7].copy()
    raw_gripper = float(action[7])
    joint = raw_joint.copy()
    gripper = raw_gripper
    debug: dict[str, float | int] = {
        "raw_joint_max_abs": float(np.max(np.abs(raw_joint))),
        "raw_gripper": float(raw_gripper),
        "velocity_time_scale": float(velocity_time_scale),
        "vel_clamped": 0,
        "accel_clamped": 0,
        "gripper_rate_clamped": 0,
        "gripper_close_blocked": 0,
        "gripper_hysteresis_hold": 0,
        "gripper_toggled": 0,
        "startup_ramp_limited": 0,
        "recovery_cooldown": 0,
    }

    joint_vel_limit = max(0.0, float(args.joint_vel_limit))
    ramp_steps = max(0, int(args.startup_ramp_steps))
    if ramp_steps > 0 and step < ramp_steps:
        # Keep early motion conservative while controller and vision settle.
        ramp_ratio = (step + 1) / float(ramp_steps)
        joint_vel_limit = joint_vel_limit * max(0.1, min(1.0, ramp_ratio))
        debug["startup_ramp_limited"] = 1

    if recovery_cooldown_remaining > 0:
        joint[:] = 0.0
        debug["recovery_cooldown"] = 1

    # Time-base compensation: if actual loop is slower than target dt,
    # scale down velocity commands to preserve per-step displacement.
    joint *= float(np.clip(velocity_time_scale, float(np.clip(args.vscale_min, 0.0, 1.0)), 1.0))

    if joint_vel_limit > 0.0:
        before = joint.copy()
        joint = np.clip(joint, -joint_vel_limit, joint_vel_limit)
        if np.any(np.abs(joint - before) > 1e-8):
            debug["vel_clamped"] = 1

    if prev_action is not None:
        prev_joint = prev_action[:7]

        joint_accel_limit = max(0.0, float(args.joint_accel_limit))
        if joint_accel_limit > 0.0:
            dv = joint_accel_limit * dt
            before = joint.copy()
            joint = np.clip(joint, prev_joint - dv, prev_joint + dv)
            if np.any(np.abs(joint - before) > 1e-8):
                debug["accel_clamped"] = 1

        alpha = float(np.clip(args.action_ema_alpha, 0.0, 1.0))
        if alpha < 1.0:
            joint = alpha * joint + (1.0 - alpha) * prev_joint

        gripper_rate_limit = max(0.0, float(args.gripper_rate_limit))
        if gripper_rate_limit > 0.0:
            dg = gripper_rate_limit * dt
            prev_gripper = float(prev_action[7])
            before = gripper
            gripper = float(np.clip(gripper, prev_gripper - dg, prev_gripper + dg))
            if abs(gripper - before) > 1e-8:
                debug["gripper_rate_clamped"] = 1

    # Prevent premature close right after reset where visual alignment is still settling.
    if step < max(0, int(args.gripper_close_delay_steps)):
        try:
            current_width = float(np.asarray(obs["observation.state"], dtype=np.float32)[-1])
            if gripper < current_width:
                gripper = current_width
                debug["gripper_close_blocked"] = 1
        except Exception:
            pass

    # Gripper anti-chatter: binary gating with hysteresis.
    # Training data has discrete open/close events (button press), but the model
    # outputs continuous width values that can oscillate near the transition.
    # Threshold the model output to stable binary states, preventing servo chatter.
    open_th = float(args.gripper_open_threshold)
    close_th = float(args.gripper_close_threshold)
    if close_th > open_th:
        close_th, open_th = open_th, close_th

    desired_binary = int(gripper_state.get("state", 1))
    if gripper <= close_th:
        desired_binary = 0
    elif gripper >= open_th:
        desired_binary = 1
    else:
        desired_binary = int(gripper_state.get("state", 1))

    # Sustain counter: only switch state after N consecutive consistent readings.
    if desired_binary != int(gripper_state.get("state", 1)):
        gripper_state["candidate"] = desired_binary
        gripper_state["candidate_count"] = int(gripper_state.get("candidate_count", 0)) + 1
    else:
        gripper_state["candidate_count"] = 0

    sustain_steps = max(1, int(args.gripper_sustain_steps))
    min_toggle_steps = max(0, int(args.gripper_toggle_min_steps))
    since_toggle = step - int(gripper_state.get("last_toggle_step", -10_000))

    if (
        int(gripper_state.get("candidate_count", 0)) >= sustain_steps
        and since_toggle >= min_toggle_steps
        and int(gripper_state.get("state", 1)) != int(gripper_state.get("candidate", 1))
    ):
        gripper_state["state"] = int(gripper_state.get("candidate", 1))
        gripper_state["last_toggle_step"] = step
        gripper_state["candidate_count"] = 0
        debug["gripper_toggled"] = 1
        # On state transition, force binary width to match training data semantics.
        gripper = 0.0 if int(gripper_state["state"]) == 0 else 0.08
    else:
        debug["gripper_hysteresis_hold"] = 1 if desired_binary != int(gripper_state.get("state", 1)) else 0
        # Hold current stable state when in hysteresis band.
        if prev_action is not None:
            gripper = 0.0 if int(gripper_state.get("state", 1)) == 0 else 0.08

    gripper = float(np.clip(gripper, 0.0, 0.08))

    filtered = np.empty((8,), dtype=np.float32)
    filtered[:7] = joint.astype(np.float32, copy=False)
    filtered[7] = np.float32(gripper)
    debug["filtered_joint_max_abs"] = float(np.max(np.abs(filtered[:7])))
    debug["filtered_gripper"] = float(filtered[7])
    return filtered, debug


def load_policy(policy_path: str, device: str, temporal_ensemble_coeff: float | None = None):
    """Load ACT policy and its processor pipelines from checkpoint.

    Args:
        temporal_ensemble_coeff: If >0, enables temporal ensembling (no retraining required).
            Smooths action chunks across replan boundaries to reduce step-to-step jitter.
    """
    print(f"[POLICY] Loading from {policy_path}...")
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(policy_path)
    # Avoid first-run online download of torchvision pretrained backbone weights.
    # Checkpoint weights are loaded from model.safetensors afterwards.
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None

    # Enable temporal ensembling at inference time (no retraining needed).
    if temporal_ensemble_coeff is not None and temporal_ensemble_coeff > 0.0:
        config.temporal_ensemble_coeff = temporal_ensemble_coeff
        print(f"[POLICY] Temporal ensembling enabled (coeff={temporal_ensemble_coeff})")

    policy = ACTPolicy.from_pretrained(policy_path, config=config)
    policy.eval().to(device)
    try:
        param_device = next(policy.parameters()).device
        print(f"[POLICY] Model parameter device: {param_device}")
    except Exception:
        pass
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        print(f"[POLICY] CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=policy_path)
    print(f"[POLICY] Loaded successfully on {device}")
    return policy, preprocessor, postprocessor


def create_robot_config(
    robot_ip: str,
    gripper_move_speed: float,
    front_camera_id: str,
    wrist_camera_id: str,
    enable_gripper: bool,
    joint_velocity_duration_ms: int,
    cameras: dict = None,
) -> RobotConfig:
    """Create Franka robot configuration."""
    if cameras is None:
        # Default: dual RealSense cameras (front and wrist)
        from lerobot.cameras.realsense import RealSenseCameraConfig

        cameras = {
            "front": RealSenseCameraConfig(
                serial_number_or_name=front_camera_id,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": RealSenseCameraConfig(
                serial_number_or_name=wrist_camera_id,
                fps=30,
                width=640,
                height=480,
            ),
        }

    # Import config after lerobot path is set up
    from lerobot.robots.franka import FrankaRobotConfig

    config = FrankaRobotConfig(
        id="franka_act_deploy",
        ip_address=robot_ip,
        enable_gripper=enable_gripper,
        gripper_move_speed=gripper_move_speed,
        joint_velocity_duration_ms=joint_velocity_duration_ms,
        cameras=cameras,
    )
    return config


def run_policy_inference(
    robot,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    args: argparse.Namespace,
    fps: float,
    duration: float,
    device: str,
    reset_before_run: bool,
    home_joints: list[float],
    reset_wait_sec: float,
    dry_run: bool = False,
) -> dict:
    """
    Main policy execution loop.

    Returns statistics dictionary.
    """
    dt = 1.0 / fps
    max_steps = int(duration * fps)
    stats = {
        "total_steps": 0,
        "successful_steps": 0,
        "errors": 0,
        "obs_time": 0.0,
        "policy_time": 0.0,
        "control_time": 0.0,
        "loop_time": 0.0,
        "missed_deadlines": 0,
        "timing_slip_time": 0.0,
        "obs_timeout_steps": 0,
        "obs_reused_steps": 0,
        "vel_clamped_steps": 0,
        "accel_clamped_steps": 0,
        "gripper_rate_clamped_steps": 0,
        "gripper_close_blocked_steps": 0,
        "gripper_hysteresis_hold_steps": 0,
        "gripper_toggled_steps": 0,
        "z_floor_blocked_steps": 0,
        "raw_joint_max_abs_sum": 0.0,
        "filtered_joint_max_abs_sum": 0.0,
    }

    print(f"\n[DEPLOY] Starting policy execution: {max_steps} steps @ {fps}Hz, duration {duration}s")
    print(f"[DEPLOY] Dry-run: {dry_run}")

    # Reset policy/processor internal states at rollout start to match eval-time behavior.
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    if not dry_run:
        robot.connect()
        print(f"[DEPLOY] Robot connected: {robot}")
        if reset_before_run:
            print(f"[RESET] Moving to home joints: {home_joints}")
            robot.move_to_home(home_joints)
            wait_s = max(0.0, float(reset_wait_sec))
            if wait_s > 0.0:
                print(f"[RESET] Waiting {wait_s:.1f}s before starting inference...")
                time.sleep(wait_s)

    try:
        start_time = time.time()
        step_times = []
        cycle_errors = []
        obs_times = []
        abort_requested = False
        prev_action = None
        next_tick = time.perf_counter()
        prev_step_end = time.perf_counter()
        ema_loop_period = dt
        velocity_time_scale = 1.0
        recovery_cooldown_remaining = 0
        last_obs = None
        ema_obs_time_ms = None
        gripper_state = {
            "state": 1,
            "candidate": 1,
            "candidate_count": 0,
            "last_toggle_step": -10_000,
        }

        for step in range(max_steps):
            if abort_requested:
                break
            step_start = time.time()

            try:
                # Get observation
                obs_start = time.time()
                obs_reused = False
                cam_refreshed = False
                vision_update_interval = max(1, int(args.vision_update_interval))

                should_refresh_camera = (last_obs is None) or (step % vision_update_interval == 0)
                if should_refresh_camera:
                    obs = robot.get_observation()
                    cam_refreshed = True
                    obs_time = time.time() - obs_start
                    obs_times.append(obs_time)
                    obs_time_ms = obs_time * 1000.0
                    if ema_obs_time_ms is None:
                        ema_obs_time_ms = obs_time_ms
                    else:
                        ema_obs_time_ms = 0.9 * ema_obs_time_ms + 0.1 * obs_time_ms

                    if (
                        args.reuse_last_observation_on_timeout
                        and obs_time_ms > float(args.obs_timeout_ms)
                    ):
                        stats["obs_timeout_steps"] += 1
                        if last_obs is not None:
                            obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in last_obs.items()}
                            obs_reused = True
                            cam_refreshed = False
                            stats["obs_reused_steps"] += 1
                        else:
                            last_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
                    else:
                        last_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
                else:
                    # Reuse last camera frames but refresh robot state every step.
                    obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in last_obs.items()}
                    if hasattr(robot, "robot") and robot.robot is not None:
                        state_q = np.array(robot.robot.state.q, dtype=np.float32)
                        width_val = float(getattr(robot.robot.state, "gripper_width", 0.0))
                        obs["observation.state"] = np.concatenate(
                            [state_q, np.array([width_val], dtype=np.float32)]
                        )
                    obs_time = time.time() - obs_start
                    obs_time_ms = obs_time * 1000.0

                # Policy inference
                policy_start = time.time()
                obs_policy = {k: v for k, v in obs.items() if k.startswith("observation.")}
                action_tensor = predict_action(
                    observation=obs_policy,
                    policy=policy,
                    device=torch.device(device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=getattr(policy.config, "use_amp", False),
                    num_z_samples=max(0, int(args.num_z_samples)),
                    z_selection=str(args.z_selection),
                    prev_action=prev_action,
                )
                policy_time = time.time() - policy_start

                action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

                # Z-axis safety floor: blocks only active downward descent.
                # Uses Z position history to detect direction; allows all other
                # motion (XY, rotation, gripper, Z ascent) even when at floor level.
                z_floor_blocked = False
                if not dry_run and args.z_safety_floor > 0.0:
                    try:
                        if hasattr(robot, "robot") and robot.robot is not None:
                            ee_z = float(robot.robot.state.O_T_EE.translation[2])
                            if ee_z < float(args.z_safety_floor):
                                z_hist = getattr(robot, "_z_hist", [])
                                z_hist.append(ee_z)
                                if len(z_hist) > 4:
                                    z_hist.pop(0)
                                robot._z_hist = z_hist
                                descending = (
                                    len(z_hist) >= 3
                                    and z_hist[-1] < z_hist[-2] - 0.00003
                                    and z_hist[-2] < z_hist[-3] - 0.00003
                                )
                                if descending:
                                    action[:7] = 0.0
                                    z_floor_blocked = True
                                else:
                                    z_floor_blocked = True
                            else:
                                robot._z_hist = []
                    except Exception:
                        pass

                action, filter_debug = filter_action_command(
                    action=action,
                    obs=obs,
                    dt=dt,
                    step=step,
                    prev_action=prev_action,
                    gripper_state=gripper_state,
                    recovery_cooldown_remaining=recovery_cooldown_remaining,
                    velocity_time_scale=velocity_time_scale,
                    args=args,
                )

                # Send action to robot
                control_start = time.time()
                if not dry_run:
                    robot.send_action({"action": action})
                control_time = time.time() - control_start
                prev_action = action

                stats["policy_time"] += policy_time
                stats["control_time"] += control_time
                stats["obs_time"] += obs_time
                stats["successful_steps"] += 1
                stats["vel_clamped_steps"] += int(filter_debug["vel_clamped"])
                stats["accel_clamped_steps"] += int(filter_debug["accel_clamped"])
                stats["gripper_rate_clamped_steps"] += int(filter_debug["gripper_rate_clamped"])
                stats["gripper_close_blocked_steps"] += int(filter_debug["gripper_close_blocked"])
                stats["gripper_hysteresis_hold_steps"] += int(filter_debug["gripper_hysteresis_hold"])
                stats["gripper_toggled_steps"] += int(filter_debug["gripper_toggled"])
                stats["z_floor_blocked_steps"] += int(z_floor_blocked)
                stats["raw_joint_max_abs_sum"] += float(filter_debug["raw_joint_max_abs"])
                stats["filtered_joint_max_abs_sum"] += float(filter_debug["filtered_joint_max_abs"])
                if recovery_cooldown_remaining > 0:
                    recovery_cooldown_remaining -= 1

                # Sleep to maintain frequency
                step_time = time.time() - step_start
                step_times.append(step_time)
                cycle_errors.append(step_time - dt)

                now_perf = time.perf_counter()
                observed_period = max(1e-4, now_perf - prev_step_end)
                prev_step_end = now_perf
                ema_loop_period = 0.9 * ema_loop_period + 0.1 * observed_period
                velocity_time_scale = min(1.0, dt / max(1e-4, ema_loop_period))
                if args.adaptive_joint_duration and not dry_run:
                    target_ms = int(round(ema_loop_period * 1000.0))
                    target_ms = int(np.clip(target_ms, args.joint_duration_min_ms, args.joint_duration_max_ms))
                    try:
                        robot.config.joint_velocity_duration_ms = target_ms
                    except Exception:
                        pass

                next_tick += dt
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    stats["missed_deadlines"] += 1
                    slip = -sleep_time
                    stats["timing_slip_time"] += slip
                    # Re-anchor to now so we don't accumulate backlog drift.
                    next_tick = time.perf_counter()

            except Exception as e:
                print(f"[ERROR] Step {step}: {e}")
                stats["errors"] += 1
                if not dry_run:
                    err_msg = str(e).lower()
                    if "user stop pressed" in err_msg and args.stop_on_user_stop:
                        print("[STOP] User Stop detected; ending deployment loop immediately")
                        abort_requested = True
                    if "reflex" in err_msg or "motion aborted" in err_msg:
                        try:
                            if hasattr(robot, "robot") and robot.robot is not None:
                                robot.robot.recover_from_errors()
                                prev_action = None
                                recovery_cooldown_remaining = max(0, int(args.recover_cooldown_steps))
                                print("[RECOVER] Recovered from reflex error; action history reset")
                        except Exception as recover_err:
                            print(f"[RECOVER] Failed to recover from reflex mode: {recover_err}")

            stats["total_steps"] += 1

            # Print progress
            if (step + 1) % max(1, int(fps * 5)) == 0:  # Every 5 seconds
                elapsed = time.time() - start_time
                eff_hz = stats["successful_steps"] / max(1e-6, elapsed)
                print(
                    f"[PROGRESS] Step {step + 1}/{max_steps} | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Errors: {stats['errors']} | "
                    f"EffHz: {eff_hz:.2f}"
                )

            should_debug = False
            if args.debug_print_first_n > 0 and step < args.debug_print_first_n:
                should_debug = True
            if args.debug_print_every > 0 and ((step + 1) % args.debug_print_every == 0):
                should_debug = True
            if should_debug:
                print(
                    f"[DEBUG] step={step + 1} "
                    f"obs_ms={obs_time * 1000.0:.2f} "
                    f"obs_ema_ms={(ema_obs_time_ms if ema_obs_time_ms is not None else 0.0):.2f} "
                    f"cam_refresh={1 if cam_refreshed else 0} "
                    f"policy_ms={policy_time * 1000.0:.2f} "
                    f"control_ms={control_time * 1000.0:.2f} "
                    f"cmd_ms={int(getattr(robot.config, 'joint_velocity_duration_ms', 0))} "
                    f"vscale={float(filter_debug['velocity_time_scale']):.3f} "
                    f"obs_reused={1 if obs_reused else 0} "
                    f"raw_joint_max={float(filter_debug['raw_joint_max_abs']):.4f} "
                    f"filtered_joint_max={float(filter_debug['filtered_joint_max_abs']):.4f} "
                    f"raw_gripper={float(filter_debug['raw_gripper']):.4f} "
                    f"filtered_gripper={float(filter_debug['filtered_gripper']):.4f} "
                    f"flags(v/a/gr/block/h/t/r/c/z)={int(filter_debug['vel_clamped'])}/"
                    f"{int(filter_debug['accel_clamped'])}/"
                    f"{int(filter_debug['gripper_rate_clamped'])}/"
                    f"{int(filter_debug['gripper_close_blocked'])}/"
                    f"{int(filter_debug['gripper_hysteresis_hold'])}/"
                    f"{int(filter_debug['gripper_toggled'])}/"
                    f"{int(filter_debug['startup_ramp_limited'])}/"
                    f"{int(filter_debug['recovery_cooldown'])}/"
                    f"{int(z_floor_blocked)}"
                )

        elapsed_total = time.time() - start_time
        stats["loop_time"] = elapsed_total

        # Compute statistics
        if step_times:
            stats["avg_step_time"] = np.mean(step_times)
            stats["max_step_time"] = np.max(step_times)
            stats["min_step_time"] = np.min(step_times)
        if cycle_errors:
            abs_cycle_err = np.abs(np.asarray(cycle_errors, dtype=np.float64))
            stats["p95_cycle_error_ms"] = float(np.percentile(abs_cycle_err, 95) * 1000.0)
            stats["p99_cycle_error_ms"] = float(np.percentile(abs_cycle_err, 99) * 1000.0)
        if obs_times:
            stats["avg_obs_time_ms"] = float(np.mean(obs_times) * 1000.0)
            stats["p95_obs_time_ms"] = float(np.percentile(np.asarray(obs_times, dtype=np.float64), 95) * 1000.0)

    finally:
        if not dry_run:
            robot.disconnect()
            print("[DEPLOY] Robot disconnected")

    return stats


def print_summary(stats: dict, policy_path: str) -> None:
    """Print deployment summary."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"Policy path: {policy_path}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Successful steps: {stats['successful_steps']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total duration: {stats['loop_time']:.2f}s")
    if stats["total_steps"] > 0:
        print(f"Success rate: {100 * stats['successful_steps'] / stats['total_steps']:.1f}%")
    if stats["successful_steps"] > 0:
        print(f"Avg observation time: {1000 * stats['obs_time'] / stats['successful_steps']:.2f}ms")
        print(f"Avg policy time: {1000 * stats['policy_time'] / stats['successful_steps']:.2f}ms")
        print(f"Avg control time: {1000 * stats['control_time'] / stats['successful_steps']:.2f}ms")
        print(f"Avg raw |joint vel| max: {stats['raw_joint_max_abs_sum'] / stats['successful_steps']:.4f}")
        print(f"Avg filtered |joint vel| max: {stats['filtered_joint_max_abs_sum'] / stats['successful_steps']:.4f}")
    if "avg_step_time" in stats:
        print(f"Avg step time: {1000 * stats['avg_step_time']:.2f}ms")
        print(f"Max step time: {1000 * stats['max_step_time']:.2f}ms")
        print(f"Min step time: {1000 * stats['min_step_time']:.2f}ms")
        eff_hz = 1.0 / max(1e-6, float(stats["avg_step_time"]))
        print(f"Effective frequency: {eff_hz:.2f}Hz")
    print(f"Vel clamp steps: {stats.get('vel_clamped_steps', 0)}")
    print(f"Accel clamp steps: {stats.get('accel_clamped_steps', 0)}")
    print(f"Gripper rate clamp steps: {stats.get('gripper_rate_clamped_steps', 0)}")
    print(f"Gripper close blocked steps: {stats.get('gripper_close_blocked_steps', 0)}")
    print(f"Gripper hysteresis hold steps: {stats.get('gripper_hysteresis_hold_steps', 0)}")
    print(f"Gripper toggled steps: {stats.get('gripper_toggled_steps', 0)}")
    print(f"Z floor blocked steps: {stats.get('z_floor_blocked_steps', 0)}")
    print(f"Missed deadlines: {stats.get('missed_deadlines', 0)}")
    print(f"Timing slip total: {1000 * stats.get('timing_slip_time', 0.0):.2f}ms")
    print(f"Observation timeout steps: {stats.get('obs_timeout_steps', 0)}")
    print(f"Observation reused steps: {stats.get('obs_reused_steps', 0)}")
    if "avg_obs_time_ms" in stats:
        print(f"Avg observation time (measured): {stats['avg_obs_time_ms']:.2f}ms")
        print(f"Observation time p95: {stats['p95_obs_time_ms']:.2f}ms")
    if "p95_cycle_error_ms" in stats:
        print(f"|cycle error| p95: {stats['p95_cycle_error_ms']:.2f}ms")
        print(f"|cycle error| p99: {stats['p99_cycle_error_ms']:.2f}ms")
    target_hz = float(stats.get("target_fps", 0.0))
    if target_hz > 0 and "avg_step_time" in stats:
        eff_hz = 1.0 / max(1e-6, float(stats["avg_step_time"]))
        if eff_hz < 0.7 * target_hz:
            print(
                f"[WARN] Effective frequency ({eff_hz:.2f}Hz) is far below target ({target_hz:.2f}Hz). "
                f"Consider setting --fps to around {max(5.0, round(eff_hz, 1))} for better control consistency."
            )
    print("=" * 60)


def main():
    args = create_parser().parse_args()

    print("\n" + "=" * 60)
    print("Franka ACT Policy Deployment")
    print("=" * 60)

    # Load policy
    policy, preprocessor, postprocessor = load_policy(
        args.policy_path, args.device, args.temporal_ensemble_coeff
    )

    # Dry-run inference without robot
    if args.dry_run:
        print("\n[DRY-RUN] Testing policy inference without robot...")
        obs_dummy = {
            "observation.state": torch.zeros(1, 8, device=args.device),
            "observation.images.front": torch.zeros(1, 3, 480, 640, device=args.device),
            "observation.images.wrist": torch.zeros(1, 3, 480, 640, device=args.device),
        }
        with torch.no_grad():
            out = policy.predict_action_chunk(obs_dummy)
        print(f"[DRY-RUN] Policy output shape: {tuple(out.shape)}")
        print("[DRY-RUN] Policy inference successful!")
        return 0

    # Create robot config
    print("[CONFIG] Creating Franka robot config...")
    joint_velocity_duration_ms = max(1, int(round(1000.0 / args.fps)))
    print(f"[CONFIG] joint_velocity_duration_ms={joint_velocity_duration_ms}ms (from fps={args.fps})")
    config = create_robot_config(
        args.robot_ip,
        args.gripper_move_speed,
        args.front_camera_id,
        args.wrist_camera_id,
        not args.disable_gripper,
        joint_velocity_duration_ms,
    )

    # Create robot
    print("[ROBOT] Instantiating Franka robot...")
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(config)

    # Run policy execution
    home_joints = resolve_home_joints(args)
    stats = run_policy_inference(
        robot,
        policy,
        preprocessor,
        postprocessor,
        args,
        args.fps,
        args.duration,
        args.device,
        args.reset_before_run,
        home_joints,
        args.reset_wait_sec,
        dry_run=False,
    )

    # Print summary
    print_summary(stats, args.policy_path)

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
