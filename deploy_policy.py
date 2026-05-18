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
        help="Control loop frequency (Hz) - matches camera fps",
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
    return parser


def load_policy(policy_path: str, device: str):
    """Load ACT policy and its processor pipelines from checkpoint."""
    print(f"[POLICY] Loading from {policy_path}...")
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(policy_path)
    # Avoid first-run online download of torchvision pretrained backbone weights.
    # Checkpoint weights are loaded from model.safetensors afterwards.
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(policy_path, config=config)
    policy.eval().to(device)
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
        "policy_time": 0.0,
        "control_time": 0.0,
        "loop_time": 0.0,
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

        for step in range(max_steps):
            step_start = time.time()

            try:
                # Get observation
                obs = robot.get_observation()

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
                )
                policy_time = time.time() - policy_start

                action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

                # Send action to robot
                control_start = time.time()
                if not dry_run:
                    robot.send_action({"action": action})
                control_time = time.time() - control_start

                stats["policy_time"] += policy_time
                stats["control_time"] += control_time
                stats["successful_steps"] += 1

                # Sleep to maintain frequency
                step_time = time.time() - step_start
                step_times.append(step_time)
                if step_time < dt:
                    time.sleep(dt - step_time)

            except Exception as e:
                print(f"[ERROR] Step {step}: {e}")
                stats["errors"] += 1

            stats["total_steps"] += 1

            # Print progress
            if (step + 1) % max(1, int(fps * 5)) == 0:  # Every 5 seconds
                elapsed = time.time() - start_time
                print(
                    f"[PROGRESS] Step {step + 1}/{max_steps} | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Errors: {stats['errors']}"
                )

        elapsed_total = time.time() - start_time
        stats["loop_time"] = elapsed_total

        # Compute statistics
        if step_times:
            stats["avg_step_time"] = np.mean(step_times)
            stats["max_step_time"] = np.max(step_times)
            stats["min_step_time"] = np.min(step_times)

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
        print(f"Avg policy time: {1000 * stats['policy_time'] / stats['successful_steps']:.2f}ms")
        print(f"Avg control time: {1000 * stats['control_time'] / stats['successful_steps']:.2f}ms")
    if "avg_step_time" in stats:
        print(f"Avg step time: {1000 * stats['avg_step_time']:.2f}ms")
        print(f"Max step time: {1000 * stats['max_step_time']:.2f}ms")
        print(f"Min step time: {1000 * stats['min_step_time']:.2f}ms")
    print("=" * 60)


def main():
    args = create_parser().parse_args()

    print("\n" + "=" * 60)
    print("Franka ACT Policy Deployment")
    print("=" * 60)

    # Load policy
    policy, preprocessor, postprocessor = load_policy(args.policy_path, args.device)

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
