#!/usr/bin/env python3
"""实时打印 Franka 末端在基坐标系下的 XYZ 位置与姿态。"""

import sys
import time
from pathlib import Path

# Setup lerobot path
LEROBOT_ROOT = Path(__file__).resolve().parents[2] / "lerobot"
if str(LEROBOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LEROBOT_ROOT / "src"))

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "10.19.131.202"

from franky import Robot as FrankyRobot, RealtimeConfig

robot = FrankyRobot(ROBOT_IP, realtime_config=RealtimeConfig.Ignore)
robot.recover_from_errors()

print(f"\n{'='*60}")
print(f"  Franka 末端位姿实时显示 (基坐标系)")
print(f"  机器人 IP: {ROBOT_IP}")
print(f"  按 Ctrl+C 退出")
print(f"{'='*60}")
print()

try:
    while True:
        state = robot.state
        t = state.O_T_EE.translation
        print(
            f"  XYZ = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})  ",
            end="\r"
        )
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\n\n退出.")
finally:
    try:
        robot.disconnect()
    except Exception:
        pass
