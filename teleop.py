#!/usr/bin/env python3
"""遥操作数据采集入口（根目录兼容入口）。

直接运行此文件等同于运行 teleop_code/record/hardware_validation/teleop.py。
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "teleop_code"))

runpy.run_path(
    str(Path(__file__).resolve().parent / "teleop_code" / "record" / "hardware_validation" / "teleop.py"),
    run_name="__main__",
)
