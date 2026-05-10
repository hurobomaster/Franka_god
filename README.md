# Franka 遥操作系统

SpaceMouse 驱动 Franka Research 3 机械臂的遥操作与数据采集系统。

---

## 快速开始

所有命令均在项目根目录 (`teleop/`) 下运行，激活对应 conda 环境后执行。

```bash
conda activate hu_lerobot
```

### 基础遥操作（不录制）

```bash
python teleop.py
```

### 遥操作 + 录制（默认开启双目 RealSense）

```bash
python teleop.py --record
```

### 指定 episode 名称

```bash
python teleop.py \
    --record \
    --episode-name pick_cup_001
```

注意：参数前不要加 `/`，例如 `/--record` 是错误写法。

---

## 完整 CLI 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--robot-ip` | `10.19.131.202` | Franka FCI 服务器 IP |
| `--loop-hz` | `30.0` | 主控制 / 数据采集循环频率 (Hz) |
| `--mouse-hz` | `250.0` | SpaceMouse 独立轮询频率 (Hz) |
| `--cmd-duration-ms` | `80` | 每条速度指令的持续时间 (ms) |
| `--dead-zone` | `0.02` | SpaceMouse 死区阈值 |
| `--normal-max-speed` | `0.08` | 末端最大平移速度 (m/s) |
| `--axis-sign` | `1,1,1,0,0,0,0` | 各轴方向符号（逗号分隔，前三位为 x/y/z） |
| `--home-strategy` | `file` | 回零策略：`fixed` / `file` / `current` / `capture` |
| `--home-file` | `info/franka_home_joints.json` | 保存回零关节角的文件路径 |
| `--fixed-home-joints` | 见代码 | 7 个固定关节角（逗号分隔） |
| `--relative-dynamics` | `0.12` | 遥操时 franky 相对动力学系数 |
| `--max-command-accel` | `0.5` | 运动中速度指令最大变化率 (m/s²) |
| `--release-command-accel` | `0.6` | 松开 SpaceMouse 后的减速率 (m/s²) |
| `--home-relative-dynamics` | `0.06` | 回零时动力学系数（更慢） |
| `--gripper-open-speed` | `0.08` | 夹爪张开速度 (m/s) |
| `--gripper-close-width` | `0.010` | 夹爪闭合目标宽度 / 最小可抓物体厚度 (m) |
| `--gripper-close-speed` | `0.05` | 夹爪闭合速度 (m/s) |
| `--gripper-close-force` | `60.0` | 夹持力 (N)，最大 70 N |
| `--gripper-epsilon-inner` | `0.001` | 夹取成功内侧容差 (m) |
| `--gripper-epsilon-outer` | `0.08` | 夹取成功外侧容差 (m) |
| `--gripper-homing` | `False` | 启动时执行夹爪回零 |
| `--dry-run-home-only` | `False` | 仅移动到 home 位置后退出 |
| `--record` | `False` | 启用状态数据录制 |
| `--record-root` | `data/` | 录制数据根目录 |
| `--episode-name` | `""` | 自定义 episode 文件夹名称 |
| `--record-realsense` | `True` | 同时录制双 RealSense 相机（`--record` 模式下默认开启，可用 `--no-record-realsense` 关闭） |
| `--realsense-config` | `info/realsense_config.json` | RealSense 相机配置文件路径 |
| `--task-name` | `franka_teleop` | 写入数据集的任务标签 |
| `--lerobot-repo-id` | `local/franka_act_teleop` | LeRobot 数据集 repo_id |

---

## 操作说明

| 操作 | 方式 |
|---|---|
| 末端平移 | 推动 SpaceMouse（x / y / z） |
| 夹爪张开 | SpaceMouse 左键 |
| 夹爪闭合 | SpaceMouse 右键 |
| 丢弃当前 episode 并重置 | 键盘 `S` |
| 保存当前 episode 并重置 | 键盘 `L` |
| 复位后开始第 N 条录制 | 键盘 `R` |
| 退出程序 | 键盘 `Q` |

---

## 硬件验证工具

### 相机预览（验证分辨率 / 帧率 / 裁剪）

```bash
python teleop_code/record/hardware_validation/cream.py
```

编辑 `cream.py` 顶部的 `FRAME_*` / `CROP_*` 参数即可调整预览设置。

---

## 文件职责说明

```
teleop/
├── teleop.py                          # 根目录兼容入口，直接运行即可
├── README.md                          # 本文档
│
├── info/
│   ├── franka_base_info.txt           # 机器人硬件基本信息
│   ├── franka_home_joints.json        # 持久化回零关节角
│   └── realsense_config.json          # RealSense 相机配置（分辨率 / 帧率 / 裁剪 / 序列号）
│
├── teleop_code/
│   ├── input/                         # 输入设备抽象层
│   │   ├── spacemouse_reader.py       # SpaceMouse 双线程采样（解耦控制延迟）
│   │   ├── keyboard_monitor.py        # 非阻塞键盘事件（UNIX tty cbreak 模式）
│   │   └── device_state.py            # 输入状态数据结构（MouseSnapshot）
│   │
│   └── record/                        # 录制 + 硬件验证
│       ├── recorder.py                # episode 元数据 / manifest / samples JSONL 写入
│       ├── realsense_recorder.py      # 双 RealSense 采集（MP4 RGB + NPZ Depth）
│       └── hardware_validation/       # 硬件调试与验证脚本
│           ├── teleop.py              # 主遥操作程序（真正的业务逻辑）
│           └── cream.py               # 相机实时预览与参数验证工具
│
└── third_repo/                        # 第三方依赖（本地源码）
    ├── franky/                        # Franka 控制库（libfranka Python 绑定）
    └── PySpaceMouse/                  # SpaceMouse HID 驱动库
```

---

## 录制数据格式

| 文件 | 格式 | 说明 |
|---|---|---|
| `metadata.json` | JSON | episode 元信息（时间戳 / 配置 / 最终状态） |
| `manifest.jsonl` | JSONL | 每帧索引与统计摘要 |
| `samples.jsonl` | JSONL | 每帧完整状态（机器人 / 输入 / 指令 / 相机引用） |
| `realsense/d455/rgb.mp4` | MP4 | D455 彩色视频（h264） |
| `realsense/d455/depth.npz` | NPZ | D455 深度帧（zstd 压缩） |
| `realsense/d435/rgb.mp4` | MP4 | D435 彩色视频（h264） |
| `realsense/d435/depth.npz` | NPZ | D435 深度帧（zstd 压缩） |

---

## 硬件信息

| 设备 | 参数 |
|---|---|
| 机器人 | Franka Research 3 |
| FCI IP | `10.19.131.202` |
| FCI 服务器版本 | Server 9 |
| 控制库 | franky-control 1.0.2 (libfranka 0.15) |
| 相机 D455 序列号 | `327522300259` |
| 相机 D435 序列号 | `153222071977` |
| SpaceMouse | 3Dconnexion，PySpaceMouse 2.0.0 |
