import sys
import os
import time
import csv
import threading
import numpy as np
import platform
import serial.tools.list_ports

# PyQt & PyQtGraph Imports
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
except ImportError as e:
    raise ImportError(
        "缺少图形依赖。请在 mjx_fast 环境执行: "
        "conda install -n mjx_fast -c conda-forge pyqt pyqtgraph"
    ) from e

# 引入 SensorConnector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from sensorConnector import SensorConnector 
from sensorConnector import SensorType
from sensorConnector import CommucationProtocol

# ================= 全局配置 =================
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'k')   
pg.setConfigOption('foreground', 'w')   

BAUDRATE = 460800
WINDOW_SEC = 5          
FPS = 30                
CALIBRATION_SAMPLES = 50 

LEVER_ARM_MM = 16.0          
LEVER_ARM_M  = LEVER_ARM_MM / 1000.0

# --- 标定参数 (Calibration) ---
# Fz 修正: 6.5(实测) / 5.0(读数) = 1.3
FZ_SCALE = 1.3

# 剪切力增益: 原3.0 * (7.5/5.0) = 4.5
SHEAR_GAIN = 4.5   

# 定义可能的设备标识 (Linux udev names)
UDEV_NAMES = ['/dev/photon_left', '/dev/photon_right']
# ===========================================

def scan_ports():
    """自动扫描并识别可能的传感器端口"""
    available_ports = []
    system_name = platform.system()
    
    print("正在扫描传感器端口...")
    
    if system_name == 'Linux':
        # 1. 优先检查 udev 绑定的固定名称
        for name in UDEV_NAMES:
            if os.path.exists(name):
                label = 'LEFT' if 'left' in name else 'RIGHT'
                available_ports.append({'port': name, 'label': label})
        
        # 2. 备用扫描
        if not available_ports:
            import glob
            usb_devs = glob.glob('/dev/ttyUSB*')
            for i, dev in enumerate(usb_devs):
                available_ports.append({'port': dev, 'label': f'UNK_{i}'})

    elif system_name == 'Windows':
        # Windows 扫描
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # 根据你的描述 COM90/COM91
            if p.device in ['COM90', 'COM91']:
                label = 'LEFT' if p.device == 'COM90' else 'RIGHT'
                available_ports.append({'port': p.device, 'label': label})
    
    if not available_ports:
        print("未检测到任何已知传感器端口！")
    else:
        print(f"检测到 {len(available_ports)} 个传感器: {[p['label'] for p in available_ports]}")
        
    return available_ports

class MultiSensorThread(threading.Thread):
    """
    数据采集线程：
    注意：CSV 文件中保存的依然是【原始数据】(Raw Fz 和 Raw Torque)，
    不包含 FZ_SCALE 和 SHEAR_GAIN 的修正。
    这样保证原始数据的纯净性，方便后续重新标定。
    """
    def __init__(self, filename, sensor_configs):
        super().__init__()
        self.filename = filename
        self.sensor_configs = sensor_configs 
        self.sensors = {} 
        self.running = True
        self.start_time = time.time()
        
        self.latest_data = {}
        self.offsets = {}
        self.calib_buffers = {}
        self.is_calibrated = {}

        # 初始化连接
        for cfg in sensor_configs:
            label = cfg['label']
            port = cfg['port']
            try:
                s = SensorConnector(CommucationProtocol.AT_Command, SensorType.PHOTON_FINGER, port, BAUDRATE)
                if s.Connect():
                    s.set_read_break(0.001)
                    s.sendCommand("AT+SZERO=1")
                    self.sensors[label] = s
                    self.latest_data[label] = np.zeros(3)
                    self.offsets[label] = np.zeros(3)
                    self.calib_buffers[label] = []
                    self.is_calibrated[label] = False
                    print(f"[{label}] 连接成功 ({port})")
                else:
                    print(f"[{label}] 连接失败")
            except Exception as e:
                print(f"[{label}] 初始化异常: {e}")

    def run(self):
        headers = ['Time_s']
        active_labels = sorted(list(self.sensors.keys()))
        for label in active_labels:
            # CSV Header 备注单位
            headers.extend([f'{label}_Fz', f'{label}_Mx', f'{label}_My'])
            
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            print(f"开始采集，等待去皮校准 ({CALIBRATION_SAMPLES}帧)...")

            while self.running:
                loop_start = time.time()
                row_data = [f"{loop_start - self.start_time:.4f}"]
                
                all_calibrated = True
                
                for label in active_labels:
                    sensor = self.sensors[label]
                    raw_data = sensor.GetData()
                    
                    vec = np.zeros(3)
                    if raw_data:
                        vec = np.array([
                            raw_data.get('Fz', 0.0) or 0.0,
                            raw_data.get('Mx', 0.0) or 0.0,
                            raw_data.get('My', 0.0) or 0.0
                        ])
                    
                    if not self.is_calibrated[label]:
                        self.calib_buffers[label].append(vec)
                        if len(self.calib_buffers[label]) >= CALIBRATION_SAMPLES:
                            self.offsets[label] = np.mean(self.calib_buffers[label], axis=0)
                            self.is_calibrated[label] = True
                            print(f"[{label}] 去皮完成")
                        all_calibrated = False
                        self.latest_data[label] = vec 
                    else:
                        clean_vec = vec - self.offsets[label]
                        self.latest_data[label] = clean_vec
                        
                    row_data.extend(self.latest_data[label])

                if all_calibrated:
                    writer.writerow(row_data)
                
                time.sleep(0.001)

    def stop(self):
        self.running = False
        for s in self.sensors.values():
            s.Close()

    def get_latest(self):
        return self.latest_data

class SingleSensorPlotWidget(QtWidgets.QWidget):
    """
    绘图组件：在此处应用标定参数 (FZ_SCALE, SHEAR_GAIN) 进行显示
    """
    def __init__(self, label, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.win = pg.GraphicsLayoutWidget(title=f"Sensor: {label}")
        layout.addWidget(self.win)
        
        self.plots = []
        self.curves = []
        self.history = np.zeros((3, int(FPS * WINDOW_SEC)))
        
        # 标题显示当前的 Gain
        titles = [
            f"{label} - Normal Fz (x{FZ_SCALE})", 
            f"{label} - Shear Y (G={SHEAR_GAIN})", 
            f"{label} - Shear X (G={SHEAR_GAIN})"
        ]
        colors = ['r', 'g', 'b']
        y_ranges = [(-20, 20), (-20, 20), (-20, 20)]
        
        # --- 核心标定逻辑 ---
        # 1. Fz: 乘以 FZ_SCALE (1.3)
        # 2. Mx/My: 除以力臂再乘以 SHEAR_GAIN (4.5)
        shear_scale = (1.0 / LEVER_ARM_M) * SHEAR_GAIN
        self.scales = [FZ_SCALE, shear_scale, shear_scale]

        for i in range(3):
            p = self.win.addPlot(title=titles[i])
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setYRange(*y_ranges[i], padding=0)
            p.setMouseEnabled(y=False)
            curve = p.plot(pen=pg.mkPen(color=colors[i], width=2))
            self.plots.append(p)
            self.curves.append(curve)
            self.win.nextRow()
            
        self.plots[0].setXLink(self.plots[1])
        self.plots[1].setXLink(self.plots[2])

    def update_data(self, vec_3d):
        self.history = np.roll(self.history, -1, axis=1)
        # 应用标定系数
        display_vals = [v * s for v, s in zip(vec_3d, self.scales)]
        self.history[:, -1] = display_vals
        for i in range(3):
            self.curves[i].setData(self.history[i])

class MainApp(QtWidgets.QMainWindow):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        
        configs = scan_ports()
        if not configs:
            print("无法启动：未找到传感器")
            sys.exit(1)
            
        self.thread = MultiSensorThread(filename, configs)
        self.thread.start()
        
        self.initUI(configs)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(int(1000/FPS))

    def initUI(self, configs):
        self.setWindowTitle(f"Multi-Sensor Calibrated (Fz x{FZ_SCALE}, G={SHEAR_GAIN})")
        self.resize(400 * len(configs), 800)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        
        self.plot_widgets = {}
        for cfg in configs:
            label = cfg['label']
            widget = SingleSensorPlotWidget(label)
            layout.addWidget(widget)
            self.plot_widgets[label] = widget

    def update_loop(self):
        data_map = self.thread.get_latest()
        for label, vec in data_map.items():
            if label in self.plot_widgets:
                self.plot_widgets[label].update_data(vec)

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.join()
        print(f"数据已保存至: {self.filename}")
        event.accept()

def main():
    print("=== 多传感器自动检测程序 (已标定) ===")
    print(f"当前参数: 力臂={LEVER_ARM_MM}mm, Fz_Scale={FZ_SCALE}, Shear_Gain={SHEAR_GAIN}")
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if not os.path.exists(data_dir): os.makedirs(data_dir)
        
    default_name = time.strftime("data_%Y%m%d_%H%M%S.csv")
    fname = input(f"文件名 (默认 {default_name}): ").strip() or default_name
    if not fname.endswith('.csv'): fname += '.csv'
    
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp(os.path.join(data_dir, fname))
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()