import sys
import os
import time
import platform

# 将 src 目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sensorConnector import SensorConnector 
from sensorConnector import SensorType
from sensorConnector import CommucationProtocol


def pick_port():
    """Select a valid serial port for current OS."""
    if platform.system() == "Windows":
        return "COM12"

    candidates = [
        "/dev/photon_left",
        "/dev/photon_right",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/dev/ttyUSB0"

def main():

    port = pick_port()
    print(f"Using serial port: {port}")
    sensor= SensorConnector(CommucationProtocol.Modbus, SensorType.PHOTON_FINGER, port, 115200)

    # 连接设备
    if not sensor.Connect():
        return

    ## 设置获取数据的间隔时长 0.03秒
    sensor.set_read_break(0.02)  
 
    # 循环读取和处理数据
    try:
        
        ## 传感器置零 
        if not sensor.set_zero_modbus() :
            return
        
        while True:
            data= sensor.GetData()
            if(data is not None):
                Fz = data.get('Fz', None)
                Mx = data.get('Mx', None)
                My = data.get('My', None)

                print(f"Fz: {Fz}, Mx: {Mx}, My: {My}")  
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down...")

    finally:
        sensor.Close()
        # 关闭连接
        
if __name__ == "__main__":
    main()
