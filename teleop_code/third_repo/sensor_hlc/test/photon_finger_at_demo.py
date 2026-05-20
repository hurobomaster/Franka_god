import sys
import os

# 将 src 目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sensorConnector import SensorConnector 
from sensorConnector import SensorType
from sensorConnector import CommucationProtocol

def main():

    sensor= SensorConnector(CommucationProtocol.AT_Command, SensorType.PHOTON_FINGER, "COM22", 460800)

    # 连接设备
    if not sensor.Connect():
        return

    ## 设置获取数据的间隔时长 0.02秒
    sensor.set_read_break(0.001)  

    # 循环读取和处理数据
    ok,resp = sensor.sendCommand("AT+SZERO=1")
    if(  not ok  ):  # 传感器置零
        return

    try:
        while True:
            data= sensor.GetData()
            if(data is not None):
                Fz = data.get('Fz', None)
                Mx = data.get('Mx', None)
                My = data.get('My', None)

                print(f"Fz: {Fz}, Mx: {Mx}, My: {My}")  
 
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down...")

    finally:
        sensor.Close()
        # 关闭连接
        
if __name__ == "__main__":
    main()
