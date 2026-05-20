import sys
import os

# 将 src 目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sensorConnector import SensorConnector 
from sensorConnector import SensorType
from sensorConnector import CommucationProtocol

def main():

    sensor= SensorConnector(CommucationProtocol.AT_Command, SensorType.Photon_FiveInOne, "COM6", 115200)

    # 连接设备
    if not sensor.Connect():
        return
    
    ##修改波特率为256000
    ok,resp = sensor.sendCommand("AT+BAUDR=?")
    if not "256000" in resp:
        ok,resp = sensor.sendCommand("AT+BAUDR=256000")
        if(  ok ):  # 传感器置零
            ok,resp = sensor.sendCommand("AT+SAVE=1")
            if(  ok  ):
                print("波特率修改成功：请重新上电传感器\n")
                return
            

    ## 设置获取数据的间隔时长 0.03秒
    sensor.set_read_break(0.019)  

    # 循环读取和处理数据
    try:
        if sensor.sensor_type.value == SensorType.Photon_FiveInOne.value:
            ret_coord,ret_SNSRN = sensor.FIO_init_check()
            if( not ret_SNSRN ):
                print("请重试。")
                return

        while True:
            data= sensor.GetData()
            if data is not None:
                for i in range(5):
                    if ret_coord:
                        Fx = data.get(f'Fx{i}', None)
                        Fy = data.get(f'Fy{i}', None)
                        Fz = data.get(f'Fz{i}', None)
                        print(f"Fx{i}: {Fx}, Fy{i}: {Fy}, Fz{i}: {Fz}")

                    else:
                        Fz = data.get(f'Fz{i}', None)
                        Mx = data.get(f'Mx{i}', None)
                        My = data.get(f'My{i}', None)
                        print(f"Fz{i}: {Fz}, Mx{i}: {Mx}, My{i}: {My}")
 
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down...")

    finally:
        sensor.Close()
        # 关闭连接
        
if __name__ == "__main__":
    main()
