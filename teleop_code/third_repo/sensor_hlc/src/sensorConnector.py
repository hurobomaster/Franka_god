import threading
import struct
from enum import Enum
try:
    from pymodbus.client import ModbusSerialClient
except ImportError:
    try:
        # Compatible with newer pymodbus package layouts.
        from pymodbus.client.serial import ModbusSerialClient
    except ImportError:
        try:
            # Compatible with pymodbus 2.x.
            from pymodbus.client.sync import ModbusSerialClient
        except ImportError as exc:
            raise ImportError(
                "Cannot import ModbusSerialClient. Install a compatible pymodbus version."
            ) from exc
import serial
from time import sleep
import platform


'''获取传感器的Fx,Fy,Fz Mx,My'''
INIT_NUM = 20                 # 偏移量的采样次数
# REGISTER_COUNT = 6            # 寄存器总数
# OFFSET_REG_COUNT  = 3                # 偏移寄存器的数量

class SensorType(Enum):
    PHOTON_56P      = "Photon_56P"
    PHOTON_FINGER   = "Photon_FINGER"
    PHOTON_SGD      = "Photon_SGD"
    PHOTON_R40      = "Photon_R40"
    Photon_FiveInOne      = "Photon_FiveInOne"
    
class CommucationProtocol(Enum):
    NoneProtocol  = "NoneProtocol"
    Serial        = "Serial"
    Modbus        = "Modbus"
    AT_Command    = "AT_Command"

 
class SensorConnector:

    ## 初始化指令
    SENSOR_CONFIGS_Modbus_GETDATA = {
        SensorType.PHOTON_56P:      {"address": 56, "slave": 1, "REGISTER_COUNT": 12},
        SensorType.PHOTON_FINGER:   {"address": 14, "slave": 1, "REGISTER_COUNT": 6,},
        SensorType.PHOTON_SGD:      {"address": 56, "slave": 1, "REGISTER_COUNT": 6 },
        SensorType.PHOTON_R40:      {"address": 56, "slave": 1, "REGISTER_COUNT": 12},
        SensorType.Photon_FiveInOne:{"address": 56, "slave": 1, "REGISTER_COUNT": 12},
    }

    ## 初始化指令
    SENSOR_CONFIGS_Serial_GETDATA = {
        SensorType.PHOTON_56P:          {"GetDataCmd" : [0x01, 0x04, 0x00, 0x38, 0x00, 0x0C, 0x71, 0xC2] , "package_len" : 29,},
        SensorType.PHOTON_FINGER:       {"GetDataCmd" : [0x01, 0x04, 0x00, 0x0E, 0x00, 0x06, 0x11, 0xCB] , "package_len" : 17,},
        SensorType.PHOTON_SGD:          {"GetDataCmd" : [0x01, 0x04, 0x00, 0x38, 0x00, 0x06, 0xF1, 0xC5] , "package_len" : 17,},
        SensorType.PHOTON_R40:          {"GetDataCmd" : [0x01, 0x04, 0x00, 0x38, 0x00, 0x0C, 0x71, 0xC2] , "package_len" : 29,},
        SensorType.Photon_FiveInOne:    {"GetDataCmd" : [0x01, 0x04, 0x00, 0x38, 0x00, 0x0C, 0x71, 0xC2] , "package_len" : 29,},
    }

    ## 初始化指令
    SENSOR_CONFIGS_AT_GETDATA = {
        SensorType.PHOTON_56P: {
            "TestCmd": "AT+TEST=?",
            "GetDataCmd": "AT+GTORQ=?",
            "setOffsetForce": "AT+SZERO=1",
            "unsetOffsetForce": "AT+SZERO=0",
            "unoffsetForce": "AT+RSZERO=1",
        },

        SensorType.PHOTON_FINGER: {
            "TestCmd": "AT+TEST=?",
            "GetDataCmd": "AT+GTORQ=?",
            "setOffsetForce": "AT+SZERO=1",
            "unsetOffsetForce": "AT+SZERO=0",
            "unoffsetForce": "AT+RSZERO=1",
        },

        SensorType.PHOTON_SGD: {
            "TestCmd": "AT+TEST=?",
            "GetDataCmd": "AT+GTORQ=?",
            "setOffsetForce": "AT+SZERO=1",
            "unsetOffsetForce": "AT+SZERO=0",
            "unoffsetForce": "AT+RSZERO=1",
        },

        SensorType.PHOTON_R40: {
            "TestCmd": "AT+TEST=?",
            "GetDataCmd": "AT+GTORQ=?",
            "setOffsetForce": "AT+SZERO=1",
            "unsetOffsetForce": "AT+SZERO=0",
            "unoffsetForce": "AT+RSZERO=1",
        },

        SensorType.Photon_FiveInOne: {
            "TestCmd": "AT+TEST=?",
            "GetDataCmd": "AT+GTORQ=?",
            "setOffsetForce": "AT+SZERO=1",
            "unsetOffsetForce": "AT+SZERO=0",
            "unoffsetForce": "AT+RSZERO=1",
            "GetSensorID": "AT+SNSRN=?",
            "GetFiveSensorData": "AT+SNSRN=5",
            "COORD": "AT+COORD=?",
            "switchSensorOne": "AT+SNSRN=1",
            "FIO_SET_OFFSET": "AT+SZERO=5",
            "GetBuad":"AT+BAUDR=?",
            "Baud_256000":"AT+BAUDR=256000",
            "Baud_921600":"AT+BAUDR=921600",
            "Baud_115200":"AT+BAUDR=115200",
            "Save":"AT+SAVE=1",
        },
    }

    

    def __init__(self, protocol, sensor_type, port, baud):
        """
        初始化传感器
        protocol: 连接协议
        :param client: Modbus RTU 客户端
        :param sensor_type: 传感器类型 (如 "Photon_56P","Photon_FINGER","Photon_SGD","Photon_R40")
        """

        if(sensor_type not in self.SENSOR_CONFIGS_Modbus_GETDATA):
            raise ValueError(f"Nokown sensor type.{sensor_type}")

        if(protocol not in CommucationProtocol):
            raise ValueError(f"Nokown sensor type.{protocol}")
        
        self.sensor_type = sensor_type
        self.protocol    = protocol
        self.port = port
        self.baud = baud
        self.init_count=0
        self.read_break = 10
        self.ret_coord = False 

 
        if(self.protocol == CommucationProtocol.Modbus):
            config = self.SENSOR_CONFIGS_Modbus_GETDATA[sensor_type]
            self.offsets = [0] * (config["REGISTER_COUNT"])
            self.address=config["address"]
            self.slave =config["slave"]
            self.count=config["REGISTER_COUNT"]
            
            is_windows = platform.system() == "Windows"
            
            if is_windows:
                self.client = ModbusSerialClient(
                method='rtu',
                port=self.port,       # win: port改成类似COM12 
                baudrate=self.baud, 
                timeout=5,       
                parity='N',      
                stopbits=1,      
                bytesize=8     
                )
            else:
                self.client = ModbusSerialClient(
                port=self.port,      #linux : # port='/dev/ttyACM0',   
                baudrate=self.baud, 
                timeout=5,       
                parity='N',      
                stopbits=1,      
                bytesize=8     
                )

        elif(self.protocol == CommucationProtocol.Serial):
            self.GetDataCMD = self.SENSOR_CONFIGS_Serial_GETDATA[sensor_type]
            config = self.SENSOR_CONFIGS[sensor_type]
            self.offsets = [0] * (config["REGISTER_COUNT"])

        elif(self.protocol == CommucationProtocol.AT_Command):
            self.GetDataCMD = self.SENSOR_CONFIGS_AT_GETDATA[sensor_type]
            

        else:    #NoneProtocol
            print(f"{self.__class__.__name__}, 类型错误 (self.protocol)")
            return ValueError     
 

    def Connect(self):
        """尝试连接到设备并返回连接状态"""
        if(self.protocol == CommucationProtocol.Modbus):
            if( self.client.connect() ):
                print(f"{self.__class__.__name__},连接成功")
                return True
            else:
                print(f"{self.__class__.__name__},连接失败")
                return False
            
        try:
            if self.protocol == CommucationProtocol.Serial:
                self.serial_connection = serial.Serial(self.port, baudrate=self.baud, timeout=1)
                print(f"{self.__class__.__name__}, 连接成功 (Serial)")
                return True

            elif self.protocol == CommucationProtocol.AT_Command:
                self.serial_connection = serial.Serial(self.port, baudrate=self.baud, timeout=1)
                print(f"{self.__class__.__name__}, 连接成功 (AT_Command)")
                return True

        except serial.SerialException as e:
            print(f"{self.__class__.__name__}, 串口连接失败: {e}")
            self.serial_connection = None
            return False

        except Exception as e:
            print(f"{self.__class__.__name__}, 未知错误: {e}")
            self.serial_connection = None
            return False

    def Close(self):
        if(self.protocol == CommucationProtocol.Modbus):
            self.client.close()  

        elif(self.protocol == CommucationProtocol.Serial):
            self.serial_connection.close()

        elif(self.protocol == CommucationProtocol.AT_Command):
            self.serial_connection.close()

    def ReadRawBuff(self):
        """从设备读取输入寄存器"""
        if(self.protocol == CommucationProtocol.AT_Command):            
            command = self.SENSOR_CONFIGS_AT_GETDATA[self.sensor_type]["GetDataCmd"]
            result = self.serial_connection.write(command.encode('utf-8'))
            sleep(self.read_break) 
            if self.serial_connection.in_waiting > 0:
                resp_data = self.serial_connection.read_all()
                # print( resp_data.hex() )
                try:
                    return resp_data
                except Exception as e:
                    print("read data filed",e)
            else:
                return None
            
        elif(self.protocol == CommucationProtocol.Modbus):
            result = self.client.read_input_registers(address=self.address, count=self.count, slave=self.slave)
            sleep(self.read_break)
            if not result.isError():
                return result.registers
            else:
                print(f"Modbus Error: {result}")
                return None
            
        elif(self.protocol == CommucationProtocol.Serial):
            command = self.SENSOR_CONFIGS_Serial_GETDATA[self.sensor_type]["GetDataCmd"]
            result = self.serial_connection.write(bytes(command))
            sleep(self.read_break) 
            if self.serial_connection.in_waiting > 0:
                resp_data = self.serial_connection.read(self.SENSOR_CONFIGS_Serial_GETDATA[self.sensor_type]["package_len"])
                try:
                    data = resp_data[ 3 : len(resp_data)-2]
                    # print(data.hex())
                    return data
                except Exception as e:
                    print("read data filed",e)
            else:
                return None
            
                
        
            
        

    def calculateCRC(self,buffer: bytes) -> bytes:
        wcrc = 0xFFFF  # 16位CRC寄存器预置
        for temp in buffer:
            wcrc ^= temp
            for _ in range(8):
                if wcrc & 0x0001:
                    wcrc >>= 1
                    wcrc ^= 0xA001  # 多项式
                else:
                    wcrc >>= 1

        crc_l = wcrc & 0xFF      # 低八位
        crc_h = (wcrc >> 8) & 0xFF  # 高八位 
        return bytes([crc_l, crc_h])

    def ProcessData(self):

        if(self.protocol == CommucationProtocol.Modbus):
            if (self.raw_data is None): #获取modbus原始数据
                return
            for i in range(0, len(self.raw_data), 2):
                # 组合高位和低位寄存器为32位整数 ， 每两个16位寄存器合成一个32位浮点数（Modbus协议中，通常32位浮点数会分成两组16位寄存器）
                combined = (self.raw_data[i] << 16) | self.raw_data[i + 1]
                curr_Val = struct.unpack('>f', struct.pack('>I', combined))[0]          #将32位整数转换为浮点数，采用大端模式('>f')来解释

                if self.init_count < INIT_NUM:
                    self.offsets[i // 2] += (curr_Val / INIT_NUM)  # 累加偏移值
                else:
                    offsetData = self.offsets[i // 2]
                    self.processed_data.append(curr_Val - offsetData )  # 计算偏移后的值

            if self.init_count <= INIT_NUM :  #偏移量
                self.init_count += 1 
                return None
            
        elif (self.protocol == CommucationProtocol.Serial):
            test = len(self.raw_data)
            
            for i in range(0,len(self.raw_data),4):
                chunk = self.raw_data[i:i+4]
                curr_Val = struct.unpack('>f',chunk)[0]

                if self.init_count < INIT_NUM:
                    self.offsets = self.offsets if self.offsets else [0.0] * int(len(self.raw_data) / 4.0) #初始化offset
                    self.offsets[i // 4] += (curr_Val / INIT_NUM)  # 累加偏移值
                else:
                    offsetData = self.offsets[i // 4]
                    self.processed_data.append(curr_Val - offsetData )  # 计算偏移后的值

            if self.init_count <= INIT_NUM :  #偏移量
                self.init_count += 1 
                return None
        
        elif (self.protocol == CommucationProtocol.AT_Command):

            if self.sensor_type.value == SensorType.Photon_FiveInOne.value:
            
                if self.raw_data[:2] != b'\xAA\x55':
                    self.serial_connection.reset_input_buffer()
                    return None
                
                packLen = self.raw_data[2] #总长度
                frameLen = self.raw_data[3] #单包长度
                total_len = len(self.raw_data)

                packCRC = self.raw_data[-2:] 
                calculCRC = self.raw_data[:-2]

                calCRC = self.calculateCRC(calculCRC)
    

                if calCRC != packCRC:
                    print("CRC 校验失败")
                    return None
                
                start_idx = 4;
                end_idx = start_idx + frameLen;


                while start_idx < total_len:
                    if end_idx > total_len -2:
                        break

                    if packLen == 0:
                        dataLen = self.raw_data[end_idx]
                        start_idx += 1
                        end_idx = start_idx + dataLen
                        continue

                    for i in range(start_idx,end_idx,4):
                        chunk = self.raw_data[i:i+4]
                        
                        chunk_hex = chunk.hex()

                        curr_Val = struct.unpack('>f',chunk)[0]
                        self.processed_data.append(curr_Val )  

                    length = self.raw_data[end_idx]
                    start_idx = end_idx + 1
                    end_idx = start_idx + length

            else:
                # 核心修正：从缓冲区末尾向前查找最新的完整包
                # 格式: [AA 55] [Len] [Data...] [CRC_L] [CRC_H]
                header = b'\xAA\x55'
                last_idx = self.raw_data.rfind(header)
                if last_idx == -1:
                    return None
                
                # 检查数据包是否完整（根据 header 后面的长度位）
                if last_idx + 2 >= len(self.raw_data):
                    return None
                packLen = self.raw_data[last_idx + 2]
                
                if last_idx + 3 + packLen + 2 > len(self.raw_data):
                    # 如果最后一包没传完，找倒数第二包
                    last_idx = self.raw_data[:last_idx].rfind(header)
                    if last_idx == -1: return None
                    packLen = self.raw_data[last_idx + 2]

                packet = self.raw_data[last_idx : last_idx + 3 + packLen + 2]
                
                # 执行 CRC 和 数据提取
                packCRC = packet[-2:] 
                calculCRC = packet[:-2]
                calCRC = self.calculateCRC(calculCRC)

                if calCRC != packCRC:
                    return None
                
                start_offset = 3
                for i in range(start_offset, start_offset + packLen, 4):
                    chunk = packet[i:i+4]
                    if len(chunk) == 4:
                        curr_Val = struct.unpack('>f', chunk)[0]
                        self.processed_data.append(curr_Val)
 
    

    def GetData(self):
        """处理读取到的数据，计算偏移值和浮点数"""
        """把modbus的小端转大端后转成浮点数"""
        self.processed_data = []

        self.raw_data = self.ReadRawBuff()

        if(self.raw_data is None):
            return
        self.ProcessData()

        if self.sensor_type.value == SensorType.PHOTON_FINGER.value:
            return {
                'Fz': self.processed_data[2] if len(self.processed_data) > 0 else None,
                'Mx': self.processed_data[0] if len(self.processed_data) > 1 else None,
                'My': self.processed_data[1] if len(self.processed_data) > 2 else None
            }
        if self.sensor_type.value == SensorType.PHOTON_56P.value:
            return {
                'Fx': self.processed_data[0] if len(self.processed_data) > 0 else None,
                'Fy': self.processed_data[1] if len(self.processed_data) > 1 else None,
                'Fz': self.processed_data[2] if len(self.processed_data) > 2 else None,
                'Mx': self.processed_data[3] if len(self.processed_data) > 3 else None,
                'My': self.processed_data[4] if len(self.processed_data) > 4 else None,
                'Mz': self.processed_data[5] if len(self.processed_data) > 5 else None
            }
        if self.sensor_type.value == SensorType.PHOTON_SGD.value:
            return {
                'Fx': self.processed_data[0] if len(self.processed_data) > 0 else None,
                'Fy': self.processed_data[1] if len(self.processed_data) > 1 else None,
                'Fz': self.processed_data[2] if len(self.processed_data) > 2 else None,
            }
        if self.sensor_type.value == SensorType.PHOTON_R40.value:
            return {
                'Fx': self.processed_data[0] if len(self.processed_data) > 0 else None,
                'Fy': self.processed_data[1] if len(self.processed_data) > 1 else None,
                'Fz': self.processed_data[2] if len(self.processed_data) > 2 else None,
                'Mx': self.processed_data[3] if len(self.processed_data) > 3 else None,
                'My': self.processed_data[4] if len(self.processed_data) > 4 else None,
                'Mz': self.processed_data[5] if len(self.processed_data) > 5 else None
            }
        if self.sensor_type.value == SensorType.PHOTON_R40.value:
            return {
                'Fx': self.processed_data[0] if len(self.processed_data) > 0 else None,
                'Fy': self.processed_data[1] if len(self.processed_data) > 1 else None,
                'Fz': self.processed_data[2] if len(self.processed_data) > 2 else None,
                'Mx': self.processed_data[3] if len(self.processed_data) > 3 else None,
                'My': self.processed_data[4] if len(self.processed_data) > 4 else None,
                'Mz': self.processed_data[5] if len(self.processed_data) > 5 else None
            }
        
        if self.sensor_type.value == SensorType.Photon_FiveInOne.value:
            result = {}
            for i in range(5):  # 处理5组数据
                if len(self.processed_data) > i*3 + 2:
                    if self.ret_coord:
                        result.update({
                            f'Fx{i}': self.processed_data[i*3] if len(self.processed_data) > 5 else None,
                            f'Fy{i}': self.processed_data[i*3+1] if len(self.processed_data) > 5 else None,
                            f'Fz{i}': self.processed_data[i*3+2] if len(self.processed_data) > 5 else None
                        })
                    else:
                        result.update({
                            f'Mx{i}': self.processed_data[i*3] if len(self.processed_data) > 5 else None,
                            f'My{i}': self.processed_data[i*3+1] if len(self.processed_data) > 5 else None,
                            f'Fz{i}': self.processed_data[i*3+2] if len(self.processed_data) > 5 else None
                        })

            return result
        return None
    

    def sendCommand(self,command):
        """发送AT指令""" 
        sp_text = command.split("+", 1)[1].split("\r", 1)[0]
        spc_command = "ACK+" + sp_text.split("=", 1)[0]
        valid_cmds_dicts = self.SENSOR_CONFIGS_AT_GETDATA.get(self.sensor_type)

        if not valid_cmds_dicts:
            print(f"[错误] 未知的 sensor_type: {self.sensor_type}")
            return False,""

        valid_cmds = list(valid_cmds_dicts.values())

        if command not in valid_cmds:
            print(f"[警告] 非法指令: {command}")
            print(f"[提示] 合法指令包括: {valid_cmds}")
            return False,""
        
        sleep(0.02)
        result = self.serial_connection.write(command.encode('utf-8'))
        sleep(0.02)

        if self.serial_connection.in_waiting > 0:
            try:
                resp_data = self.serial_connection.read_all()
                resp_str = resp_data.decode("utf-8", "ignore") 
                if resp_str.startswith(spc_command):
                    return True,resp_str
                
                elif "failed" in resp_str.lower():
                    return False,resp_str
                else:
                    return False,""
                
            except Exception as e:
                print("读取数据失败,请稍后重试", e)
                return False,resp_str
        else:
            print("读取数据失败,请加大读取时间")
            return False,""
    
  
    def FIO_init_check(self):
        """执行一次传感器数据读取"""        
        command = self.SENSOR_CONFIGS_AT_GETDATA[self.sensor_type]["switchSensorOne"]
        ok,resp = self.sendCommand(command) 
        if(ok):
            command = self.SENSOR_CONFIGS_AT_GETDATA[self.sensor_type]["COORD"]
            ok,resp = self.sendCommand(command)
            value = resp.split("=", 1)[1].split("\r\n", 1)[0]
            value = value.strip("()")      # 去括号
            get_sw = value.split(",", 1)[0] 
            if get_sw == "1" : 
                self.ret_coord = True

        ##  SNSRN
        command = self.SENSOR_CONFIGS_AT_GETDATA[self.sensor_type]["GetFiveSensorData"]
        ok,resp = self.sendCommand(command)
        ret_SNSRN = False
        if ok:
            if resp.startswith("ACK+SNSRN=5"):

                command = self.SENSOR_CONFIGS_AT_GETDATA[self.sensor_type]["FIO_SET_OFFSET"]
                ok,resp = self.sendCommand(command) ##置零
                ret_SNSRN = True
                return self.ret_coord,ret_SNSRN
            
        else:
            return self.ret_coord,ret_SNSRN
        

    def set_read_break(self,set_break):
        """设置获取数据的间隔时长"""      
        self.read_break = set_break
        return self.read_break
    
    
    def set_zero_modbus(self):

        values = [0x0000, 0x0001]

        result = self.client.write_registers(
            address=12,        
            values=values,    
            slave=1
        )
        sleep(2) ##缓冲时间

        if not result.isError(): 
            return True
        else:
            return False