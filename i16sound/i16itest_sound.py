#-*- encoding: utf-8 -*-
import socket
import time

# 继电器 WJ94 - RJ45
# 继电器的ip
#10.248.10.49
if __name__=="__main__":
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(("192.168.0.7",23))
    print("连接成功")
    #DO-1 ON
    s.send(b'\x02\x03\x03')
    print(s.recv(128))
    time.sleep(2)
    #DO-1 OFF
    s.send(b"\x02\x00\x03")
    time.sleep(1)
    s.close()
#'''
