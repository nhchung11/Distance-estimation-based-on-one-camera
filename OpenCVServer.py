import cv2
import socket
import pickle
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ip = "192.168.2.236"
ip = "127.0.0.1"
port = 6666
s.bind((ip, port))

while True:
    x = s.recvfrom(1000000)
    # print(type(x));
    clientip = x[1][0]
    # print(type(clientip))
    data = x[0]

    data = pickle.loads(data)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # print(type(img))
    cv2.imshow('Img Server', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()