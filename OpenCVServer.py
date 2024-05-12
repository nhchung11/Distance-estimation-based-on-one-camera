import cv2
import socket
import pickle
import numpy as np
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.0.9"
# ip = "127.0.0.1"
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
    # Calculate the center point
    h, w = img.shape[:2]
    center = (w//2, h//2)

    # Draw the center point
    cv2.circle(img, center, radius=5, color=(0, 255, 0), thickness=-1)
    # print(type(img))
    cv2.imshow('Img Server', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    time.sleep(10)
    cv2.imwrite('captured_image01.jpg', img)
    # Destroy all windows
    cv2.destroyAllWindows()
    break