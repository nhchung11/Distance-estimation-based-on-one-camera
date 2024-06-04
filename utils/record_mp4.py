import cv2
import socket
import pickle
import numpy as np
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.0.102"
# ip = "127.0.0.1"
port = 6666
s.bind((ip, port))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec and create a VideoWriter object
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1024, 768))

start_time = time.time()  # Start time for capturing video

while(True):
    x = s.recvfrom(1000000)
    clientip = x[1][0]
    data = x[0]

    data = pickle.loads(data)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    current_time = time.time()

    # If 10 seconds have passed since the start, start capturing video
    if current_time - start_time >= 10:
        out.write(img)
        cv2.imshow('frame', img)

        # If 20 seconds have passed since the start, stop capturing video
        if current_time - start_time >= 20:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()