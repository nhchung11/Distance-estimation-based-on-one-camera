import cv2
import socket
import pickle
import numpy as np
import time
from ultralytics import YOLO

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.0.104"
# ip = "127.0.0.1"
port = 6666
s.bind((ip, port))
counter = 15
start_time = time.time()
model = YOLO("yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.75                             # Confidence threshold
GREEN = (0, 255, 0)                                     # BGR color

while True:
    x = s.recvfrom(1000000)
    clientip = x[1][0]
    data = x[0]
    data = pickle.loads(data)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    detections = model(img, verbose=False)[0]
    for detection in detections.boxes.data.tolist():
        confidence = detection[4]
        class_id = detection[5]
        if float(confidence) < CONFIDENCE_THRESHOLD or int(class_id) != 0:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), GREEN, 2)
    h, w = img.shape[:2]
    center = (w//2, h//2)

    # Draw the center point
    cv2.circle(img, center, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow('Img Server', img)
    # if cv2.waitKey(1) & 0xFF == ord('s'):
    # if time.time() - start_time >= 10:
    #     cv2.imwrite(f'frame_human_{counter}.jpg', img)
    #     counter += 1
    #     print(f"Frame {counter} saved")
    #     break

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    # Destroy all windows
cv2.destroyAllWindows()