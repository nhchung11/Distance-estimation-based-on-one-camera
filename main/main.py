import numpy as np
import cv2
from ultralytics import YOLO
import cv2
import math
import folium
import matplotlib.pyplot as plt
import supervision as sv
from collections import defaultdict


# INPUTS
model = YOLO("yolov8n.pt")                              # YOLO model
cap = cv2.VideoCapture(r"D:\python\test_video.mp4")     # Video capture
H = 1.22                                                # UAV height (m)
d = 4.2                                                 # UAV distance to world coordiante origin (m)
p = 50                                                  # Compass angle (degree)
alpha = 75                                              # Camera angle (degree)
f = 3.29                                                # Focal length (mm)
s_w, s_h = 3.67, 2.74                                   # Sensor size (mm)
w, h = 1024, 768                                        # Resolution (pixcel)
n_people = 0                                            # Number of people
fx, fy = 917.17, 922.16                                 # Focal length
CONFIDENCE_THRESHOLD = 0.8                              # Confidence threshold
GREEN = (0, 255, 0)                                     # BGR color
track_history = defaultdict(lambda: [])                 # Track history
P = (int(w/2), int(h/2), 0)                             # Principle point
a, b, c = 0, 0, alpha

# Intrinsic matrix
mtrx = matrix = np.array([[fx, 0, w/2], 
                          [0, fy, h/2], 
                          [0, 0, 1]])

# Rotation matrix
Rz = np.array([[np.cos(np.radians(a)), -np.sin(np.radians(a)), 0],
               [np.sin(np.radians(a)), np.cos(np.radians(a)), 0],
               [0, 0, 1]])
Ry = np.array([[np.cos(np.radians(b)), 0, np.sin(np.radians(b))],
               [0, 1, 0],
               [-np.sin(np.radians(b)), 0, np.cos(np.radians(b))]])  
Rx = np.array([[1, 0, 0],
               [0, np.cos(np.radians(c)), -np.sin(np.radians(c))],
               [0, np.sin(np.radians(c)), np.cos(np.radians(c))]])
R = np.dot(Rz, np.dot(Ry, Rx))
R_inv = np.linalg.inv(R)

# Camera in world coordinate
cw = np.array([[0], [d], [-H]])
t = np.dot(R, cw)
tx, ty, tz = t[0][0], t[1][0], t[2][0]

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, classes=[0], persist=True, save=False, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
        # cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Frame', frame)
    
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()