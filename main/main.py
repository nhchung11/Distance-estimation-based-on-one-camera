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
cap = cv2.VideoCapture(r"D:\python\output.mp4")         # Video capture
H = 1.22                                                # UAV height (m)
d = 4.2                                                 # UAV distance to world coordiante origin (m)
p = 50                                                  # Compass angle (degree)
alpha = 75                                              # Camera angle (degree)
f = 3.29                                                # Focal length (mm)
s_w, s_h = 3.67, 2.74                                   # Sensor size (mm)
w, h = 1024, 768                                        # Resolution (pixcel)
n_people = 0                                            # Number of people
fx, fy = 917.17, 922.16                                 # Focal length
CONFIDENCE_THRESHOLD = 0.75                             # Confidence threshold
GREEN = (0, 255, 0)                                     # BGR color
track_history = defaultdict(lambda: [])                 # Track history
P = (int(w/2), int(h/2), 0)                             # Principle point
UAV_lat, UAV_lon = 21.0261309, 105.8328612              # UAV latitude and longitude
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
R = np.array([[1, 0, 0],
            [0, np.cos(math.radians(alpha)), np.sin(math.radians(alpha))],
            [0, -np.sin(math.radians(alpha)), np.cos(math.radians(alpha))]])
R_inv = np.linalg.inv(R)
r31 = R_inv[2][0]
r32 = R_inv[2][1]
r33 = R_inv[2][2]

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
        detections = model(frame, verbose=False)[0]
        for data in detections.boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]
            class_id = data[5]
            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD or int(class_id) != 0:
                continue

            # if the confidence is greater than the minimum confidence,
            # draw the bounding box on the frame
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            xf = int((xmin + xmax) / 2)
            yf = int(ymax)
            x_over_z = (xf - P[0]) / fx
            y_over_z = (yf - P[1]) / fy
            zc = (r31*tx + r32*ty + r33*tz) / (r31*x_over_z + r32*y_over_z + r33)
            Xc = np.array([[x_over_z*zc], [y_over_z*zc], [zc]])
            Xw = -np.dot(R_inv, (Xc - t))
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
            GO = [0, d]
            GF = [0 - Xw[0][0], d - Xw[1][0]]
            cos_theta = (GO[0]*GF[0] + GO[1]*GF[1]) / (math.sqrt(GO[0]**2 + GO[1]**2) * math.sqrt(GF[0]**2 + GF[1]**2))
            theta = np.degrees(np.arccos(cos_theta))
            distance = math.sqrt(GF[0]**2 + GF[1]**2)
            length = 0.0001
            with open ('compass.txt', 'a') as f:
                if P[0] < xf:
                    end_lat_direct = UAV_lat + length * math.cos(math.radians(p + theta))
                    end_lon_direct = UAV_lon + length * math.sin(math.radians(p + theta))
                else:
                    end_lat_direct = UAV_lat + length * math.cos(math.radians(p - theta))
                    end_lon_direct = UAV_lon + length * math.sin(math.radians(p - theta))
                f.write(f"{end_lat_direct} {end_lon_direct}\n")
            print("Target: ", end_lat_direct, end_lon_direct)
            # print("Xw: ", np.round(Xw.T, 6))
        cv2.imshow("Frame", frame)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()