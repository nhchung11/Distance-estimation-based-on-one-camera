import numpy as np
import cv2
from ultralytics import YOLO
import cv2
import math
from collections import defaultdict
import pickle
import socket
from tkinter import *
import tkintermapview
from PIL import Image, ImageTk
import threading
import math

# INPUTS
model = YOLO("yolov8n.pt")                              # YOLO model
H = 4.47                                                # UAV height (m)
d = 17.65                                                 # UAV distance to world coordiante origin (m)
p = 90                                                  # Compass angle (degree)
alpha = 78                                             # Camera angle (degree)
f = 3.29                                                # Focal length (mm)
s_w, s_h = 3.67, 2.74                                   # Sensor size (mm)
w, h = 1024, 768                                        # Resolution (pixcel)
n_people = 0                                            # Number of people
fx, fy = 917.17, 922.16                                 # Focal length
CONFIDENCE_THRESHOLD = 0.75                             # Confidence threshold
GREEN = (0, 255, 0)                                     # BGR color
track_history = defaultdict(lambda: [])                 # Track history
P = (int(w/2), int(h/2), 0)                             # Principle point
UAV_lat, UAV_lon = 21.0058026,105.8425124               # UAV latitude and longitude
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


# Open video source
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ip = "127.0.0.1"
ip = "192.168.253.8"
port = 6666
s.bind((ip, port))

root = Tk()
root.geometry('2048x768')
main_frame = Frame(root, width=1024, height=768)
main_frame.pack(fill="both", expand=True)
# cap = cv2.VideoCapture(0)

map_frame = Frame(main_frame, background="blue")
map_label = Label(map_frame, text = "Map")
map_label.place(relx=0.5, rely=0.5,anchor=CENTER)
map_frame.pack(expand = True, fill = BOTH, side=LEFT)
map_widget = tkintermapview.TkinterMapView(map_frame, width=1024, height=768)
map_widget.set_position(UAV_lat, UAV_lon)

def calculate_end_point(lat, lon, distance, angle):
    # Earth's radius in kilometers
    R = 6378.1
    
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Convert distance to kilometers
    distance_km = distance / 1000
    
    # Calculate end point
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(angle_rad))
    
    lon2 = lon1 + math.atan2(math.sin(angle_rad) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))
    
    # Convert back to degrees
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    
    return lat2, lon2

# Create markers
marker_1 = map_widget.set_marker(21.003970, 105.842779, "UAV")
marker_2 = map_widget.set_marker(21.003970, 105.842779, "Target")
end_lat, end_lon = calculate_end_point(UAV_lat, UAV_lon, 10, p)
direct = map_widget.set_path([(UAV_lat, UAV_lon), (end_lat, end_lon)])
map_widget.set_zoom(20)
map_widget.pack(fill="both", expand=True)

camera_frame = Frame(main_frame, background="red", width=1024, height=768)
camera_label = Label(camera_frame, text= "Camera")
camera_label.place(relx=0.5, rely=0.5,anchor=CENTER)
camera_frame.pack(expand=True, fill=BOTH, side=LEFT)

def show_frame():
    marker_1.set_position(UAV_lat, UAV_lon)
    data, addr = s.recvfrom(1000000)
    frame = pickle.loads(data)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    results = model(frame, verbose=False)[0]
    for result in results.boxes.data.tolist():
        confidence = result[4]
        class_id = result[5]
        if float(confidence) < CONFIDENCE_THRESHOLD or int(class_id) != 0:
            # marker_2.delete()
            continue
        
        xmin, ymin, xmax, ymax = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
        xf = int((xmin + xmax) / 2)
        yf = int(ymax)
        x_over_z = (xf - P[0]) / fx
        y_over_z = (yf - P[1]) / fy
        zc = (r31*tx + r32*ty + r33*tz) / (r31*x_over_z + r32*y_over_z + r33)
        Xc = np.array([[x_over_z*zc], [y_over_z*zc], [zc]])
        Xw = -np.dot(R_inv, (Xc - t))
        GO = [0, d]
        GF = [0 - Xw[0][0], d - Xw[1][0]]
        cos_theta = (GO[0]*GF[0] + GO[1]*GF[1]) / (math.sqrt(GO[0]**2 + GO[1]**2) * math.sqrt(GF[0]**2 + GF[1]**2))
        theta = np.degrees(np.arccos(cos_theta))
        distance = math.sqrt(GF[0]**2 + GF[1]**2)
        length = 0.00001
        if P[0] < xf:
            end_lat_direct = UAV_lat + length * math.cos(math.radians(p + theta)) * distance
            end_lon_direct = UAV_lon + length * math.sin(math.radians(p + theta)) * distance
        else:
            end_lat_direct = UAV_lat + length * math.cos(math.radians(p - theta)) * distance
            end_lon_direct = UAV_lon + length * math.sin(math.radians(p - theta)) * distance
            
        marker_2.set_position(end_lat_direct, end_lon_direct)
        cv2.putText(frame, f"Distance: {distance:.2f} m", (xmin, ymin+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {theta:.2f} degree", (xmin, ymin+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    
    camera_label.config(image=img)
    camera_label.image = img
    root.after(10, show_frame)

thread = threading.Thread(target=show_frame)
thread.daemon = True
thread.start()
root.mainloop()
