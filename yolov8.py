from ultralytics import YOLO
import cv2
import numpy as np
import math

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

cx = 160
cy = 208
fx = 3.6    # mm
fy = 3.6    # mm
alpha = 18  # Degree
H = 5       # m
Zp = H * (1 / math.cos(math.radians(alpha)))

know_distance = 40
know_width = 21
face_detector = YOLO("yolov8n.pt")

def FocalLength(measure_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measure_distance)/real_width
    return focal_length

def Distance_finder (Focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width*Focal_length)/face_width_in_frame
    return distance

def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), RED, 1)
        face_width = w
    return face_width
# Image source
video_cap = cv2.VideoCapture(r'D:\python\Human_tracking\scripts\medium27.jpg')
# ref_image_face_width = face_data(video_cap)


# load the pre-trained YOLOv8n model
# model = YOLO("yolov8n.pt")
ret, frame = video_cap.read()
detections = face_detector(frame, verbose=False)[0]

for data in detections.boxes.data.tolist():
    confidence = data[4]

    if float(confidence) < CONFIDENCE_THRESHOLD:
        continue

    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
    xf = 0.5 * (xmin + xmax)
    yf = ymax
    xn = (xf - cx) / fx
    yn = (yf - cy) / fy
    


# show the frame to our screen
cv2.imshow("Frame", frame)
cv2.waitKey(0)