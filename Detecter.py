import datetime
from ultralytics import YOLO
import cv2
import socket
import pickle
import numpy as np
# from helper import create_video_writerq

# Define Server
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.2.236"
# ip = "127.0.0.1"
port = 6666
s.bind((ip, port))

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# initialize the video capture object
# video_cap = cv2.VideoCapture(0)
# initialize the video writer object
# writer = create_video_writer(video_cap, "output.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
while True:
    x = s.recvfrom(1000000)
    clientip = x[1][0]
    data = x[0]
    data = pickle.loads(data)
    # start time to compute the fps
    start = datetime.datetime.now()

    # ret, frame = video_cap.read()
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # if there are no more frames to process, break out of the loop
    # if not ret:
    #     break

    # run the YOLO model on the frame
    detections = model(img, verbose=False)[0]
    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), GREEN, 2)
        # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(img, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", img)
    # writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

# video_cap.release()
# writer.release()
cv2.destroyAllWindows()