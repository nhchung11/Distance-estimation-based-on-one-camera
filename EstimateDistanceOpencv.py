import cv2
from ultralytics import YOLO

know_distance = 40
know_width = 21

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts =cv2.FONT_HERSHEY_COMPLEX
# model = YOLO("yolov8n.pt")
# face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
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

ref_image = cv2.imread(r"D:\python\Human_tracking\scripts\medium27.jpg")
ref_image_face_width = face_data(ref_image)
Focal_length_found = FocalLength(know_distance,know_width,ref_image_face_width)
print(Focal_length_found)
cv2.imshow("ref_image", ref_image)
print(ref_image_face_width)
cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    face_width_in_frame = face_data(frame)
    if face_width_in_frame !=0:
        Distance = Distance_finder(Focal_length_found, know_width, face_width_in_frame)
        cv2.putText(frame, f"Distance={Distance}",(50,50),fonts, 0.6, (RED),2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
