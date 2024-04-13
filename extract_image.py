import cv2

cap = cv2.VideoCapture(r'D:\python\Human_tracking\medium.mp4')    
i = 0

while(cap.isOpened()):
    flag, frame = cap.read()
    if flag == False:
        break
    path = 'D:\\python\\Human_tracking\\medium2\\medium' + str(i) + '.jpg'
    cv2.imwrite(path, frame)
    i += 1
cap.release()
cv2.destroyAllWindows()