import cv2 
import numpy as np 
import os 
import glob 
  
  
# Define the dimensions of checkerboard 
rows = 8
cols = 6

# Termination and criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
  
# Prepare object points
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []

# Save gray scale
gray = None
  
   
images = glob.glob('D:\python\CheckerBoards\*.jpg') 
  
for (index, path) in enumerate(images): 
    image = cv2.imread(path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None) 
  
    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret: 
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPointsArray.append(objectPoints) 
        imgPointsArray.append(corners)
        cv2.drawChessboardCorners(image, (rows, cols), corners, ret)
    else:
        print(f"Unable to detect corners in image: {path}")
    cv2.imshow('img', image) 
    cv2.waitKey(500) 
  
# cv2.destroyAllWindows() 
  
h, w = image.shape[:2] 
  
  
# Perform camera calibration by 
# passing the value of above found out 3D points (threedpoints) 
# and its corresponding pixel coordinates of the 
# detected corners (twodpoints) 
def getParamters(objectPointsArray, imgPointsArray, gray):
    return cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)

ret, matrix, distortion, r_vecs, t_vecs = getParamters(objectPointsArray, imgPointsArray, gray)
  
  

# Displaying required output 
print(" Camera matrix:") 
print(matrix) 
  
print("\n Distortion coefficient:") 
print(distortion) 
  
print("\n Rotation Vectors:") 
print(r_vecs) 
  
print("\n Translation Vectors:") 
print(t_vecs) 

