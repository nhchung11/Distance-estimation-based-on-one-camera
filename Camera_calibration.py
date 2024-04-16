import numpy as np
import cv2
import glob
import sys
import random

showGui = not ('--no-gui' in sys.argv)
imagesBasePath = sys.argv[-1]

# imagesBasePath ending with '.py' implies that the user did not pass any arguments
if '--help' in sys.argv or imagesBasePath.endswith('.py'):
    print('Usage: python cameraCalib.py [--no-gui] images_path')
    print('  --no-gui: disable OpenCV GUI (may be required on Linux systems with GTK)')
    print('  images_path: path to directory containing calibration images')
    sys.exit()

# Define the chess board rows and columns
rows = 8
cols = 6

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []

# Save the grayscale version of the last image
gray = None

# Loop over the image files
print(f"Reading images from directory: {imagesBasePath}")
imagesToParse = glob.glob(imagesBasePath+'/*.jp*g')

if len(imagesToParse) == 0:
    print('Unable to find any jpeg images in the passed directory.')
    sys.exit()

for (index, path) in enumerate(imagesToParse):
    print(f"Reading image: {path} ({index+1}/{len(imagesToParse)})")
    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # Make sure the chess board pattern was found in the image
    if ret:
        # Refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)
        if showGui:
            # Draw the corners on the image
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

    if showGui:
        # Display the image
        cv2.imshow('chess board', img)
        cv2.waitKey(500)

# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez(imagesBasePath+'/calib_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("ret", ret)
print("mtx", mtx)
print("dist", dist)
print("rvecs", rvecs)
print("tvecs", tvecs)
print("imageSize", gray.shape[::-1])

# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))

# Load one of the test images
one_file = random.choice(imagesToParse)
img = cv2.imread(one_file)
h, w = img.shape[:2]

# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)

# Crop the undistorted image
# x, y, w, h = roi
# undistortedImg = undistortedImg[y:y + h, x:x + w]
fx, fy, height, ppx, ppy, width = mtx[0][0], mtx[1][1], h, mtx[0][2], mtx[1][2], w
rk1, rk2, tp1, tp2, rk3 = dist[0]
print(
    f'\n'
    f'"intrinsic_parameters": {{\n'
    f'   "fx": {fx},\n'
    f'   "fy": {fy},\n'
    f'   "height_px": {height},\n'
    f'   "ppx": {ppx},\n'
    f'   "ppy": {ppy},\n'
    f'   "width_px": {width}\n'
    f' }},\n'
    f' "distortion_parameters": {{\n'
    f'   "rk1": {rk1},\n'
    f'   "rk2": {rk2},\n'
    f'   "rk3": {rk3},\n'
    f'   "tp1": {tp1},\n'
    f'   "tp2": {tp2}\n'
    f' }},\n'
)

# Display the final result
if showGui:
    print('Showing original vs undistorted image')
    print('Press \'0\' to close the window')
    cv2.imshow('chess board', np.hstack((img, undistortedImg)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
Output:
ret 0.6908937569159879
mtx [[2.44095401e+03 0.00000000e+00 1.24813626e+03]
 [0.00000000e+00 2.43467286e+03 9.57016786e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
dist [[ 1.04610123e-01 -5.43544468e-03  6.06591989e-04  4.44426274e-03
  -1.04927194e+00]]
rvecs (array([[-0.21481338],
       [-0.13106442],
       [-0.05167473]]), array([[-0.27433151],
       [-0.64540804],
       [-0.02226883]]), array([[-0.5102189 ],
       [-0.0513879 ],
       [-0.04282204]]), array([[-0.57903023],
       [-0.31846375],
       [-0.049676  ]]), array([[-0.44265656],
       [-0.19542225],
       [-0.96998478]]), array([[-0.35282999],
       [-0.0411734 ],
       [-0.78086343]]), array([[-0.32489513],
       [-0.31440527],
       [ 0.43756401]]))
tvecs (array([[-0.49730172],
       [-2.84310108],
       [17.2890832 ]]), array([[-3.79340414],
       [-1.89318878],
       [17.64128512]]), array([[-2.92199046],
       [-0.53134388],
       [14.47700857]]), array([[ 3.0373909 ],
       [ 2.0321178 ],
       [29.74373142]]), array([[-6.57768215],
       [ 1.30555726],
       [23.69775083]]), array([[-4.79538214],
       [-3.42377143],
       [26.13729506]]), array([[-1.48638208],
       [-5.42140425],
       [24.43715573]]))
imageSize (2592, 1944)
Total error:  0.09477064110633412

"intrinsic_parameters": {
   "fx": 2440.9540138328043,
   "fy": 2434.6728574963054,
   "height_px": 1944,
   "ppx": 1248.1362594555521,
   "ppy": 957.0167856547552,
   "width_px": 2592
 },
 "distortion_parameters": {
   "rk1": 0.10461012298345916,
   "rk2": -0.005435444683453119,
   "rk3": -1.0492719403629378,
   "tp1": 0.0006065919889218954,
   "tp2": 0.004444262737182748
 },
'''