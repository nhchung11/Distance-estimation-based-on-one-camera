import cv2
import numpy as np
import glob

# Chuẩn bị các điểm đối tượng, ví dụ: (0,0,0), (1,0,0), (2,0,0), ... (6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Tạo danh sách để lưu trữ các điểm đối tượng và các điểm hình ảnh từ tất cả các ảnh.
objpoints = []  # Điểm 3D trong không gian thế giới
imgpoints = []  # Điểm 2D trong mặt phẳng hình ảnh.

# Đọc các ảnh hiệu chỉnh
images = glob.glob(r'D:\python\Distance_measurement\scripts\test_image\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tìm các góc của chessboard
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    
    # Nếu tìm thấy, thêm các điểm đối tượng và các điểm hình ảnh (sau khi làm chính xác các điểm)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Vẽ và hiển thị các góc
        cv2.drawChessboardCorners(img, (7, 6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Hiệu chỉnh camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# In ra ma trận nội tại của camera
print("Camera Matrix:\n", camera_matrix)

# In ra các hệ số biến dạng
print("Distortion Coefficients:\n", dist_coeffs)

# Chuyển đổi vector xoay của ảnh đầu tiên thành ma trận xoay
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])

print("Rotation Matrix for the first view:")
print(rotation_matrix)
