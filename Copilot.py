import cv2

# Load the image
image = cv2.imread(r"D:\python\Distance_measurement\height4_1.jpg")

# Get the dimensions of the image
height, width, _ = image.shape

# Calculate the center coordinates
center_x = width // 2
center_y = height // 2

# Draw a red circle at the center point
cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
print(center_x, center_y)

cv2.imshow("Rescaled Image with Center Point", image)
cv2.waitKey(0)
cv2.destroyAllWindows()