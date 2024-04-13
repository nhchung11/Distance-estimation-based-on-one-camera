import cv2
import os

# Đường dẫn thư mục chứa ảnh ban đầu
input_folder = r'D:\python\Human_tracking\medium2'

# Đường dẫn thư mục để lưu ảnh đã resize
output_folder = r'D:\python\Human_tracking\medium'

# Tạo thư mục đầu ra nếu nó không tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lặp qua tất cả các file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Chỉ xử lý các file ảnh
        # Đường dẫn đầy đủ đến file ảnh đầu vào
        input_path = os.path.join(input_folder, filename)
        
        # Đọc ảnh từ đường dẫn
        image = cv2.imread(input_path)
        
        # Resize ảnh về kích thước 416x416
        resized_image = cv2.resize(image, (416, 416))
        
        # Tạo đường dẫn đến file ảnh đầu ra
        output_path = os.path.join(output_folder, filename)
        
        # Lưu ảnh đã resize
        cv2.imwrite(output_path, resized_image)

print("Quá trình resize đã hoàn thành!")
