import os
import shutil

def move_txt_files(source_folder, destination_folder):
    # Kiểm tra xem thư mục đích có tồn tại không
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Duyệt qua tất cả các tệp trong thư mục nguồn
    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)
        # Kiểm tra xem tệp có đuôi .txt không
        if filename.endswith(".txt") and os.path.isfile(source_file_path):
            # Di chuyển tệp sang thư mục đích
            destination_file_path = os.path.join(destination_folder, filename)
            shutil.move(source_file_path, destination_file_path)
            print(f"Đã di chuyển {filename} sang thư mục {destination_folder}")

# Thư mục nguồn và thư mục đích
source_folder = r"D:\python\Human_tracking\valid"
destination_folder = r"D:\python\Human_tracking\dataset\labels"

# Gọi hàm để di chuyển các tệp .txt
move_txt_files(source_folder, destination_folder)
