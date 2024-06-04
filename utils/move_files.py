# import os
# import shutil

# def move_txt_files(source_folder, destination_folder):
#     # Kiểm tra xem thư mục đích có tồn tại không
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # Duyệt qua tất cả các tệp trong thư mục nguồn
#     for filename in os.listdir(source_folder):
#         source_file_path = os.path.join(source_folder, filename)
#         # Kiểm tra xem tệp có đuôi .txt không
#         if filename.endswith(".txt") and os.path.isfile(source_file_path):
#             # Di chuyển tệp sang thư mục đích
#             destination_file_path = os.path.join(destination_folder, filename)
#             shutil.move(source_file_path, destination_file_path)
#             print(f"Đã di chuyển {filename} sang thư mục {destination_folder}")

# # Thư mục nguồn và thư mục đích
# source_folder = r"D:\python\Human_tracking\valid"
# destination_folder = r"D:\python\Human_tracking\dataset\labels"

# # Gọi hàm để di chuyển các tệp .txt
# move_txt_files(source_folder, destination_folder)

# Importing Necessary Modules
import matplotlib.pyplot as plt
import imageio
import os

# List to store the paths of the map images
paths = []

# Open and read the file
with open('D:\python\compass.txt', 'r') as f:
    for i, line in enumerate(f):
        # Split the line into latitude and longitude
        lat, lon = map(float, line.split())
        
        # Create a plot with the coordinates
        plt.figure()
        plt.plot(lon, lat, 'o')
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)

        # Save the plot as an image
        path = f'map_{i}.png'
        plt.savefig(path)

        # Store the path
        paths.append(path)

# Create a GIF from the images
images = [imageio.imread(path) for path in paths]
imageio.mimsave('map.gif', images)

# Delete the temporary images
for path in paths:
    os.remove(path)