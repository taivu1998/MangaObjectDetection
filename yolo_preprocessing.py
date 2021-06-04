# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)

# cd 'drive/MyDrive/CS 231N Project/'

import os

# First, copy all the images to the correct folder and rename them
manga_dirs = os.listdir('Manga109/images/')
manga_dirs.sort()

for manga_dir in manga_dirs:
    dir_name = "Manga109/images/"+manga_dir+"/"
    images = os.listdir(dir_name)
    images.sort()
    print(dir_name)
    for image in images:
        original_image_path = dir_name + image
        new_image_path = "YOLO/PyTorch-YOLOv3/data/custom/images/" + manga_dir + "-" + image
        os.system("cp \"%s\" \"%s\""%(original_image_path, new_image_path))

