# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)

# cd 'drive/MyDrive/CS 231N Project/NST/PyTorch-Style-Transfer/experiments'

# !pip3 install torchfile

import os
from os import listdir

manga_dirs = os.listdir('../../../Manga109/images/')
manga_dirs.sort()


import time

timer = time.time()
for i in range(0, 20):
    manga_dir = manga_dirs[i]
    dir_name = '../../../Manga109/images/'+manga_dir+'/'
    images = os.listdir(dir_name)
    images.sort()
    print(images)
    for image in images:
        im_path = dir_name + image
        output_location = '../../../Manga109/duplicate_images/' + manga_dir + '/' + image
        print("output location: ", output_location)
        os.system("python main.py eval --content-image %s --style-image images/21styles/eeveelutions.jpg --model models/21styles.model --content-size 1024 --cuda 0 --output-image %s"%(im_path, output_location))
print("time elapsed: %f"%(time.time() - timer))

# Colab kicked me off after 90 minutes ugh
timer = time.time()
for i in range(14, 20):
    manga_dir = manga_dirs[i]
    dir_name = '../../../Manga109/images/'+manga_dir+'/'
    images = os.listdir(dir_name)
    images.sort()
    print(images)
    for image in images:
        im_path = dir_name + image
        output_location = '../../../Manga109/duplicate_images/' + manga_dir + '/' + image
        print("output location: ", output_location)
        os.system("python main.py eval --content-image %s --style-image images/21styles/eeveelutions.jpg --model models/21styles.model --content-size 1024 --cuda 0 --output-image %s"%(im_path, output_location))
print("time elapsed: %f"%(time.time() - timer))

