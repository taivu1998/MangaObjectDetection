# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)

# cd 'drive/MyDrive/CS 231N Project/'

import pandas as pd

METADATA_PATHS = {
    "train": "./Manga109_metadata/data_condensed_train.pkl",
    "valid": "./Manga109_metadata/data_condensed_valid.pkl",
    "test": "./Manga109_metadata/data_condensed_test.pkl",
}
LABELS = ["body", "face", "frame", "text"]
NUM_LABELS = len(LABELS)
LABEL_MAP = {LABELS[i] : i for i in range(NUM_LABELS)}


df = pd.read_pickle(METADATA_PATHS["train"])
for i in range(len(df)):
    if('AppareKappore/036.jpg' in df.iloc[i].image_path):
        print(i)

print(df.iloc[7326])
new_path = imagepath2newpath(df.iloc[7326].image_path)
annotation2file(new_path, df.iloc[7326].image_annotation)

pd.read_pickle(METADATA_PATHS["test"])

df = pd.read_pickle(METADATA_PATHS["test"])

def imagepath2newpath(image_path):
    new_path = "YOLO/PyTorch-YOLOv3/data/custom/labels"+image_path[image_path.find("images")+6:-4]+".txt"
    new_path = new_path[:-8] + "-" + new_path[-7:]
    return new_path

def annotation2file(file_path, annotation_dict):
    """
    annotation_dict corresponds to df.iloc[1].image_annotation
    annotation_list corresponds to df.iloc[1].image_annotation["contents"]
    """
    annotation_list = annotation_dict["contents"]
    output_strs = []
    for i in range(len(annotation_list)):
        label_idx = LABEL_MAP[annotation_list[i]['type']]
        x_center = (annotation_list[i]['@xmax'] + annotation_list[i]['@xmin']) / 2 / annotation_dict['@width']
        y_center = (annotation_list[i]['@ymax'] + annotation_list[i]['@ymin']) / 2 / annotation_dict['@height']
        width = (annotation_list[i]['@xmax'] - annotation_list[i]['@xmin']) / annotation_dict['@width']
        height = (annotation_list[i]['@ymax'] - annotation_list[i]['@ymin']) / annotation_dict['@height']
        assert width >= 0
        assert height >= 0
        output_str = str(label_idx) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + '\n'
        output_strs.append(output_str)
    output_strs[-1] = output_strs[-1][:-1]
    file1 = open(file_path,"w")
    file1.writelines(output_strs)
    file1.close()

# for the test df
for i in range(len(df)):
    if(i % 100 == 0):
        print(i)
    new_path = imagepath2newpath(df.iloc[i].image_path)
    annotation2file(new_path, df.iloc[i].image_annotation)

df = pd.read_pickle(METADATA_PATHS["valid"])
# for the val df
for i in range(len(df)):
    if(i % 100 == 0):
        print(i)
    new_path = imagepath2newpath(df.iloc[i].image_path)
    annotation2file(new_path, df.iloc[i].image_annotation)

df = pd.read_pickle(METADATA_PATHS["train"])
# for the train df
for i in range(len(df)):
    if(i % 100 == 0):
        print(i)
    new_path = imagepath2newpath(df.iloc[i].image_path)
    annotation2file(new_path, df.iloc[i].image_annotation)

"""
Below is some testing
"""

new_path = imagepath2newpath(df.iloc[0].image_path)
annotation2file(new_path, df.iloc[0].image_annotation)

annotation2file("hi", df.iloc[1].image_annotation)

imagepath2newpath(df.iloc[0].image_path)

df.iloc[1].image_annotation

"""
Below is some more preprocessing code
"""

def imagepath2newpath2(image_path):
    new_path = "YOLO/PyTorch-YOLOv3/data/custom/images"+image_path[image_path.find("images")+6:-4]+".txt"
    new_path = new_path[:-8] + "-" + new_path[-7:]
    return new_path

df = pd.read_pickle(METADATA_PATHS["valid"])

path_list = []
for i in range(len(df)):
    new_path = imagepath2newpath2(df.iloc[i].image_path)
    new_path_short = new_path[new_path.find("PyTorch-YOLOv3")+15:-4] + ".jpg" + " \n"
    path_list.append(new_path_short)
path_list[-1] = path_list[-1][:-2]
print(path_list[0])
print(path_list[-1])
file1 = open('YOLO/PyTorch-YOLOv3/data/custom/valid.txt',"w")
file1.writelines(path_list)
file1.close()

df = pd.read_pickle(METADATA_PATHS["train"])

path_list = []
for i in range(len(df)):
    new_path = imagepath2newpath2(df.iloc[i].image_path)
    new_path_short = new_path[new_path.find("PyTorch-YOLOv3")+15:-4] + ".jpg" + " \n"
    path_list.append(new_path_short)
path_list[-1] = path_list[-1][:-2]
print(path_list[0])
print(path_list[-1])
file1 = open('YOLO/PyTorch-YOLOv3/data/custom/train.txt',"w")
file1.writelines(path_list)
file1.close()

