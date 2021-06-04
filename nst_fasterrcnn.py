"""
References: 

https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# 
# pip install cython
# # Install pycocotools, the version by default in Colab
# # has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
# pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# 
# # Download TorchVision repo to use some files from
# # references/detection
# git clone https://github.com/pytorch/vision.git
# cd vision
# git checkout v0.3.0
# 
# cp references/detection/utils.py ../
# cp references/detection/transforms.py ../
# cp references/detection/coco_eval.py ../
# cp references/detection/engine.py ../
# cp references/detection/coco_utils.py ../

# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)

# %cd 'drive/MyDrive/CS 231N Project/'

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from engine import train_one_epoch, evaluate
import utils
import transforms

METADATA_PATHS = {
    "train": "./Manga109_metadata/data_condensed_train.pkl",
    "valid": "./Manga109_metadata/data_condensed_valid.pkl",
    "test": "./Manga109_metadata/data_condensed_test.pkl",
}
LABELS = ["body", "face", "frame", "text"]
NUM_LABELS = len(LABELS)
LABEL_MAP = {LABELS[i] : i + 1 for i in range(NUM_LABELS)}

class Manga109Dataset(torch.utils.data.Dataset):
    def __init__(self, split, transforms=None, alternate_path=None):
        self.metadata = pd.read_pickle(METADATA_PATHS[split])
        self.transforms = transforms
        self.alternate_path = alternate_path

    def __getitem__(self, idx):
        image_info = self.metadata.iloc[idx]
        image_path = image_info["image_path"]
        if(self.alternate_path is not None):
            image_path = "./Manga109/" + self.alternate_path + image_path[image_path.find("images")+6:]
        annotation = image_info["image_annotation"]
        image = self.load_img(image_path)
        if(self.alternate_path is not None):
            image = image.resize((1654, 1170))
        num_objects, boxes, labels = self.load_annotation(annotation)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.metadata)

    def load_img(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    def load_annotation(self, annotation):
        objects = annotation["contents"]
        num_objects = len(objects)
        boxes = []
        labels = []

        for obj in objects:
            xmin = obj["@xmin"]
            xmax = obj["@xmax"]
            ymin = obj["@ymin"]
            ymax = obj["@ymax"]
            if xmax <= xmin or ymax <= ymin:
                continue
            label = LABEL_MAP[obj["type"]]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return num_objects, boxes, labels

def get_transforms(train):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform_list)

dataset_train = Manga109Dataset(split="train", transforms=get_transforms(train=False), alternate_path='duplicate_images_wave')
dataset_valid = Manga109Dataset(split="valid", transforms=get_transforms(train=False), alternate_path='duplicate_images_wave')
dataset_test = Manga109Dataset(split="test", transforms=get_transforms(train=False), alternate_path='duplicate_images_wave')

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)



def get_model(num_classes, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = NUM_LABELS + 1
model = get_model(num_classes, pretrained=True)
model.load_state_dict(torch.load('models_NST-FasterRCNN-2/NST-FasterRCNN-2-8.ckpt'))
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
    evaluate(model, data_loader_valid, device=device)
    torch.save(model.state_dict(), 'models_NST-FasterRCNN-2/NST-FasterRCNN-2-%d.ckpt'%(epoch))

num_epochs = 10
for epoch in range(4, 10):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
    evaluate(model, data_loader_valid, device=device)
    torch.save(model.state_dict(), 'models_NST-FasterRCNN-2/NST-FasterRCNN-2-%d.ckpt'%(epoch))

num_epochs = 10
for epoch in range(9, 10):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
    evaluate(model, data_loader_valid, device=device)
    torch.save(model.state_dict(), 'models_NST-FasterRCNN-2/NST-FasterRCNN-2-%d.ckpt'%(epoch))