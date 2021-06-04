"""
References: 

https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://github.com/eriklindernoren/PyTorch-YOLOv3
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

# !pip install git+https://github.com/aleju/imgaug.git

# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)
# %cd 'drive/MyDrive/CS 231N Project/'

import os
import sys
import glob
from pathlib import *
import numpy as np
import pandas as pd
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T

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

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, image, target):
        boxes = target["boxes"]
        bbox_list = []
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            xmin, ymin, xmax, ymax = xmin.item(), ymin.item(), xmax.item(), ymax.item()
            bbox_list.append(BoundingBox(xmin, ymin, xmax, ymax))
        
        bounding_boxes = BoundingBoxesOnImage(bbox_list, shape=image.shape)
        image, bounding_boxes = self.augmentations(
            image=image,
            bounding_boxes=bounding_boxes)
        bounding_boxes = bounding_boxes.clip_out_of_image()

        boxes = np.zeros((len(bounding_boxes), 4))
        for box_idx, box in enumerate(bounding_boxes):
            xmin = box.x1
            ymin = box.y1
            xmax = box.x2
            ymax = box.y2
            boxes[box_idx, :] = np.array([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target["boxes"] = boxes

        return image, target


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, image, target):
        h, w, _ = image.shape
        boxes = target["boxes"]
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        target["boxes"] = boxes
        return image, target


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, image, target):
        h, w, _ = image.shape
        boxes = target["boxes"]
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        target["boxes"] = boxes
        return image, target


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return image, target

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])



DEFAULT_TRANSFORMS = transforms.Compose([
    # AbsoluteLabels(),
    PadSquare(),
    # RelativeLabels(),
    ToTensor(),
])


AUGMENTATION_TRANSFORMS = transforms.Compose([
    # AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    # RelativeLabels(),
    ToTensor(),
])

class Manga109Dataset(torch.utils.data.Dataset):
    def __init__(self, split, transforms=None):
        self.metadata = pd.read_pickle(METADATA_PATHS[split])
        self.transforms = transforms

    def __getitem__(self, idx):
        image_info = self.metadata.iloc[idx]
        image_path = image_info["image_path"]
        annotation = image_info["image_annotation"]

        image = self.load_img(image_path)
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
        return np.array(image)

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
    if train:
        return AUGMENTATION_TRANSFORMS
    else:
        return DEFAULT_TRANSFORMS

dataset_train = Manga109Dataset(split="train", transforms=get_transforms(train=True))
dataset_valid = Manga109Dataset(split="valid", transforms=get_transforms(train=False))
dataset_test = Manga109Dataset(split="test", transforms=get_transforms(train=False))

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)



def get_model(num_classes, pretrained=False):
    model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=num_classes, pretrained_backbone=True)
    return model

CHECKPOINT_FOLDER = "./checkpoints/"
def load_checkpoint(model, optimizer, exp_name, load_mode="best"):
    checkpoint_dir = os.path.join(CHECKPOINT_FOLDER, exp_name)
    checkpoints = sorted(list(Path(checkpoint_dir).glob("*")))
    if len(checkpoints) == 0:
        return 0, -1
    else:
        best_checkpoint_path = checkpoints[0]
        best_checkpoint = torch.load(best_checkpoint_path)
        best_map = best_checkpoint['map']
        if load_mode == "best":
            model.load_state_dict(best_checkpoint['model_state_dict'])
            optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
            start_epoch = best_checkpoint['epoch']
        elif load_mode == "recent":
            last_checkpoint_path = checkpoints[-1]
            model.load_state_dict(torch.load(last_checkpoint_path))
            start_epoch = int(last_checkpoint_path[last_checkpoint_path.rfind("_") + 1 : last_checkpoint_path.rfind(".")])
        return start_epoch, best_map

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = NUM_LABELS + 1
model = get_model(num_classes, pretrained=True)
model.to(device)

lr = 0.0001
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=lr)

exp_name = "retinanet_pretrained_augment_lr=0.0001_imgaug"
checkpoint_dir = os.path.join(CHECKPOINT_FOLDER, exp_name)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
load_mode = "best"

start_epoch, best_map = load_checkpoint(model, optimizer, exp_name, load_mode=load_mode)

num_epochs = 15
for epoch in range(start_epoch, num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
    evaluator = evaluate(model, data_loader_valid, device=device)
    map = evaluator.coco_eval['bbox'].stats[0]

    epoch_save_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.ckpt")
    torch.save(model.state_dict(), epoch_save_path)
    if map >= best_map:
        best_map = map
        best_save_path = os.path.join(checkpoint_dir, "best_model.tar")
        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "map": map,
        }
        torch.save(save_dict, best_save_path)