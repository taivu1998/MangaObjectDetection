"""
References: 

https://github.com/eriklindernoren/PyTorch-YOLOv3
"""

# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)

# cd 'drive/MyDrive/CS 231N Project/YOLO/PyTorch-YOLOv3'

# get images and put them into a samples directory
import pandas as pd
import os

METADATA_PATHS = {
    "train": "../../Manga109_metadata/data_condensed_train.pkl",
    "valid": "../../Manga109_metadata/data_condensed_valid.pkl",
    "test": "../../Manga109_metadata/data_condensed_test.pkl",
}
LABELS = ["body", "face", "frame", "text"]
NUM_LABELS = len(LABELS)
LABEL_MAP = {LABELS[i] : i for i in range(NUM_LABELS)}


# !pip install importlib-metadata==3.7.3
# !pip install -q --pre poetry
# !poetry --version

# !poetry install



os.system("poetry run yolo-detect --images data/samples/")

os.system("poetry run yolo-detect --images data/samples/ --weights checkpoints/yolov3_ckpt_7.pth --model yolov3-awesomeconfig.cfg --classes data/custom/classes.names")

# !pip3 install --upgrade imgaug

# from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/detect.py
#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, to_cpu, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def detect_directory(model_path, weights_path, img_path, classes, output_path,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    img_detections, imgs = detect(
        model,
        dataloader,
        output_path,
        img_size,
        conf_thres,
        nms_thres)
    _draw_and_save_output_images(
        img_detections, imgs, img_size, output_path, classes)


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")
    
    print(input_img.shape)

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return to_cpu(detections).numpy()


def detect(model, dataloader, output_path, img_size, conf_thres, nms_thres):
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        print(input_imgs.shape)
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    # Iterate through images and save plot of detections
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(
            image_path, detections, img_size, output_path, classes)


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    # Create plot
    img = np.array(Image.open(image_path))
    print("A1: ", img.shape)
    plt.figure(figsize=(30, 22.5), dpi=2)
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, 1654)
    ax.set_ylim(0, 1170)
    ax.imshow(img)
    
    
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader

detect_directory(
        "config/yolov3-awesomeconfig.cfg",
        "checkpoints/yolov3_ckpt_7.pth",
        "data/samples/",
        "data/custom/classes.names",
        "output_new",
        batch_size=1,
        img_size=416,
        n_cpu=8,
        conf_thres=0.5,
        nms_thres=0.4)

