"""
References:

https://github.com/manga109/manga109api
"""

# !pip install manga109api

# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)
# %cd 'drive/MyDrive/CS 231N Project/'

import manga109api
from pprint import pprint
import os
import pandas as pd


manga109_root_dir = "./Manga109"
p = manga109api.Parser(root_dir=manga109_root_dir)
path = p.img_path(book="ARMS", index=0)
os.path.join(".", path[path.find("Manga109"):])
annotation = p.get_annotation(book="ARMS")
annotation_ordered = p.get_annotation(book="ARMS", separate_by_tag=False)


data = {"image_path": [], "book": [], "page": [], "image_annotation": []}
for book in p.books:
    annotation_ordered = p.get_annotation(book=book, separate_by_tag=False)
    num_pages = len(annotation_ordered["page"])
    for page in range(num_pages):
        path = p.img_path(book=book, index=page)
        path = os.path.join(".", path[path.find("Manga109"):])
        annotation = annotation_ordered["page"][page]
        data["book"].append(book)
        data["page"].append(page)
        data["image_path"].append(path)
        data["image_annotation"].append(annotation)


df = pd.DataFrame.from_dict(data)
df.to_pickle("data_full.pkl")
df2 = pd.read_pickle("data_full.pkl")


data_condensed = {"image_path": [], "book": [], "page": [], "image_annotation": []}
for book in p.books:
    annotation_ordered = p.get_annotation(book=book, separate_by_tag=False)
    num_pages = len(annotation_ordered["page"])
    for page in range(num_pages):
        path = p.img_path(book=book, index=page)
        path = os.path.join(".", path[path.find("Manga109"):])
        annotation = annotation_ordered["page"][page]
        if len(annotation["contents"]) > 0:
            data_condensed["book"].append(book)
            data_condensed["page"].append(page)
            data_condensed["image_path"].append(path)
            data_condensed["image_annotation"].append(annotation)

df_condensed = pd.DataFrame.from_dict(data_condensed)
df_condensed.to_pickle("data_condensed.pkl")



data_condensed_fixed = {"image_path": [], "book": [], "page": [], "image_annotation": []}
for book in p.books:
    annotation_ordered = p.get_annotation(book=book, separate_by_tag=False)
    num_pages = len(annotation_ordered["page"])
    for page in range(num_pages):
        path = p.img_path(book=book, index=page)
        path = os.path.join(".", path[path.find("Manga109"):])
        annotation = annotation_ordered["page"][page]
        if len(annotation["contents"]) > 0:
            data_condensed_fixed["book"].append(book)
            data_condensed_fixed["page"].append(page)
            data_condensed_fixed["image_path"].append(path)
            data_condensed_fixed["image_annotation"].append(annotation)