# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)
# %cd 'drive/MyDrive/CS 231N Project/'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1

df = pd.read_pickle("data_full.pkl")

y = df["image_annotation"]
X = df.drop("image_annotation", axis=1)

X_train_valid, X_test, y_train_valid, y_test = \
    train_test_split(X, y, test_size=TEST_SPLIT, random_state=0)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train_valid, y_train_valid, test_size=VAL_SPLIT/(1-TEST_SPLIT), random_state=0)
data_train = pd.concat([X_train, y_train], axis=1)
data_valid = pd.concat([X_valid, y_valid], axis=1)
data_test = pd.concat([X_test, y_test], axis=1)

data_train.to_pickle("data_full_train.pkl")
data_valid.to_pickle("data_full_valid.pkl")
data_test.to_pickle("data_full_test.pkl")



df = pd.read_pickle("data_condensed.pkl")

y = df["image_annotation"]
X = df.drop("image_annotation", axis=1)

X_train_valid, X_test, y_train_valid, y_test = \
    train_test_split(X, y, test_size=TEST_SPLIT, random_state=0)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train_valid, y_train_valid, test_size=VAL_SPLIT/(1-TEST_SPLIT), random_state=0)
data_train = pd.concat([X_train, y_train], axis=1)
data_valid = pd.concat([X_valid, y_valid], axis=1)
data_test = pd.concat([X_test, y_test], axis=1)

data_train.to_pickle("data_condensed_train.pkl")
data_valid.to_pickle("data_condensed_valid.pkl")
data_test.to_pickle("data_condensed_test.pkl")



SPLIT_PATH = "Manga109_metadata/data_condensed_train.pkl"
df = pd.read_pickle(SPLIT_PATH)
df.iloc[0]["image_annotation"].copy()

df = pd.read_pickle("Manga109_metadata/data_condensed_test.pkl")
for i in range(len(df)):
    annotation = df.iloc[i]["image_annotation"]
    annotation_fixed = {
        "@height": annotation["@height"],
        "@index": annotation["@index"],
        "@width": annotation["@width"],
        "contents": [],
    }
    for box in annotation["contents"]:
        if box["@xmin"] < box["@xmax"] and box["@ymin"] < box["@ymax"]:
            annotation_fixed["contents"].append(box)
    df.iloc[i]["image_annotation"] = annotation_fixed
df.to_pickle("Manga109_metadata/data_condensed_fixed_test.pkl")

df.iloc[0]["image_annotation"]