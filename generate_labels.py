import numpy as np
import json
import pandas as pd
import os
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import glob
import cv2
from torchsummary import summary
import copy
from tqdm import tqdm

JSON_PATH = "dataset/annotation.json"
IMAGE_PATH = "dataset/images/"
ROWS, COL = (300, 300)


annotation_data = None

with open(JSON_PATH, "r") as f:
    annotation_data = json.load(f)


image_df = pd.DataFrame(annotation_data["images"])
annotation_df = pd.DataFrame(annotation_data["annotations"])
full_df = pd.merge(
    annotation_df, image_df, how="left", left_on="image_id", right_on="id"
).dropna()

print(full_df.head())

# image_df = pd.DataFrame(annotation_data)


# print(image_df.head())
