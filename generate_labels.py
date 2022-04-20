from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

JSON_PATH = "dataset/annotation.json"
IMAGE_PATH = "dataset/images/"
ROWS, COL = (300, 300)


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"
    
# Initialize the COCO api for instance annotations
coco=COCO(JSON_PATH)


# #--------- getting specific classes----------------------
filterClasses = ['building']

# -------------getting all classes ---------------------------
catIDs = coco.getCatIds()
imgIds = coco.getImgIds(catIds=catIDs)


catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats);


# Get all images containing the above Category IDs
print("Number of images containing all the  classes:", len(imgIds))

#-----------------------------------------------------------------------
# load and display a random image
#-----------------------------------------------------------------------

LABEL_PATH = "dataset/labels/"

for i in tqdm(imgIds):
    img = coco.loadImgs(i)[0]
    I = io.imread(IMAGE_PATH + img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIDs, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = np.zeros((img['height'],img['width']))
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        pixel_value = filterClasses.index(className)+1
        mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
    mask = mask.astype(np.int16) * 255
    # fig = plt.figure(figsize=(8, 8))
    # fig.add_subplot(2, 1, 1)
    # plt.imshow(I)
    # fig.add_subplot(2, 1, 2)
    # plt.imshow(mask)
    # plt.show()
    cv2.imwrite(LABEL_PATH + img['file_name'], mask)
    # break


# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# I = io.imread(IMAGE_PATH  + img['file_name'])/255.0

# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIDs, iscrowd=None)
# anns = coco.loadAnns(annIds)
# # print("annotation ids -> ", anns)
# # coco.showAnns(anns)

# print("Number of annotations in the image:", len(anns))


# #-----------------------------------------------------------------------
# # -------------------------------------adding masks
# #-----------------------------------------------------------------------
# mask = np.zeros((img['height'],img['width']))
# for i in range(len(anns)):
#     className = getClassName(anns[i]['category_id'], cats)
#     pixel_value = filterClasses.index(className)+1
#     mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)

# fig = plt.figure(figsize=(8, 8))
# fig.add_subplot(2, 1, 1)
# plt.imshow(I)
# fig.add_subplot(2, 1, 2)
# plt.imshow(mask)
# plt.show()