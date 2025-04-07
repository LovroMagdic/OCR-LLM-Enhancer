import cv2 
import pytesseract
import os
import numpy as np
import math
from pathlib import Path
from cv2 import dnn_superres

def createFolders():
    os.mkdir("dataset_filled")

#here starts main script
dir = os.getcwd()
folders = []
for each in os.scandir(dir):
    folders.append(str(each))
if "<DirEntry 'dataset_filled'>" not in folders:
    createFolders()
dir = os.path.join(dir,"dataset")
dir = dir.replace("\\", "/")
arr = [] # sadrzi imena svih dataset slika

#we are saving each file stored in dataset in arr array so that is easier to iterate after
for filename in os.scandir(dir):
    if filename.is_file():
        arr.append(filename.path)

#iterating through threshold value of 100 - 235 and calculating minimal remainder like that we get most optimal threshold which gives us best contour
for each in arr:
    image = cv2.imread(each)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('demo/dataset-result/Z05353401-bw.jpg', thresh)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    c = max(contours, key = cv2.contourArea)

    black_canvas = np.zeros_like(img_gray)
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(black_canvas, c, -1, 255, cv2.FILLED) # this gives a binary mask
    each = each.replace("dataset","dataset_filled")
    cv2.imwrite(each, black_canvas)