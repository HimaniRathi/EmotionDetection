import csv
import glob

import cv2
import numpy as np


x_data = []
label = []
files = glob.glob("data/frames/*.png")
for myFile in files:
    temp = myFile.split("_")
    l = ''.join([i for i in temp[1] if not i.isdigit()])
    # print (labels)
    if l == 'a':
        # Angry
        label = np.append(label, ([1, 0, 0, 0, 0, 0, 0]))
    elif l == 'd':
        # Disgust
        label = np.append(label, ([0, 1, 0, 0, 0, 0, 0]))
    elif l == 'f':
        # Fear
        label = np.append(label, ([0, 0, 1, 0, 0, 0, 0]))
    elif l == 'h':
        # Happy
        label = np.append(label, ([0, 0, 0, 1, 0, 0, 0]))
    elif l == 'n':
        # Neutral
        label = np.append(label, ([0, 0, 0, 0, 1, 0, 0]))
    elif l == 'sa':
        # Sad
        label = np.append(label, ([0, 0, 0, 0, 0, 1, 0]))
    elif l == 'su':
        # Surprise
        label = np.append(label, ([0, 0, 0, 0, 0, 0, 1]))
    image = cv2.imread(myFile)
    # print (image)
    x_data.append(image)

images = np.array(x_data)
print(label)