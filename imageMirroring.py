# Simple script written to quickly mirror an image about the y-axis and save

import cv2
import os
import numpy as np

img_dir = 'Training_images/Negatives/toMirror/'
img_save_dir = 'Training_images/Negatives/'

for filename in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,filename))
    if img is not None:
        img = cv2.flip(img,1)
        cv2.imwrite(img_save_dir + 'inv_' + filename, img)
