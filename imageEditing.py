# Simple script written to mirror all images, then rotates all images
# (including those previously mirrored). These can be used as further 
# training instances. Scaling is not necessary as HOG scales all images
# to the same size anyway.

import cv2
import os
import numpy as np
import datetime

img_dir = 'image_capture/'
img_save_dir = 'image_capture/'
now = datetime.datetime.now()
year = now.year
month = now.month
day = now.day
hour = now.hour
minute = now.minute
date = str(year-2000) + str(month) + str(day) + "_" + str(hour) + str(minute)
print date
# use precise date and time to generate unique image names

#image mirroring
for filename in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,filename))
    if img is not None:
        img_mirr = cv2.flip(img,1)
        cv2.imwrite(img_save_dir + 'mirrored_' + filename, img_mirr)
        
#image rotating (6 angles, +- 4deg, 8deg and 12deg)
for filename in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,filename))
    if img is not None:
        rows,cols,_ = np.shape(img)
        for ang in [4,8,12]:
            M_pos = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
            M_neg = cv2.getRotationMatrix2D((cols/2,rows/2),-ang,1)
            img_rot_pos = cv2.warpAffine(img,M_pos,(cols,rows))
            img_rot_neg = cv2.warpAffine(img,M_neg,(cols,rows))
            cv2.imwrite(img_save_dir + date + 'pos_rotate' + str(ang) + filename, img_rot_pos)
            cv2.imwrite(img_save_dir + date + 'neg_rotate' + str(ang) + filename, img_rot_neg)
