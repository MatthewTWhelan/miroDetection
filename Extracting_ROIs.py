# cropping the MiRo from the images using the ROI
import numpy as np
import cv2
import os
import time

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def ROI(img):
    # thresholding the standard image
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_stddev_img = cv2.meanStdDev(img_grey)
    mean = mean_stddev_img[0]
    stddev = mean_stddev_img[1]
    thresh = mean*(1 + -1.2*(stddev/128 - 1))
    if thresh>230:
        thresh = 230
    img_thresh = cv2.inRange(img_grey, thresh, 255)
    im2, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi = []
    i = 0
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (float(w)/float(h) > 0.25) & (w/h < 2):

            roi.append((x,y,w,h))
            i += 1
    return roi

j = 0
for i in range(10000):
    img = cv2.imread('/home/matt/images/Pre_processed/' + str(i) + 'left.png')
    if img is None:
        continue
    #imshow(img)
    roi = ROI(img)
    for r in range(len(roi)):
        (x,y,w,h) = roi[r]
        if w > h: # if width is greater than height, set ROI height to be same as width
            h = w
        else: # else set the ROI width to be the same as the height
            w = h

        # Saving the ROI's
        for step in range(2): # remember, we increase the ROI size 4 (or 3 (or 2)) times, classifying each time
            img_roi = img[y:(y+h), x:(x+w)]
            #imshow(img_roi)
            cv2.imwrite('/home/matt/images/ROI_images/ROI' + str(j) + '.png', img_roi)
            w_new = int(w*1.25)
            h_new = int(h*1.25)
            x_new = x - (w_new - w)/2
            y_new = y - (h_new - h)/2
            if x_new>0:
                x = x_new
            else:
                x = 0
            if y_new>0:
                y = y_new
            else:
                y = 0
            w = w_new
            h = h_new
            j += 1
