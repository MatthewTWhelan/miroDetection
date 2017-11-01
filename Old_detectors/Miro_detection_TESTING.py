# Matt Whelan
# Here we will load up the trained SVM using HOG feature extraction. Then the
# adaptive threshold will be applied for finding a ROI, before attempting to
# classify the ROI

import numpy as np
import cv2
import os

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

# Let's set up HOG and load the SVM data
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

svm = cv2.ml.SVM_load('svm.dat')

# Defining now the ROI algorithm
def ROI(img):
    # thresholding the standard image
    #img_ROI = img
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

# Let us now load the test images and run them through the ROI algorithm
for filename in os.listdir('Test_images/'):
    img = cv2.imread(os.path.join('Test_images/',filename))
    if img is None:
        continue
    roi = ROI(img)

    # The SVM classifier will now be run on each ROI. The ROI is first set to a square
    # size, reduced (or increased) to size 64x64, then tested using HOG and SVM.
    for r in range(len(roi)):
        (x,y,w,h) = roi[r]
        if w > h: # if width is greater than height, set ROI height to be same as width
            h = w
        else: # else set the ROI width to be the same as the height
            w = h

        # Now the exciting part, let's classify!
        for step in range(4): # remember, we increase the ROI size 4 times, classifying each time
            img_roi = img[y:(y+h), x:(x+w)]
            img_roi = cv2.resize(img_roi,(64,64)) # resizing the image to 64x64 size
            hog_feature = np.transpose(hog.compute(img_roi)) # computing the HOG features
            hog_feature.astype(np.float32) # preparing for input into SVM
            result = svm.predict(hog_feature)[1]
            if result==1:
                cv2.rectangle(img, (x,y-h), (x+int(w*1.25),y+h), (255,0,0), 2)
                print 'MiRo has been detected from the left side'
                break
            if result==2:
                cv2.rectangle(img, (x,y-h), (x+int(w*1.25),y+h), (255,0,0), 2)
                print 'MiRo has been detected from the right side'
                break
            if result==3:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                print 'MiRo has been detected from the back'
                break
            #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            w_new = int(w*1.25)
            h_new = int(h*1.25)
            x_new = x - (w_new - w)/2
            y_new = y - (h_new - h)/2
            if x_new > 0:
                x = x_new
            if y_new > 0:
                y = y_new
            w = w_new
            h = h_new
    imshow(img)
