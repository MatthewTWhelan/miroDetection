# Matt Whelan
# Here we will load up the trained SVM using HOG feature extraction. Then the
# adaptive threshold will be applied for finding a ROI, before attempting to
# classify the ROI

import numpy as np
import cv2
import os
import sys

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

class miroDetector:
    
    def __init__(self):        

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
        self.hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        self.svm = cv2.ml.SVM_load('svm.dat')
        

    def ROI_expand(self,(x,y,w,h)):
        w_new = int(w*1.5)
        h_new = int(h*1.5)
        x_new = x - (w_new - w)/2
        y_new = y - (h_new - h)/2
        if x_new >= 0:
            x = x_new
        if y_new >= 0:
            y = y_new
        return (x, y, w_new, h_new)

    def ROI(self,img):
        # thresholding the standard image
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean,stddev = cv2.meanStdDev(img_grey)
        thresh = mean*(1 + -0.75*(stddev/128 - 1))
        if thresh>230:
            thresh = 230
        img_thresh = cv2.inRange(img_grey, thresh, 255)
        imshow(img_thresh)
        _,contours,_ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        roi = []
        for c in contours:
            if cv2.contourArea(c) < 256:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            (x, y, w, h) = self.ROI_expand((x,y,w,h))
            roi.append((x,y,w,h))
        return roi

    def detector(self,img):        
        # Let us now load the test images and run them through the ROI algorithm
        roi = self.ROI(img)

        # The SVM classifier will now be run on each ROI. The ROI is first set to a square
        # size, reduced (or increased) to size 64x64, then tested using HOG and SVM.
        detected = []
        for r in range(len(roi)):
            (x,y,w,h) = roi[r]
            winSize = 32
            while winSize < min(w,h):
                for x_win in range(x, x + w - winSize, 4):
                    for y_win in range(y, y + h - winSize, 4):
                        win = img[y_win:(y_win+winSize), x_win:(x_win+winSize)]
                        img_roi = cv2.resize(win,(64,64))
                        hog_feature = np.transpose(self.hog.compute(img_roi)).astype(np.float32)
                        result = self.svm.predict(hog_feature)[1]
                        if result>0:
                            detected.append(((x_win,y_win,winSize),result))
                winSize += 8
        self.Overlap(detected)
        for det in detected:
            (x_win,y_win,winSize) = det[0]
            result = det[1]
            cv2.rectangle(img, (x_win,y_win), (x_win+winSize,y_win+winSize), (255,0,0), 2)
            if result==1:
                print 'MiRo has been detected from the left side'
            if result==2:
                print 'MiRo has been detected from the right side'
            if result==3:
                print 'MiRo has been detected from the back side'
            imshow(img)

    def Overlap(self,detected):
        # This method counts the number of overlapping detected zones, determines most likely orientation, 
        # and returns single detected regions
        x = []
        y = []
        win = []
        for det in detected:
            (x_win,y_win,winSize) = det[0]
            x.append(x_win)
            y.append(y_win)
            win.append(winSize)
        print x
        print y
        print win
        groups = np.zeros((10,50)) # store a max of 10 groups, each with a max of 50 detected regions
        for i in range(len(detected)):
            for j in range(i+1,len(detected)):
                if i==j:
                    continue
                a1 = x[j] <= (x[i] + win[i]) and x[j] >= x[i]
                a2 = y[j] <= (y[i] + win[i]) and y[j] >= y[i]
                a3 = x[i] <= (x[j] + win[j]) and x[i] >= x[j]
                a4 = y[i] <= (y[j] + win[j]) and y[i] >= y[j]
                
                if a1 and (a2 or a4):
                    print "these overlap: ",i,j
                elif a3 and (a2 or a4):
                    print "these overlap: ",i,j
                else:
                    print "no overlap: ",i,j
        
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        f = arg.find('=')
        if f == -1:
            key = arg
            val = ""
        else:
            key = arg[:f]
            val = arg[f+1:]    
        if key == "image_path":
            image = val
        else:
            print("argument \"image_path\" must be specified")
            sys.exit(0)
    img = cv2.imread(image)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if img is None:
        print("Invalid image location specified")
        sys.exit(0)
    miroDetector().detector(img)
