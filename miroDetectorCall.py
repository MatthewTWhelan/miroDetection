import cv2
import time
from miroDetector import miroDetector

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows   

miroDetector = miroDetector()

for i in range(8):
    
    img = cv2.imread('Test_images/test' + str(i+1) + '.png')
    imshow(img)
    t0 = time.time()
    # pass the image to the detector and outputs will be region coordinates of detected zones
    detectedZones = miroDetector.detector(img)
    print "Number of detected regions is ", len(detectedZones)
    t1 = time.time()
    print t1 - t0
    miroDetector.displayZones(img,detectedZones)
