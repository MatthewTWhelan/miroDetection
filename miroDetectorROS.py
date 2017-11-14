#!/usr/bin/python

import numpy as np
import cv2
import time
from miroDetector import miroDetector

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
miroDetector = miroDetector()
count = 0

def imshow(img, detectedZones):
    if len(detectedZones) != 0:
        for zone in detectedZones:
            (xMin,yMin) = zone[0:2]
            (xMax,yMax) = zone[2:4]
            (xMin,yMin) = (int(xMin),int(yMin))
            (xMax,yMax) = (int(xMax),int(yMax))
                        
            cv2.rectangle(img, (xMin,yMin), (xMax,yMax), (0,0,255), 2)
            
    cv2.imshow("image", img)
    cv2.waitKey(50)

def camCallback(msg):
    global count 
    count += 1
    #print count
    if count % 10 == 0:
        #print "Detect!"
        img = bridge.imgmsg_to_cv2(msg,'bgr8')
        cv2.imwrite("images_stored/frame.png", img)
        t0 = time.time()
        detectedZones = miroDetector.detector(img)
        print detectedZones
        imshow(img, detectedZones)

        t1 = time.time()
        print "Processing time was: ", (t1 - t0)

if __name__ == '__main__':
    print "Initialising MiRo Detector.."
    t0 = time.time()

    rospy.init_node('miroDetector',anonymous=True)
    
    count = 0
    sub = rospy.Subscriber ('/miro/rob01/platform/caml', Image,
                            camCallback)
    # Loop
    rospy.spin()
