#!/usr/bin/python

import numpy as np
import cv2
import time
from miroDetector import miroDetector
<<<<<<< HEAD
import numpy as np
import sys
=======
>>>>>>> 53480592b08e5e97f89f44ba0814e0a596992916

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

<<<<<<< HEAD
count = 0
#F = open("areas.txt","a")

def drawRegions(img, detectedZones):
    
    for zone in detectedZones:
        (xMin,yMin) = zone[0:2]
        (xMax,yMax) = zone[2:4]
        (xMin,yMin) = (int(xMin),int(yMin))
        (xMax,yMax) = (int(xMax),int(yMax))
                    
        cv2.rectangle(img, (xMin,yMin), (xMax,yMax), (0,0,255), 2)
        
        return img
        
def drawFaces(img, faces):
    
    for face in faces:
        (xMin,yMin) = face[0:2]
        (xMax,yMax) = (xMin + face[2], yMin + face[2])
        (xMin,yMin) = (int(xMin),int(yMin))
        (xMax,yMax) = (int(xMax),int(yMax))
        
        cv2.rectangle(img, (xMin,yMin), (xMax,yMax), (0,0,255), 2)
        
    return img

def imshow(img):
    cv2.imshow("image", img)
    cv2.waitKey(50)

def camCallback(msg):
    global count
    #global F
    count += 1
    
    if count % 10 == 0:
        img = bridge.imgmsg_to_cv2(msg,'bgr8')
        cv2.imwrite("image_capture/frame.png", img)
        t0 = time.time()
        
        detectedZones, orientations, confidence, faces = miroDetector.detector(img, faces=True)
        
        if not detectedZones is None:
            img = drawRegions(img, detectedZones)
            distance = miroDetector.distanceEstimation(detectedZones)
            print "Distance estimate: ", distance
            print "Orientation information: ", orientations
            print "Confidence levels: ", confidence
        
        if not faces is None:
            img = drawFaces(img, faces)
            print "Faces: ", faces
        
        imshow(img)
        t1 = time.time()
        print "Processing time was: ", (t1 - t0)
    
miroDetector = miroDetector()

if __name__ == '__main__':
    print "Initialising MiRo Detector.."
    
    for arg in sys.argv[1:]:
        f = arg.find('=')
        if f == -1:
            key = arg
            val = ""
        else:
            key = arg[:f]
            val = arg[f+1:]    
        if key == "robot":
            robot = val
        else:
            print("argument \"robot\" must be specified")
            sys.exit(0)
    
    rospy.init_node('miroDetector',anonymous=True)
    
    #~ subL = rospy.Subscriber ("/miro/" + robot + "/platform/caml", Image,
                            #~ camCallback)
    subR = rospy.Subscriber ("/miro/" + robot + "/platform/camr", Image,
=======
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
>>>>>>> 53480592b08e5e97f89f44ba0814e0a596992916
                            camCallback)
    # Loop
    rospy.spin()
