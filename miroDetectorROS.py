#!/usr/bin/python

import cv2
import time
from miroDetector1 import miroDetector

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def camCallback(msg):

    img = bridge.imgmsg_to_cv2(msg,'bgr8')
    detectedZones = miroDetector.detector(img)
    miroDetector.displayZones(img,detectedZones)
    
miroDetector = miroDetector()

if __name__ == '__main__':
    print "Initialising MiRo Detector.."
    t0 = time.time()
    
    rospy.init_node('miroDetector',anonymous=True)
    
    sub = rospy.Subscriber ('/miro/rob01/platform/caml', Image,
                            camCallback)
    # Loop
    rospy.spin()
