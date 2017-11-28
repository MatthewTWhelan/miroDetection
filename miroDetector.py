#!/usr/bin/python

# Matt Whelan
# MiRo detection using HOG features and an SVM kernal machine classifier

import numpy as np
import cv2
import os
import sys
import time
import datetime

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
        
        self.hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
                                cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,
                                gammaCorrection,nlevels)
        winSizeFace = (32,32)
        blockSizeFace = (8,8)
        blockStrideFace = (4,4)
        cellSizeFace = (2,2)
        
        self.hogFace = cv2.HOGDescriptor(winSizeFace,blockSizeFace,
                                blockStrideFace,cellSizeFace,nbins,
                                derivAperture,winSigma,histogramNormType,
                                L2HysThreshold,gammaCorrection,nlevels)        
        
        self.svm = cv2.ml.SVM_load("svm.dat")
        self.svmCon = cv2.ml.SVM_load("svmCon.dat")
        self.svmSupportVector = self.svmCon.getSupportVectors()
        
<<<<<<< HEAD
        self.svmFace = cv2.ml.SVM_load("svmFace.dat")
        
    def detector(self,img,faces=False):
        t0 = time.time()
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        date = str(year-2000) + str(month) + str(day) + "_" + str(hour) + str(minute)

        (self.resY, self.resX, _) = np.shape(img)
        #print help(self.svmCon.getDecisionFunction)
=======
    def detector(self,img):
        t0 = time.time()

>>>>>>> 53480592b08e5e97f89f44ba0814e0a596992916
        # cropping off the top third of the image
        img_crop = img[int(self.resY/3):self.resY, 0:self.resX]
        
        roi = self.ROI(img_crop)
        
        detected = []
        detectedFaces = []
        i = 1
        for r in range(len(roi)):
<<<<<<< HEAD
            if faces:
                (x,y,w,h) = roi[r]
                winSize = 32
                while winSize < max(w,h):
                    for x_win in range(x, x + w - winSize, 4):
                        for y_win in range(y, y + h - winSize, 4):
                            win = img_crop[y_win:(y_win+winSize), x_win:(x_win+winSize)]
                            if np.size(win) == 0:
                                continue
                            
                            img_roi = cv2.resize(win,(64,64))
                            img_roi_face = cv2.resize(win,(32,32))
                            
                            hog_feature = np.transpose(self.hog.compute(img_roi)).astype(np.float32) 
                            hog_feature_face = np.transpose(self.hogFace.compute(img_roi_face)).astype(np.float32)                           
                            
                            result = self.svm.predict(hog_feature)[1]
                            resultSvmCon = self.svmCon.predict(hog_feature)[1]
                            faceResult = self.svmFace.predict(hog_feature_face)[1]
                            
                            #print (faceResult)

                            if faceResult == 1:
                                detectedFaces.append([x_win,y_win,winSize])
                            
                            elif result > 0 or resultSvmCon == 1:
                                # the below two lines are useful for storing classified regions, if needed for adding to negative image database etc..
                                #cv2.imwrite(date + "_" + str(i) + 'image.png', win)
                                #i += 1
                                
                                decisionFuncVal = self.decisionFuncValue(hog_feature)
                                
                                #~ print "svmCon result is: ", resultSvmCon
                                #~ print decisionFuncVal
                                #~ print result
                                
                                detected.append(((x_win,y_win,winSize),result,decisionFuncVal))
                            
                            t = time.time() - t0
                            #~ if t > 2:
                                #~ return None, None, None, None # to break the detector if it's taking more than 2 seconds
                            
                    winSize += 8
            else:
                (x,y,w,h) = roi[r]
                winSize = 32
                while winSize < max(w,h):
                    for x_win in range(x, x + w - winSize, 4):
                        for y_win in range(y, y + h - winSize, 4):
                            win = img_crop[y_win:(y_win+winSize), x_win:(x_win+winSize)]
                            if np.size(win) == 0:
                                continue
                            img_roi = cv2.resize(win,(64,64))
                            
                            hog_feature = np.transpose(self.hog.compute(img_roi)).astype(np.float32) 
                            
                            result = self.svm.predict(hog_feature)[1]
                            resultSvmCon = self.svmCon.predict(hog_feature)[1]

                            if result > 0 or resultSvmCon == 1:                                
                                decisionFuncVal = self.decisionFuncValue(hog_feature)
                                detected.append(((x_win,y_win,winSize),result,decisionFuncVal))
                            
                            t = time.time() - t0
                            if t > 2:
                                return None, None, None # to break the detector if it's taking more than 2 seconds
                            
                    winSize += 8
=======
            (x,y,w,h) = roi[r]
            winSize = 32
            while winSize < max(w,h):
                for x_win in range(x, x + w - winSize, 4):
                    for y_win in range(y, y + h - winSize, 4):
                        win = img_crop[y_win:(y_win+winSize), x_win:(x_win+winSize)]
                        if np.size(win) == 0:
                            continue
                        img_roi = cv2.resize(win,(64,64))
                        hog_feature = np.transpose(self.hog.compute(img_roi)).astype(np.float32)
                        result = self.svm.predict(hog_feature)
                        weights = self.svm.getClassWeights
                        if result>0:
                            # the below two lines are useful for storing classified regions, if needed for adding to negative images etc..
                            cv2.imwrite('images_stored/' + str(i) + 'image.png', win)
                            i += 1
                            detected.append(((x_win,y_win,winSize),result))
                        t1 = time.time()
                        t = t1 - t0
                        if t > 2:
                            print "exiting"
                            sys.exit(0)
                winSize += 8
>>>>>>> 53480592b08e5e97f89f44ba0814e0a596992916

        if len(detected) == 0:
            print "No MiRo detected"
            if faces:
                return None, None, None, None
            else:
                return None, None, None
            
        else:
            detectedZones = self.overlap(detected) # detectedZones returns
            # a list of the detected zones in the following form
            # [xMin, xMax, yMin, yMax, noLeft, noRight, noBack, avgSVMDist]
            # for each detected zone.

            # adding 80 to the y detection coordinates to account for cropped image
            for det in detectedZones:
                det[1] += int(self.resY / 3)
                det[3] += int(self.resY / 3)
            if faces:
                for face in detectedFaces:
                    face[1] += int(self.resY / 3)
                    pass
        # organising the data for output        
        detectedCoords, orientations, confidences = self.Organise(detectedZones)

        if faces:
            return detectedCoords, orientations, confidences, detectedFaces
        else:
            return detectedCoords, orientations, confidences

    def ROI(self,img):
        # thresholding the standard image using the histogram data to 
        # set the threshold so that only 15%, or less, is thresholded
        
        k = 0
        thresh_prop = 1.0
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean,stddev = cv2.meanStdDev(img_grey)
        
        thresh = mean*(1 + -k*(stddev/128 - 1))
        img_thresh = cv2.inRange(img_grey, thresh, 255)
        
        thresh_prop = ( 
                        float(cv2.countNonZero(img_thresh)) / 
                (float(np.shape(img)[0]) * float(np.shape(img)[1])) 
                                                                    )
        if thresh_prop > 0.3:
            thresh = self.histogramThresh(img)
        elif thresh_prop > 0.15:
            while thresh_prop > 0.15:
                thresh = mean*(1 + -k*(stddev/128 - 1))
                img_thresh = cv2.inRange(img_grey, thresh, 255)
                thresh_prop = (
                                float(cv2.countNonZero(img_thresh)) / 
                        (float(np.shape(img)[0]) * float(np.shape(img)[1]))
                                                                    )
                k += 0.05
        _,contours,_ = cv2.findContours(img_thresh, cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE)
        
        roi = []
        for c in contours:
            if cv2.contourArea(c) < 256:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            (x, y, w, h) = self.ROI_expand((x,y,w,h))
            #imshow(img[y:y+h,x:x+w])
            roi.append((x,y,w,h))
        
        return roi
    
    def histogramThresh(self,img):
        
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        total = sum(hist)
        summed = 0
        thresh = 0
        for x in hist:
            summed += x
            thresh += 1
            if summed / total > 0.85:
                break
        
        return thresh
    
    def ROI_expand(self,(x,y,w,h)):
                
        w_new = int(w*1.75)
        h_new = int(h*1.75)
        x_new = x - (w_new - w)/2
        y_new = y - (h_new - h)/2
        
        # edge cases
        x = 0 if x_new < 0 else x_new
        y = 0 if y_new < 0 else y_new
        w = (self.resX - 1 - x) if x + w_new > self.resX else w_new
        h = (self.resY - 1 - y) if y + h_new > self.resY else h_new
        
        return (x, y, w, h)
    
    def decisionFuncValue(self, hog_feature):
        # Method computes the decision function value manually. Rho is
        # first extracted, from which ther kernal fucntion is subtracted.
        # The kernel function is computed by taking the dot product 
        # of the hog feature vector with the the support vector.
        # See https://docs.opencv.org/3.0-beta/modules/ml/doc/support_vector_machines.html#prediction-with-svm
        
        supportVector = self.svmSupportVector
        rho = self.svmCon.getDecisionFunction(i=0)[0]
        kernalFunc = np.dot(supportVector, np.transpose(hog_feature))
        
        decisionFuncVal = rho - kernalFunc
        
        return decisionFuncVal
    
    def overlap(self,detected):
        # This method counts the number of overlapping detected zones, 
        # and returns single detected regions with a count of the total 
        # number of detections per orientation
        
        x = []
        y = []
        win = []
        groups = np.zeros(len(detected))
        
        for det in detected:
            (x_win,y_win,winSize) = det[0]
            x.append(x_win)
            y.append(y_win)
            win.append(winSize)
        
        groups[0] = 1
        for i in range(len(detected)):
            for j in range(i+1,len(detected)):
                if i==j:
                    continue
                a1 = x[j] <= (x[i] + win[i]) and x[j] >= x[i]
                a2 = y[j] <= (y[i] + win[i]) and y[j] >= y[i]
                a3 = x[i] <= (x[j] + win[j]) and x[i] >= x[j]
                a4 = y[i] <= (y[j] + win[j]) and y[i] >= y[j]
                
                if a1 and (a2 or a4):
                    groups[j] = groups[i]
                elif a3 and (a2 or a4):
                    groups[j] = groups[i]
                elif groups[j] == 0:
                    groups[j] = max(groups) + 1
        
        no_groups = int(max(groups))
        groupRegions = np.zeros((no_groups,4))
        for i in range(len(groups)):            
            if groupRegions[int(groups[i]-1),3] == 0:
                groupRegions[int(groups[i]-1),:] = (x[i],y[i],x[i] + win[i],y[i] + win[i])
            else:
                if x[i] < groupRegions[int(groups[i]-1),0]:
                    groupRegions[int(groups[i]-1),0] = x[i]
                if y[i] < groupRegions[int(groups[i]-1),1]:
                    groupRegions[int(groups[i]-1),1] = y[i]
                if x[i] + win[i] > groupRegions[int(groups[i]-1),2]:
                    groupRegions[int(groups[i]-1),2] = x[i] + win[i]
                if y[i] + win[i] > groupRegions[int(groups[i]-1),3]:
                    groupRegions[int(groups[i]-1),3] = y[i] + win[i]
        
        noOrientations = np.zeros((no_groups,3)) # each row represents a
        # group, and is organised as (noLSide, noRSide, noBack)
        for i in range(len(groups)):
            if detected[i][1] == 1.0:
                noOrientations[int(groups[i])-1,0] += 1
            if detected[i][1] == 2.0:
                noOrientations[int(groups[i]-1),1] += 1
            if detected[i][1] == 3.0:
                noOrientations[int(groups[i]-1),2] += 1
        
        svmDistances = np.zeros((no_groups,2)) # confidences are taken as
        # an average of the SVM distances within each group
        for i in range(len(groups)):
            svm_dist = detected[i][2]
            svmDistances[int(groups[i])-1,0] += svm_dist
            svmDistances[int(groups[i])-1,1] += 1
        confidences = np.zeros((no_groups,1))
        for i in range(len(svmDistances)):
            confidences[i,0] = svmDistances[i,0] / svmDistances[i,1]
        
        detectedZones = np.zeros((no_groups,8))
        detectedZones[:,0:4] = groupRegions
        detectedZones[:,4:7] = noOrientations
        detectedZones[:,7] = confidences[:,0]
        
        # delete regions that are clearly too small
        group = 0
        delIndex = []
        for zone in detectedZones:
            (xMin,yMin) = zone[0:2]
            (xMax,yMax) = zone[2:4]            
            x = xMax - xMin
            y = yMax - yMin
            
            if x < 10 or y < 10 or (y/x) < 0.5 or (x/y) < 0.5:
                delIndex.append(group)
            
            group += 1
            
        detectedZones = np.delete(detectedZones, delIndex, axis=0) 
        
        return detectedZones
                
    def Organise(self, detectedZones):
        
        zonesCoords = []
        orientations = []
        confidences = []
        
        for zone in detectedZones:
            zonesCoords.append([zone[0],zone[1],zone[2],zone[3]])
            orientation = self.Orientation(zone[4:7])
            orientations.append(orientation)
            confidences.append(zone[7])
        
        return zonesCoords, orientations, confidences
        
    def Orientation(self, zone):
        
        total = sum(zone)
        left = zone[0] / total
        right = zone[1] / total
        back = zone[2] / total
            
        return [left, right, back]
                
    def displayZones(self, img, detectedZones, confidences):
        
        i = 0
        for zone in detectedZones:
            (xMin,yMin) = zone[0:2]
            (xMax,yMax) = zone[2:4]
            (xMin,yMin) = (int(xMin),int(yMin))
            (xMax,yMax) = (int(xMax),int(yMax))
            
            cv2.rectangle(img, (xMin,yMin), (xMax,yMax), (0,0,255), 2)

<<<<<<< HEAD
            cv2.putText(img, str(round(confidences[i],2)), (xMin,yMin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            
            i += 1
=======
            #~ cv2.putText(img, str(noLSide), (xMin,yMin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            #~ cv2.putText(img, str(noRSide), (xMin,yMin+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            #~ cv2.putText(img, str(noBack), (xMin,yMin+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            imshow(img)
>>>>>>> 53480592b08e5e97f89f44ba0814e0a596992916
            
        return img
    
    def distanceEstimation(self, detectedZones):
        
        distances = []
        for zone in detectedZones:
            (xMin,yMin) = zone[0:2]
            (xMax,yMax) = zone[2:4]
            area = (xMax - xMin) * (yMax - yMin)
            dist = -(1 / 3.6) * np.log(area / 19000)
            distances.append(dist)
            
        return distances

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(100)

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
    image = "image_capture/frame.png"
    img = cv2.imread(image)
    #imshow(img)
    if img is None:
        print("Invalid image location specified")
        sys.exit(0)
    t0 = time.time()
    detectedZones, orientations, confidences, faces = miroDetector().detector(img,faces=True)
    for face in faces:
        (xMin,yMin) = face[0:2]
        (xMax,yMax) = (xMin + face[2], yMin + face[2])
        (xMin,yMin) = (int(xMin),int(yMin))
        (xMax,yMax) = (int(xMax),int(yMax))
        cv2.rectangle(img, (xMin,yMin), (xMax,yMax), (0,0,255), 2)
    imshow(img)
    print faces
    t1 = time.time()
    if not detectedZones is None:
        imgDetected = miroDetector().displayZones(img,detectedZones,confidences)
        imshow(imgDetected)

