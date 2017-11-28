#!/usr/bin/python

#!/usr/bin/python

# Matt Whelan

# WARNING: Running this script will overwrite the svm.dat and svmCon.dat
# files, which are used in the main Miro Detector.

# This script is used to train the SVM on a set of training images, and
# performs validation of the SVM classifier too.

# Classification is performed on three of MiRo's body parts - side views 
# (left and right) and back view.

# A second SVM is generated that is later used for extracting confidence
# levels of detection. This is a simple binary classifier, using the 
# positive and negative images as the training data.

import numpy as np
import cv2
import os
import time

test = False # If one wants to validate only, set this to true. If one is only
<<<<<<< HEAD:HOG_w_svm_TRAIN.py
             # interested in training the SVM, set to false.
=======
            # interested in training the SVM/k-NN, set to false.
>>>>>>> 53480592b08e5e97f89f44ba0814e0a596992916:HOG_w_knn_svm_TRAIN.py

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

# setting up the HOG feature extractor
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
                        
# Setting up the face detector parameters. It's important the parameters 
# keep the resultant feature vector having the same size as the body extractor.
# This is so it's easy to use the negative image features in training.
winSizeFace = (32,32)
blockSizeFace = (8,8)
blockStrideFace = (4,4)
cellSizeFace = (2,2)
hogFace = cv2.HOGDescriptor(winSizeFace,blockSizeFace,blockStrideFace,cellSizeFace,nbins,derivAperture,
                        winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
print "Collecting image data..."

# We have three classes for the body: MiRo's two side views and the back. 
# Here we compute the hog features for each
img_side_l_dir = 'Training_images/MiRo_body/Left/'
img_side_r_dir = 'Training_images/MiRo_body/Right/'
img_back_dir = 'Training_images/MiRo_body/Back/'
no_side_l = len(os.listdir(img_side_l_dir))
no_side_r = len(os.listdir(img_side_r_dir))
no_back = len(os.listdir(img_back_dir))
hog_side_l = np.zeros((no_side_l,1764))
hog_side_r = np.zeros((no_side_r,1764))
hog_back = np.zeros((no_back,1764))
j = 0
for filename in os.listdir(img_side_l_dir):
    img = cv2.imread(os.path.join(img_side_l_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_side_l[j,:] = np.reshape(h,np.size(h),1)
        j += 1
j = 0
for filename in os.listdir(img_side_r_dir):
    img = cv2.imread(os.path.join(img_side_r_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_side_r[j,:] = np.reshape(h,np.size(h),1)
        j += 1
j = 0
for filename in os.listdir(img_back_dir):
    img = cv2.imread(os.path.join(img_back_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_back[j,:] = np.reshape(h,np.size(h),1)
        j += 1

# Computing HOG features for the negative images        
negative_images_dir = 'Training_images/Negatives/'
no_negs = len(os.listdir(negative_images_dir))
j = 0
hog_negs = np.zeros((no_negs,1764))
for filename in os.listdir(negative_images_dir):
    img = cv2.imread(os.path.join(negative_images_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_negs[j,:] = np.reshape(h,np.size(h),1)
        j += 1

# and the same for the face negs (needs doing for difference in feature desciptor size)
j = 0
hog_negs_face = np.zeros((no_negs,7056))
for filename in os.listdir(negative_images_dir):
    img = cv2.imread(os.path.join(negative_images_dir,filename))
    if img is not None:
        img = cv2.resize(img,(32,32)) # resizing the image to 64x64 size
        h = hogFace.compute(img)
        hog_negs_face[j,:] = np.reshape(h,np.size(h),1)
        j += 1

# combining the training data
train = np.vstack((
                    hog_side_l, 
                    hog_side_r, 
                    hog_back, 
                    hog_negs)).astype(np.float32)

# create the training labels: left side = 1, right side = 2, back = 3, negs = -1
side_l_label = 1
side_r_label = 2
back_label = 3
negs_label = -1
train_labels = np.concatenate(
              (np.repeat(side_l_label,no_side_l),
               np.repeat(side_r_label,no_side_r),
               np.repeat(back_label,no_back),               
               np.repeat(negs_label,no_negs)), 
               axis=0)[:,np.newaxis]

# Training the SVM for MiRo faces
faces_image_dir = '/home/matt/miroDetection/Training_images/MiRo_face/'
no_faces = len(os.listdir(faces_image_dir))
j = 0
hog_faces = np.zeros((no_faces,7056))
for filename in os.listdir(faces_image_dir):
    img = cv2.imread(os.path.join(faces_image_dir,filename))
    if img is not None:
        img = cv2.resize(img,(32,32)) # resizing the image to 32x32 for faces
        h = hogFace.compute(img)
        hog_faces[j,:] = np.reshape(h,np.size(h),1)
        j += 1
trainFaces = np.vstack((hog_faces, hog_negs_face)).astype(np.float32)
train_face_labels = np.concatenate(
                    (np.repeat(1,no_faces),
                     np.repeat(-1,no_negs)),
                     axis=0)[:,np.newaxis]
                     
if not test:
    print "Training only"
    
    # training the SVM, and saving as a dat file
    print "Training the SVM..."
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    # here we can save the SVM data and then load it back in for classifying
    svm.save('svm.dat')
    #svm = cv2.ml.SVM_load('svm.dat')
    
    # training a second, binary classifier in order to get confidence levels
    svmCon = cv2.ml.SVM_create()
    svmCon.setKernel(cv2.ml.SVM_LINEAR)
    svmCon.setType(cv2.ml.SVM_C_SVC)
    
    train_labels_con = np.zeros(np.shape(train_labels), dtype = int)
    for i in range(np.size(train_labels)):
        train_labels_con[i] = train_labels[i] / abs(train_labels[i])
    svmCon.train(train, cv2.ml.ROW_SAMPLE, train_labels_con)
    svmCon.save('svmCon.dat')
    
    # finally training the MiRo face detector
    svmFace = cv2.ml.SVM_create()
    svmFace.setKernel(cv2.ml.SVM_LINEAR)
    svmFace.setType(cv2.ml.SVM_C_SVC)
    svmFace.train(trainFaces, cv2.ml.ROW_SAMPLE, train_face_labels)
    svmFace.save('svmFace.dat')
    
    print "Training complete"

else:
    # Below lines for testing purposes. It will train the SVM on a
    # subset of the training data, then validate on the rest.
    import random
    print "Running validation"
    print "Training the SVM..."
    for i in range(50):
        index_l = random.sample(range(no_side_l),50)
        index_r = random.sample(range(no_side_l, no_side_l + no_side_r),50)
        index_b = random.sample(range(no_side_l + no_side_r, no_side_l + no_side_r + no_back),100)
        index_neg = random.sample(range(no_side_l + no_side_r + no_back, no_side_l + no_side_r + no_back + no_negs),1000)
    index = np.concatenate((index_l,index_r,index_b,index_neg),axis=0)

    val_data = train[index,:]
    val_labels = train_labels[index]

    train = np.delete(train,index,axis=0)
    train_labels = np.delete(train_labels,index,axis=0)

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    result = svm.predict(val_data)[1]
    print "Training complete"
    no_correct = 0
    no_false = 0
    i = 0
    for res in result:
        if val_labels[i] == res:
            no_correct += 1
        else:
            no_false += 1
        i += 1
    print "Percentage of correctly classified images is: ", float(no_correct) / len(val_labels)
