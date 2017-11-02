# Matt Whelan
# This is an example script of how k-NN and SVM can be trained using
# HOG features, and is the script run for training the MiRo detector

# WARNING: Running this script will overwrite the knn.npz and svm.dat
# files, which are used in testing

# Here we will train both a k-NN and SVM on a set of training data.
# Classification is performed on three of MiRo's body parts - side views (left and right)
# and back view
import numpy as np
import cv2
import os
import time

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

# We have three classes for the body: MiRo's two side views and the back. Here we compute the
# hog features for each
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


# Doing now the same for MiRo's head: two side views and the back.
img_head_l_dir = 'Training_images/MiRo_head/Left/'
img_head_r_dir = 'Training_images/MiRo_head/Right/'
img_head_back_dir = 'Training_images/MiRo_head/Back/'
no_head_l = len(os.listdir(img_head_l_dir))
no_head_r = len(os.listdir(img_head_r_dir))
no_head_back = len(os.listdir(img_head_back_dir))
hog_head_l = np.zeros((no_head_l,1764))
hog_head_r = np.zeros((no_head_r,1764))
hog_head_back = np.zeros((no_head_back,1764))
j = 0
for filename in os.listdir(img_head_l_dir):
    img = cv2.imread(os.path.join(img_head_l_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_head_l[j,:] = np.reshape(h,np.size(h),1)
        j += 1
j = 0
for filename in os.listdir(img_head_r_dir):
    img = cv2.imread(os.path.join(img_head_r_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_head_r[j,:] = np.reshape(h,np.size(h),1)
        j += 1
j = 0
for filename in os.listdir(img_head_back_dir):
    img = cv2.imread(os.path.join(img_head_back_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_head_back[j,:] = np.reshape(h,np.size(h),1)
        j += 1

# Computing HOG features for the negative images        
negative_images_dir = 'Training_images/Negatives/'
no_negs = len(os.listdir(negative_images_dir))
print no_negs
j = 0
hog_negs = np.zeros((no_negs,1764))
for filename in os.listdir(negative_images_dir):
    img = cv2.imread(os.path.join(negative_images_dir,filename))
    if img is not None:
        img = cv2.resize(img,(64,64)) # resizing the image to 64x64 size
        h = hog.compute(img)
        hog_negs[j,:] = np.reshape(h,np.size(h),1)
        j += 1

# combining the training data
train = np.vstack((
                    hog_side_l, 
                    hog_side_r, 
                    hog_back, 
                    hog_head_l,
                    hog_head_r,
                    hog_head_back,
                    hog_negs)).astype(np.float32)
print np.shape(train)

# create the training labels: left side = 1, right side = 2, back = 3, negs = -1
side_l_label = 1
side_r_label = 2
back_label = 3
head_l_label = 4
head_r_label = 5
head_back_label = 6
negs_label = -1
train_labels = np.concatenate(
              (np.repeat(side_l_label,no_side_l),
               np.repeat(side_r_label,no_side_r),
               np.repeat(back_label,no_back),               
               np.repeat(head_l_label,no_head_l), 
               np.repeat(head_r_label,no_head_r),
               np.repeat(head_back_label,no_head_back),
               np.repeat(negs_label,no_negs)), 
               axis=0)[:,np.newaxis]

# first we train using k-NN
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# let's save the data in order to retrieve it for classification later
np.savez('knn.npz', train=train, train_labels=train_labels)

# the data can later be loaded in the following way, ready for use in classifying
with np.load('knn.npz') as data:
    train = data['train']
    train_labels = data['train_labels']
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

## retrieving the result for k-NN is done as follows
#result = knn.findNearest(test,k=5)[1]

# next we train for SVM, and save as a dat file
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# here we can save the SVM data and then load it back in for classifying
svm.save('svm.dat')
svm = cv2.ml.SVM_load('svm.dat')