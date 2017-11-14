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

test = False # If one wants to validate only, set this to true. If one is only
            # interested in training the SVM/k-NN, set to false.

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

print "Collecting image data..."

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

# combining the training data
train = np.vstack((
                    hog_side_l, 
                    hog_side_r, 
                    hog_back, 
                    hog_negs)).astype(np.float32)
print np.shape(train)

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
print np.shape(train_labels)

if not test:
    print "Training only"
    # first we train using k-NN
    print "Training the k-NN..."

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
    print "Training the SVM..."
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    # here we can save the SVM data and then load it back in for classifying
    svm.save('svm.dat')
    svm = cv2.ml.SVM_load('svm.dat')
    
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
    #~ print np.shape(val_data)
    #~ print np.shape(val_labels)

    train = np.delete(train,index,axis=0)
    train_labels = np.delete(train_labels,index,axis=0)
    #~ print np.shape(train)
    #~ print np.shape(train_labels)

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
