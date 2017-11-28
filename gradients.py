import cv2
import numpy as np

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('Training_images/MiRo_face/11.png')
img = cv2.resize(img,(64,64))

# gradient image can be obtained using Sobel edges, as below. Or, they
# can be found using the hog.computeGradients method (see below for how to
# initialise hog)
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
imshow(mag)
cv2.imwrite("SobelEdges.png",mag)
cell = img[39:47,47:55]
cv2.imwrite("cell8size.png",cell)



# Computing HOG features for the image
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
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection)
                        
h = hog.compute(img)
# The hog feature vector contains, in most cells, two sets of histograms.
# Therefore, for plotting, they will be renormalised anyway (relative to the
# individual histogram) so which one is taken is irrelevent. There are 64 cells
# in total, so an array of size 64x9 will be used to store the normalised
# histograms for each cell. Note that, if the blocks slide left to right,
# then the side cells will contain only one histogram. The rest of the cells
# each contain two histograms in the hog feature vector. This must be accounted
# for.

cell_hists = np.zeros((64,9))
index_no = [ # The HOG descriptor is built on the 2x2 blocks. Hence, for each cell,
            # more than one histogram is built. These indices are the cell numbers that
            # have been built for the descriptor. Index 0 represents h[0:9], 182 represents
            # h[182:191] etc.
            0, 4, 8, 12, 16, 20, 24, 25, 
            28, 32, 36, 40, 44, 48, 52, 53,
            56, 60, 64, 68, 72, 76, 80, 81,
            84, 88, 92, 96, 100, 104, 108, 109,
            112, 116, 120, 124, 128, 132, 136, 137,
            140, 144, 148, 152, 156, 160, 164, 165,
            168, 172, 176, 180, 184, 188, 192, 193,
            170, 174, 178, 182, 186, 190, 194, 195
            ]   

i = 0
for ind in index_no:
    cell_hists[i,:] = np.reshape(h[ind*9:ind*9 + 9],9)
    i += 1

i = 0
for x in range(8):
    for y in range(8):
        cell_grads = cell_hists[i,:]
        #print np.size(cell_grads)
        img = cv2.resize(img,(64*8,64*8))
        for j in range(9):
            total = sum(cell_grads)
            ang = j*20
            x_len = int( np.sin(ang) * cell_grads[j] / total * 50 )
            y_len = int( np.cos(ang) * cell_grads[j] / total * 50 )
            cv2.line(img, ((x+1)*8*8-x_len/2-32,(y+1)*8*8-y_len/2-32), ((x+1)*8*8+x_len/2-32,(y+1)*8*8+y_len/2-32), (0,0,255), 1)
        i += 1

imshow(img)
cv2.imwrite("Gradients.png",img)
