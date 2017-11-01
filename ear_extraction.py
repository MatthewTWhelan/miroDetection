import cv2
import numpy as np

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    
img = cv2.imread('/home/matt/Miro_images/Miro_Ears/2.png')
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x_size = np.shape(img)[1]
y_size = np.shape(img)[0]

lower_thresh = np.array([110,50,50])
upper_thresh = np.array([130,255,255])

# cut off the bottom half of the image
#img_grey = img_grey[0:(y_size / 2),0:x_size]
imshow(img_grey)

img_thresh = cv2.inRange(img_grey, 245, 255) / 255
#img_thresh = img_thresh[:,:,np.newaxis]

img2 = img_grey * img_thresh
imshow(img2)

miro_body_vals = []
print x_size
for x in range(x_size):
    for y in range(y_size):
        print y,x
        if img2[y,x] != 0:
            miro_body_vals.append(img2[y,x])

print np.average(miro_body_vals)
print np.std(miro_body_vals)
