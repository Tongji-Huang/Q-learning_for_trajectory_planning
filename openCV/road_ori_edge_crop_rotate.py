# coding:utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt



'''
# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511), (0,255,0), 20)

# 显示 
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

#cv2.imshow('line',img)
#cv2.waitKey()
'''

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
 

# mask and bitwise 
def mask_bitwise_keep(image):
    # mask and bitwise the image, the fillpoly area will be kept 
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask,
                    np.array([[[0, 434], [905, 8], [1672, 619], [706, 1262]]]),
                    color=255) 
    masked_edge_img = cv2.bitwise_and(image, mask) 

    return masked_edge_img

def mask_bitwise_dele(image):
    # the fillpoly area will be deleleted
    mask = np.zeros_like(image)
    area1 = np.array([[[482, 892], [703, 764], [897, 1037], [697, 1146]]])
    area2 = np.array([[[1022, 592], [1356, 387], [1583, 602], [1187, 853]]])
                    
    mask = cv2.fillPoly(mask, [area1, area2], color=255)
    mask = cv2.bitwise_not(mask)
    masked_edge_img = cv2.bitwise_and(image, mask) 

    return masked_edge_img

img_ori = cv2.imread('cs_google.jpg')

# Canny edge detection 
img_edge = cv2.Canny(img_ori, 158, 203)

# Crop 
img_crop = mask_bitwise_keep(img_edge)
img_crop = mask_bitwise_dele(img_crop)

# rotate the picture 
angle = 32
img_rotate = rotate_bound(img_crop, angle)
img_rotate = img_rotate[300:1520, 365:1866] # [y0:y1, x0:x1]
#cv2.imshow('ww',imag)
#cv2.waitKey()

# save files
cv2.imwrite('cs_google' + '_extraction.jpg', img_edge)
cv2.imwrite('cs_google' + '_crop.jpg', img_crop)
cv2.imwrite('cs_google' + '_rotate.jpg', img_rotate)

# numRows, numCols, plotNum
plt.subplot(141),plt.imshow(img_ori)
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(142),plt.imshow(img_edge,cmap = 'gray')
plt.title('Road Feature Extraction'), plt.xticks([]), plt.yticks([])

plt.subplot(143),plt.imshow(img_crop, cmap = 'gray')
plt.title('Crop'), plt.xticks([]), plt.yticks([])

plt.subplot(144),plt.imshow(img_rotate, cmap = 'gray')
plt.title('Rotate'), plt.xticks([]), plt.yticks([])

plt.show()



'''
import cv2

cv2.namedWindow('edge_detection')
cv2.createTrackbar('minThreshold', 'edge_detection', 50, 1000, lambda x: x)
cv2.createTrackbar('maxThreshold', 'edge_detection', 100, 1000, lambda x: x)

img = cv2.imread('cs_google.jpg', cv2.IMREAD_GRAYSCALE)
while True:
    minThreshold = cv2.getTrackbarPos('minThreshold', 'edge_detection')
    maxThreshold = cv2.getTrackbarPos('maxThreshold', 'edge_detection')
    edges = cv2.Canny(img, minThreshold, maxThreshold)
    cv2.imshow('edge_detection', edges)
    cv2.waitKey(10)

'''
