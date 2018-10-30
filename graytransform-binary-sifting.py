# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2


def mid_shade(array):
    # return a middle integer value of an array
    # not yet using
    return np.int(np.sum(array.flatten())/ len(array.flatten()))

# read image, return a gray scale array of numbers
img = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)

# thresholding, return a binary image with turn value is 127
# array has lots of integer numbers which each starts from 0 to 255, so turn point will be set as 127
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# SIFT - Scale-Invariant Feature Transforming
# Read more: https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
sift = cv2.xfeatures2d.SIFT_create()

# detecting key points
kps = sift.detect(thresh, None)

# drawing key points on top of thresholding gray scale image
new_img = cv2.drawKeypoints(thresh, kps, img)

# showing, writing new image to file
# cv2.imshow("image", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("keypoints.jpg", new_img)
