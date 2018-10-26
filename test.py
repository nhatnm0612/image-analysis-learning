# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2


def mid_shade(array):
    return np.int(np.sum(array.flatten())/ len(array.flatten()))

img = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
# print(img)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
sift = cv2.xfeatures2d.SIFT_create()
kps = sift.detect(thresh, None)
new_img = cv2.drawKeypoints(thresh, kps, img)
cv2.imshow("image", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("keypoints.jpg", new_img)
