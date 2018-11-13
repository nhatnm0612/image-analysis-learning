import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

img = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
img1 = plt.imread("1.png")
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

fig = plt.figure(1)
ax1 = plt.subplot(211)
ax1.imshow(img1)

ax2 = plt.subplot(212, sharex=ax1)
ax2.imshow(thresh)

plt.show()
