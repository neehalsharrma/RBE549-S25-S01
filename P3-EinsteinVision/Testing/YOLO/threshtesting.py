import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Testing/YOLO/vid_10/frame_570/crop_2.png
img = cv2.imread('vid_10/frame_570/crop_2.png', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

img = cv2.split(img)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img[0], cmap='gray')
ax[0].set_title('Y Channel')
ax[1].imshow(img[1], cmap='gray')
ax[1].set_title('Cr Channel')
ax[2].imshow(img[2], cmap='gray')
ax[2].set_title('Cb Channel')
plt.show()
y = img[0]


sobel = cv2.Sobel(y, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
plt.imshow(sobel, cmap='gray')
plt.show()
