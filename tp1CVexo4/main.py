import numpy as np
import cv2
from matplotlib import pyplot as plt

# import de l'image
img = cv2.imread('code-route.jpg', 0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:120, 100:120] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)

mask2 = np.zeros(img.shape[:2], np.uint8)
mask2[120:200, 120:200] = 255
masked_img2 = cv2.bitwise_and(img, img, mask=mask2)

# Calculate histogram with mask and without mask
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Check third argument for mask
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
hist_mask2 = cv2.calcHist([img], [0], mask2, [256], [0, 256])
plt.subplot(221), plt.imshow(masked_img2, 'gray')
plt.subplot(222), plt.imshow(masked_img, 'gray')
plt.subplot(223), plt.plot(hist_full), plt.plot(hist_mask)
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask2)
plt.xlim([0, 256])
plt.show()

