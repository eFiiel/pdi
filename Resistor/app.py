import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('src\\10M.bmp', 0)
w = img.shape[0]
h = img.shape[1]
print(img[0][185])
M = cv2.getRotationMatrix2D((h/2, w/2), 23, 1)
rot = cv2.warpAffine(img, M, (h, w))
somaVert = np.sum(rot, 0)
somaHoriz = np.sum(rot, 1)
plt.figure()
plt.plot(somaVert)
plt.figure()
plt.plot(somaHoriz)
# plt.show()
cv2.imshow('teste', rot)
cv2.waitKey(0)