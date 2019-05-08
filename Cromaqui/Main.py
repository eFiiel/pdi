import cv2
import numpy as np
import time
import sys

imagens = []
for i in range(9):
    img = cv2.imread("src\\{}.bmp".format(i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imagens.append(img)

imgTest = cv2.imread("src\\2.bmp")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2HSV)
width = imgTest.shape[1]
height = imgTest.shape[0]
h, s, v = cv2.split(imgTest)
for y in range(height):
    for x in range(width):
        if 45 < h[y][x] < 75:
            v[y][x] = 0
outImg = cv2.merge((h, s, v))
print(outImg.shape)
outImg = cv2.cvtColor(outImg, cv2.COLOR_HSV2BGR)
cv2.imwrite("Masks\\2mask.bmp",outImg)