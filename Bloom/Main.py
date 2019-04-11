import cv2
import numpy as np
import time


def bloom(img):





img = cv2.imread("mao.bmp")
bloomed = bloom(img)
cv2.imwrite("Output/Bloomed.bmp", bloomed)
