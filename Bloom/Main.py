import cv2
import numpy as np
import time
from pprint import pprint


def bloom(img, thres, iter, baseKernel, sigma):
    # mask = np.where(img[2] < 0.4*255, 0, img)
    mask = np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(img[y, x][2] < thres*255):
                mask[y, x] = [0, 0, 0]
            else:
                mask[y, x] = img[y, x]
    mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # mask = cv2.inRange(img, (thres*0.114, thres*0.587, thres*0.299),)
    #blur = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite("Output/mask.bmp", mask)
    pprint(mask)
    blur = mask
    for i in range(iter):
        blurredMask = cv2.GaussianBlur(mask, (iter*baseKernel+1, iter*baseKernel+1), iter*sigma)
        blur += blurredMask

    blur = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imwrite("Output/mask.bmp", blur)
    return img+blur


img = cv2.imread("src/car.bmp", 1)
cv2.imwrite("Output/greyCar.bmp", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
bloomed = bloom(img, 0.8, 10, 3, 2)
# bloomed = cv2.normalize(bloomed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imwrite("Output/Bloomed.bmp", bloomed)
