import cv2
import numpy as np
import time
from pprint import pprint


def bloom(img, thres, iter):
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = np.where(img[1] < 0.6, 0, img)
    print(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_HLS2BGR)
    # mask = cv2.inRange(img, (thres*0.114, thres*0.587, thres*0.299),)
    cv2.imwrite("Output/mask.bmp", mask)
    blur = mask
    pprint(mask)
    for i in range(iter):
        blurredMask = cv2.GaussianBlur(mask, (iter*10+1, iter*10+1), iter*3)
        blur += blurredMask

    blur = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imwrite("Output/mask.bmp", blur)
    return img+blur


img = cv2.imread("src/car.bmp", 1)
cv2.imwrite("Output/greyCar.bmp", img)
img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
bloomed = bloom(img, 0.8, 5)
# bloomed = cv2.normalize(bloomed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imwrite("Output/Bloomed.bmp", bloomed)
