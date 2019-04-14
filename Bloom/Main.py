import cv2
import numpy as np
import time
from pprint import pprint


def bloom(img, thres, iter, sigma):
    start = time.time()
    print(start)
    mask = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if (mask[y, x][0]*0.114 + mask[y, x][1]*0.587 + mask[y, x][2]*0.299) < thres:
                mask[y, x] = [0, 0, 0]

    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite("Output\premask.bmp", mask)
    blur = np.zeros(mask.shape)
    # blur = mask
    for i in range(iter):
        blurredMask = cv2.GaussianBlur(mask, (12*(i+1)*sigma + 1, 12*(i+1)*sigma + 1), 8*(i+1)*sigma)
        blur += blurredMask

    blur = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite("Output/mask.bmp", blur)
    print(time.time() - start)
    return img+blur

img = cv2.imread("src\car.bmp")
cv2.imwrite("Output\greyCar.bmp", img)
bloomed = bloom(img, 0.65, 5, 3)
cv2.imwrite("Output\Bloomed.bmp", bloomed)
