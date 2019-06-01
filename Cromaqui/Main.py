import cv2
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
import statistics as st

imagens = []
for i in range(9):
    img = cv2.imread("src\\{}.bmp".format(i))
    imagens.append(img)


def croma(img, name):
    imgTest = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bg = cv2.imread("masks\\car.bmp")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    width = imgTest.shape[1]
    height = imgTest.shape[0]
    bg = cv2.resize(bg, (width, height))
    h, s, v = cv2.split(imgTest)
    n, bins = np.histogram(h, bins=25)
    # plt.bar(bins[:-1], n)
    # plt.show()
    matiz = h.flatten()
    for i in range(len(matiz)):
        matiz[i] = int(matiz[i])
    print(matiz.std())
    std = matiz.std()
    argmax = np.argmax(n)
    green = (bins[argmax] + bins[argmax+1])/2
    print(green)
    while 30 > green < 70:
        n[argmax] = 0
        newargmax = np.argmax(n)
        green = (bins[newargmax] + bins[newargmax + 1]) / 2
        print(green)
    print("Verde encontrado")
    for y in range(height):
        for x in range(width):
            if green - std/3 - 5 < h[y][x] < green + std/3 + 5:
                try:
                    img[y][x] = bg[y][x]
                    img[y][x][3] = 255 - abs(green - h[y][x])*(240/(std/3))
                except Exception:
                    imgTest[y][x] = [0, 0, 0]
    outImg = cv2.merge((h, s, v))
    print(outImg.shape)
    outImg = cv2.cvtColor(imgTest, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"Masks\\{name}mask.bmp", img)


for i in range(len(imagens)):
    print(f"imagem {i}")
    croma(imagens[i], i)
