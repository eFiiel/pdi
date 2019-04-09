import cv2
import numpy as np
import time


def naive(image, height, width):
    start = time.time()
    print(start)

    if not height % 2 or not width % 2:
        print("Box height and width must be odd")
        return

    temp = np.copy(image)
    w = image.shape[1]# Largura  w = 234
    h = image.shape[0]# Altura   h = 226
    for ch in range(3):
        for y in range(height//2, h-height//2):
            for x in range(width//2, w-width//2):
                sum = 0
                for j in range(y-height//2, y+height//2):
                    if 0 < j <= h-1:
                        for i in range(x-width//2, x+width//2):
                            if 0 < i < w-1:
                                sum += image[j][i][ch]
                temp[y][x][ch] = sum/(width*height)
    print(time.time() - start)
    return temp


def sep(image, height, width):
    start = time.time()
    print(start)
    if not height % 2 or not width % 2:
        print("Box height and width must be odd")
        return

    saida = np.copy(image)
    medias = np.copy(image)
    try:
        channels = image.shape[2]  # 3 Canais
    except IndexError:
        channels = 1
    w = image.shape[1]  # Largura  w = 234
    h = image.shape[0]  # Altura   h = 226
    for ch in range(channels):
        for y in range(height//2, h-height//2):
            for x in range(width//2, w-width//2):
                sum = 0
                for i in range(x-width//2, x+width//2):
                    if 0 < i <= w - 1:
                        sum += image[y][i][ch]
                medias[y][x][ch] = sum/width

    for ch in range(channels):
        for y in range(height//2, h-height//2):
            for x in range(width//2, w-width//2):
                sum = 0
                for j in range(y - height // 2, y + height // 2):
                    if 0 < j <= h - 1:
                        sum += medias[j][x][ch]
                saida[y][x][ch] = sum / height

    print(time.time() - start)
    return saida

img = cv2.imread("mao.bmp")
blurred = sep(img, 51, 51)
# blurred = naive(img, 21, 21)
cv2.imwrite("blurred51x51.bmp", blurred)
