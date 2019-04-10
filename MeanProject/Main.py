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
    w = image.shape[1]  # Largura  w = 234
    h = image.shape[0]  # Altura   h = 226
    for ch in range(3):
        for y in range(height // 2, h - height // 2):
            for x in range(width // 2, w - width // 2):
                sum = 0
                for j in range(y - height // 2, y + height // 2):
                    if 0 < j <= h - 1:
                        for i in range(x - width // 2, x + width // 2):
                            if 0 < i < w - 1:
                                sum += image[j][i][ch]
                temp[y][x][ch] = sum / (width * height)
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
        for y in range(height // 2, h - height // 2):
            for x in range(width // 2, w - width // 2):
                sum = 0
                for i in range(x - width // 2, x + width // 2):
                    if 0 < i <= w - 1:
                        sum += image[y][i][ch]
                medias[y][x][ch] = sum / width

    for ch in range(channels):
        for y in range(height // 2, h - height // 2):
            for x in range(width // 2, w - width // 2):
                sum = 0
                for j in range(y - height // 2, y + height // 2):
                    if 0 < j <= h - 1:
                        sum += medias[j][x][ch]
                saida[y][x][ch] = sum / height

    print(time.time() - start)
    return saida


def integral(img, width, height):
    print(img.shape)
    start = time.time()
    print(start)
    if not height % 2 or not width % 2:
        print("Box height and width must be odd")
        return

    temp = np.full_like(img, 0, dtype=np.float32)
    saida = np.full_like(img, 0, dtype=np.float32)
    try:
        channels = img.shape[2]  # 3 Canais
    except IndexError:
        channels = 1

    h = img.shape[0]
    w = img.shape[1]
    for ch in range(channels):
        for y in range(h):
            for x in range(w):

                if (y > 0) and (x > 0):
                    temp[y, x][ch] = (img[y, x][ch] +
                                      temp[y, x - 1][ch] +
                                      temp[y - 1, x][ch] -
                                      temp[y - 1, x - 1][ch])
                elif y > 0:
                    temp[y, x][ch] = img[y, x][ch] + temp[y - 1, x][ch]
                elif x > 0:
                    temp[y, x][ch] = img[y, x][ch] + temp[y, x - 1][ch]
                else:
                    temp[y, x][ch] = img[y, x][ch]

    for ch in range(channels):
        for y in range(h):
            for x in range(w):
                left = x - width//2
                right = x + width//2
                upper = y - height//2
                lower = y + height//2
                if left <= 0 and upper <= 0:
                    saida[y, x][ch] = (temp[lower, right][ch])/((right+1)*(lower+1))
                elif left <= 0:
                    if lower < h:
                        saida[y, x][ch] = (temp[lower, right][ch] -
                                           temp[upper-1, right][ch])/((right+1)*height)
                    else:
                        saida[y, x][ch] = (temp[h-1, right][ch] -
                                           temp[upper - 1, right][ch])/((right+1)*(h-upper))
                elif upper <= 0:
                    if right < w:
                        saida[y, x][ch] = (temp[lower, right][ch] -
                                           temp[lower, left-1][ch])/(width*(lower+1))
                    else:
                        saida[y, x][ch] = (temp[lower, w-1][ch] -
                                          temp[lower, left - 1][ch])/((w-left)*(lower+1))
                else:
                    if right < w and lower < h:
                        saida[y, x][ch] = (temp[lower, right][ch] -
                                          temp[lower, left-1][ch] -
                                          temp[upper-1, right][ch] +
                                          temp[upper-1, left-1][ch])/(width*height)

                    elif right >= w and lower < h:
                        saida[y, x][ch] = (temp[lower, w-1][ch] -
                                           temp[lower, left-1][ch] -
                                           temp[upper-1, w-1][ch] +
                                           temp[upper-1, left-1][ch])/((w-left)*height)

                    elif lower >= h and right < w:
                        saida[y, x][ch] = (temp[h-1, right][ch] -
                                           temp[h-1, left - 1][ch] -
                                           temp[upper - 1, right][ch] +
                                           temp[upper - 1, left - 1][ch])/(width*(h-upper))

                    else:
                        saida[y, x][ch] = (temp[h-1, w-1][ch] -
                                           temp[h-1, left - 1][ch] -
                                           temp[upper - 1, w-1][ch] +
                                           temp[upper - 1, left - 1][ch])/((w-left)*(h-upper))
    print(time.time()-start)
    return saida


img = cv2.imread("mao.bmp")
# blurred = sep(img, 21, 51)
# blurred = naive(img, 21, 21)
blurred = integral(img, 51, 51)
cv2.imwrite("Output\integBlurred51x51.bmp", blurred)
# for i in blurred:
#     print(i)