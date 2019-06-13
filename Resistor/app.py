import numpy as np
import cv2
from matplotlib import pyplot as plt


def find_res(img_orig_rgb, d_graus):
    img_orig = cv2.cvtColor(img_orig_rgb, cv2.COLOR_BGR2GRAY)

    w1 = img_orig.shape[1]
    h1 = img_orig.shape[0]

    # deixa a imagem quadrada e preenche com branco
    size = max(w1, h1)
    print(size)

    img = np.ones((size, size), np.uint8) * 255

    if w1 > h1:
        img[int((w1 - h1) / 2):int((w1 - (w1 - h1) / 2)), 0:w1] = img_orig
    elif h1 > w1:
        img[0:h1, int((h1 - w1) / 2):int((h1 - (h1 - w1) / 2))] = img_orig
    else:
        img = img_orig

    w2 = img.shape[1]
    h2 = img.shape[0]
    M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), d_graus, 1)

    min_s = np.uint32(-1)
    min_s_idx = -1

    # rotaciona a imagem de d_graus em d_graus graus ate achar a rotacao que tenha a menor diferenca entre o grafico
    # de somaVert e somaHoriz
    # quando somaVert e somaHoriz forem parecidos o resistor esta girado 45 graus
    for i in range(int(180 / d_graus)):
        img = cv2.warpAffine(img, M, (w2, h2), borderValue=(255, 255, 255, 255))

        somaVert = np.sum(img, 0)
        somaHoriz = np.sum(img, 1)

        # ve a diferença entre o valor minimo do grafico horiz e vert
        s = abs(int(np.min(somaVert)) - int(np.min(somaHoriz)))

        if s < min_s:
            min_s = s
            min_s_idx = i

    print("min = " + str(min_s))
    print("min_i = " + str(min_s_idx))

    # gira a imagem original para que o resistor fique reto
    M = cv2.getRotationMatrix2D((w1 / 2, h1 / 2), (min_s_idx + 1) * d_graus + 45, 1)
    img_orig_rgb_rot = cv2.warpAffine(img_orig_rgb, M, (w1, h1), borderValue=(255, 255, 255, 255))
    img = cv2.cvtColor(img_orig_rgb_rot, cv2.COLOR_BGR2GRAY)

    # calcula soma horizontal e vertical de novo para achar onde esta o resistor
    somaVert = np.sum(img, 0)
    somaHoriz = np.sum(img, 1)

    # binariza o grafico com otsu, onde for zero no grafico  é o resistor
    thresh_v, binar_v = cv2.threshold(np.uint8((somaVert / max(somaVert)) * 255), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_h, binar_h = cv2.threshold(np.uint8((somaHoriz / max(somaHoriz)) * 255), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    x1, x2 = maior_seq_zero(binar_v)
    y1, y2 = maior_seq_zero(binar_h)

    print(x1)
    print(x2)
    print(y1)
    print(y2)
    wR = abs(x1 - x2)
    hR = abs(y1 - y2)
    # corta resistor da imagem
    img_res = img_orig_rgb_rot[int(y1 + hR*0.1):int(y2 - 0.1*hR), int(x1 + 0.1*wR):int(x2 - 0.1*wR)]
    if x2 - x1 < y2 - y1:  # se estiver de pe, deita
        img_res = cv2.rotate(img_res, cv2.ROTATE_90_CLOCKWISE)

    return img_res


# acha a maior sequecia de zeros no grafico
def maior_seq_zero(arr):
    last = 255
    st = 0
    bigger = 0
    st_bigger = 0
    for i in range(arr.size):
        if arr[i] == 0 and last == 255:  # borda de descida
            st = i
        if (arr[i] == 255 and last == 0) or (
                arr[i] == 0 and i == arr.size - 1):  # borda de subida ou fim do grafico
            if i - st > bigger:
                bigger = i - st
                st_bigger = st
        last = arr[i]
    return st_bigger, st_bigger + bigger - 1


def getColors(resistor):
    res = cv2.cvtColor(resistor, cv2.COLOR_BGR2HLS)
    h = res.shape[0]
    w = res.shape[1]
    hue, s, v, = cv2.split(res)
    somaHue = np.average(hue, 0)
    somaSat = np.average(v, 0)
    somaVib = np.average(s, 0)
    fig = plt.figure()
    plt.plot(somaHue)
    fig.savefig('HSLVisu/HueHist.png')
    fig = plt.figure()
    plt.plot(somaSat)
    fig.savefig('HSLVisu/SatHist.png')
    fig = plt.figure()
    plt.plot(somaVib)
    fig.savefig('HSLVisu/LumHist.png')

    hue, s, v, = cv2.split(resistor)
    somaHue = np.average(hue, 0)
    somaSat = np.average(v, 0)
    somaVib = np.average(s, 0)
    fig = plt.figure()
    plt.plot(somaHue)
    fig.savefig('RGBVisu/BlueHist.png')
    fig = plt.figure()
    plt.plot(somaSat)
    fig.savefig('RGBVisu/GreenHist.png')
    fig = plt.figure()
    plt.plot(somaVib)
    fig.savefig('RGBVisu/RedHist.png')
    return []


def findMaxima(numbers):
    maxima = []
    length = len(numbers)
    if length >= 2:
        if numbers[0] > numbers[1]:
            maxima.append(numbers[0])

        if length > 3:
            for i in range(1, length - 1):
                if abs(numbers[i] - numbers[i - 1]) > 10 or abs(numbers[i] - numbers[i + 1]) > 10:
                    maxima.append(numbers[i])

        if numbers[length - 1] > numbers[length - 2]:
            maxima.append(numbers[length - 1])
    return maxima


# img_orig = cv2.imread('src/18k.jpeg')
# img_orig = cv2.imread('src/33.bmp')
# img_orig = cv2.imread('src/resistor.jpeg')
# img_orig = cv2.imread('src/330.png')
img_color = cv2.imread('src/33.bmp')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

resistor = find_res(img_color, 10)
colors = getColors(resistor)
cv2.imwrite('Output/img_certa.bmp', resistor)
