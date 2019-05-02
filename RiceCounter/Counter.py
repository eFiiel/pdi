import cv2
import numpy as np
import time
import sys
import statistics as st
import matplotlib.pyplot as plt
sys.setrecursionlimit(2000)

originais = []
paths = [
    {
        "path": "src\\60.bmp",
        "nRice": 60
    },
    {
        "path": "src\\82.bmp",
        "nRice": 82
    },
    {
        "path": "src\\114.bmp",
        "nRice": 114
    },
    {
        "path": "src\\150.bmp",
        "nRice": 150
    },
    {
        "path": "src\\205.bmp",
        "nRice": 205
    },

]

unix_paths = [
    {
        "path": "src/60.bmp",
        "nRice": 60
    },
    {
        "path": "src/82.bmp",
        "nRice": 82
    },
    {
        "path": "src/114.bmp",
        "nRice": 114
    },
    {
        "path": "src/150.bmp",
        "nRice": 150
    },
    {
        "path": "src/205.bmp",
        "nRice": 205
    },

]


class Pixel:
    def __init__(self, ponto):
        self.x = ponto[1]
        self.y = ponto[0]

    def l(self):
        return self.y, self.x - 1

    def r(self):
        return self.y, self.x + 1

    def u(self):
        return self.y - 1, self.x

    def d(self):
        return self.y + 1, self.x


class Set:

    def __init__(self):
        self.blobs = []
        self.qnt = 0
        self.size = 0
        self.avg = self.size
        self.normQnt = 0
        self.maxSize = 0
        self.sizes = []

    def add(self, blob):
        self.blobs.append(blob)
        self.size += blob.size
        self.qnt += 1
        self.avg = self.size / self.qnt
        self.maxSize = max(blob.size, self.maxSize)
        self.sizes.append(blob.size)



class Component:

    def __init__(self, label):
        self.size = 0
        self.pixels = []
        self.label = label

    def add(self, pixel):
        self.pixels.append(pixel)
        self.size += 1


def trataImagem():
    imagens = []
    for imagem in paths:
        original = cv2.imread(imagem['path'])
        originais.append(original)
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)
        output = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, -50)
        # ret, output = cv2.threshold(img, 0.7, 1, cv2.THRESH_BINARY)
        output = cv2.erode(output, (5, 5), iterations=3)
        output = cv2.dilate(output, (5, 5))
        imagens.append(output)
        cv2.imwrite("Outputs\\tratada{}.bmp".format(imagem['nRice']), output)

    return imagens


def floodFill(img, mask, blob, pixel):
    mask.itemset(pixel.y, pixel.x, blob.label)
    blob.add(pixel)

    if img.item(pixel.d()) == 255:
        if mask.item(pixel.d()) == -1:
            floodFill(img, mask, blob, Pixel(pixel.d()))

    if img.item(pixel.u()) == 255:
        if mask.item(pixel.u()) == -1:
            floodFill(img, mask, blob, Pixel(pixel.u()))

    if img.item(pixel.r()) == 255:
        if mask.item(pixel.r()) == -1:
            floodFill(img, mask, blob, Pixel(pixel.r()))

    if img.item(pixel.l()) == 255:
        if mask.item(pixel.l()) == -1:
            floodFill(img, mask, blob, Pixel(pixel.l()))


def counter(imgs):
    imgIndex = 0
    imags = []
    for img in imgs:
        rices = Set()
        label = 1
        height = img.shape[0]
        width = img.shape[1]
        mask = np.full_like(img, -1, dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if img.item(y, x) == 255:
                    if mask.item(y, x) == -1:
                        try:
                            seed = Pixel((y, x))
                            blob = Component(label)
                            floodFill(img, mask, blob, seed)
                            label += 1
                            rices.add(blob)
                        except cv2.error as err:
                            print(y, x, err)
                            raise
                else:
                    mask.itemset(y, x, 0)

        cv2.imwrite("Masks\mask{}.bmp".format(paths[imgIndex]['nRice']), mask)
        imgIndex += 1
        imags.append(rices)
    return imags


def analyze(res):
    index = 0
    for img in res:
        # plt.hist(img.sizes, bins='fd')
        # plt.show()
        coef = 0.157/((st.pvariance(img.sizes))**(1/2) / img.avg)
        n, bins = np.histogram(img.sizes, bins='auto')
        argmax = np.argmax(n)
        riceSize = (bins[argmax] + bins[argmax+1])/2
        for rice in img.blobs:
            qnt = rice.size // riceSize
            if rice.size % riceSize >= coef * riceSize:
                qnt += 1
            img.normQnt += qnt

        # print("\nContagem da imagem de {} arroz : {}".format(paths[index]['nRice'], img.qnt))
        print("\nContagem normalizada da imagem de {} arroz : {}".format(paths[index]['nRice'], img.normQnt))
        # print("MaxHist: ", riceSize)
        # print("Média: ", img.avg)
        # print("Maior: ", img.maxSize)
        # print("Variância: ", st.pvariance(img.sizes))
        # print("Var/Med: ", (st.pvariance(img.sizes))**(1/2) / img.avg)
        # print(coef)
        index += 1


start = time.time()
imagens = trataImagem()
resultados = counter(imagens)
analyze(resultados)
print("Tempo total: {}".format(time.time() - start))
