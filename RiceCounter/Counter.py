import cv2
import numpy as np


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



class Component:
    def __init__(self):
        self.size = 0
        self.pixels = []
        self.label = None


def trataImagem():
    imagens = []
    for imagem in paths:
        img = cv2.imread(imagem['path'], 0)
        img = cv2.medianBlur(img, 3)
        output = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, -50)
        # ret, output = cv2.threshold(img, 0.7, 1, cv2.THRESH_BINARY)
        output = cv2.erode(output, (5, 5), iterations=3)
        output = cv2.dilate(output, (5, 5))
        imagens.append(output)
        cv2.imwrite("Outputs\\tratada{}.bmp".format(imagem['nRice']), output)

    return imagens

# def floodFill(img, seed):


def counter(imgs):
    count = 0
    for img in imgs:
        label = 1
        height = img.shape[0]
        width = img.shape[1]
        mask = np.copy(img)
        mask = np.resize(mask, (height+2, width+2))
        # mask = np.zeros((height+2, width+2), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if img.item(y, x) == 255:
                    try:
                        ret, img2, mask, rect = cv2.floodFill(img, mask, (x, y), label, flags=cv2.FLOODFILL_MASK_ONLY)
                        # print(rect)
                        label += 1
                    except cv2.error as err:
                        print(y, x, err)
                        raise
        cv2.imwrite("Masks\mask{}.bmp".format(paths[count]['nRice']), mask)
        count += 1

imagens = trataImagem()
counter(imagens)