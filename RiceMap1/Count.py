import cv2
import numpy as np
label = 0
blobs = []
arrozes = []


def binarization(img):
    image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if image.shape == (2048, 1536):
        bin = np.where(image < 0.7, 1, 0)
    else:
        bin = np.where(image > 0.8, 1, 0)
    #bin = cv2.normalize(bin, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bin


def floodFill(img, temp, j, i, label):
    # rotula o blob com o label atual
    temp.itemset(j, i, label)
    # aumenta o número de blobs do blob correspondente ao label atual
    blobs[label] += 1
    #Adiciona a posição do blob encontrado ao vetor do blob
    arrozes[label].append((j, i))

    # Continua o algoritmo para os 4 vizinhos
    # Checagem de limite de imagem, se o blob é inexplorado e se ele é branco
    if j + 1 < img.shape[0] and temp.item(j + 1, i) == -1 and img.item(j + 1, i) == 1:
        floodFill(img, temp, j + 1, i, label)
    if j - 1 >= 0 and temp.item(j - 1, i) == -1 and img.item(j - 1, i) == 1:
        floodFill(img, temp, j - 1, i, label)
    if i + 1 < img.shape[1] and temp.item(j, i + 1) == -1 and img.item(j, i + 1) == 1:
        floodFill(img, temp, j, i + 1, label)
    if i - 1 >= 0 and temp.item(j, i - 1) == -1 and img.item(j, i - 1) == 1:
        floodFill(img, temp, j, i - 1, label)


def blobber(img):
    # cria uma matriz do mesmo tamanho da imagem preenchida com -1 representando o status de exploração dos blobs
    temp = np.full(img.shape, -1)
    h = img.shape[0]
    w = img.shape[1]
    label = 0

    for j in range(0, h - 1):
        for i in range(0, w - 1):
            # Checa se o blob não foi explorado
            if temp.item(j, i) == -1:
                # Checa se o blob é branco
                if img.item(j, i) == 1:
                    # Inicia um novo contador do Blob
                    blobs.append(0)
                    # Inicia um novo Vetor de posições dos blobs do blob
                    arrozes.append([])
                    # Inicia o Flood Fill
                    floodFill(img, temp, j, i, label)
                    # Itera o contador
                    label += 1
                else:
                    # Muda o valor do blob preto da matriz de blobs inexplorados
                    temp.itemset(j, i, 0)

    return temp


def contorna(img):
    for blob in arrozes:
        if len(blob) >= 50:
            bottom = max(blob, key=lambda blob: blob[0])[0]
            top = min(blob, key=lambda blob: blob[0])[0]
            left = min(blob, key=lambda blob: blob[1])[1]
            right = max(blob, key=lambda blob: blob[1])[1]
            #print("bot", bottom, "top", top, "left", left, "right", right)

            cv2.rectangle(img, (left, top), (right, bottom),  (0, 0, 255), 1)
    return img


# Carrega a imagem
img = cv2.imread('arroz.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('arroz.bmp')
#img = cv2.imread('documento-3mp.bmp', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('documento-3mp.bmp')

# Binariza a imagem
bin = binarization(img)
# Faz a contagem de blobs
output = blobber(bin)    
rices = np.array(blobs)
# Faz a eliminação de ruídos
if img.shape == (2048, 1536):
    rices = rices[(rices >= 50)]
else:
    rices = rices[(rices >= 50)]
print("Blobs Encontrados :", len(rices))
img2 = contorna(img2)
bin = cv2.normalize(bin, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imwrite("TESTE.bmp", img2)
# cv2.imshow("Teste", img2)
# print(arrozes[0])
# print(min(arrozes[0], key=lambda arrozes: arrozes[:][1]))
cv2.waitKey(0)
