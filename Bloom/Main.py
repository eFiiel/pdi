import cv2
import numpy as np
import time


# Função que cria a máscara usando boxFilter
def maskBox(mask, iter, boxes, kernel):

    # Cria uma imagem preta para fazer as somas das máscaras borradas
    blur = np.zeros(mask.shape)
    blurredMask = mask

    # Fazendo as borragens
    for i in range(iter):

        # Aplica o boxFilter varias vezes para se aproximar ao GaussianFilter
        for j in range(boxes):
            blurredMask = cv2.boxFilter(blurredMask, blurredMask.shape[2], kernel)
        blur += blurredMask

    # Normaliza a máscara pra valores entre 0 e 255, tirando os overflows
    blur = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return blur


# Função que cria a máscara usando GaussianFilter
def maskGaus(mask, iter, sigma):

    # Cria uma imagem preta para fazer as somas das máscaras borradas
    blur = np.zeros(mask.shape)

    # Fazendo as borragens
    for i in range(iter):
        blurredMask = cv2.GaussianBlur(mask, (8 * (i + 1) * sigma + 1, 8 * (i + 1) * sigma + 1), 2 * (i + 1) * sigma)
        blur += blurredMask

    # Normaliza a máscara pra valores entre 0 e 255, tirando os overflows
    blur = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return blur


# Funçao para realizar a binarização manca e a fusão da máscara com a imagem
def bloom(img, thres, iter, sigma=None, kernel=None):
    start = time.time()
    print(start)

    # Converte os valores dos pixels para escala de 0 a 1
    mask = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Percorre a imagem deixando preto tudo que está abaixo do threshold
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):

            # Faz a comparação do threshold com o valor em escala de cinza em cada pixel
            if (mask[y, x][0]*0.114 + mask[y, x][1]*0.587 + mask[y, x][2]*0.299) < thres:
                mask[y, x] = [0, 0, 0]

    # Volta os pixels para escala de 0 a 255
    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Decide qual filtro vai usar baseado nos argumentos
    if sigma:
        blur = maskGaus(mask, iter, sigma)
    else:
        blur = maskBox(mask, iter, 5, kernel)

    print(time.time() - start)

    # Retorna uma soma empiricamente ponderada da imagem com a máscara
    return 0.7*img+1.3*blur


# Leitura da imagem
img = cv2.imread("src\car.bmp")

# Chama a função de bloom
bloomed = bloom(img, 0.5, 5, sigma=3)
# bloomed = bloom(img, 0.5, 5, kernel=(15, 15))

# Gera o arquivo da imagem
cv2.imwrite("Output\GaussBloomed.bmp", bloomed)
