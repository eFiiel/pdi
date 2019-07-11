import numpy as np
import cv2
from matplotlib import pyplot as plt
from colors import color
import os


def find_res(img_orig_rgb, d_graus):
    w1 = img_orig_rgb.shape[1]
    h1 = img_orig_rgb.shape[0]

    # deixa a imagem quadrada e preenche com branco
    size = max(w1, h1)
    # print(size)

    img_rgb = np.ones((size, size, 3), np.uint8) * 255

    if w1 > h1:
        img_rgb[int((w1 - h1) / 2):int((w1 - (w1 - h1) / 2)), 0:w1, :] = img_orig_rgb
    elif h1 > w1:
        img_rgb[0:h1, int((h1 - w1) / 2):int((h1 - (h1 - w1) / 2)), :] = img_orig_rgb
    else:
        img_rgb = img_orig_rgb

    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

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
        # plt.plot(somaVert)
        # plt.plot(somaHoriz)
        # plt.show()

        # ve a diferença entre o valor minimo do grafico horiz e vert
        s = abs(int(np.min(somaVert)) - int(np.min(somaHoriz)))

        if s < min_s:
            min_s = s
            min_s_idx = i

    # print("min = " + str(min_s))
    # print("min_i = " + str(min_s_idx))

    # gira a imagem original para que o resistor fique reto
    M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), (min_s_idx + 1) * d_graus + 45, 1)
    img_rgb_rot = cv2.warpAffine(img_rgb, M, (w2, h2), borderValue=(255, 255, 255, 255))
    img = cv2.cvtColor(img_rgb_rot, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("img_rgb_rot", img_rgb_rot)
    # cv2.waitKey(0)

    # calcula soma horizontal e vertical de novo para achar onde esta o resistor
    somaVert = np.sum(img, 0)
    somaHoriz = np.sum(img, 1)

    # "borra" o grafico para deixar curvas mais suaves
    blurSize = int(size / 10)
    somaVert = np.convolve(somaVert, np.ones(blurSize) * (1 / blurSize), 'valid')
    somaHoriz = np.convolve(somaHoriz, np.ones(blurSize) * (1 / blurSize), 'valid')

    # binariza o grafico com otsu, onde for zero no grafico  é o resistor
    thresh_v, binar_v = cv2.threshold(np.uint8((somaVert / max(somaVert)) * 255), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_h, binar_h = cv2.threshold(np.uint8((somaHoriz / max(somaHoriz)) * 255), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    x1, x2 = maior_seq_zero(binar_v)
    y1, y2 = maior_seq_zero(binar_h)
    x1 = x1 + int(blurSize / 2)
    x2 = x2 + int(blurSize / 2)
    y1 = y1 + int(blurSize / 2)
    y2 = y2 + int(blurSize / 2)

    # print(x1)
    # print(x2)
    # print(y1)
    # print(y2)
    wR = abs(x1 - x2)
    hR = abs(y1 - y2)

    # corta resistor da imagem
    img_res = img_rgb_rot[int(y1 + hR * 0.13):int(y2 - 0.13 * hR), int(x1 + 0.05 * wR):int(x2 - 0.05 * wR)]
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


def getResistance(resistor):
    resist = resistor.copy()
    res = cv2.cvtColor(resistor, cv2.COLOR_BGR2HLS)
    res = cv2.normalize(res, None, alpha=0, beta=360, dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)
    resistor = cv2.normalize(resistor, None, alpha=0, beta=360, dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)
    h = res.shape[0]
    w = res.shape[1]
    hue, sat, lum, = cv2.split(res)
    luminanceAV = np.average(lum)
    print("Luminancia média: ", luminanceAV)
    somaHue = np.average(hue, 0)

    somaSat = np.average(sat, 0)
    somaLum = np.average(lum, 0)
    fig = plt.figure()
    medianH = np.median(somaHue)
    diffH = np.array([abs(x - medianH) for x in somaHue])
    plt.plot(diffH, 'r')
    fig.savefig('HSLVisu/HueHist.png')
    plt.clf()
    plt.cla()
    plt.close()
    fig = plt.figure()
    medianS = np.median(somaSat)
    diffS = np.array([abs(x - medianS) for x in somaSat])
    plt.plot(diffS, 'g')
    fig.savefig('HSLVisu/SatHist.png')
    plt.clf()
    plt.cla()
    plt.close()
    fig = plt.figure()
    medianL = np.median(somaLum)
    diffL = np.array([abs(x - medianL) for x in somaLum])
    plt.plot(diffL, 'b')
    fig.savefig('HSLVisu/LumHist.png')
    plt.clf()
    plt.cla()
    plt.close()

    diff = diffS + diffH + diffL

    thresh = np.average(diff)

    fig = plt.figure()
    plt.plot(diff)
    plt.hlines(thresh, 0, len(diff))
    fig.savefig('Output/Diff.png')
    plt.clf()
    plt.cla()
    plt.close()

    fig = plt.figure()
    binDiff = np.where(diff < thresh, 0, 255)
    plt.plot(binDiff)
    fig.savefig('Output/BinDiff.png')
    plt.clf()
    plt.cla()
    plt.close()
    listras = findStripes(binDiff)

    # means = [np.mean(i) for i in listras]
    # differ = [abs(means[i] - means[i + 1]) for i in range(len(listras) - 1)]
    # max = np.argmax(differ)
    if len(listras) > 3:
        num = min(listras, key=lambda listras: abs(listras[0] - listras[1]))
        listras.remove(num)

    if np.mean(listras[0]) > abs(np.mean(listras[2]) - res.shape[1]):
        listras.reverse()

    j = 0

    stripes = []
    for i in listras:
        start = i[0]
        end = i[1]
        w = resist.shape[1]
        h = resist.shape[0]
        cv2.rectangle(resist, (start, 0), (end, h-1), (255, 255, 255), 1)
        if start != end:
            listra = resistor[:, start:end]
            listr = res[:, start:end]
        else:
            listra = resistor[:, end]
            listr = res[:, end]
            sp = listr.shape
            listr = np.reshape(listr, (sp[0], 1, sp[1]))
        # cv2.imwrite(f'Stripe{j}.png', listra)
        j += 1
        h, s, l = cv2.split(listr)
        listr = cv2.merge((np.ravel(h), np.ravel(s), np.ravel(l)))
        stripes.append(listr)

    hue = []
    sat = []
    lum = []

    for i in stripes:
        h, s, l = cv2.split(i)
        n, bins = np.histogram(h, bins=12)
        argmax = np.argmax(n)
        hue.append((bins[argmax] + bins[argmax + 1]) / 2)

        sat.append(np.median(s))
        lum.append(np.average(l))

    print("H: ", hue)
    print("S: ", sat)
    print("L: ", lum)
    output = getValues(hue, sat, lum)
    values, resC = output[0], output[1]
    print(resC)
    print(values)
    if 'Color Not Recognized' in values:
        return resist
    try:
        resistance = (values[0] * 10 + values[1]) * 10 ** values[2]
    except Exception as err:
        print("Processing failed")
        return resist
    print(resistance)
    milhar = resistance / 1000
    if milhar > 1:
        if milhar < 1000:
            print(resistance / 1000, "k Ohms", sep="")
        elif milhar < 1000000:
            print(resistance / 1000000, "M Ohms", sep="")
        elif milhar < 1000000000:
            print(resistance / 1000000000, "G Ohms", sep="")
    else:
        print(resistance, "Ohms")
    return resist


def findStripes(vect):
    i = 0
    cumes = []
    length = len(vect)
    found = True
    while i < length - 1:
        if vect[i] != 0:
            start = i
            found = False
            while i < length:
                if vect[i] == 0:
                    end = i - 1
                    # print("Start", start, " End", end)
                    if abs(start - end) > 0.03*length:
                        cumes.append((start, end))
                    found = True
                    break

                i += 1
        i += 1
    if not found:
        cumes.append((start, length - 1))
    return cumes


def getValues(hue, sat, lum):
    values = []
    resColors = []
    for i in range(len(hue)):
        found = False
        for j in color:
            if j['hue'][0] <= hue[i] <= j['hue'][1]:
                if j['sat'][0] <= sat[i] <= j['sat'][1]:
                    if j['lum'][0] <= lum[i] <= j['lum'][1]:
                        values.append(j['value'])
                        resColors.append(j['name'])
                        found = True
                        break
        if not found:
            resColors.append('Color Not Recognized')
            values.append('Color Not Recognized')

    return values, resColors


def execution(file):
    img_color = cv2.imread(f'src/{file}')
    img_color = cv2.normalize(img_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    resistor = find_res(img_color, 10)
    cv2.imwrite('Output/img_certa.bmp', resistor)
    resistor = getResistance(resistor)
    print(resistor.shape)
    cv2.imshow(f"{file}", resistor)
    cv2.waitKey()


def main():
    files = []
    for root, dirs, files in os.walk("./src"):
        pass

    for f in files:
        execution(f)
        print(f)


if __name__ == "__main__":
    main()
