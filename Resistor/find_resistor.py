import numpy as np
import cv2
from matplotlib import pyplot as plt

def find_resistor(img, d_graus):

	img_orig = cv2.cvtColor(img_orig_rgb, cv2.COLOR_BGR2GRAY)

	w1 = img_orig.shape[1]
	h1 = img_orig.shape[0]


	#deixa a imagem quadrada e preenche com branco
	size = max(w1,h1) 
	print(size)

	img = np.ones((size,size), np.uint8) * 255

	if w1>h1 :
		img[int((w1-h1)/2):int((w1-(w1-h1)/2)), 0:w1] = img_orig
	elif h1>w1 :
		img[0:h1, int((h1-w1)/2):int((h1-(h1-w1)/2))] = img_orig
	else :
		img = img_orig

	#cv2.imshow("img", img)
	#cv2.waitKey(0)


	w2 = img.shape[1]
	h2 = img.shape[0]
	M = cv2.getRotationMatrix2D((w2/2, h2/2), d_graus, 1)

	min_s = np.uint32(-1)
	min_s_idx = -1

	#rotaciona a imagem de d_graus em d_graus graus ate achar a rotacao que tenha a menor diferenca entre o grafico de somaVert e somaHoriz
	#quando somaVert e somaHoriz forem parecidos o resistor esta girado 45 graus
	for i in range(int(180 / d_graus)):
		img = cv2.warpAffine(img, M, (w2, h2),  borderValue=(255,255,255,255))

		#print(i)
		#cv2.imshow("img", img)
		#cv2.waitKey(0)

		somaVert = np.sum(img, 0)
		somaHoriz = np.sum(img, 1)

		#compara as duas funcoes
		#s = 0
		#for j in range(somaVert.size):
		#	s = s + abs(int(somaVert[j]) - int(somaHoriz[j]))

		#ve a diferença entre o valor minimo do grafico horiz e vert
		s = abs(int(np.min(somaVert)) - int(np.min(somaHoriz)))

		if s < min_s:
			min_s = s
			min_s_idx = i

		#print(s)
		#fig = plt.figure()
		#plt.plot(somaVert)
		#plt.plot(somaHoriz)
		#fig.savefig('fig' + str(i) + '.jpg')

	print("min = " + str(min_s))
	print("min_i = " + str(min_s_idx))

	#gira a imagem original para que o resistor fique reto
	M = cv2.getRotationMatrix2D((w1/2, h1/2), (min_s_idx+1)*d_graus+45, 1)
	img_orig_rgb_rot = cv2.warpAffine(img_orig_rgb, M, (w1, h1),  borderValue=(255,255,255,255))
	img = cv2.cvtColor(img_orig_rgb_rot, cv2.COLOR_BGR2GRAY)

	#cv2.imshow("img_certa", img)
	#cv2.waitKey(0)

	#calcula soma horizontal e vertical de novo para achar onde esta o resistor
	somaVert = np.sum(img, 0)
	somaHoriz = np.sum(img, 1)
	#fig = plt.figure()
	#plt.plot(somaVert)
	#plt.plot(somaHoriz)
	#fig.savefig('fig_certa.jpg')

	#binariza o grafico com otsu, onde for zero no grafico  é o resistor
	thresh_v,binar_v = cv2.threshold(np.uint8((somaVert/max(somaVert))*255),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	thresh_h,binar_h = cv2.threshold(np.uint8((somaHoriz/max(somaHoriz))*255),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#fig = plt.figure()
	#plt.plot(binar_v)
	#plt.plot(binar_h)
	#fig.savefig('fig_certa_bin.jpg')

	#acha a maior sequecia de zeros no grafico
	def maior_seq_zero(arr):
		last = 255
		st = 0
		bigger = 0
		st_bigger = 0
		for i in range(arr.size):
			if arr[i] == 0 and last == 255: #borda de descida
				st = i
			if (arr[i] == 255 and last == 0) or (arr[i] == 0 and i == arr.size-1): #borda de subida ou fim do grafico
				if i - st > bigger:
					bigger = i - st
					st_bigger = st
			last = arr[i]
		return st_bigger,st_bigger+bigger-1

	x1,x2 = maior_seq_zero(binar_v)
	y1,y2 = maior_seq_zero(binar_h)
				
	print(x1)
	print(x2)
	print(y1)
	print(y2)

	#cv2.rectangle(img_orig_rgb_rot, (x1,y1), (x2,y2), (0,255,0,255), 2)
	#cv2.imshow("img_certa", img_orig_rgb_rot)
	#cv2.waitKey(0)

	#corta resistor da imagem
	img_res = img_orig_rgb_rot[y1:y2, x1:x2]
	if x2-x1 < y2-y1: #se estiver de pe, deita
		img_res=cv2.rotate(img_res, cv2.ROTATE_90_CLOCKWISE) 

	#cv2.imshow("img_res", img_res)
	#cv2.waitKey(0)
	
	return img_res



#img_orig_rgb = cv2.imread('src/18k.jpeg', cv2.IMREAD_COLOR)
img_orig_rgb = cv2.imread('src/33.png', cv2.IMREAD_COLOR)
#img_orig_rgb = cv2.imread('src/resistor.jpeg', cv2.IMREAD_COLOR)
#img_orig_rgb = cv2.imread('src/330.jpg', cv2.IMREAD_COLOR)
#img_orig_rgb = cv2.imread('src/10M.bmp', cv2.IMREAD_COLOR)

cv2.imshow("img", img_orig_rgb)
cv2.waitKey(0)

img_res = find_resistor(img_orig_rgb, 10)
cv2.imshow("img_res", img_res)
cv2.waitKey(0)

