import numpy as np
import math as mt
from PIL import Image as im
from matplotlib import pyplot as plt
import scipy.ndimage as img

def toPolar(image):
	h, w = image.shape
	new_image = np.empty([h, w])
	
	phi = np.linspace(0, 2*mt.pi, num=h)
	r = np.linspace(0, np.sqrt(w**2 + h**2)/2, num=w)
	
	for i in range(h):
		for j in range(w):
			u = int(r[j]*mt.cos(phi[i])) + w/2
			v = int(r[j]*mt.sin(phi[i])) + h/2
			if (u >= 0 and v >= 0 and u < w and v < h):
				new_image[i][j] = image[v][u]
	return new_image
11
def gaussian(image):
	sigma = 5
	image = img.filters.gaussian_filter1d(image, sigma, axis=0)
	return image

def toCartesian(image):
	h,w = image.shape
	new_image = np.empty([h, w])
	
	r_max = np.sqrt(w**2 + h**2)/2
	r_scale = w / r_max
	phi_scale = h / (2*mt.pi)
	
	for x in range(h):
		for y in range(w):
			r = mt.sqrt((x-w/2)**2 + (y-h/2)**2)
			phi = mt.atan2((x-w/2),(y-w/2))
			r_coord = int(r*r_scale)
			phi_coord = int(phi*phi_scale)
			if (r_coord < w and phi_coord < h):
				new_image[x][y] = image[phi_coord][r_coord]
	return new_image
	
def main():
	image = np.array(im.open("../img/clock.jpg"))
	
	new_image = toPolar(image)
	new_image = gaussian(new_image)
	new_image = toCartesian(new_image)

	plt.imshow(new_image,'gray')
	plt.savefig('clock3.jpg')
	plt.show()

main()
