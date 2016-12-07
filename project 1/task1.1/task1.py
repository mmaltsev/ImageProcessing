import numpy as np
import math as mt
from PIL import Image as im
from matplotlib import pyplot as plt

raw_image = im.open("../img/clock.jpg")
image = np.array(raw_image)
new_image = np.empty([image[:][0].size, image[0][:].size])
w = image[0][:].size
h = image[:][0].size

r_min = 19
r_max = 20

def norm_calc(i,j):
	return mt.sqrt(mt.pow((i - w/2),2) + mt.pow((j - h/2),2))

for i in range(w):
	for j in range(h):
		if (norm_calc(i,j) >= r_min and norm_calc(i,j) <= r_max):
			new_image[i][j] = 0
		else:
			new_image[i][j] = image[i][j]
			
plt.imshow(new_image,'gray')
plt.savefig('new_clock3.jpg')
plt.show()
