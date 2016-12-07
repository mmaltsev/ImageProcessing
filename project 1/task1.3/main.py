import numpy as np
import math as mt
import numpy.fft as fft
import matplotlib.pyplot as plt
from PIL import Image as im

raw_image = im.open("../img/clock.jpg")
image = np.array(raw_image)

fourier = fft.fft2(image)
log = lambda t: np.log(np.absolute(t))
fourier2 = np.vectorize(log)(fourier)

shifted = fft.fftshift(fourier)
shifted2 = np.vectorize(log)(shifted)

def norm_calc(i, j):
  return mt.sqrt(mt.pow((i - w/2), 2) + mt.pow((j - h/2), 2))

r_min = 1
r_max = 90
w = image[0][:].size
h = image[:][0].size
bounded = np.empty([image[:][0].size, image[0][:].size])
for i in range(w):
  for j in range(h):
    if (norm_calc(i,j) >= r_min and norm_calc(i,j) <= r_max):
      bounded[i][j] = shifted[i][j]
    else:
      bounded[i][j] = 0

bounded2 =  np.vectorize(log)(bounded)

ishifted = fft.ifftshift(bounded)
ishifted2 = np.vectorize(log)(ishifted)

inversed = fft.ifft2(ishifted)
inversed2 = np.vectorize(lambda t: np.absolute(t))(inversed)

plt.imshow(inversed2, 'gray')
plt.show()
