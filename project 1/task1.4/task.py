import numpy as np
import math as mt
import Image as im
from matplotlib import pyplot as plt
import numpy.fft as fft

def read_im(name="/home/phil/clock.jpg"):
    import Image as im
    raw_image = im.open(name)
    image = np.array(raw_image)
    new_image = np.empty(image.shape)
    global h,w
    h,w = image.shape
    return image


def norm_calc(i,j):
    return mt.sqrt(mt.pow((i - w/2),2) + mt.pow((j - h/2),2))

img = read_im()
im_fft = fft.fft2(img)
im_shft = fft.fftshift(im_fft)
for i in range(w):
    for j in range(h):
        if not (norm_calc(i,j) >= 15 and norm_calc(i,j) <= 25):
            im_shft[i][j] = 0

im_new = fft.ifft2( fft.ifftshift(im_shft ))
plt.imshow(np.abs(im_new),'gray')
plt.imshow(img, 'gray')

plt.show()