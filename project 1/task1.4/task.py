import numpy as np
import math
from matplotlib import pyplot as plt
import numpy.fft as fft
from pylab import imshow
from PIL import Image as im
from matplotlib import pyplot as plt
import numpy.fft as fft
##method to read im.. with some redundant code
def read_im(name="../img/clock.jpg"):
    raw_image = im.open(name)
    image = np.array(raw_image)
    new_image = np.empty(image.shape)
    global h,w
    h,w = image.shape
    return image


def norm_calc(i,j):
    return math.sqrt(math.pow((i - w/2),2) + math.pow((j - h/2),2))

def rnd():
    img = read_im("/home/phil/Downloads/bauckhage.jpg")
    img = read_im('/home/phil/potato.jpg')
    im_fft = fft.fft2(img)
    im_shft = fft.fftshift(im_fft)
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    im_shft[crow - 15:crow + 15, ccol - 15:ccol + 15] = 0

    for i in range(w):
        for j in range(h):
            if not (norm_calc(i,j) >= 0 and norm_calc(i,j) <= 25):
                im_shft[i][j] = 255


    im_new = fft.ifft2( fft.ifftshift(im_shft ))
    plt.imshow(np.absolute(im_new) ,'gray')
    plt.imshow(np.absolute(fft.ifft2(im_fft)), 'gray')
    plt.imshow(np.log2(np.abs(im_shft)), 'gray')
    plt.show()

###actual 4th task
def getRI(im_fft):
    magnitudes = np.sqrt(np.power(im_fft.real,2) + np.power(im_fft.imag,2))
    vatan2 = np.vectorize(math.atan2)
    phases = vatan2(im_fft.imag , im_fft.real)
    return magnitudes, phases
def main():
    img = read_im("../img/bauckhage.jpg")
    img1 = read_im("../img/clock.jpg")
    #img1 = read_im("../img/lena.jpg")

    im_fft = fft.fft2(img)
    im_fft1 = fft.fft2(img1)

    mag, phases = getRI(im_fft)
    mag1,phases1 = getRI(im_fft1)

    im_fft = mag * np.cos(phases1) + 1j * mag * np.sin(phases1)
    im_fft1 = mag1 * np.cos(phases) + 1j * mag1 * np.sin(phases)
    im_fft2 = mag1 * np.cos(phases1) + 1j* mag1 * np.sin(phases1)
    im_new = fft.ifft2(im_fft)
    im_new1 = fft.ifft2(im_fft1)
    im_new2 = fft.ifft2(mag1 * np.cos(phases1))

    plt.imshow(np.absolute(im_new),'gray')
    plt.savefig('crossover11.jpg')
    plt.imshow(np.absolute(im_new1),'gray')
    plt.savefig('crossover12.jpg')
    plt.imshow(np.absolute(im_new2), 'gray')
    plt.savefig('lenaR.jpg')

    #plt.show()
    plt.show()

if __name__ == '__main__':
    main()
