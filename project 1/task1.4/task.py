import numpy as np
import math
import Image as im
from matplotlib import pyplot as plt
import numpy.fft as fft
from pylab import imshow
def read_im(name="/home/phil/clock.jpg"):
    import Image as im
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
    img = read_im('/home/phil/lena.jpg')
    im_fft = fft.fft2(img)
    im_shft = fft.fftshift(im_fft)
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    im_shft[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    for i in range(w):
        for j in range(h):
            if not (norm_calc(i,j) >= 0 and norm_calc(i,j) <= 25):
                im_shft[i][j] = 255


    im_new = fft.ifft2( fft.ifftshift(im_shft ))
    plt.imshow(np.abs(im_new),'gray')
    plt.imshow(img, 'gray')
    plt.imshow(np.log2(np.abs(im_shft)), 'gray')
    plt.show()

###actual 4th task
def getRI(im_fft):
    magnitudes = np.sqrt(np.power(im_fft.real,2) + np.power(im_fft.imag,2))
    vatan2 = np.vectorize(math.atan2)
    phases = vatan2(im_fft.imag , im_fft.real)
    return magnitudes, phases
def main():
    img = read_im("/home/phil/Downloads/bauckhage.jpg")
    img1 = read_im("/home/phil/lena.jpg")
    im_fft = fft.fft2(img)
    im_fft1 = fft.fft2(img1)

    mag, phases = getRI(im_fft)
    mag1,phases1 = getRI(im_fft1)

    im_fft = mag * np.cos(phases1) + mag * np.sin(phases1)
    im_fft1 = mag1 * np.cos(phases) + mag1 * np.sin(phases)
    im_fft2 = mag * np.cos(phases) + mag * np.sin(phases)
    im_new = fft.ifft2(im_fft)
    im_new1 = fft.ifft2(im_fft1)
    im_new2 = fft.ifft2(im_fft2)

    plt.imshow(np.abs(im_new),'gray')
    plt.savefig('crossover1.jpg')
    plt.imshow(np.abs(im_new1),'gray')
    plt.savefig('crossover2.jpg')
    plt.imshow(im_new2.real, 'gray')

    #plt.show()

if __name__ == '__main__':
    main()
