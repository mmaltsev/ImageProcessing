from PIL import Image as im
from pylab import *

# naive 1D approach, 2D approach just didn't want to work fine
def warp(image, amplitude, period, phase, direction):

    height, width = shape(image)

    # calculating size of new image
    if (direction == 1):                # vertical warping
        h = height + amplitude * 2
        if (period > height * 2 - 1):           # in this case, less than one full oscillation will fit into the image
            h = height + amplitude
        result_image = np.zeros((h, width))
    else:
        w = width + amplitude * 2
        if (period > width * 2 - 1):
            w = width + amplitude
        result_image = np.zeros((height, w))

    # calculating new values
    for i in xrange(0, width - 1):
        for j in xrange(0, height - 1):
            if (direction == 1):
                # amplitude*sin(frequency*X+phase)
                # we calculate frequency using period T and formula for their relation 'w=2*pi/T'
                x = amplitude * (sin(i * (2.0 * pi / period) + phase) + 1)
                # move each j value from original picture adding calculated delta
                result_image[x + j, i] = image[j, i]
            else:
                y = amplitude * (sin(j * (2.0 * pi / period) + phase) + 1)
                result_image[j, y + i] = image[j, i]
    return result_image


def main():

    image = np.asarray(im.open('clock.jpg'))

    parameters = [[140, 256, pi, 1],                    # 1st picture, one period
                 [140, 512, pi, 1],                         # 2nd, half period
                 [10, 120, (2.5 * pi), 1], [10, 120, pi, 2],# 3rd, ci. 2 periods along each axis
                 [10, 60, pi, 1], [10, 45, (0.5 * pi), 2],  # and so on
                 [10, 33, (1.5 * pi), 1], [10, 240, pi, 2]]

    for i in range(0, 5):
        if i < 2:
            plt.imshow(warp(image, parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3]), 'gray')
            #plt.savefig('result'+str(i)+'.jpg')
            plt.show()
        else:
            plt.imshow(warp(
                warp(image, parameters[2 * i - 2][0], parameters[2 * i - 2][1], parameters[2 * i - 2][2], parameters[2 * i - 2][3]),
                parameters[2 * i - 1][0], parameters[2 * i - 1][1], parameters[2 * i - 1][2], parameters[2 * i - 1][3]), 'gray')
            #plt.savefig('result' + str(i) + '.jpg')
            plt.show()

main()
