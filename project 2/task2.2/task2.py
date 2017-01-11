import sys
import numpy as np
import math as mt
from PIL import Image as im
from matplotlib import pyplot as plt
import scipy.ndimage as img


def gaussian(x, y, sigma):
    gauss_exp = np.exp(-(x ** 2 + y ** 2) / float(2 * sigma ** 2))
    gauss_factor = 1 / float(sigma * (2 * mt.pi) ** 0.5)
    return gauss_factor * gauss_exp


def gridCompute(image, grid_w, grid_h):  # kernel grid computing
    sigma = (grid_h - 1) / float(2 * 2.575)
    # multiplication in the frequency domain
    grid = np.zeros([grid_w, grid_h])
    center_w = int(grid_w / 2)
    center_h = int(grid_h / 2)
    for i in range(0, grid_w - center_w):
        for j in range(0, grid_h - center_h):
            grid[center_w + i, center_h + j] = gaussian(i, j, sigma)
            grid[center_w - i, center_h + j] = gaussian(i, j, sigma)
            grid[center_w + i, center_h - j] = gaussian(i, j, sigma)
            grid[center_w - i, center_h - j] = gaussian(i, j, sigma)
            grid = grid / np.sum(grid)
    return grid


def smooth(image, grid_w, grid_h):
    image_w, image_h = image.shape
    new_image_x = np.empty([image_w, image_h])
    new_image_y = np.empty([image_w, image_h])
    new_image = np.empty([image_w, image_h])
    grid = gridCompute(image, grid_w, grid_h)
    grid_y = np.diff(grid, axis=0)
    grid_x = np.diff(grid)
    new_image_x = img.filters.convolve(image, grid_x)
    new_image_y = img.filters.convolve(image, grid_y)
    new_image = ((new_image_x ** 2) + (new_image_y ** 2)) ** 0.5
    return new_image


def main(grid_w, grid_h):
    image = np.array(im.open("../img/clock.jpg"))
    plt.imshow(smooth(image, grid_w, grid_h), 'gray')
    plt.savefig('clock' + str(grid_w) + '.jpg')
    plt.show()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))  # grid width, grid height and type of blurring from command line