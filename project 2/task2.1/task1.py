import sys
import numpy as np
import math as mt
	from PIL import Image as im
from matplotlib import pyplot as plt
import numpy.fft as fft

#sigma = 20
#sigma = (n-1)/float(2*2.575)
#gauss_factor = 1/float(sigma*(2*mt.pi)**0.5)

def gaussian(x,y,sigma):
	gauss_exp = np.exp(-(x**2+y**2)/float(2*sigma**2))
	gauss_factor = 1/float(sigma*(2*mt.pi)**0.5)
	return gauss_factor*gauss_exp
	
def gridCompute(image,grid_w,grid_h,type):  # kernel grid computing
	sigma = (grid_h-1)/float(2*2.575)
	if type == 2:  # multiplication in the frequency domain
		grid_w, grid_h = image.shape
	grid = np.zeros([grid_w,grid_h])
	center_w = int(grid_w/2)
	center_h = int(grid_h/2)
	for i in range(0,grid_w-center_w):
		for j in range(0,grid_h-center_h):
			grid[center_w+i,center_h+j] = gaussian(i,j,sigma)
			grid[center_w-i,center_h+j] = gaussian(i,j,sigma)
			grid[center_w+i,center_h-j] = gaussian(i,j,sigma)
			grid[center_w-i,center_h-j] = gaussian(i,j,sigma)
	grid = grid/np.sum(grid)
	return grid

def multiplication(image,grid):
	#-- Fourier transform of original image --#
	image_fourier = fft.fft2(image)
	image_shifted = fft.fftshift(image_fourier)
	#-- Multiplication of image and grid --#
	combination = np.multiply(grid,image_shifted)
	#-- Inverse Fourier transform of produced image --#
	image_ishifted = fft.ifftshift(combination)
	new_image = fft.ifft2(image_ishifted)
	return new_image
	
def convolution(image,x,y,grid,type):
	grid_w, grid_h = grid.shape
	grid_w_offset = int(grid_w/2)
	grid_h_offset = int(grid_h/2)
	conv = [0]*grid_w  # list of 0s 1 x grid width
	conv_sum = 0
	image_part = np.empty([grid_w, grid_h])
	for i in range (-grid_w_offset, grid_w_offset):
		for j in range(-grid_h_offset, grid_h_offset):
			image_part[grid_w_offset-i][grid_h_offset-j] = image[x-i][y-j]
	if type == 0: # naive convolution
		conv_sum = np.sum(np.multiply(image_part,grid))
	elif type == 1:  # separable convolution
		conv = np.multiply(image_part,grid[:,0].reshape([grid_h,1])).sum(axis=0)
		conv_sum = np.sum(np.multiply(conv,grid[0,:]))
	return conv_sum

def smooth(image,grid_w,grid_h,type):
	image_w,image_h = image.shape
	new_image = np.empty([image_w, image_h])
	grid = gridCompute(image,grid_w,grid_h,type)
	grid_w_offset = int(grid_w/2)
	grid_h_offset = int(grid_h/2)
	if type == 0 or type == 1:
		for x in range(grid_w_offset, image_w-grid_w_offset):
			for y in range(grid_h_offset, image_h-grid_h_offset):
				new_image[x][y] = convolution(image,x,y,grid,type)
	elif type == 2:
		new_image = multiplication(image,grid)
		#-- Computing log of the image for visualization --#
		log = lambda t: np.log(np.absolute(t))
		new_image = np.vectorize(log)(new_image)
	else:
		print 'error: unknown type'
	return new_image

def main(grid_w,grid_h,type):
	print grid_w, grid_h, type
	#-- types of blurring: 0 - naive conv., 1 - separable conv., 2 - multiplication in the frequency domain --#
	#image = np.array(im.open("../img/bauckhage.jpg"))
	image = np.array(im.open("../img/maltsev.jpg"))
	#plt.imshow(smooth(image,grid_w,grid_h,type),'gray')  
	plt.imshow(smooth(image[:,:,2],grid_w,grid_h,type),'gray')
	plt.savefig('maltsev'+str(type)+'.jpg')
	plt.show()
	
if __name__ == "__main__":
	main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])) # grid width, grid height and type of blurring from command line