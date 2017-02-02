import numpy as np
from PIL import Image as im
#photo =
image = np.array(im.open("%s.jpg" % photo))
import scipy.misc as msc

im2 = msc.toimage(image, 2, mode='constant')
im2.show()

from xalglib import spline2dbuildbicubicv

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def mapPicture():
    pass

x, y = np.mgrid[-1:1:20j, -1:1:20j]
z = (x+y) * np.exp(-6.0*(x*x+y*y))

plt.figure()
plt.pcolor(x, y, z)
plt.colorbar()
plt.title("Sparsely sampled function.")
plt.show()
xnew, ynew = np.mgrid[-1:1:140j, -1:1:140j]
tck = interpolate.bisplrep(x, y, z, s=0)
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
plt.figure()
plt.pcolor(xnew, ynew, znew)
plt.colorbar()
plt.title("Interpolated function.")
plt.show()

#####
import scipy.misc as msc
import scipy.ndimage as img

g = msc.imread('task5/isle.jpg', flatten=True).astype('float')
g1 = msc.imread('task5/asterixGrey.jpg', flatten=True).astype('float')

corrpoints = [(1,1), (1,729),
              (495, 1), (495, 729)]
points = [(56, 215),(10, 365),
          (258, 218),(296,364 )]
#AX = B
#X= A^(-1)B
u =np.array(corrpoints).astype(float)[:,0]
v= np.array(corrpoints).astype(float)[:,1]
x = np.array(points).astype(float)[:,0]
y = np.array(points).astype(float)[:,1]

A = np.array([
    [u[0], v[0], 1, 0 ,0, 0, -u[0]*x[0], -v[0]*x[0]],
    [u[1], v[1], 1, 0, 0, 0, -u[1]*x[1], -v[1]*x[1]],
    [u[2], v[2], 1, 0, 0, 0, -u[2]*x[2], -v[2]*x[2]],
    [u[3], v[3], 1, 0, 0, 0, -u[3]*x[3], -v[3]*x[3]],
    [0, 0, 0, u[0], v[0], 1, -u[0]*y[0], -v[0]*y[0]],
    [0, 0, 0, u[1], v[1], 1, -u[1]*y[1], -v[1]*y[1]],
    [0, 0, 0, u[2], v[2], 1, -u[2]*y[2], -v[2]*y[2]],
    [0, 0, 0, u[3], v[3], 1, -u[3]*y[3], -v[3]*y[3]],
]).astype(float)
a,b,c,d,e,f,g,h = np.dot(np.linalg.inv( A),  np.append(x,y).T)
def getCoords(u,v):
    xnew = (a*u+b*v + c)/(g*u+h*v+1)
    ynew = (d*u + e*v+f)/(g*u + h*v + 1)
    return  np.vstack((xnew,ynew))
hight,width = g1.shape
#
X = np.array(np.meshgrid(np.arange(width), np.arange(hight))).T.reshape(-1,2).T
new = getCoords(X[0],X[1])
ret = np.array(img.map_coordinates(g1, new, order=3))
new = np.round(new).astype(int)
for i in range(len(new.T)):
    g[new[0,i], new[1,i]] = g1[X[0,i],X[1,i]]
#ret = np.reshape(ret,g1.shape)
im2 = msc.toimage(g, cmin=0,cmax=255)#.save('asterix-eye.png')
im2.show()

out = np.zeros((300,400))
for i in range(len(new.T)):
    out[new[0,i], new[1,i]] = ret[i]
im2 = msc.toimage(out, cmin=0,cmax=255)#.save('asterix-eye.png')
im2.show()




###trash
src =np.array(corrpoints).astype(float)
dst = np.array(points).astype(float)
xs = src[:, 0]
ys = src[:, 1]
rows = src.shape[0]
A = np.zeros((rows * 2, 8))
A[:rows, 0] = 1
A[:rows, 1] = xs
A[:rows, 2] = ys
A[rows:, 3] = 1
A[rows:, 4] = xs
A[rows:, 5] = ys
A[:rows, 6] = - dst[:, 0] * xs
A[:rows, 7] = - dst[:, 0] * ys
A[rows:, 6] = - dst[:, 1] * xs
A[rows:, 7] = - dst[:, 1] * ys
b = np.zeros((rows * 2,))
b[:rows] = dst[:, 0]
b[rows:] = dst[:, 1]
params = np.linalg.lstsq(A, b)[0]
#a0, a1, a2, b0, b1, b2, c0, c1
#X = (a0 + a1 * x + a2 * y) / (1 + c0 * x + c1 * y)
#Y = (b0 + b1 * x + b2 * y) / (1 + c0 * x + c1 * y)
#    xnew = (a*u+b*v + c)/(g*u+h*v+1)
#    ynew = (d*u + e*v+f)/(g*u + h*v + 1)
a0, a1, a2, b0, b1, b2, c0, c1 = [c, a, b, f, d, e, g,h]
newx = (a2 * b0 - a0 * b2 + (b2 - b0 * c1) * x + (a0 * c1 - a2) * y) \
            / (a1 * b2 - a2 * b1 + (b1 * c1 - b2 * c0) * x + (a2 * c0 - a1 * c1) * y)
newy = (a0 * b1 - a1 * b0 + (b0 * c0 - b1) * x + (a1 - a0 * c0) * y) \
            / (a1 * b2 - a2 * b1 + (b1 * c1 - b2 * c0) * x + (a2 * c0 - a1 * c1) * y)




from numpy import ones,vstack
from numpy.linalg import lstsq
points = [(56, 215), (10, 365),  (296, 364),(258, 218)]
coefficients =np.zeros((points.__len__(), 2))

for i in range(len(points)):
    x_coords, y_coords = zip(*[points[i-1], points[i]])
    coefficients[i,:] = np.polyfit(x_coords, y_coords , 1)

import matplotlib.pyplot as plt

for i in range(4):
    plt.plot(np.arange(0, 400,1), coefficients[i,0]*np.arange(0, 400,1)+coefficients[i,1] )

#y = kx + b
#x = (y-b)/k
x1,y1 = np.mgrid[0:400:100j, 0:400:100j]
xy = np.vstack((x1.flatten(), y1.flatten())).T
def inpoly1(xy, coefficients):
    return ((xy[:,1]>= xy[:,0]*coefficients[0,0]+coefficients[0,1] )
            and xy[:,0]>= (xy[:,1]-coefficients[1,1])/coefficients[1,0]
            and xy[:,1]<= xy[:,0]*coefficients[2,0]+coefficients[2,1]
            and xy[:,0] <= (xy[:,1] - coefficients[3, 1]) / coefficients[3, 0]
            )

def inpoly(xy, coefficients):
    ind = (xy[:,1]>= xy[:,0]*coefficients[0,0]+coefficients[0,1]).astype(int) *\
    (xy[:,0]>= (xy[:,1]-coefficients[1,1])/coefficients[1,0]).astype(int)*\
    (xy[:,1]<= xy[:,0]*coefficients[2,0]+coefficients[2,1]).astype(int)*\
    (xy[:,0] <= (xy[:,1] - coefficients[3, 1]) / coefficients[3, 0]).astype(int)
    return ind
xyIn= xy[inpoly(xy,coefficients).astype(bool)]
plt.scatter(xyIn[:,0], xyIn[:,1])

#point in polygon problem, buggy however
# def inpoly(nvert, vertx, verty, testx, testy):
#     c = False
#     j = nvert-1
#     for i in range(nvert):#i = 0, j = nvert-1; i < nvert; j = i++) :
#         if  ((verty[i]>testy) != (verty[j]>testy)) and (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]):
#             c = not c
#         j = i
#     return c
# xyIn= np.array([pair for pair in xy if inpoly(4, x, y, pair[0], pair[1])])
# plt.scatter(xyIn[:,0], xyIn[:,1])


#fitting elipse
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
image_rgb = data.coffee()[0:220, 160:420]
image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=2.0,
              low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
###
xy = np.vstack((cy, cx)).T
center = np.array([yc,xc])
dists = np.sqrt(np.sum((xy - center)**2, axis=1))
p1 = xy[np.argmin(dists)]
p2 = xy[np.argmax(dists)]
p3 = 2 * center - p1
p4 = 2 * center - p2
im = np.zeros((300,300))
im[p1[0],p1[1]] =255
im[p2[0],p2[1]] =255
im[p3[0],p3[1]] =255
im[p4[0],p4[1]] =255
im[cy, cx] = 255
im[yc, xc] = 255

msc.toimage(image_gray*255).save('task5/fitted_elipse.jpg')
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(edges)
edges[cy, cx] = (250, 0, 0)
#image_gray[cy, cx] = 0.
fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()