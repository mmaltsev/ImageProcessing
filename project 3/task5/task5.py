import scipy.misc as msc
import scipy.ndimage as img
import numpy as np
from skimage import data, color
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
def getCoordinatePairs(img):
    hight,width = img.shape
    X = np.array(np.meshgrid(np.arange(width), np.arange(hight))).T.reshape(-1, 2)
    return X

def inpoly(xy, points):
    coefficients = np.zeros((points.__len__(), 2))
    #fitting multiple lines, given points which are ordered subsequantly
    for i in range(len(points)):
        x_coords, y_coords = zip(*[points[i - 1], points[i]])
        coefficients[i, :] = np.polyfit(x_coords, y_coords, 1)
    ind = (xy[:, 1] >= xy[:, 0] * coefficients[0, 0] + coefficients[0, 1]).astype(int) * \
          (xy[:, 0] >= (xy[:, 1] - coefficients[1, 1]) / coefficients[1, 0]).astype(int) * \
          (xy[:, 1] <= xy[:, 0] * coefficients[2, 0] + coefficients[2, 1]).astype(int) * \
          (xy[:, 0] <= (xy[:, 1] - coefficients[3, 1]) / coefficients[3, 0]).astype(int)
    return ind


def main(isle,asterix, points = None, corrpoints = None, method = None):
    if points is None:
        corrpoints = [(0,0), (0,asterix.shape[1]-1),
                      tuple(s - 1 for s in asterix.shape),(asterix.shape[0]-1, 0)]
        points = [(56, 215),(10, 365),
                  (296,364 ),(258, 218)]
    u =np.array(corrpoints).astype(float)[:,0]
    v= np.array(corrpoints).astype(float)[:,1]
    x = np.array(points).astype(float)[:,0]
    y = np.array(points).astype(float)[:,1]

    A = np.array([
        [u[0], v[0], 1, 0, 0, 0, -u[0]*x[0], -v[0]*x[0]],
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
    def invertCoord(x,y):
        newx = (b * f - c * e + (e - f * h) * x + (c * h - b) * y) \
               / (a * e - b * d + (d * h - e * g) * x + (b * g - a * h) * y)
        newy = (c * d - a * f + (f * g - d) * x + (a - c * g) * y) \
               / (a * e - b * d + (d * h - e * g) * x + (b * g - a * h) * y)
        return  np.vstack((newx,newy))

    #calculate corresponding coords on isle

    if method == 'naive':
        X = getCoordinatePairs(asterix)
        #only for clock approach, hack though
        center = (corrpoints[0] +corrpoints[2])/2
        r = np.sqrt(np.sum((corrpoints[0] - center)**2))
        X = X[np.sqrt(np.sum((X - center) ** 2, axis=1)) < r].T
        new = getCoords(X[0],X[1])
        new = np.round(new).astype(int)
        xyIn = np.vstack({tuple(row) for row in new.T})
        #very naive approach, just to round the mapped coordinates and put the intensities on them, works though
#        for i in range(len(new.T)):
 #           isle[new[0,i], new[1,i]] = asterix[X[0,i],X[1,i]]
    #im2 = msc.toimage(isle, cmin=0,cmax=255)#.save('asterix-eye.png')
    #im2.show()
    else:
        # better way to get coordinates from area in "isle"
        # points = [(56, 215), (10, 365), (296, 364), (258, 218)]
        xy = getCoordinatePairs(isle)
        xyIn = xy[inpoly(xy, points).astype(bool)]


        #dummy way of getting corresponding coordinates in "isle" pic
    #   xyIn = np.vstack({tuple(row) for row in new.T})


    #find coordinates in source picture by invert transform
    mapX = invertCoord(xyIn[:,0], xyIn[:,1])
    #get intensities by interpolation
    intensities = np.array(img.map_coordinates(asterix, mapX, order=3))
    for i in range(len(xyIn )):
        isle[xyIn[i,0],xyIn[i,1]] = intensities[i]

    im2 = msc.toimage(isle, cmin=0, cmax=255)  # .save('asterix-eye.png')
    #im2.show()
    return im2
def getElipsePoints(src = None):


    # Load picture, convert to grayscale and detect edges
    if src is None :
        image_rgb = data.coffee()[0:220, 160:420]
        src = color.rgb2gray(image_rgb)
    edges = canny(src, sigma=2.0,
                  low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    # Find the most distant coordinates, i.e. biggest and smallest horde
    xy = np.vstack((cy, cx)).T
    center = np.array([yc, xc])
    dists = np.sqrt(np.sum((xy - center) ** 2, axis=1))
    p1 = xy[np.argmin(dists)]
    p2 = xy[np.argmax(dists)]
    p3 = 2 * center - p1
    p4 = 2 * center - p2
    return (p1,p2,p3,p4)
if __name__ == '__main__':
    isle = msc.imread('task5/isle.jpg', flatten=True).astype('float')
    #let asterix be abstract pic to be projected
    asterix = msc.imread('task5/asterixGrey.jpg', flatten=True).astype('float')
    im2 = main(isle,asterix)
    im2.save("task5/asterix_on_isle.jpg")
    clock = msc.imread('task5/clock.jpg', flatten=True).astype('float')
    im2 = main(isle,clock)
    im2.save("clock_on_isle.jpg")

    for i in range(10):
        im2 = main(isle, np.array(im2,dtype=float))

    im2.show()
    im2.save("iterated.jpg")
    points = getElipsePoints()
    image_rgb = data.coffee()[0:220, 160:420]
    cup = np.array(color.rgb2gray(image_rgb)) *255
    pairs = getCoordinatePairs(clock)
    corrpoints = np.array([[127, 8], [3, 127], [127,245],[252,127]])
    for p in points:
        cup[p[0], p[1]] = 255
    im3 = main(cup, clock, points, corrpoints,method='naive')
    im3.show()
    msc.toimage(cup).save('task5/cup.jpg')