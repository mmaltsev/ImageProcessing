import scipy.misc as msc
import scipy.ndimage as img
import numpy as np

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


def main(isle,asterix):

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

#    X = getCoordinatePairs(asterix).T
    #calculate corresponding coords on isle
#    new = getCoords(X[0],X[1])

#    new = np.round(new).astype(int)

    #very naive approach, just to round the mapped coordinates and put the intensities on them, works though
    #for i in range(len(new.T)):
    #    isle[new[0,i], new[1,i]] = asterix[X[0,i],X[1,i]]
    #im2 = msc.toimage(isle, cmin=0,cmax=255)#.save('asterix-eye.png')
    #im2.show()


    def invertCoord(x,y):
        newx = (b * f - c * e + (e - f * h) * x + (c * h - b) * y) \
               / (a * e - b * d + (d * h - e * g) * x + (b * g - a * h) * y)
        newy = (c * d - a * f + (f * g - d) * x + (a - c * g) * y) \
               / (a * e - b * d + (d * h - e * g) * x + (b * g - a * h) * y)
        return  np.vstack((newx,newy))
    #dummy way of getting corresponding coordinates in "isle" pic
#   xyIn = np.vstack({tuple(row) for row in new.T})

    #better way to get coordinates from area in "isle"
    #points = [(56, 215), (10, 365), (296, 364), (258, 218)]
    xy = getCoordinatePairs(isle)
    xyIn = xy[inpoly(xy, points).astype(bool)]

    #find coordinates in source picture by invert transform
    mapX = invertCoord(xyIn[:,0], xyIn[:,1])
    #get intensities by interpolation
    intensities = np.array(img.map_coordinates(asterix, mapX, order=3))
    for i in range(len(xyIn )):
        isle[xyIn[i,0],xyIn[i,1]] = intensities[i]

    im2 = msc.toimage(isle, cmin=0, cmax=255)  # .save('asterix-eye.png')
    #im2.show()
    return im2
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
