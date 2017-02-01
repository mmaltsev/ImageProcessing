import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
def r2(r, rmax):
    mask = r < rmax
    r[ mask] = 1. - r[mask]**2 / rmax**2
    r[-mask] = 0.
    return r
def r3(r, rmax):
    mask = r < rmax
    r[ mask] = np.sqrt(rmax**2 - r[mask]**2) / rmax
    r[-mask] = 0.
    return r
def r4(r, rmax):
    return 1. - np.tanh(r/rmax)
def r5(r, rmax):
    return np.exp(-0.5 * (r/rmax)**2)
def fisheye(image, mu, rfct, rmax):
    h1,w1 = image.shape
    xs, ys = np.meshgrid(np.arange(w1), np.arange(h1))
    X = np.vstack((ys.flatten(),xs.flatten()))
    C = mu - X
    # vectors pointing to mu
    r= np.sqrt(np.sum(C**2, axis=0)) # distances to mu
    d = rfct(r, rmax)
    X += (C * d).astype(int)
    from scipy import interpolate
#    tck = interpolate.bisplrep(xs, ys, Z, s=0)
#    znew = interpolate.bisplev(X[:, 0], X[0, :], tck)
    ret = np.array(img.map_coordinates(image, X, order=3))
    ret = np.reshape(ret,image.shape)
    return ret
import sys
image = msc.imread('task5/asterixGrey.jpg', flatten=True).astype('float')
hh = fisheye(image, mu=np.array([[300],[540]]), rfct=r5, rmax=150)
tt = msc.toimage(hh, cmin=0,cmax=255)#.save('asterix-eye.png')
tt.show()