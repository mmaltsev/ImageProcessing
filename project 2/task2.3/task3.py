import math
import numpy as np
from PIL import Image as im
def main(sigma = 0.5, photo ='phil'):
    alfa = [1.68, -0.68]
    beta = [3.7350 , -0.2598 ]
    gamma = [1.7830 ,1.7230 ]
    omega = [0.6318, 1.9970]
    aplus = [0,0,0,0]
    aplus[0] = sum(alfa)
    aplus[1] = math.exp(-gamma[1]/sigma )*(beta[1] * math.sin(omega[1]/sigma) -(alfa[1]+2*alfa[0])*math.cos(omega[1]/sigma)) + \
             math.exp(-gamma[0]/sigma)*(beta[0]*math.sin(omega[0]/sigma) -(2*alfa[1] +alfa[0]))*math.cos(omega[0]/sigma)

    aplus[2] = 2*math.exp(-sum(gamma) / sigma)*(sum(alfa)*math.cos(omega[1]/sigma)*math.cos(omega[0]/sigma) -math.cos(omega[1]/sigma) * beta[0]*math.sin(omega[1]/sigma) - math.cos(omega[0]/sigma) * beta[1]*math.sin(omega[1]/sigma)) + \
             alfa[1] * math.exp(-2*gamma[0]/sigma) + alfa[0] *math.exp(-2*gamma[1]/sigma)
    aplus[3] = math.exp(-(gamma[1]+ 2* gamma[0])/sigma) * (beta[1]*math.sin(omega[1]/sigma) - alfa[1]*math.cos(omega[1]/sigma)) + \
             math.exp(-(gamma[0] + 2 * gamma[1]) / sigma) * (beta[0] * math.sin(omega[0] / sigma) - alfa[0] * math.cos(omega[0] / sigma))
    bplus = [0,0,0,0,0]
    bplus[1] = -2*math.exp(-gamma[1]/sigma)*math.cos(omega[1]/sigma) -\
             2*math.exp(-gamma[0]/sigma)*math.cos(omega[0]/sigma)
    bplus[2] = 4*math.cos(omega[1]/ sigma)*math.cos(omega[0]/ sigma)*math.exp(-sum(gamma)/sigma) + \
             math.exp(-2*gamma[1]/ sigma)+ \
             math.exp(-2*gamma[0]/ sigma)
    bplus[3] = -2*math.cos(omega[0]/ sigma)*math.exp(-(gamma[0] + 2*gamma[1])/sigma) - 2*math.cos(omega[1]/ sigma)*math.exp(-(gamma[1] + 2*gamma[0])/sigma)
    bplus[4] = math.exp(-2*sum(gamma) / sigma)

    n= 1
    bminus = bplus

    aminus = [0,0,0,0,0]
    aminus[1] = aplus[1] - bplus[1]*aplus[0]
    aminus[2] = aplus[2] - bplus[2]*aplus[0]
    aminus[3] = aplus[3] - bplus[3]*aplus[0]
    aminus[4] = -bplus[4]*aplus[0]

    image = np.array(im.open("/home/phil/Dropbox/Uni_Bonn/ws1617/ImageProcessing/ImageProcessing/project 2/img/%s.jpg" % photo))
    h,w = image.shape
    image1 = np.append(image,image[:,image.shape[1]-4:],axis=1)
    image1 = np.append(image[:,:4],image1,axis=1)
    image2 = np.append(image1, image1[(image1.shape[0] - 4):, :], axis=0)
    image2 = np.append(image1[:4,:], image2 , axis=0)
    yminus = image2.copy()
    yplus = image2.copy()
    y = np.zeros(image2.shape)
    y.fill(255)

    for k in np.arange(h)+4:
        for n in np.arange(w)+4:
            yplus[k,n] = sum([aplus[m] * image2[k,n-m] for m in range(4)]) -sum([bplus[m] * yplus[k,n-m] for m in range(1,5)])
            yminus[k,n] =sum([aminus[m] * image2[k,n+m] for m in range(1,5)]) -\
                    sum([bminus[m] * yminus[k,n+m] for m in range(1,5)])
            y[k,n] = (yminus[k,n] + yplus[k,n])/(sigma *math.sqrt(2*math.pi))

    for n in np.arange(w) + 4:
        for k in np.arange(h)+4:
            yplus[k,n] = sum([aplus[m] * y[k-m,n] for m in range(4)]) -sum([bplus[m] * yplus[k-m,n] for m in range(1,5)])
            yminus[k,n] =sum([aminus[m] * y[k+m,n] for m in range(1,5)]) -\
                    sum([bminus[m] * yminus[k+m,n] for m in range(1,5)])
            y[k,n] = (yminus[k,n] + yplus[k,n])/(sigma *math.sqrt(2*math.pi))


    import scipy.misc as msc

    im1 = msc.toimage(
        y
    )
    im1.show()
    im1.save('task2.3/phil_sigma%s.jpg' % sigma)

    if __name__ == '__main__':
        for photo in ['bauckhage', 'phil']:
            for s in np.arange(.5,1.5,0.2):
                main(s, photo)
