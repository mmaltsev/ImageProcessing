import math
import numpy as np
from PIL import Image as im
#line = image[100,:]
def causal(line, aplus,bplus):
    yplus =  [0]*(len(line) + 4) #np.append(line[-4:], [0]*(len(line)))
    line = np.append([0]*4,line)

    #line = np.pad(line, [4, 4], 'constant', constant_values=[0, 0])
    #yplus =np.zeros_like(line)
    for n in range(4, len(line)) : # -4 #for evidence
        yplus[n] = np.dot(aplus[::-1] , line[n-3:n+1] ) - np.dot(bplus[1:][::-1], yplus[n-4:n-1+1])
        #yplus[n] = sum([aplus[m] * line[n-m] for m in range(4)]) -np.sum([bplus[m] * yplus[n-m] for m in range(1,5)])
    return yplus[4:]
def anticausal(line, aminus, bminus):
    yminus =np.append( [0] * (len(line)) , line[:4])
    line =  np.append(line, [0] * 4)
#    line = np.pad(line, [4, 4], 'constant', constant_values=[0, 0])
#    yminus =np.zeros_like(line)
    for n in range(len(line)-5,-1,-1 ) : #for evidence
        yminus[n] = np.dot(aminus[1:] , line[n + 1:n+5]) - np.dot(bminus[1:], yminus[n+1:n+5])
        #yminus[n] = sum([aminus[m] * line[n + m] for m in range(1, 5)]) -sum([bminus[m] * yminus[n + m] for m in range(1, 5)])
    return yminus[:-4:]

def main(sigma = 4, photo ='phil'):
    import time
    startT=time.time()
    alfa = [1.68, -0.6803]
    beta = [3.7350 , -0.2598 ]
    gamma = [1.7830 ,1.7230 ]
    omega = [0.6318, 1.9970]
    aplus = [0,0,0,0]
    aplus[0] = sum(alfa)
    aplus[1] = math.exp(-gamma[1]/sigma )*\
               (beta[1] * math.sin(omega[1]/sigma) -(alfa[1]+2*alfa[0])*math.cos(omega[1]/sigma)) + \
             np.exp(-gamma[0]/sigma)*(beta[0]*np.sin(omega[0]/sigma) -(2*alfa[1] +alfa[0])*np.cos(omega[0]/sigma))

    aplus[2] = 2*math.exp(-sum(gamma) / sigma)\
               *( sum(alfa)*math.cos(omega[1]/sigma)*math.cos(omega[0]/sigma) -\
                  math.cos(omega[1]/sigma) * beta[0]*math.sin(omega[0]/sigma) -\
                  math.cos(omega[0]/sigma) * beta[1]*math.sin(omega[1]/sigma)) + \
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

    image = np.array(im.open("%s.jpg" % photo))
    h,w = image.shape

    y = np.zeros(image.shape)

    for i in range(h):
        y[i,:] = (causal(image[i,:], aplus, bplus) + anticausal(image[i,:], aminus, bminus))/(sigma *math.sqrt(2*math.pi))
    y1 = np.zeros_like(image)
    for k in range(w):
         y1 [:,k]= (causal(y[:, k], aplus, bplus) + anticausal(y[:, k], aminus, bminus)) / (
        sigma * math.sqrt(2 * math.pi))

    import scipy.misc as msc
    im1 = msc.toimage(y1)
    print(str(time.time() - startT) + " seconds")

#    from  scipy.ndimage.filters import gaussian_filter
#    im2 = msc.toimage(gaussian_filter(image, 4, mode='constant'))

#   y1 = y1*255/np.max(y1)
    im1.show(title="Recursive filter")

    im2.show(title="Built-in scipy gaussian filter")
    im1.save('%s_sigma%s.jpg' % (photo,sigma))
    print(str(time.time() - startT) + " seconds")

if __name__ == '__main__':
    import logging as log
    log.info("Starting processing")

    main(2, photo='bauckhage')
