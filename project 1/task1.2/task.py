#Explore the behavior of the Fourier transform. To this end, when working
#with Python, import the following
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
#Next, create a uni-variate function f (x) to transform, say
n = 512
x = np.linspace(0, 2*np.pi, n)
f = np.sin(x)
# This roduces an array f of type float. To plot this (discrete) function (of
# finite support), you could use
plt.plot(x, f, 'k-')
plt.show()
#Next, compute the (discrete) Fourier transform F (ω) of f (x) by means of
F = fft.fft(f)
# This will produce an array F of type complex. Hence, we you try the fol-
# lowing plotting commands
w = fft.fftfreq(n)
plt.plot(w, F, 'k-')
plt.show()
# you should get a warning message. Therefore, see what happens, if you
# execute
plt.plot(w, np.abs(F), 'k-')
plt.show()
# instead. Also, have a look at the outcome of
plt.plot(w, np.log(np.abs(F)), 'k-')
plt.show()
# Now, generalize the function f (x) as follows: f (x) = o + α · sin(νx + φ). For
# example
offset = 1.
amplitude= 2.
frequency = 16.
phase = np.pi
f = offset + amplitude * np.sin(frequency*x + phase)
###
fprime = fft.ifft(F)
plt.plot(x, f, 'k-')
plt.show()