
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt


# In[3]:

n = 20
sigma = 0.5 # 0.5, 1, 2, 4 


# In[4]:

def calc_phi(s):
    return np.exp(-s**2 / ( 2 * sigma**2))

Phi = np.zeros(shape=(n, n))

def distance(x1, x2):
    return abs(x1 - x2)

def fill_Phi():
    for i in range(0, n):
        for j in range(0, n):
            Phi[i, j] = calc_phi(distance(x[i], x[j]))
            # print x[i], x[j], distance(x[i], x[j]), Phi[i, j]


# In[5]:

def calc_y(val):
    sum = 0
    for i in range(0, n):
        sum += w[i] * calc_phi(distance(val, x[i]))
    return sum


# In[6]:

def redraw():
    plt.plot(xs, calc_y(xs))
    plt.plot(x, y, color='r')
    plt.plot(x, y, marker='o', color='g')
    plt.show()


# In[7]:

x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
fill_Phi()
w = np.linalg.inv(Phi).dot(y)
xs = np.linspace(0, n, 200)
redraw()


# In[ ]:



