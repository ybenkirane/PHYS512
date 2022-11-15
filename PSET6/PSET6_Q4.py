# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 00:50:06 2022

@author: Yacine Benkirane
Contributions, worked with various classmates on the discord (From Recollection): 
    Matt, Nic, Simon, Guillaume, Jeff, Louis, Chris, Steve
"""

import numpy as np
import matplotlib.pyplot as plt


N = 201 
k0 =  51 
x = np.linspace(0, N, N) 

def SinTransform(k1, k2, N):

    kMinus = k1 - k2
    kPlus = k1 + k2    
    
    transform = np.abs((1 - np.exp(2*np.pi*1j*kMinus)) / (1 - np.exp(2*np.pi*1j*kMinus/N)) - (1 - np.exp(-2*np.pi*1j*kPlus)) / (1 - np.exp(-2*np.pi*1j*kPlus/N)))
    transformNormalized = transform/transform.sum()
    return transformNormalized


sine = 2j * np.sin(2 * np.pi * k0 *x /N) #sine for numerical transform
numTransf = np.abs(np.fft.fft(sine))
numTransfNormalized = numTransf/numTransf.sum()

plt.plot(SinTransform(k0, x, N),  'r--',label='Anal Transform')
plt.plot(numTransfNormalized, 'g-.', label='Num Transform')
plt.xlabel("k-value")
plt.ylabel("Intensity")
plt.legend()
plt.grid()
plt.title('Analytic and Numerical Transforms')
plt.savefig('PSET6_Q4c.png')
plt.show()

#Window Function
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
window_transform = np.abs(np.fft.fft(sine*window))
winTransf_Normalized = window_transform/window_transform.sum()


plt.plot(numTransfNormalized,  'r--', label='Without Window')
plt.plot(np.abs(winTransf_Normalized), 'g-.', label='With Window')
plt.xlabel("k-value")
plt.ylabel("Intensity")
plt.grid()
plt.legend()
plt.title('Frequency Leakage')
plt.savefig('PSET6_Q4d.png')
plt.show()

