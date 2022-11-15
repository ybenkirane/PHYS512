# -*- coding: utf-8 -*-
"""
Created on Sun Nov 7 20:50:49 2022

@author: Yacine Benkirane
Contributions, worked with various classmates on the discord (From Recollection): 
    Matt, Nic, Simon, Guillaume, Jeff, Louis, Chris, Steve
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft


def dftOffset(fun:np.ndarray, x0:int):
    
    assert len(fun)%2==0, f"array length {fun.shape} must be even 1-d"
    N = len(fun)
    
    funft = rfft(fun)
    CTFactor = np.exp(-2j * np.pi * x0/N * np.arange(0, N//2 + 1)) #Got reccomendation from Classmates for this factor
    offsetted = irfft(funft * CTFactor)
    
    return offsetted

def correlation(u,v):
    
    return irfft(rfft(u) * np.conj(rfft(v)))
    #This function will throw back the correlation of (u,v)

def offsettedAutoCorrelation(u, offset):
    
        #Correlating some fun with its shifted self 
    u_offset = np.roll(u,offset)
    return correlation(u, u_offset)


def wraplessCorrelation(u,v):
    
        #Question of Padding: pad u and v with zeroes to maintain the center of convolution
    zeroes = np.zeros(len(v)) 
    
        #To avoid circulant conditions, padding input with zeroes should be sufficent. 
    u_Padding = np.hstack((u, zeroes))
    v_Padding = np.hstack((v, zeroes))
    
        #Does the correlation of (u,v) but without the circulant, as it can be problematic... 
    return correlation(u_Padding, v_Padding)

x = np.linspace(-5,10, 500)
fun = np.exp(-x**2) #Gaussian

offset = len(fun)//5
offsetted_fun = dftOffset(fun, offset)

plt.figure(figsize=(10,5))
plt.plot(x,fun,label="Original Gaussian")
plt.plot(x,offsetted_fun,label="Shifted Gaussian")
plt.title("Convolution Gaussian Shift")
plt.xlabel("Phase")
plt.ylabel("Intensity")
plt.legend()
plt.savefig("ShiftedGaussian.png")
plt.show()


# Correlation of gaussian with its self
plt.figure(figsize=(10,5))
plt.plot(correlation(fun,fun),label="AutoCorr Gaussian")
offsets=(5,10,15,20,25,30,40,50,75, 100, 125, 150)
for offset in offsets:
    plt.plot(offsettedAutoCorrelation(fun,offset),label=f"Offsetted by {offset}")
plt.title("Gaussian Correlations")
plt.xlabel("Phase")
plt.ylabel("Intensity")
plt.legend()
plt.tight_layout()
plt.savefig("Q2Correlations.png")
plt.show()
