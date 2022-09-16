# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:21:25 2022

@author: Yacine Benkirane
Solution to Q2 of PSET1 
Comp Physics at McGill University: PHYS 512
"""

import numpy as np

#Using exp functions as they are often presented as examples.
def funNorm(x):
    return np.e**(x)

def funStretch(x):
    return np.e**(10*x)

#Defining ndiff, function of interest...
def ndiff(fun,x,full):
    epsil = 1e-15 #Epsil used in Q1 for consistency (mentioned in class)
    Df3 = 1e-3 #Picked this for no particular reason 
   
    f3 = (3*fun(x-Df3) - 3*fun(x+Df3) + fun(x+3*Df3) - fun(x-3*Df3)) * (1/(8*Df3**3))
    dx = ((epsil)*(fun(x)/f3))**(1/3)
    
    f1 = (fun(x + dx) - fun(x - dx))/(2*dx)
    error = epsil*(fun(x)/dx) + 2 * f3 * (dx**2)
    
    if full == True:
        return f1, dx, error
    elif full == False:
        return f1

#Above is the assignement, Lines under test the defined function(s)
fprime, dx, error_fPrime = ndiff(funStretch, 1, True)
print('Exp(10x) about x=1')
print('dx:', dx)
print('Computational Deriv:',fprime, 'Analytical Deriv:',10*np.e**10)
print('Computed Error:', error_fPrime, 'Analytical Error:', np.abs(fprime - 10*np.e**10))
print('Ratio btwn Comp and Analyt Errors: ', error_fPrime/np.abs(fprime - 10*np.e**10), '\n')


fprime, dx, error_fPrime = ndiff(funStretch, 0, True)
print('Exp(10x) about x=0')
print('dx:', dx)
print('Computational Deriv:', fprime, 'Analytical Deriv:', 10)
print('Computed Error:', error_fPrime, 'Analytical Error:', np.abs(fprime-10))
print('Ratio btwn Comp and Analyt Errors: ', error_fPrime/np.abs(fprime-10), '\n')


fPrimeValue, dx, error_fPrime = ndiff(funNorm, 1, True)
print('Exp(x) about x=1')
print('dx:', dx)
print('Computational Deriv:',fPrimeValue, 'Analytical Deriv:', np.e)
print('Computed Error:',error_fPrime, 'Analytical Error:', np.abs(fPrimeValue-np.e))
print('Ratio btwn Comp and Analyt Errors: ', error_fPrime/np.abs(fPrimeValue-np.e), '\n')


fPrimeValue, dx, error_fPrime = ndiff(funNorm, 0, True)
print('Exp(x) about x=0')
print('dx:', dx)
print('Computational Deriv:',fPrimeValue,'Analytical Deriv:', 1)
print('Computed Error:', error_fPrime, 'Analytical Error:', np.abs(fPrimeValue-1), )
print('Ratio btwn Comp and Analyt Errors: ', error_fPrime/np.abs(fPrimeValue-1), '\n')


