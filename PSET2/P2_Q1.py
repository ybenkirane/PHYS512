# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:56:08 2022

@author: Yacine Benkirane
Code for PSET 2, Problem 1
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scp

eps0 = 8.8541878*(10**(-12)) #Vacuum electric permittivity
R = 1 #Sphere Radius
sig = 0.1*(4 * np.pi * R**2)**-1 #Surface Charge Density

def E_func(theta,zVal, R = 1, sigma = sig, AtPosR = False):  #Electric Field Function As Shown in Derivation
    if AtPosR:
        return R
    return R**2*sigma/(2*eps0)*((zVal - R*np.cos(theta)) * np.sin(theta))/(R**2 + zVal**2 - 2*R*zVal*np.cos(theta))**(3/2)

#Integrator as discussed with Prof Sievers (integrates with zValues now)
def E_z(func,  zVal, a, b, tol): 
    x = np.linspace(a, b, 5)
    y = func(x, zVal)
    dx = x[1] - x[0]
    
    #Integrates with Three Points
    IP1 = (y[0] + 4*y[2] + y[4])/3*(2*dx)
    IP2 = (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/3*dx
    absErr = np.abs(IP1 - IP2)
    
    if absErr < tol:
        return IP2
    else:
        middlePoint = (a + b)/2
        INT1 = E_z(func, zVal, a, middlePoint, tol/2) #From a to middle point
        INT2 = E_z(func, zVal, middlePoint, b, tol/2) #from middle point to b
        return INT1 + INT2 #from a to b sum 

#Test 
#ax1.set_xlabel('Step Size')
#ax1.set_ylabel('Exp(x) Error', color = 'red', fontsize = 14)
#ax1.tick_params(axis="y", labelcolor='red')
#
#ax2.set_ylabel('Exp(0.01x) Error', color = 'green', fontsize = 14)
#ax2.tick_params(axis="y", labelcolor='green')
    
def compute_field(domain=np.linspace(0, 15, 51)): #Electric Field for domain (iterates through varying z distances)
    
    R = E_func(1, 1, AtPosR = True)
    ElecField = np.zeros(len(domain))
    
    for i in range(len(domain)):
        zVal = domain[i]
        
        if zVal != R:
            ElecField[i] = E_z(E_func, zVal, 0, np.pi, 1) 
            
            
    fig, ax = plt.subplots()
    ax.plot(domain,ElecField, 'tab:purple')
    ax.set_xlabel('Distance (z)')
    ax.set_ylabel('E Field')
    ax.grid(True)
    fig.suptitle('Electric Field from Integral')
    plt.savefig('IntegralElectricField.png')
    return ElecField
    
ElecField = compute_field()

#Scipy Quad funcction
domain = np.linspace(0, 15, 201)
ElecErr = np.zeros(len(domain))
ElecField = np.zeros(len(domain))

#Quad iteration
for i in range(len(domain)):
    zVal = domain[i]   
    ElecField[i], ElecErr[i] = scp.quad(E_func, 0, np.pi, args = zVal)
    #Based on scipy documentation, quad returns integral of function from a to b, and abserr (float) an estimate of the absolute error in the result

#Plotting
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios': [4, 1]})
fig.suptitle('Electric Field from Quad')
ax[0].grid(True)
ax[1].grid(True)
ax[0].plot(domain, ElecField, 'tab:green')
ax[1].set_xlabel('Distance (z)')
ax[0].set_ylabel('E Field')
ax[1].plot(domain, ElecErr, 'tab:red')
ax[1].set_ylabel('Error')
plt.savefig('QuadElectricField.png')
