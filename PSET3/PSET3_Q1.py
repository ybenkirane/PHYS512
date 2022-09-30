# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:29:33 2022

@author: Yacine Benkirane
"""

import numpy as np
from matplotlib import pyplot as plt

#Standard RK4 Evalutation Below 
def RationalExpression(x,y):
    dyBYdx = y/(1+x**2)
    return dyBYdx


def f(x,y): 
    dyBYdx = np.asarray([y[1],-y[0]])
    return dyBYdx

def rk4_step(fun,x,y,h):
    k1 = h * fun(x,y)
    k2 = h * fun(x+h/2,y+k1/2)
    k3 = h * fun(x+h/2,y+k2/2)
    k4 = h * fun(x+h,y+k3)
    dy = (k1 + 2*k2 + 2*k3 + k4)/6
    return y + dy

#Stepper: takes a step length h, compares to h/2, cancels out leading order error from RK4

def rk4_stepd(fun,x,y,h):
    y1 = rk4_step(fun, x, y, h)
    y2a = rk4_step(fun, x, y, h/2)
    y2b = rk4_step(fun, x+h/2, y2a, h/2)
    return y2b + (y2b - y1)/15


# Let's Integrate using 200 Steps as requested in Question
    
y0 = 1
xSpace1 = np.linspace(-20, 20, 200)
h = np.median(np.diff(xSpace1))
ySpace1 = np.zeros(len(xSpace1))
ySpace1[0]=y0

for i in range(len(xSpace1)-1):
    ySpace1[i+1]=rk4_step(RationalExpression, xSpace1[i], ySpace1[i], h)

#stepd should use roughly 4/11 times as many steps to evaluate the same number of function evaluations as the original RK4 method

y0 = 1
xSpace2 = np.linspace(-20, 20, 73) #Should be roughly 72.7 steps, rounded up
h = np.median(np.diff(xSpace2))
ySpace2 = np.zeros(len(xSpace2))
ySpace2[0] = y0

for i in range(len(xSpace2)-1):
    ySpace2[i+1]=rk4_stepd(RationalExpression, xSpace2[i], ySpace2[i], h)    

# Compare error to real functions
yAnalytical1 = np.e**(np.arctan(xSpace1) + np.arctan(20))
yAnalytical2 = np.e**(np.arctan(xSpace2) + np.arctan(20))

yError1 = np.abs(yAnalytical1 - ySpace1)
yError2 = np.abs(yAnalytical2 - ySpace2)

STND = 'Standard RK4 with 200 Steps'
ENH = 'Enhanced RK4 with 72 Steps'

# Numerical and Analytical Solutions Plot

plt.figure()
plt.title('Plot of Numerical and Analytical Solutions')
plt.plot(xSpace1,ySpace1,'k', label = 'Original Function')
plt.plot(xSpace1,ySpace1,'r^', markerfacecolor = 'none', label = STND)
plt.plot(xSpace2,ySpace2,'b^', markerfacecolor = 'none', label = ENH)
plt.legend()
plt.grid()
plt.savefig('SolutionsP3Q1.png')
plt.show()

# Error Plot

plt.figure()
plt.title('RK4 Error')
plt.semilogy(xSpace1,yError1,'r', label = STND)
plt.semilogy(xSpace2,yError2,'b', label =  ENH)
plt.legend()
plt.grid()
plt.savefig('ErrorPlotP1Q1.png')
plt.show()

# Error Comparison 

print('Standard RK4 RMS Error: ', np.std(yError1))
print('Enhanced RK4 RMS Error: ', np.std(yError2))
print('RK4 Enhancement is ', np.std(yError1)/np.std(yError2), ' times better on average...')