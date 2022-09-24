# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:56:49 2022

@author: Yacine Benkirane
Code for PSET 2, Problem 2
"""

import numpy as np


#Note: integrate_adaptative very similar to Problem 1 -> could have used it in Q1? 

# This is the enhanced (faster) integrator --> Less counts 
#Note: MUST NOT CALL f(x) multiple times for the same x...

def integrate_adaptive(func, a, b, tol, extra=None):
    if extra == None:
        x = np.linspace(a,b,5)
        dx = (b - a)/(len(x) - 1)
        y = func(x)
        
        
        INT1 = 2*dx*(y[0]+4*y[2]+y[4])/3
        INT2 = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 
        absErr = np.abs(INT1 - INT2)
        
        integrate_adaptive.counter += 5 #Let's keep track of the number of integration iterations...
        
        if absErr < tol:
            return INT2
        
            
        else:
            middlePoint = (a + b)/2
            lowerBound = integrate_adaptive(func, a, middlePoint, tol/2, extra=[y[0], y[1], y[2], dx])
            upperBound = integrate_adaptive(func, middlePoint, b, tol/2, extra=[y[2], y[3], y[4], dx])
            return lowerBound + upperBound
    
    else:   

        x=np.array([a+0.5*extra[3],b-0.5*extra[3]])
        y=func(x)
        integrate_adaptive.counter += 2
        dx=extra[3]/2
        area1 = 2*dx*(extra[0]+4*extra[1]+extra[2])/3
        area2 = dx*(extra[0]+4*y[0]+2*extra[1]+4*y[1]+extra[2])/3 
        absErr = np.abs(area1-area2)
        
        if absErr < tol:
            return area2
        
        else:
            
            middlePoint = (a + b)/2
            lowerBound = integrate_adaptive(func, a, middlePoint, tol/2, extra=[extra[0],y[0],extra[1],dx])#a to mid ;; Quickly hinted in class as for the extras
            upperBound = integrate_adaptive(func,middlePoint,b,tol/2, extra=[extra[1],y[1],extra[2],dx])# mid to b
            return lowerBound + upperBound     
        
#def E_z(func,  zVal, a, b, tol): #Using Q1 for Q2?
#    x = np.linspace(a, b, 5)
#    y = func(x, zVal)
#    dx = x[1] - x[0]
#    
#    #Integrates with Three Points
#    IP1 = (y[0] + 4*y[2] + y[4])/3*(2*dx)
#    IP2 = (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/3*dx
#    absErr = np.abs(IP1 - IP2)
#    
#    if absErr < tol:
#        return IP2
#    else:
#        middlePoint = (a + b)/2
#        INT1 = E_z(func, zVal, a, middlePoint, tol/2) #From a to middle point
#        INT2 = E_z(func, zVal, middlePoint, b, tol/2) #from middle point to b
#        return INT1 + INT2 #from a to b sum 
            
#Defining our Bounds
a = -50
b = 50

def lorentz(x): #defining our lorentzian function to test the opposing methods
    return 1/(1+x**2)

#Lorentz bound from -50 to 50 ;;;; Exp bound from 0 to 10
    
integrate_adaptive.counter = 0

soln = integrate_adaptive(lorentz, a, b, 1e-8)
print('Faster Integration Method:')
print('\n Lorentzian Function')
print('Error: ',soln - (np.arctan(b) - np.arctan(a))) #Integral of Lorentz
print('Number of Calls: ', integrate_adaptive.counter)

LorentzImprovedCount = integrate_adaptive.counter


integrate_adaptive.counter = 0

soln = integrate_adaptive(np.exp,0, 10,1e-8)
print('\n Exponential Function')
print('Error: ',soln-(np.exp(10)-np.exp(0)))
print('Number of calls: ',integrate_adaptive.counter)

ExponentialImprovedCount = integrate_adaptive.counter









# This is the "Lazy Way" we coded in class
def integrate_adaptive(func,a,b,tol):
    x=np.linspace(a,b,5)
    y=func(x)
    integrate_adaptive.counter += 5
    dx=(b-a)/(len(x)-1)
    area1 = 2*dx*(y[0]+4*y[2]+y[4])/3
    area2 = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    absErr = np.abs(area1-area2)
    if absErr < tol:
        return area2
    else:
        middlePoint = (a + b)/2
        lowerBound = integrate_adaptive(func, a, middlePoint, tol/2)
        upperBound = integrate_adaptive(func, middlePoint, b, tol/2)
        return lowerBound + upperBound

integrate_adaptive.counter = 0
soln = integrate_adaptive(lorentz, a, b, 1e-8)
print('\n------------------------------------')
print('\n Lazy Integration Method:')
print('\n Lorentzian Function')
print('Error' ,soln-(np.arctan(b)-np.arctan(a)))
print('Number of Calls: ',integrate_adaptive.counter)

LorentzPoorCount = integrate_adaptive.counter


integrate_adaptive.counter = 0
soln = integrate_adaptive(np.exp, 0, 10, 1e-8)
print('\n Exponential Function ')
print('Error: ', soln-(np.exp(10)-np.exp(0)))
print('Number of Calls: ', integrate_adaptive.counter)

ExponentialPoorCount = integrate_adaptive.counter

print('\n \n=================================== \n')
print('The count on the Lorentzian Integrator was improved by a factor of: ', LorentzPoorCount/LorentzImprovedCount)
print('The count on the Exponential Integration was improved by a factor of: ', ExponentialPoorCount/ExponentialImprovedCount)
print('This is roughly equivalent to our expected ratio of: ', 5/2)