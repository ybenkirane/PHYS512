# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:36:12 2022

@author: Yacine Benkirane
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import astropy.units as u


elementList = ['U238','Th234','Pa234', 'U234', 'Th230','Ra226', 'Rn223','Po218','Pb214','Bu214', 'Po214','Pb210','Bi210','Po210', 'Pb206']

lifeSet = [4.468e9 * u.yr, 24.10 * u.day, 6.70 * u.hr, 245500 * u.yr,75380 * u.yr,1600 * u.yr,3.8235 * u.d,3.10 * u.min,26.8 * u.min,19.9 * u.min,164.3e-6 * u.s,22.3 * u.yr,5.015 * u.yr,138376 * u.d]

life = []
for l in lifeSet:
    life.append(l.to(u.yr).value)
    

def fun(x,y, lifeTime = life):
    
    dyBYdx = np.zeros(len(lifeTime)+1)
    for n in range(len(lifeTime)+1):    
        if n == 0:
            dyBYdx[n] = -y[n] / lifeTime[n]
        elif n == len(lifeTime):
            dyBYdx[n] = y[n-1] / lifeTime[n-1]
        else:
            dyBYdx[n] = -y[n] / lifeTime[n] + y[n-1] / lifeTime[n-1]
    return dyBYdx   

y0 = np.zeros(len(lifeSet)+1)
y0[0] = 1
x = np.linspace(0, 10, 201)
xLowerBound=0
xUpperBound = 10e17

soln = integrate.solve_ivp(fun, [xLowerBound, xUpperBound], y0, method='Radau') #Using Radau Method

plt.figure(figsize=(9,5))
for sol in range(len(soln.y)):
    plt.plot(soln.t, soln.y[sol], label = elementList[sol])
    
plt.xscale('log')
plt.xlabel('Time (Years)')
plt.ylabel('Substance Quantity')
plt.title('U238 Radioactive Decay - Log Scale')
plt.legend()
plt.savefig('SolutionsP3Q2_LogScale.png')
plt.show()



plt.figure(figsize=(8,8))
for sol in range(len(soln.y)):
    plt.loglog(soln.t, soln.y[sol], label=elementList[sol])
    
plt.xlabel('Time (Years)')
plt.ylabel('Substance Quantity')
plt.title('U238 Radioactive Decay')
plt.legend()
plt.savefig('SolutionsP3Q2.png')
plt.show()


#Th to U Ratio

plt.loglog(soln.t, soln.y[4]/soln.y[3], label = "Ratio of Th230 - U238", color = "red")
plt.title("Th230 - U238 Ratio")
plt.xlabel("Time (Years)")
plt.ylabel("Concentration")
plt.savefig("Th230U238Ratio.png")
plt.legend()
plt.show()


#Pb to U Ratio
soln = integrate.solve_ivp(fun, [xLowerBound, 10e8], y0, method='Radau') #Using Radau Method
plt.loglog(soln.t, soln.y[-1]/soln.y[0], label = "Ratio of Pb206 - U238", color = "blue")
plt.title("Pb206/U238 ratio")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.savefig("Pb206U238Ratio.png")
plt.legend()
plt.show()
