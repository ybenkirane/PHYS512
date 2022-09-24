# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:57:09 2022

@author: Yacine Benkirane
Code for PSET 2, Problem 3
"""

import numpy as np
import matplotlib.pyplot as plt

#Very tricky problem (for me at least), discussed extensively with classmates so solution / thought process may resemble in theory


################################################# LOG 2 ##################################################
def chebyLog2(a= 1/5 , b = 10, ord = 200, tol = 1e-8):
    x = np.linspace(a, b, 2001) #Domain
    y = np.log2(x) #Log Base 2
    cheby = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg=ord, domain = (a,b)) #Least Square
    cheby = cheby.trim(tol) #Trims all vals in cheby array below tol
    coeffCount = len(cheby)
    cheby_evl = cheby(x) #Polyn for every domain point from a to b
    print(cheby_evl, 'cheby Log 2')

    dataSet = np.vstack((x,cheby_evl)).T #vstack for full polyn evaluation (function)
    return dataSet, coeffCount, cheby 

data,coeffCount,cheby = chebyLog2()
logError = np.log2(data[:,0])-data[:,1]

#
#fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios': [4, 1]})
#fig.suptitle('Electric Field from Quad')
#ax[0].grid(True)
#ax[1].grid(True)
#ax[0].plot(domain, ElecField, 'tab:green')
#ax[1].set_xlabel('Distance (z)')
#ax[0].set_ylabel('E Field')
#ax[1].plot(domain, ElecErr, 'tab:red')
#ax[1].set_ylabel('Error')
#plt.savefig('QuadElectricField.png')
#




fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [4,1]}, sharex=True)

ax[0].plot(data[:,0],data[:,1], linewidth = 5, label = 'Chebyshev Log Base 2', color = 'blue')
ax[0].plot(data[:,0],np.log2(data[:,0]),'--', linewidth = 2, color = 'r', label = 'Numpy Log Base 2')
ax[0].legend(loc = 'lower right')
ax[0].set_ylabel('Series')

ax[1].set_xlabel('X')
ax[1].set_ylabel('Error')
ax[1].plot(data[:,0], logError, color = 'orange')
fig.suptitle('Log Base 2 Chebyshev')
plt.savefig('Log2Cheb.png')


#################################################### NATURAL LOG ######################################################

def log2Func(num, tol = 1e-8, ErrVal = False):
    mantis_num, exp_num = np.frexp(num)
    mantis_e, expon = np.frexp(np.e)
    print(mantis_num, exp_num, mantis_e, expon, 'log 2 cheb')
    cheby = chebyLog2(tol = tol)[2] 
    log2_num = cheby(mantis_num) + exp_num 
    log2_e = cheby(mantis_e) + expon
    natLog = log2_num / log2_e 
    logError = np.abs(np.log(num) - natLog)
    print(logError, 'log error')
    if ErrVal:
        return natLog, logError
    else:
        return natLog
   
    

#Test 
#ax[].set_xlabel('Step Size')
#ax[].set_ylabel('Exp(x) Error', color = 'red', fontsize = 14)
#ax[].tick_params(axis="y", labelcolor='red')
#
#ax[].set_ylabel('Exp(0.01x) Error', color = 'green', fontsize = 14)
#ax[].tick_params(axis="y", labelcolor='green')

x = np.linspace(1, 151, 2001)
lns = log2Func(x, ErrVal = True)

fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]}, sharex=True)
fig.suptitle('Natural Log Chebyshev')

ax[0].plot(x,lns[0],linewidth = 4, color='r', label = 'Chebyshev Natural Log')
ax[0].plot(x,np.log(x), linewidth = 2,linestyle='--', color = 'blue', label='Numpy Natural Log')
ax[0].legend(loc = 'lower right')
ax[0].set_ylabel('Series')

ax[1].set_xlabel('X')
ax[1].set_ylabel('Error')
ax[1].plot(x,lns[1], color = 'orange')

plt.savefig('NatLogCheb.png')