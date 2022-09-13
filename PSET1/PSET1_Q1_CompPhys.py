# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:17:16 2022

@author: Yacine Benkirane
Solution to Q1b of PSET1 
Comp Physics at McGill University: PHYS 512
"""

import numpy as np
from matplotlib import pyplot as plt

logdx = np.linspace(-15, -1, 1001)
dx = 10**logdx

func = np.exp
x0 = 1

y00 = func(x0)
y01 = func(x0 + dx)
y02 = func(x0 - dx)
y03 = func(x0 + 2*dx)
y04 = func(x0 - 2*dx)

y10 = func(x0*0.01)
y11 = func(x0*0.01 + dx)
y12 = func(x0*0.01 - dx)
y13 = func(x0*0.01 + 2*dx)
y14 = func(x0*0.01 - 2*dx)

d1_Norm = (y04 - 8*y02 + 8*y01 - y03)/(12*dx)
d1_Stretched = (y14 - 8*y12 + 8*y11 - y13)/(12*dx)




fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.loglog(dx, np.abs(d1_Norm - np.exp(x0)), label = 'exp(x)', color = 'red')
ax2.loglog(dx, np.abs(d1_Stretched - np.exp(x0)), label = 'exp(0.01x)', color = 'green')

ax1.set_xlabel('Step Size')
ax1.set_ylabel('Exp(x) Error', color = 'red', fontsize = 14)
ax1.tick_params(axis="y", labelcolor='red')

ax2.set_ylabel('Exp(0.01x) Error', color = 'green', fontsize = 14)
ax2.tick_params(axis="y", labelcolor='green')

plt.title('Error Estimation of the Exponential about x = 1')
fig.tight_layout()

plt.savefig('ErrorEst_Q1.png')
plt.show()

