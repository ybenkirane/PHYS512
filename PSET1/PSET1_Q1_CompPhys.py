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


d1_Stretched_Optim = (func(x0*0.01 - 2*(0.0758)) - 8*func(x0*0.01 - (0.0758)) + 8*func(x0*0.01 + (0.0758)) - func(x0*0.01 + 2*(0.0758)))/(12*(0.0758))
d1_Norm_Optim = (func(x0 - 2*0.000758) - 8*func(x0 - (7.58 * 10**(-4))) + 8*func(x0 + (7.58 * 10**(-4))) - func(x0 + 2*(7.58 * 10**(-4))))/(12*(7.58 * 10**(-4)))

print('Error on Exp(0.01x) is ', np.abs(d1_Stretched_Optim - np.exp(x0)), 'at a delta of ', 0.0758)
print('Error on Exp(x) Exp is ', np.abs(d1_Norm_Optim - np.exp(x0)), 'at a delta of ', (7.58 * 10**(-4)))


fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.loglog(dx, np.abs(d1_Norm - np.exp(x0)), label = 'exp(x)', color = 'red')
ax2.loglog(dx, np.abs(d1_Stretched - np.exp(x0)), label = 'exp(0.01x)', color = 'green')

ax1.set_xlabel('Step Size')
ax1.set_ylabel('Exp(x) Error', color = 'red', fontsize = 14)
ax1.tick_params(axis="y", labelcolor='red')

ax2.set_ylabel('Exp(0.01x) Error', color = 'green', fontsize = 14)
ax2.tick_params(axis="y", labelcolor='green')


xNorm = [0.000758]
yNorm = [9.725553695716371* 10**(-14)]

xStretch = [(0.0758)]
yStretch = [1.7082327736073493]

ax1.plot(xNorm, yNorm, "o", color="blue", markersize=10, label='Estimate for Optimal Delta on Exp(x)')
ax2.plot(xStretch, yStretch, "o", color="black", markersize=10, label='Estimate for Optimal Delta on Exp(0.01x)')

plt.title('Error Estimation of the Exponential about x = 1')
fig.tight_layout()

ax1.grid(True)
ax1.legend(loc = 'lower center')
ax2.legend(loc='upper center')

plt.savefig('ErrorEst_Q1.png')
plt.show()

