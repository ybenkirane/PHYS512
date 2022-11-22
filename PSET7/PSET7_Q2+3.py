# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:00:42 2022

@author: Yacine Benkirane
"""

import numpy as np
from matplotlib import pyplot as plt


def Gaussian(x):
    return np.exp(-x**2/2)

def PowLaw(x):
    return 1/(x+1)

def Lorentz(x):
    return 1/(1+x**2)

def CauchyCurve(x):
    return 1/(x**2 + 1)

#We must use a rejection method...
def CauchyRando(tail):

    rando = np.random.rand(2)*tail
    rando[0] = rando[0]

    if rando[1] < CauchyCurve(rando[0]) * np.pi/tail:
        return rando[0]
    else:
        return CauchyRando(tail)

def ExpRando(tail):
    
    randoCauch = np.array([CauchyRando(tail) for i in range(int(1e5))])
    randoNum = np.random.rand(int(1e5))*tail
    
    approved = randoNum < np.exp(-randoCauch) / CauchyCurve(randoCauch)
    
    return randoCauch[approved]


xDom = np.linspace(0, 10, 1001)

u = np.random.rand(int(1e5))
v = np.random.rand(int(1e5))*2/np.e

approved = u <= np.sqrt(np.exp(-v/u))

plt.plot(xDom , PowLaw(xDom), ls = '--', label = "Power Law")
plt.plot(xDom , Lorentz(xDom), ls = '--', label = 'Lorentzian')
plt.plot(xDom , Gaussian(xDom), ls = '--', label = 'Gaussian')
plt.plot(xDom, np.exp(-xDom), c= 'k', label ="Exponential, $\lambda = 1$")
plt.plot(xDom, xDom*0 + 1, label = "Uniform")
plt.plot(xDom, CauchyCurve(xDom), label = "Cauchy")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.title("Distributions")
plt.legend()
plt.savefig('PSET7_Q2_GraphComparison.png')
plt.show()

tail=5
dx = np.linspace(0, tail, 101)

CauchSamp = np.array([CauchyRando(tail) for i in range(int(1e5))])

print("Acceptance Rate:", len(CauchSamp)/int(1e5))

plt.hist(CauchSamp, dx, density = True, label = "Cauchy Bins")
plt.plot(dx, CauchyCurve(dx)/np.arctan(tail), label = "Cauchy Curve", color = 'red')
plt.plot(dx, np.exp(-dx)*(1  + np.sinh(tail) - np.cosh(tail)), ls = '--', label = "Exponential Curve", color = 'green')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Cauchy Sampling')
plt.grid()
plt.legend()
plt.savefig("PSET7_Q2_CauchyDistribution.png")
plt.show()


ExpSamp = ExpRando(tail)

print("Acceptance Rate:", len(ExpSamp)/int(1e5))

plt.hist(ExpSamp, dx, density = True, label = "Exponential Bins")
plt.plot(dx, np.exp(-dx)/(1  + np.sinh(tail) - np.cosh(tail)), label = "Exponential Curve", color = 'green')
plt.plot(dx, CauchyCurve(dx)/np.arctan(tail), ls = '--', label = "Cauchy Curve", color = 'red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exponential Sampling')
plt.grid()
plt.legend()
plt.savefig("PSET7_Q2_ExponentialDistribution.png")
plt.show()

pdfPoints = (v/u)[approved]

print("Acceptance Rate of PDF:", len(pdfPoints)/len(approved))

plt.hist(pdfPoints, bins = dx, density=True, label = "PDF Bins")
plt.plot(dx, np.exp(-dx)/(1  + np.sinh(tail) - np.cosh(tail)), label = "Exponential PDF", color = 'green')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('PDF Sampling')
plt.grid()
plt.legend()
plt.savefig("PSET7_Q3_PDF.png")
plt.show()