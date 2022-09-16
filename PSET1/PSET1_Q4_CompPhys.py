"""
Created on Wed Sep 14 11:15:28 2022

@author: Yacine Benkirane
Solution to Q3 of PSET1 
Comp Physics at McGill University: PHYS 512
"""


import numpy as np
from numpy.polynomial import polynomial
from scipy.interpolate import splev, splrep
from matplotlib import pyplot as plt



def Lorentzian(X):
    return 1/(1+X**2)

#Evaluating the Rational Expression
def RationalEval(p, q, x): #Directly from Prof Sievers  lines 20 to 29
    num=0
    for i in range(len(p)):
        num = num + p[i] * (x**i)
    den = 1
    
    for i in range(len(q)):
        den = den + q[i] * (x**(i+1))
    return num/den

#Fitting the Rational Model
def RationalFit(x,y,n,m):   #Directly from Prof Sievers lines 31 to 48
    
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    conc=np.zeros([n+m-1,n+m-1])
    
    for i in range(n):
        conc[:,i]=x**i
        
    for i in range(1,m):
        conc[:,i-1+n]= -y * (x**i)
    
    print(conc)
    
    prod = np.dot( np.linalg.inv(conc), y)
    return prod[:n], prod[n:]

#pinv... Lecture Notes Inspired!
def rat_fit_pinv_version(x, y, n, m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    conc = np.zeros([n+m-1,n+m-1])
    for i in range(n):
        conc[:,i]=x**i
    for i in range(1,m):
        conc[:, i-1+n]=-y*x**i
    
    print(conc)
    prod = np.dot(np.linalg.pinv(conc), y)
    return prod[:n], prod[n:]


 
def RoutineOutput(func, a, b): 
    Ptns = 10
    X = np.linspace(a,b, Ptns)
    Y = func(X)
    
    #Fitting Polynomials
    PolyfitVals = polynomial.polyfit(X,Y,11)
    PolyFunc = polynomial.Polynomial(PolyfitVals)
    spl = splrep(X,Y)

    #Such order was recommended during conversation
    ratp, ratq = RationalFit(X,Y,6,5)
    ratp2, ratq2 = rat_fit_pinv_version(X,Y,6,5)
    
    RationalFitPoints = RationalEval(ratp, ratq, np.linspace(a, b, 10))
    RatPinvFitPoints = RationalEval(ratp2,ratq2,np.linspace(a, b, 10))
    realPoints = np.cos(np.linspace(a, b, 10))
    PolynomialFitPoints = PolyFunc(np.linspace(a, b, 10))
    CubicSplinePoints = splev(np.linspace(a, b, 10),spl)
   
   
    print("Rational Fit Summed Error:", np.sum(np.abs(RationalFitPoints - realPoints)))
    print("Rational (pinv) Summed Error:", np.sum(np.abs( RatPinvFitPoints - realPoints )))
    
    print("Polynomial Fit Summed Error:", np.sum(np.abs( PolynomialFitPoints - realPoints )))
    print("Cubic Spline Summed Error:", np.sum(np.abs( CubicSplinePoints - realPoints )))
    plt.plot(np.linspace(a, b, 10), RationalFitPoints, 'bs', label = "Rational Fit")
    plt.plot(np.linspace(a, b, 10), RatPinvFitPoints, 'bh', label = "Rational pinv Fit")
    plt.plot(np.linspace(a, b, 10), realPoints, 'g.', label = "Real")
    plt.plot(np.linspace(a, b, 10), PolynomialFitPoints, 'rp', label = "Polynomial FIt " )
    plt.plot(np.linspace(a, b, 10), CubicSplinePoints, 'r.', label = "Cub Spline Fit")
    plt.legend()
    plt.xlabel("Domain")
    plt.ylabel("Range")
    plt.title('Comparison of Various Interpolation Techniques on Cos(x)') 
#    plt.savefig('CosineInterpolation.png')

#print("\n Lorentzian Fits:")
#RoutineOutput(Lorentzian, -1, 1)
#plt.show()

print("Cos(x) Fits")
RoutineOutput(np.cos, -np.pi/2, np.pi/2)
#plt.show()
