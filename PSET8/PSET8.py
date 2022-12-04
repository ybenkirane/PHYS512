"""
Created on Sun Nov 25 19:55:42 2022

@author: Yacine Benkirane

"""

import numpy as np
import matplotlib.pyplot as plt

n = 300
x = np.arange(n)

def avg(v):
    return 0.25 * (np.roll(v,1,0) + np.roll(v,-1,0) + np.roll(v,1,1) + np.roll(v,-1,1))

def FFT_V(rho1, k_ft1, msk1, out = True):
    
    PotMat = np.zeros(msk1.shape)
    PotShape = PotMat.shape
    PotMat[msk1] = rho1

    PotMat = np.pad(PotMat, (0, PotMat.shape[0]), 'constant')
    PotMat_ft = np.fft.rfftn(PotMat)
    V = np.fft.irfftn(PotMat_ft * k_ft1)
    V = V[:PotShape[0],:PotShape[1]]
    if out:    
        return V[msk1]
    else:   
        return V
    
def compGrad(Voltmsk, rho1, msk1, k_ft1, n_iter = 50):
    
    VPrediction = FFT_V(rho1, k_ft1, msk1)
    r = Voltmsk - VPrediction
    p = r.copy()
    x = rho1.copy()
    r_sq = np.sum(r*r)
    
    for i in range(n_iter):
        
        A = FFT_V(p, k_ft1, msk1)
        a = np.sum(r*r)/np.sum(A*p)
        x = x + a*p
        r = r - a*A
        new_r_sq = np.sum(r*r)
        b = new_r_sq/r_sq
        p = r + b*p
        r_sq = new_r_sq
    
    return x


x[n//2:] = x[n//2:] - n
xx, yy = np.meshgrid(x, x)
r = np.sqrt(xx**2 + yy**2)
r[0,0] = 1 

V = np.log(r)/(2*np.pi)
V[0,0] = 0.25 * (V[1,0] - V[-1,0] + V[0,1] + V[0,-1])
V = V - V[n//2, n//2]
V[0,0] = 4*V[1,0] - V[2,0] - V[1,1] - V[1,-1]

rhoSet = V - avg(V)
V = V/rhoSet[0,0]
V = V - V[0,0] + 1
rho = V - avg(V)

#print("V at [0,0] is {}".format(V[0,0]))
#print("V at [1,0] is {}".format(V[1,0]))
#print("V at [2,0] is {}".format(V[2,0]))
#print("V at [5,0] is {}".format(V[5,0]))


plt.figure(1)
plt.imshow(np.roll(rho, rho.shape[0]//2, axis = (0,1))[n//2 - 5: n//2 + 5, n//2 - 5: n//2 + 5], cmap = "flag")
plt.title("Green's Function Charge Density")
plt.colorbar()
plt.savefig('PSET8_GreenFunctionChargeDensity.png')
plt.show()

plt.figure(2)
plt.imshow(np.roll(V, V.shape[0]//2, axis = (0,1)), cmap = "gist_rainbow")
plt.title("Green's Function Potential")
plt.colorbar()
plt.savefig('PSET8_GreenFunctionPotential.png')
plt.show()

m = n//2
bc = np.zeros([m,m])
msk = np.zeros([m,m], dtype = "bool")
bc[0,:] = 0
bc[-1,:] = 0
bc[:,0] = 0
bc[:,-1] = 0
msk[0,:] = True
msk[-1,:] = True
msk[:,0] = True
msk[:,-1] = True
bc[3*m//8:5*m//8, 3*m//8:5*m//8] = 1
msk[3*m//8:5*m//8, 3*m//8:5*m//8] = True

k = V.copy()
k_ft = np.fft.rfft2(k)

Voltmsk = bc[msk]
rho_i = 0*Voltmsk
rho_f = compGrad(Voltmsk, rho_i, msk, k_ft)
PotMat = np.zeros(msk.shape)
PotShape = PotMat.shape
PotMat[msk] = rho_f

plt.figure(1)
plt.imshow(PotMat, cmap = "gist_rainbow")
plt.title("Charge Density")
plt.colorbar()
plt.savefig('PSET8_ChargeDensity.png')
plt.show()

PotSide = PotMat[:,5*m//8-1]
plt.figure(3)
plt.plot(PotSide)
plt.ylabel("Charge Density", c = "k")
plt.title("Charge Density on One Side of Box")
plt.savefig('PSET8_ChargeDensitySideProfile.png')
plt.show()

potValu = FFT_V(rho_f, k_ft, msk, out = False)
plt.figure(1)
plt.imshow(potValu, cmap = "inferno")
plt.title("Potential on the Box")
plt.colorbar()
plt.savefig('PSET8_BoxPotential.png')
plt.show()

V_inside = potValu[bc == 1]
print("Mean Potential = {} with STD = {}".format(np.mean(V_inside), np.std(V_inside)))

dX, dY = np.gradient(potValu)
Ey = -dY
Ex = -dX
x = np.arange(potValu.shape[0])
X, Y = np.meshgrid(x,x)

plt.figure(1)
plt.quiver(X, Y, Ex, Ey)
plt.title("Electric Field on Box")
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('PSET8_Efield.png')
plt.show()