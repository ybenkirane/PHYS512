# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:19:12 2022

@author: Yacine Benkirane
"""


import numpy as np 
from matplotlib import pyplot as plt
#from mpl_toolkits import mplot3d

#Let's do a LeastSquars Fit to dish_zenith.txt : photgrammetry data for a prototype telescope dish...


data = np.loadtxt("dish_zenith.txt").T # Importing raw data
def paraboloid(x, y, fit):
    return fit[0]*(x**2 + y**2) + fit[1]*x + fit[2]*y + fit[3]

fns = np.array(
    [
        lambda x : x[0]**2 + x[1]**2 ,
        lambda x : x[0] ,
        lambda x : x[1] ,
        lambda x : 1 ,
    ]
)

#Data Provided by Prof

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[0], data[1], data[2], color = 'red')
ax.set_title("Photogrammetry Data for Prototype Telescope Dish")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('DataDish.png')
plt.show()




xSet = data[0]
ySet = data[1]
zSet = data[2] 
xyVector = np.array([xSet, ySet]).T

#Array with analytic lambda functions to form matrix


matrixA = np.zeros([len(xSet), len(fns)])
for n in range(len(fns)):
    matrixA[:, n] = np.array([fns[n](m) for m in xyVector])

lhs = matrixA.T@matrixA
rhs = matrixA.T@zSet
fitp = np.linalg.inv(lhs)@rhs




# Paraboloid Surface and Raw Data
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(min(xSet), max(xSet), 201)
y = np.linspace(min(ySet), max(ySet), 201)

X, Y = np.meshgrid(x, y)

zGeo = np.array(paraboloid(np.ravel(X), np.ravel(Y), fitp))
Z = zGeo.reshape(X.shape)

ax.plot_surface(X, Y, Z)
ax.scatter(data[0], data[1], data[2], c = 'purple')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Paraboloid Surface on Dish Data')
plt.savefig('SurfaceOnDish.png')
plt.show()


#Bit Confused as to how to carry out noise operations... 