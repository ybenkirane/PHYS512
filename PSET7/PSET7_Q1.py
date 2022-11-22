# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:12:47 2022

@author: Yacine Benkirane
"""

import numpy as np
from matplotlib import pyplot as plt
#from mpl_toolkits import mplot3d


data = np.loadtxt('rand_points.txt')

x = data[:,0]
y = data[:,1]
z = data[:,2]

size = len(x)
limit = int(1e8 + 1)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, marker='.', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('C Random Points')
plt.savefig('PSET7_Q1_CRandomData.png')
ax.view_init(elev=55, azim=32 )
plt.show()

#Testing different iterations to find valuable a and b candidates

#while True:
#    a = np.random.rand()*5 - 2.5
#    b = np.random.rand()*5 - 2.5
#    print(a, b)
#    
#    xCross = a*x + b*y
#    yCross = z 
#
#    plt.scatter(xCross, yCross, s = 1)
#    plt.show()


a = 2.1318438704144382
b = -0.9297303705485978
xCross = a*x + b*y
yCross = z 

plt.scatter(xCross, yCross, s = 1)
plt.xlabel('a*x + b*y')
plt.ylabel('z')
plt.title('Cross Section')
plt.savefig('PSET7_Q1_CrossSection.png')
plt.show()
    






py_rand = np.random.randint(0, limit, size=(size, 3))

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(py_rand[:,0],py_rand[:,1],py_rand[:,2],color = 'blue', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Numpy Random Points')
plt.savefig('PSET7_Q1_NumpyRandom.png')
plt.show()

#Couldn't find any correlating cross-sections for numpy.random !