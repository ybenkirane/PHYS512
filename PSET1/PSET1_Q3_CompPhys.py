# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:09:13 2022

@author: Yacine Benkirane
Solution to Q3 of PSET1 
Comp Physics at McGill University: PHYS 512
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sp

dat = np.loadtxt("lakeshore.txt")
Temperature = dat[:,0]
Voltage = dat[:,1]

#print(min(volt), max(volt))
plt.clf();
plt.plot(Voltage, Temperature, "b.", label = 'Data')

#Let's iterate through even and odd segments (Technique mentioend by Prof)
TempEven = Temperature[::2]
TempOdd = Temperature[1::2]
VoltEven = Voltage[::2]
VoltOdd = Voltage[1::2]

#Interpolate using Polynomial
def PlyInt(Volt,Temp,x):
    Ply = 0
    
    for m in range(0, len(Volt), 4):
        min = m
        max = m + 4
        
        if m == (len(Volt)-4): 
            max= m + 3
            
        if x <= Volt[min] and x >= Volt[max]:
            for i in range(min, max):
                
                xx = np.append(Volt[min:i],Volt[i + 1:max])
                y0 = Temp[i]
                x0 = Volt[i]
                num = 1
                den = np.prod((x0 - xx))
                
                for j in xx:
                   num=num*(x - j)
                Ply = Ply + (y0*num)/den
        
    return Ply



interpolVolt = np.linspace(Voltage[1],Voltage[-1],501)
polyPlot = np.zeros(len(interpolVolt)) #Polynomial Interpolation Graphing

for j in range(len(interpolVolt)): #Looping through Voltage Domain
   polyPlot[j]=PlyInt(Voltage, Temperature, interpolVolt[j])
plt.plot(interpolVolt, polyPlot)


#Requesting User for Input (Arbitrary) Voltage to be interpolated
userResp = float(input('Hello! Input a voltage between 0.090681 and 1.64429 Volts: \n'))

print('\n', "As per the Cubic Spline Fit, the Temperature is  " , sp.splev(userResp, sp.splrep(Voltage[::-1], Temperature[::-1])), "Kelvin.")
print("As per the Polynomial Fit, the Temperature is ", PlyInt(Voltage, Temperature, userResp),"Kelvin.")

#def lakeshore(V,data):

PlyEvenDomain = np.zeros(len(VoltEven)) 
for i in range(len(VoltEven)):
    PlyEvenDomain[i]=PlyInt(VoltOdd, TempOdd, VoltEven[i])
plt.plot(VoltEven, PlyEvenDomain, color='red')
PlyError = np.mean(np.abs(PlyEvenDomain- TempEven)) #Error in Polynomial Fit

splineOdd =sp.splrep(VoltOdd[::-1],TempOdd[::-1])
CubSplnError = np.mean(np.absolute(sp.splev(VoltEven, splineOdd)- TempEven)) #Cubic Spline Error

print("Polynomial Fit Error: ", PlyError)
print("Cubic Spline Fit Error: ", CubSplnError, '\n')


plt.plot(VoltOdd, sp.splev(VoltOdd, splineOdd), label = 'Cubic Spline Fit') #Plotting Interpolation
plt.ylabel("Temperature (˚C)")
plt.xlabel("Voltage (V)")
plt.title("Lakeshore 670 Diode Temp vs Volt Interpolation")
plt.grid()
plt.legend()
plt.savefig('lakeshoreDiagram.png')
plt.show()

