# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 16:13:37 2022

@author: Yacine Benkirane
Collaboration with countless students in PHYS 512 
    Primarly Jeff, Guillaume, Christien, Liam, Simon, and David
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

# Loading data
data = np.load('mcmc/sidebands.npz')
t = data['time']
d = data['signal']

plt.plot(t ,d, color='red')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (sec)', fontsize=15)
plt.ylabel('Signal Intensity', fontsize=15)
plt.title('Raw Data', fontsize=15)
plt.grid()
plt.show()

def lorentz(p, t): 
    a, t0 , w = p
    return a / (1 + ((t - t0) / w)**2)

def cacul_lorentz(p,t):
    a, t0, w = p
    y = lorentz(p, t)
    gradient = np.zeros([len(t), len(p)])
    gradient[:,0] = 1.0 / (1 + (t-t0)**2 / w**2)
    gradient[:,1] = (2 * (t-t0) / (t**2 - 2 * t0 * t + t0**2 + w**2) ) * y
    gradient[:,2] = (2 / w - 2 * w / (t**2 - 2 * t0 * t + t0**2 + w**2)) * y 
    return y , gradient

def cacul_lortenzN(p, t):
    a, t0, w = p
    y = lorentz(p, t)
    gradient = np.zeros([len(t), len(p)])
    
    llor = lambda p: lorentz(p, t)
    gradient = Ndf(llor, p).transpose()
    
    return y , gradient

def Ndf(f, x):
    diffs = []
    eps = 1e-16 # Precision of the Machine
    dx = np.sqrt(eps)
    for i in range(len(x)):
        iter1 = np.zeros(len(x))
        iter1[i] += 1
        
        m1 = x.copy()
        m2  = x.copy()
        p1  = x.copy()
        p2 = x.copy()
        
        m2 -= 2 *dx * iter1
        m1  -= dx * iter1
        p1  += dx * iter1
        p2 += 2*dx * iter1
        diffs.append((f(m2) + 8 * f(p1) - 8 * f(m1) - f(p2))/(12 * dx))
    return np.array(diffs)

def ThreeLorentz(p, t):
    a, b, c, t0, w, dt = p
    return a/(1 + (t-t0)**2 / w**2) + b/(1 + (t-t0+dt)**2 / w**2) + c/(1 + (t-t0-dt)**2 / w**2)

def cacul_ThreeLorentz(p, t):
    y = ThreeLorentz(p, t)
    gradient = np.zeros([len(t), len(p)])
    TriLorentz = lambda p: ThreeLorentz(p, t)
    gradient = Ndf(TriLorentz, p).transpose()
    return y , gradient

steps = 20

# Initial guess for parameters
a = 1.4
w = 0.0001
t0 = 0.0002

p0 = np.array([a, t0, w]) 
p = p0.copy()
Np = p0.copy()

for i in range(steps):
    pred, gradient = cacul_lorentz(p,t)
    r = d - pred
    r = np.matrix(r).transpose()
    gradient = np.matrix(gradient)

    lhs = gradient.transpose() @ gradient
    rhs = gradient.transpose() @ r
    curv_mat = np.linalg.inv(lhs)
    dp = curv_mat@(rhs)
    for j in range(len(p)):
        p[j] = p[j] + dp[j]

plt.plot(t, d, 'k--', label = 'Data', color = 'blue')
plt.plot(t, pred, 'r-', label = 'Newtonian Fit', color = 'red')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (sec)', fontsize=15)
plt.ylabel('Signal Intensity', fontsize=15)
plt.title('Newton\'s Method', fontsize=15)
plt.grid()
plt.legend()
plt.savefig("NewtonMethod.png")
plt.show()


print('parameters: ', p)
print('a = ', p[0] , '     t0 = ', p[1], '      w = ', p[2])

rms_err = np.sqrt(np.sum((d-pred)**2)/len(d))
p_err = np.sqrt(np.diag(curv_mat * rms_err**2))
print('Noise: ', p_err, 'a = ', p[0] , ' +-', p_err[0], 't0 = ', p[1], ' +-', p_err[1], 'w = ', p[2], ' +-', p_err[2],)

for i in range(steps):
    pred, gradient = cacul_lortenzN(Np,t)
    r = d - pred
    r = np.matrix(r).transpose()
    gradient = np.matrix(gradient) 

    lhs = gradient.transpose() @ gradient
    rhs = gradient.transpose() @ r
    curv_mat = np.linalg.inv(lhs)
    dp = curv_mat@(rhs)
    for j in range(len(Np)):
        Np[j] = Np[j] + dp[j]

plt.plot(t, d, label = 'Data')
plt.plot(t, pred, 'r-', label = 'Newtonian Fit')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (sec)', fontsize=15)
plt.ylabel('Signal Intensity', fontsize=15)
plt.title('Numerical Deriv Newton\'s Method', fontsize=15)
plt.grid()
plt.legend()
plt.savefig("NumericalNewtonMethod.png")
plt.show()

print('parameters: ', p)
print('a = ', p[0] , '     t0 = ', p[1], '      w = ', p[2])


print("Equiv Diff Fit:".rjust(30),p, "N diff Fit:".rjust(30), Np)
print("Uncertainty):".rjust(30),p_err,"\n\n", "Difference: ".rjust(30), p-Np)

steps = 20

#Initial Guess for Parameters
a = 1.4
b = 0.1
c = 0.1
w = 0.000015
t0 = 0.000192
dx = 5e-5

p0 = np.array([a, b, c, t0, w, dx])
p = p0.copy()

for i in range(steps):
    pred, gradient = cacul_ThreeLorentz(p,t)
    r = d - pred
    r = np.matrix(r).transpose()
    gradient = np.matrix(gradient)

    lhs = gradient.transpose() @ gradient
    rhs = gradient.transpose() @ r
    curv_mat = np.linalg.inv(lhs)
    dp = curv_mat@(rhs)
    for j in range(len(p)):
        p[j] += dp[j]
    
plt.plot(t, d, label = 'Data')
plt.plot(t, cacul_ThreeLorentz(p0, t)[0], 'g--', label = "Guess", zorder = -1)
plt.plot(t, pred, 'r-', label = 'Newton\'s Fit')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (sec)', fontsize=15)
plt.ylabel('Signal Intensity', fontsize=15)
plt.title('Newton\'s Method with Three Lorentzians', fontsize=15)
plt.grid()    
plt.legend()
plt.savefig("NewtonMethodThreeLor.png")
plt.show()

rms_err = np.sqrt(np.sum((d-pred)**2)/len(d))
p_err = np.sqrt(np.diag(curv_mat * rms_err**2))

print('Best Parameter Fits: \n'.ljust(30), p, "\n Error RMS:".ljust(30), rms_err,"\n", 'Unc: \n'.ljust(30), p_err)

plt.plot(t, d-pred, 'r.', ms = 1)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel('Residuals')
plt.xlabel('Time (t)')
plt.title('Residual', fontsize=15)
plt.grid()
plt.savefig("Residuals.png")
plt.show()


for i in range(5):
    p_rand = p + np.linalg.cholesky(curv_mat * rms_err**2)@np.random.randn(len(p))
    p_rand = p_rand.tolist()[0]
    plt.plot(t, cacul_ThreeLorentz(p_rand, t)[0] - cacul_ThreeLorentz(p, t)[0], lw = 1, label = f"{i+1}")
    plt.legend()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Time (t)', fontsize=15)
    plt.ylabel('Residuals')
    plt.title('Alternate Fit Residuals', fontsize=15)
    plt.grid()
plt.savefig("AlternateFitRes.png")
plt.show()    
    
stepsizes = np.sqrt(np.diag(curv_mat * rms_err**2))
print(stepsizes)

def chi2(params):
    ChiSqrd = np.sqrt(np.sum((ThreeLorentz(params, t) - d)**2))/rms_err
    return ChiSqrd

steps = 25000
scaling = 4


pp = [1.44299, 1.039108e-01, 6.4733e-02, 1.9258e-04, 1.6065e-05, 4.4567e-05] #NLS Fitting Data from Above
ChiSqrd = chi2(pp)

chain = np.zeros([steps, len(pp)])
chi2L = np.zeros(steps)

for s in np.arange(steps):
    p_new = pp + np.random.randn(6) * stepsizes * scaling
    Chi2Nov = chi2(p_new)    
    del_chi = Chi2Nov - ChiSqrd    
    take = None   
    if del_chi >= 0 :
        if np.random.rand() < np.exp(- (1/2)*del_chi):
            take = True
        else:
            take = False         
    if del_chi < 0:
        take = True  
    if take == True:
        ChiSqrd = Chi2Nov      
        pp = p_new            
    chi2L[s] = ChiSqrd
    chain[s, :] = pp  
    
fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()
param_NameList = ["a","b","c", "t_0", "w", "dt"]

for i in np.arange(6):
    axes[i].plot(chain[:, i], color = "red")
    axes[i].set_xlabel('Steps')
    axes[i].set_ylabel(f"${param_NameList[i]}$ Param.")
    axes[i].set_title(f"${param_NameList[i]}$ MCMC")
    axes[i].grid()
plt.savefig("MCMCParams.png")
plt.show()    

fig2 = plt.figure(figsize = (15,5),constrained_layout = True)
axes2 = fig2.subplots(2,3).flatten()

param_NameList = ["a","b","c", "t_0", "w", "dt"]

#multidimensional cross-section
fig = corner.corner(chain, labels = param_NameList,label_kwargs=dict(fontsize=12),quantiles = [0.25,.50,.75],show_titles=True,title_fmt='g',title_kwargs = dict(fontsize = 7))

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=5)
fig.set_size_inches((10, 10))



for i in np.arange(6):
    freq = np.fft.rfftfreq(chain[:,i].shape[-1])
    period = 2*np.pi/freq
    
    axes2[i].loglog(period,np.abs(np.fft.rfft(chain[:,i])))
    axes2[i].set_xlabel('Period Domain')
    axes2[i].set_ylabel('Spectral Intensity')
    axes2[i].grid()
    axes2[i].set_title(f"${param_NameList[i]}$ : MCMC")
plt.savefig("MCMCRoutes.png")
plt.show()

#_______________________________________________________________________________________

plt.plot(chi2L, color = "red")
plt.ylabel("$\chi^2$")
plt.xlabel("steps")
plt.grid()
plt.title("Chi Squared vs Steps")
plt.savefig("Chi2Steps.png")
plt.show()
#_______________________________________________________________________________________
freq = np.fft.rfftfreq(chi2L.shape[-1])
plt.plot(freq,np.fft.rfft(chi2L), color = "red")
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.title("Power Spectrum")
plt.savefig("PowerSpectrum.png")
plt.show()
#_______________________________________________________________________________________


print("Fit using MCMC:".rjust(30), chain[np.argmin(chi2L)])
print("Uncertainty using MCMC:".rjust(30),np.std(chain,axis = 0),"\n\n")
print("Minimum Chi2 Value Found at:".rjust(30), np.min(chi2L),"\n\n")


#From the MCMC Fit: w = 1.60*10e-5 and dt = 4.437*10e-5 and e_err = 1.418*10e-7
w = 1.60*10e-5
dt = 4.437*10e-5
w_err = 1.418*10e-7
width = dt/w*9
err_width = width*(w_err/w)
print(width, err_width)