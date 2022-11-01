# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:44:26 2022

@author: Yacine Benkirane
#Worked with David, Simon, Louis, Mathieu, Chris, and countless others... 
"""


import camb
import numpy as np
import matplotlib.pylab as plt
import time
from scipy.stats import chi2
import corner


 
 
data = np.loadtxt("mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt")
print(data.shape)


 # Values as discussed with other classmates, new params for power spectrum
pars=np.asarray([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
plnck=np.loadtxt('mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
scl=plnck[:,0] 
spec=plnck[:,1]
errors=(1/2)*(plnck[:,2]+plnck[:,3]);

def calculateChiVal(y):
    return np.sum((y - spec)**2/errors**2)

def get_spectrum(pars,lmax=3000):
    
    H0 = pars[0]
    ombh2 = pars[1]
    omch2 = pars[2]
    tau = pars[3]
    As = pars[4]
    ns = pars[5]
    
    #CAMB Library (took a while to understand... with help from classmates)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    
    results=camb.get_results(pars)
    
    powers = results.get_cmb_power_spectra(pars,CMB_unit='muK')
    
    cmb = powers['total']
    tt = cmb[:,0]
   
    return tt[2:]

#Taken from Previous PSET
def Ndf(f, x, dxArr):
    diffs = []
    for i in range(len(x)):
        iter1 = np.zeros(len(x))
        iter1[i] += 1
        dx = dxArr[i]

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


def calculateSpectrum(params, lmax = 3000):
   
    f = lambda params: get_spectrum(params, lmax)
    y = f(params)
    gradient = np.zeros([lmax, len(params)])
    gradient = Ndf(f, params, dx).transpose()
    return y, gradient

#Very Confused about this, helped by classmates
def amend(dampFac, succ):
    if succ:
        dampFac *= .3
    if dampFac <= 0.1:
        dampFac = 0
    else:
        if dampFac == 0:
            dampFac = 1
        else:
            dampFac *= 2
    return dampFac

model = get_spectrum(pars)
model = model[:len(spec)]

resid = spec - model
chisq = np.sum((resid/errors)**2)

print("Chi Squared ",chisq," with ",len(resid)-len(pars)," DOF")

plnck_binned=np.loadtxt('mcmc/COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)

errors_binned=0.5*(plnck_binned[:,2]+plnck_binned[:,3]);
chi2.sf(chisq, len(resid)-len(pars))

pars = np.array([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec - model
chisq2=np.sum( (resid/errors)**2)
dof = len(resid) - len(pars)

print("Param Set 1:", chisq, '\n Probability:' , chi2.sf(chisq, dof),"\n")
print("Param Set 1:", chisq2, '\n Probability:' , chi2.sf(chisq2, dof))

#np.save('obj/fit_p.npy', pars)
info = [10,1e9,True]
#np.save('obj/info.npy', info)

dx = pars *1e-3
stepCount = 10
p0 = np.load('obj/fit_p.npy')
p = np.array(p0.copy())
invN = np.diag(1/errors**2)
dampFac, chisq, succ = np.load('obj/info.npy')

#Got Help from David for this step
for i in range(stepCount):
    
    print("Damping Factor: ".rjust(30), dampFac)
    print("Parameter Values: ".rjust(30), p)
    
    pred, gradient = calculateSpectrum(p)
    pred = pred[:len(spec)]
    gradient = np.matrix(gradient)[:len(spec),:]
    
    r = spec - pred
    r = np.matrix(r).transpose()
    
    lhs = gradient.transpose() @ invN @ gradient
    curv_mat = np.linalg.inv(lhs)
    
    lhs += dampFac * np.diag(np.diag(gradient.transpose() @ invN @ gradient))
    rhs = gradient.transpose() @ invN @ r
    
    dp = np.linalg.inv(lhs)@(rhs)
    newPValu = p.copy()
    
    for j in range(len(p)):
        newPValu[j] = p[j] + dp[j]
        
    chiSqNew = calculateChiVal(get_spectrum(newPValu)[:len(spec)])
    
    if chiSqNew < chisq:
        
        succ = True
        chisq = chiSqNew
        
        p = newPValu
        dampFac = amend(dampFac, succ)
        
    else:
        succ = False
        dampFac = amend(dampFac, succ)
        
#print("Done For Loop 1")

#    np.save('obj/fit_p.npy', p)
#    np.save('obj/fit_curvmat.npy', curv_mat)
#    np.save('obj/info.npy', [dampFac, chisq, succ])

p = np.load('obj/fit_p.npy')
curv_mat = np.load('obj/fit_curvmat.npy')

paramNames = ['Hubble', "Baryon", "CDM", "Optical Depth", "Primordial Amp.", "Primordial Tilt"]
paramError = np.sqrt(np.diag(curv_mat))
for i, n in enumerate(paramNames):
    print(f"{n} = ".ljust(20), f"{p[i]:.2E}", "+/-", f"{paramError[i]:.2E}")
    
fit_string = []
paramNames = ['Hubble', "Baryon", "CDM", "Optical Depth", "Primordial Amp.", "Primordial Tilt"]
paramError = np.sqrt(np.diag(curv_mat))

for i, n in enumerate(paramNames):
    fit_string.append(f"{n} = ".ljust(20)+f"{p[i]:.2E}"+ " +/- "+ f"{paramError[i]:.2E}\n")

#np.savetxt('planck_fit_params.txt', fit_string, delimiter="\n", fmt="%s")

ppChain = np.load('obj/chain.npy')[-1]
stepCount = 3000
scaleFactor = .75



chisqr = np.load('obj/chiChain.npy')[-1]

chain = np.zeros([stepCount, len(ppChain)])
chiChain = np.zeros(stepCount)

for s in np.arange(stepCount):

    newPValu = ppChain + np.random.multivariate_normal(mean = np.zeros(len(curv_mat)), cov = curv_mat) * scaleFactor
    chi2New = calculateChiVal(get_spectrum(newPValu)[:len(spec)])
    del_chi = chi2New - chisqr
    take = None
    if del_chi >= 0 :
        if np.random.rand() < np.exp(- 0.5 * del_chi):
            11
            take = True
        else:
            take = False
    else:
        take = True
        
    if take == True:
        ppChain = newPValu
        chisqr = chi2New
        
    chiChain[s] = chisqr
    chain[s, :] = ppChain
        
#print("Done For Loop 2")

chiTestChain = np.hstack([np.load('obj/chiChain.npy'), chiChain])
test_chain = np.vstack([np.load('obj/chain.npy'), chain])
#np.save('obj/chain.npy', chain)
#np.save('obj/chiChain.npy', chiTestChain)
chain = np.load('obj/chain.npy')[:9000]
chiChain = np.load('obj/chiChain.npy')

plt.plot(chiChain)
plt.ylabel("$\chi^2$")
plt.xlabel("steps")
plt.show()

plt.plot(chiChain[250:])
plt.ylabel("$\chi^2$")
plt.xlabel("steps")
plt.show()

plt.loglog(np.abs(np.fft.rfft(chiChain[300:])))
plt.ylabel("$\chi^2$")
plt.xlabel("steps")

fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()
for i in np.arange(6):
    axes[i].plot(chain[300:, i])
    axes[i].set_xlabel('Steps')
    axes[i].set_title(f"{paramNames[i]}")



fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()
for i in np.arange(6):
    axes[i].loglog(np.abs(np.fft.rfft(chain[300:, i])))
    axes[i].set_xlabel('freq')
    axes[i].set_title(f"{paramNames[i]}")

#corner plots like in PSET4
plt.ioff()
corner.corner(chain, labels=paramNames, quantiles=[0.18, 0.4, 0.75], show_titles=True, title_fmt = '.1E')
#print("Corner Loop 1 Compiled")


mp = np.mean(chain, axis = 0)
mp_error = np.std(chain,axis = 0)

#not a pro in Astrophysics so don't fully comprehend the derivation for these
h = mp[0]/100
eh = mp_error[0]/100
Ob = mp[1]/h**2 #WOW
eOb = np.sqrt(1/h**4 *mp_error[1]**2 + 4*Ob**2/h**6 * eh**2)

Oc = mp[2]/h**2 #Wild
eOc = np.sqrt(1/h**4 *mp_error[2]**2 + 4*Oc**2/h**6 * eh**2)

Ol = 1 - Ob - Oc #Shocking
eOl = np.sqrt(eOc**2 + eOb**2)

print("Dark energy = ", f"{Ol:.2f}", "+/-", f"{eOl:.2f}")

tau = .0540
tauError = .0074


w = 1/np.sqrt(((chain[:,3] - tau)**2/tauError**2))
ip = np.average(chain, axis = 0, weights = w)

prevCov = np.cov(chain.T, aweights = w)
ip_error = np.sqrt(np.diag(prevCov))

paramNames = ['Hubble', "Baryon", "CDM", "Optical Depth", "Primordial Amp.", "Primordial Tilt"]


print("Parameter Values from Importance Sampling")
for i, n in enumerate(paramNames):
    print(f"\t{n} = ".ljust(20), f"{ip[i]:.2E}", "+/-", f"{ip_error[i]:.2E}")
    
    
paramNames = ['Hubble', "Baryon", "CDM", "Opt Depth Tau", "Amp.", "Tilt"]
print("\n Fit:")


for i, n in enumerate(paramNames):
    print(f"\t{n} = ".ljust(20), f"{p[i]:.2E}", "+/-", f"{paramError[i]:.2E}")
    
ppChain = pars.copy()
# ppChain = np.load('obj/tau_chain.npy')[-1]
stepCount = 10000
scaleFactor = .75

chisqr = calculateChiVal(get_spectrum(ppChain)[:len(spec)])
# chisqr = np.load('obj/tau_chiChain.npy')[-1]



#Next Run
chain = np.zeros([stepCount, len(ppChain)])
chiChain = np.zeros(stepCount)

for s in np.arange(stepCount):
    
    newPValu = ppChain + np.random.multivariate_normal(mean = np.zeros(len(curv_mat)), cov = prevCov) * scaleFactor
   
    chi2New = calculateChiVal(get_spectrum(newPValu)[:len(spec)])
    del_chi = chi2New - chisqr + (newPValu[3] - tau)**2/tauError**2
    
    take = None
    if del_chi >= 0 :
        if np.random.rand() < np.exp(- 0.5 * del_chi):
            take = True
        else:
            take = False
    else:
        take = True
        
    if take == True:
        ppChain = newPValu
        chisqr = chi2New
    chiChain[s] = chisqr
    chain[s, :] = ppChain
        
#np.save('obj/tau_chain.npy', chain)
#np.save('obj/tau_chiChain.npy', chiChain)

chain = np.load('obj/tau_chain.npy')
chiChain = np.load('obj/tau_chiChain.npy')

plt.plot(chiChain[250:])
plt.ylabel("$\chi^2$")
plt.xlabel("steps")
plt.show()

plt.loglog(np.abs(np.fft.rfft(chiChain[300:])))
plt.ylabel("$\chi^2$")
plt.xlabel("steps")
plt.show()

plt.ion()
fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()

for i in np.arange(6):
    axes[i].plot(chain[300:, i])
    axes[i].set_xlabel('Steps')
    axes[i].set_title(f"{paramNames[i]}")
    
fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()
for i in np.arange(6):
    axes[i].loglog(np.abs(np.fft.rfft(chain[300:, i])))
    axes[i].set_xlabel('Frequency')
    axes[i].set_title(f"{paramNames[i]}")
    


plt.ioff()
corner.corner(chain,labels=paramNames, quantiles=[0.18, 0.4, 0.75],show_titles=True, title_fmt = '.1E')

#print("Corner Loop 2 Compiled")


taup = np.mean(chain, axis = 0)
taup_error = np.std(chain,axis = 0)

paramNames = ['Hubble', "Baryon", "CDM", "Optical Depth", "Primordial Amp.", "Primordial Tilt"]
print("IMPORTANCE SAMPLING:")


for i, n in enumerate(paramNames):
    print(f"\t{n} = ".ljust(20), f"{ip[i]:.2E}", "+/-", f"{ip_error[i]:.2E}")
    
    
paramNames = ['Hubble', "Baryon", "CDM", "Optical Depth", "Primordial Amp.", "Primordial Tilt"]
print("\n Parameter Values from Revised MCMC Simulation:")
for i, n in enumerate(paramNames):
    print(f"\t{n} = ".ljust(20), f"{taup[i]:.2E}", "+/-", f"{taup_error[i]:.2E}")