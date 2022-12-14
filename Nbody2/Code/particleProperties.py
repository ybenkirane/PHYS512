import numpy as np
import numba as nb
import numpy.fft


#Used as a reference in our grid... 
#Was having trouble figuring out how to reference the grid squares so after discussing with peers, using a corner of the grid is 
#the most simple way of doing this. Using the floor functions was recommended as a simple solution. This makes sense as using the same
#corner for every grid is consistent ... 
def computeCornerGridPoint(nMesh, gridLength, position, periodicity):

    if periodicity == True:
        
        CornerGridPoint = np.floor(position/gridLength) % nMesh
        
    else: 
        
        CornerGridPoint = np.floor(position/gridLength)
        
    return CornerGridPoint

#Computing the Rho of our cell grid here.
@nb.njit
def computeRho(nMesh, gridLength, CornerGridPoint, m, periodicity):

    mass = m.copy()

    if periodicity == True:

        rho = np.zeros((nMesh,nMesh))
       
        for i in range(CornerGridPoint.shape[0]):
           
            CartesianPoint  = CornerGridPoint[i]
            rho[int(CartesianPoint[0])][int(CartesianPoint[1])] += mass[i]
    else:

        rho = np.zeros((nMesh*2,nMesh*2))
        for i in range(CornerGridPoint.shape[0]):
           
            CartesianPoint  = CornerGridPoint[i]
           
            #Critical padding condition to limit wrap-around 
            if not(CartesianPoint[0] < nMesh/2 or CartesianPoint[0] > 3*nMesh/2 or CartesianPoint[1] < nMesh/2 or CartesianPoint[1]> 3*nMesh/2): 
                
                rho[int(CartesianPoint[0])][int(CartesianPoint[1])] += mass[i]

    
    rho /= gridLength**2
    
    return rho



#Prof Sievers' get_kernel function
def get_kernel(nMesh,r0,periodicity):
    if not(periodicity):
        x=np.fft.fftfreq(nMesh*2)*nMesh*2
        rsqr=np.outer(np.ones(nMesh*2),x**2)
    else:
        x=np.fft.fftfreq(nMesh)*nMesh
        rsqr=np.outer(np.ones(nMesh),x**2)
    rsqr=rsqr+rsqr.T
    rsqr[rsqr<r0**2]=r0**2
    kernel =rsqr**-0.5
    return kernel


#Convolution between density and kernel
def computePotential(rho, kernel, nMesh, periodicity):

    rhoFFT= np.fft.rfft2(rho)
    kernelFFT = np.fft.rfft2(kernel)
    
    if periodicity:
        potential = np.fft.irfft2(rhoFFT * kernelFFT, [nMesh, nMesh])
    else:
        potential = np.fft.irfft2(rhoFFT * kernelFFT, [nMesh*2, nMesh*2])
    return potential 
    
#Central difference scheme implied to compute the force from poisson's equation... 
@nb.njit 
def computeForce(nMesh, gridLength, CornerGridPoint,potential,periodicity):

    force = np.zeros((CornerGridPoint.shape[0],2))

    for i in range(CornerGridPoint.shape[0]):
        x  = int(CornerGridPoint[i][0])
        y =  int(CornerGridPoint[i][1])

        if periodicity:
            force[i][0] = (potential[(x+1) % nMesh][y]-potential[(x-1) % nMesh][y])/(2*gridLength)
            force[i][1] = (potential[x][(y+1) % nMesh]-potential[x][(y-1) % nMesh])/(2*gridLength)
        else:
            if (x < nMesh/2 or x > 3*nMesh/2 or y < nMesh/2 or y > 3*nMesh/2 ):
                
                force[i][0] = 0
                force[i][1] = 0
                
            else:
                
                force[i][0] = (potential[(x+1)][y]-potential[(x-1)][y])/(2*gridLength)
                force[i][1] = (potential[x][(y+1)]-potential[x][(y-1)])/(2*gridLength)
                
    return force

#Prof. Sievers' Functions:
def compDerivatives(nMesh,gridLength, xRange, potential,periodicity):
    
    nRange = xRange.shape[0]//2
    x = xRange[:nRange,:]
    v = xRange[nRange:,:]
    
    f = computeForce(nMesh,gridLength, x, potential, periodicity)
    
    return np.vstack([v,f])

#Leapfrog Step 
def stepLeapfrog(position,vSet,f,dt):
    
    position[:]+= dt*vSet
    vSet[:] += f*dt
    
    return position, vSet

#RK4 Step 
def stepRK4(nMesh,gridLength, CornerGridPoint, position, v, potential,periodicity, dt):

    xRange = np.vstack([CornerGridPoint, v])

    D1 = compDerivatives(nMesh,gridLength, xRange, potential,periodicity)
    D2 = compDerivatives(nMesh,gridLength, xRange + D1*dt/2, potential, periodicity)
    D3 = compDerivatives(nMesh,gridLength, xRange + D2*dt/2, potential, periodicity)
    D4 = compDerivatives(nMesh,gridLength, xRange + D3*dt, potential, periodicity)
    
    tot = (D1 + 2*D2 + 2*D3 + D4)/6

    if periodicity:
        
        nRange = position.shape[0]
        position += (tot[:nRange,:])*dt
        v += tot[nRange:,:]*dt
        
    else:
        
        nRange = position.shape[0]
        position += tot[:nRange,:]*dt
        v += tot[nRange:,:]*dt
        
    return position, v


class Particles:
    
    def __init__(self, partCount = 25000, nMesh = 600, dx = 1, dt = 0.075, flatConst = 2, periodicity = True):
        
        self.position = np.empty([partCount,2])
        self.partCount = partCount
        self.CornerGridPoint = np.empty([self.partCount,2])
        self.m = np.empty(partCount)
        self.f = np.empty([partCount,2])
        self.v = np.empty([partCount,2])
        self.kernel=[]
        self.nMesh=nMesh
        self.gridLength = dx
        self.dt = dt
        self.flatConst = flatConst
        self.periodicity = periodicity
        
        if self.periodicity:
            
            self.rho=np.empty([self.nMesh,self.nMesh])
            self.potential=np.empty([self.nMesh,self.nMesh])
            
        else:
            
            self.rho=np.empty([self.nMesh*2,self.nMesh*2])
            self.potential=np.empty([self.nMesh*2,self.nMesh*2])

    def oneBodyProblem(self):
        
        self.partCount = 1
        self.nMesh = 50
        
        if self.periodicity:
            
            self.position = np.array([20, self.nMesh/2])
            
        else:
            
            self.position = np.array([35., self.nMesh - 1])
            
        self.v = np.array([0.,0.])
        self.m = 1

    def twoBodyProblem(self):
        
        self.nMesh = 50
        self.partCount = 2
        
        if self.periodicity:
            
            self.position = np.array([[20., self.nMesh/2],[30., self.nMesh/2]])
            
        else:
            
            self.position = np.array([[40, 45],[self.nMesh, 35]])
            
        self.v = np.array([[0,-1],[0,1]])
        self.m[:] = 8

    def uniformDistribution(self):
        
        seed = 24986541
        rng = np.random.default_rng(seed)
        self.nMesh = 500
        
        if self.periodicity:
            
            self.position[:]=rng.random((self.partCount,2))*self.nMesh
       
        else:
           
            self.position[:]=rng.random((self.partCount,2))*self.nMesh + self.nMesh/2

        self.v[:] = 0
        self.m[:] = 1
