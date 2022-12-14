import numpy as np
import numba as nb
from matplotlib import pyplot as plt
import numpy.fft
import imageio 
from particleProperties import*


atomos = Particles(dt = 0.008)
atomos.oneBodyProblem()
atomos.kernel = get_kernel(atomos.nMesh, atomos.flatConst, atomos.periodicity)

fig = plt.figure()
ax = fig.add_subplot(111)
timeDomain = 50

def computeRhoSingleParticle(nMesh, gridLength, CornerGridPoint, m, periodicity):

    if periodicity == True:

        rho = np.zeros((nMesh,nMesh))
        rho[int(CornerGridPoint[0])][int(CornerGridPoint[1])] += m
    else:
        rho = np.zeros((nMesh*2,nMesh*2))
        if not(CornerGridPoint[0] < nMesh/2 or CornerGridPoint[0] > 3*nMesh/2 or CornerGridPoint[1] < nMesh/2 or CornerGridPoint[1]> 3*nMesh/2):
            rho[int(CornerGridPoint[0])][int(CornerGridPoint[1])] += m
        else:
            print("Particle went out of bounds")
    
    rho /= gridLength**2
    
    return rho

@nb.njit 
def computeForceSingleParticle(nMesh,gridLength, CornerGridPoint,potential,periodicity):

    force = np.zeros(2)

    x  = int(CornerGridPoint[0])
    y =  int(CornerGridPoint[1])

    if periodicity:
        force[0] = (potential[(x+1) % nMesh][y]-potential[(x-1) % nMesh][y])/(2*gridLength)
        force[1] = (potential[x][(y+1) % nMesh]-potential[x][(y-1) % nMesh])/(2*gridLength)
    else:
        if (x < nMesh/2 or x > 3*nMesh/2 or y < nMesh/2 or y > 3*nMesh/2 ):
            force[0] = 0
            force[1] = 0
        else:
            force[0] = (potential[(x+1)][y]-potential[(x-1)][y])/(2*gridLength)
            force[1] = (potential[x][(y+1)]-potential[x][(y-1)])/(2*gridLength)
    return force

frames = []
KinE = np.zeros(timeDomain)
PotE = np.zeros(timeDomain)
MechE = np.zeros(timeDomain)

for i in range(timeDomain):
    
    for j in range(3):
        
        atomos.CornerGridPoint = computeCornerGridPoint(atomos.nMesh, atomos.gridLength, atomos.position, atomos.periodicity)
        atomos.rho = computeRhoSingleParticle(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.m, atomos.periodicity)
        atomos.potential = computePotential(atomos.rho, atomos.kernel, atomos.nMesh, atomos.periodicity)
        atomos.force = computeForceSingleParticle(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.potential,atomos.periodicity)
        atomos.position, atomos.v = stepLeapfrog(atomos.position, atomos.v, atomos.force, atomos.dt)
        
        
    KinE[i] += np.sum(atomos.v**2) #Kinetic Energy
    PotE[i] += np.sum(atomos.rho*atomos.potential) #Potential Energy
    #Summing the total mechanical energy
    MechE[i] += KinE[i] + PotE[i]
    plt.imshow(atomos.rho, cmap='gnuplot', interpolation='nearest');
    plt.title(f'Frame {i:003}')
    plt.colorbar()
    

    
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    
    plt.savefig(f'Figures/SingleParticle/{i:003}', dpi = 200)
    plt.show()

plt.plot(KinE[:], 'g', label = "Kinetic Energy", )
plt.plot(PotE[:], 'b', label = "Potential Energy")
plt.plot(MechE[:], 'r.', label = "Mechanical Energy")
plt.xlabel("Timesteps")
plt.ylabel("Energy")
plt.grid()
plt.legend()
plt.title("Single Particle Energy Evolution")
plt.savefig("Figures/SingleParticle.jpg")
plt.show()


with imageio.get_writer("Videos/" + "SingleBody" +".gif", mode="I") as writer:
    for filename in ["Figures/SingleParticle/" + f'{tally:003}' + ".png" for tally in range(timeDomain)]:
        image = imageio.imread(filename)
        writer.append_data(image)
      

