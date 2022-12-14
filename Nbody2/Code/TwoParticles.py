import numpy as np
from matplotlib import pyplot as plt
import numpy.fft
import imageio 
from particleProperties import*

atomos = Particles(dt = 0.1)
atomos.twoBodyProblem()
atomos.kernel = get_kernel(atomos.nMesh, atomos.flatConst, atomos.periodicity)

fig = plt.figure()
ax = fig.add_subplot(111)
timeDomain = 250

frames = []
KinE = np.zeros(timeDomain)
PotE = np.zeros(timeDomain)
MechE = np.zeros(timeDomain)

for i in range(timeDomain):
    
    for j in range(3):
        
        atomos.CornerGridPoint = computeCornerGridPoint(atomos.nMesh, atomos.gridLength, atomos.position, atomos.periodicity)
        atomos.rho = computeRho(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.m,atomos.periodicity)
        atomos.potential = computePotential(atomos.rho, atomos.kernel, atomos.nMesh , atomos.periodicity)
        atomos.f = computeForce(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.potential,atomos.periodicity)
        atomos.position, atomos.v = take_frog_step(atomos.position, atomos.v, atomos.f, atomos.dt)
        
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

    
    plt.savefig(f'Figures/DoubleParticles/{i:003}', dpi = 200)
    plt.show()
    plt.pause(0.0001)

      
plt.plot(KinE[:], 'g', label = "Kinetic Energy", )
plt.plot(PotE[:], 'b', label = "Potential Energy")
plt.plot(MechE[:], 'r.', label = "Mechanical Energy")
plt.xlabel("Timesteps")
plt.ylabel("Energy")
plt.grid()
plt.legend()
plt.title("Two Particles Energy Evolution")
plt.savefig("Figures/TwoParticlesEnergyPlot.jpg")
plt.show()


with imageio.get_writer("Videos/" + "TwoBodies" +".gif", mode="I") as writer:
    for filename in ["Figures/DoubleParticles/" + f'{tally:003}' + ".png" for tally in range(timeDomain)]:
        image = imageio.imread(filename)
        writer.append_data(image)