import numpy as np
from matplotlib import pyplot as plt
import numpy.fft
import imageio 
from particleProperties import*

#Leapfrog Simulation: Non-Periodic

atomos = Particles(partCount = 100000, dt = 0.025, periodicity = False)
atomos.uniformDistribution()
atomos.kernel = get_kernel(atomos.nMesh, atomos.flatConst, atomos.periodicity)

fig = plt.figure()
ax = fig.add_subplot(111)
timeDomain = 500

frames = []
KinE = np.zeros(timeDomain)
PotE = np.zeros(timeDomain)
MechE = np.zeros(timeDomain)

for i in range(timeDomain):
    for j in range(3):
        
        atomos.CornerGridPoint = computeCornerGridPoint(atomos.nMesh, atomos.gridLength, atomos.position, atomos.periodicity)
        atomos.rho = computeRho(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.m, atomos.periodicity)
        atomos.potential = computePotential(atomos.rho, atomos.kernel, atomos.nMesh , atomos.periodicity)
        atomos.force = computeForce(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.potential,atomos.periodicity)
        atomos.position, atomos.v = stepLeapfrog(atomos.position, atomos.v, atomos.force, atomos.dt)

    KinE[i] += np.sum(atomos.v**2) #Kinetic Energy
    PotE[i] += np.sum(atomos.rho*atomos.potential) #Potential Energy
    
    #Summing the total mechanical energy
    MechE[i] += KinE[i] + PotE[i]
    
    plt.imshow(atomos.rho[250:750, 250:750], cmap='terrain', interpolation='nearest');
    plt.title(f'Frame {i:003}')
    plt.colorbar()

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    
    plt.savefig(f'Figures/NonPeriodic/Leapfrog/{i:003}', dpi = 200)
    plt.show()
    plt.pause(0.0001)

    
plt.plot(KinE[:], 'g', label = "Kinetic Energy", )
plt.plot(PotE[:], 'b', label = "Potential Energy")
plt.plot(MechE[:], 'r.', label = "Mechanical Energy")
plt.xlabel("Timesteps")
plt.ylabel("Energy")
plt.grid()
plt.legend()
plt.title("Non-Periodic Leapfrog Energy Evolution")
plt.savefig("Figures/NonPeriodicLeapfrogEnergy.jpg")
plt.show()
    
    
with imageio.get_writer("Videos/" + "NonPeriodicLeapfrog" +".gif", mode="I") as writer:
    for filename in ["Figures/NonPeriodic/Leapfrog/" + f'{tally:003}' + ".png" for tally in range(timeDomain)]:
        image = imageio.imread(filename)
        writer.append_data(image)
      

################################################################################################

#RK4 Simulation: Non-Periodic
        
atomos= Particles(partCount = 100000, dt =0.025, periodicity=False)
atomos.uniformDistribution()
atomos.kernel = get_kernel(atomos.nMesh, atomos.flatConst, atomos.periodicity)

fig = plt.figure()
ax = fig.add_subplot(111)
timeDomain = 500

frames = []
KinE = np.zeros(timeDomain)
PotE = np.zeros(timeDomain)
MechE = np.zeros(timeDomain)

for i in range(timeDomain):
    for j in range(3):
        atomos.CornerGridPoint = computeCornerGridPoint(atomos.nMesh, atomos.gridLength, atomos.position, atomos.periodicity)
        atomos.rho = computeRho(atomos.nMesh, atomos.gridLength, atomos.CornerGridPoint, atomos.m,atomos.periodicity)
        atomos.potential = computePotential(atomos.rho, atomos.kernel, atomos.nMesh , atomos.periodicity)
        atomos.position, atomos.v = stepRK4(atomos.nMesh,atomos.gridLength, atomos.CornerGridPoint, atomos.position,atomos.v, atomos.potential,atomos.periodicity, atomos.dt)
        
    KinE[i] += np.sum(atomos.v**2) #Kinetic Energy
    PotE[i] += np.sum(atomos.rho*atomos.potential) #Potential Energy
    
    #Summing the total mechanical energy
    MechE[i] += KinE[i] + PotE[i]
    
    
    plt.imshow(atomos.rho[250:750, 250:750], cmap='terrain', interpolation='nearest');
    plt.title(f'Frame {i:003}')
    plt.colorbar()

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    
    plt.savefig(f'Figures/NonPeriodic/RK4/{i:003}', dpi = 200)
    plt.show()  

plt.plot(KinE[:], 'g', label = "Kinetic Energy", )
plt.plot(PotE[:], 'b', label = "Potential Energy")
plt.plot(MechE[:], 'r.', label = "Mechanical Energy")
plt.xlabel("Timesteps")
plt.ylabel("Energy")
plt.grid()
plt.legend()
plt.title("Non-Periodic RK4 Energy Evolution")
plt.savefig("Figures/NonPeriodicRK4.jpg")
plt.show()


with imageio.get_writer("Videos/" + "NonPeriodicRK4" +".gif", mode="I") as writer:
    for filename in ["Figures/NonPeriodic/RK4/" + f'{tally:003}' + ".png" for tally in range(timeDomain)]:
        image = imageio.imread(filename)
        writer.append_data(image)
#      
      


