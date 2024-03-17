# Start by importing the `apocharmm` package
import charmm.apocharmm as ac

# setup data location
testDataPath = "../../test/data/"

# Import the parameters anf the system's topology (prm and psf)
psf = ac.CharmmPSF(testDataPath + "waterbox.psf")
prm = ac.CharmmParameters(testDataPath + "toppar_water_ions.str")

# Set up the ForceManager that will drive the simulation.
# Create the ForceManager object using the psf and prm
fm = ac.ForceManager(psf, prm)
# Setup box size, FFT options, cutoffs...
fm.setBoxDimensions([50., 50., 50.])
fm.setFFTGrid(48, 48, 48)   # optional
fm.setCtonnb(9.0)           # optional, default value is 10.0
fm.setCtofnb(10.0)          # optional, default value is 12.0
fm.setCutoff(12.0)          # optional, default value is 14.0

# The simulation state will be handled by a CharmmContext object, created from the ForceManager.
ctx = ac.CharmmContext(fm)

# This CharmmContext handles the coordinates and velocities.

crd = ac.CharmmCrd(testDataPath + "waterbox.crd")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300.0)

# We start by a short minimization of our system.
minimizer = ac.Minimizer()
minimizer.setSimulationContext(ctx)
minimizer.minimize(1000)

# Here we will integrate using a Langevin thermostat, with a bath temperature of 300K and a friction constant of 12 ps^-1. We create the Integrator object, then attach it to the CharmmContext. Finally, we propagate for 10 steps.

integrator = ac.LangevinThermostatIntegrator(.001, 300, 12)
integrator.setSimulationContext(ctx)
integrator.propagate(10)

# The simulation can be monitored using various Susbcribers, responsible for creating output files at a given frequency:
# * StateSubscribers (time, energy, temperature, pressure...);
# * RestartSubscriber: outputs a Charmm-like restart file
# * DcdSubscriber: saves the trajectory to a .dcd file
# * ...
#

stateSub = ac.StateSubscriber("waterboxState.txt", 500)
dcdSub = ac.DcdSubscriber("waterboxTraj.dcd", 1000)
restartSub = ac.RestartSubscriber("waterboxRestart.res", 2000)

# Once created, a Subscriber needs to be subscribed to an Integrator.
integrator.subscribe(stateSub)
integrator.subscribe([dcdSub, restartSub])
integrator.propagate(3000)
