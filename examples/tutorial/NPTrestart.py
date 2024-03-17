# This tutorial aims at illustrating how to create a restart file, and how to read it.

import charmm.apocharmm as ac
testDataPath = "../../test/data/"

# Import prm and psf
psf = ac.CharmmPSF(testDataPath + "waterbox.psf")
prm = ac.CharmmParameters(testDataPath + "toppar_water_ions.str")
# Create forceManager
fm = ac.ForceManager(psf, prm)
fm.setBoxDimensions([50., 50., 50.])

# Create CharmmContext
ctx = ac.CharmmContext(fm)
crd = ac.CharmmCrd(testDataPath + "waterbox.crd")
# Even if using a restart file, giving coordinates is mandatory to setup atom number
ctx.setCoordinates(crd)

# This is how to create a restart file
# =====================================
# Create integrator.
# Make sure to follow the order in which parameters are setup (create
# integrator, set context, set friction, set crystal, possibly set piston mass)
integrator = ac.LangevinPistonIntegrator(0.002)
integrator.setSimulationContext(ctx)
integrator.setPistonFriction(12.0)
integrator.setCrystalType(ac.CRYSTAL.TETRAGONAL)
integrator.setPistonMass([500.0, 500.0])
# Create restart subscriber
restartsub = ac.RestartSubscriber("waterboxRestart.res", 1000)
# Subscribe the subscriber to the integrator
integrator.subscribe(restartsub)
# Run !
integrator.propagate(1000)
integrator.unsubscribe(restartsub)

# This is how to read a restart file
# ======================================
# Create a new integrator
# As said above, make sure to follow order of parameters setup
integrator2 = ac.LangevinPistonIntegrator(0.002)
integrator2.setSimulationContext(ctx)
integrator2.setPistonFriction(12.0)
integrator2.setCrystalType(ac.CRYSTAL.TETRAGONAL)
integrator2.setPistonMass([500.0, 500.0])
# Create a restart subscriber, using the file name that we want to read as input
readrestartsub = ac.RestartSubscriber("waterboxRestart.res", 1000)
# Subscribe the subscriber to the integrator
integrator2.subscribe(readrestartsub)
# Read the restart file
readrestartsub.readRestart()
# Run !
integrator2.propagate(100)
