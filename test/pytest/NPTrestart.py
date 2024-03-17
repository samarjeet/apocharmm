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
# Create integrator

# integrator = ac.LangevinPistonIntegrator(0.002)
# integrator.setSimulationContext(ctx)
# integrator.setPistonFriction(12.0)

# integrator.setCrystalType(ac.CRYSTAL.CUBIC)
# restartsub = ac.RestartSubscriber("waterboxRestartCubic.res", 1000)

# integrator.setCrystalType(ac.CRYSTAL.TETRAGONAL)
# restartsub = ac.RestartSubscriber("waterboxRestartTetragonal.res", 1000)

# integrator.setCrystalType(ac.CRYSTAL.ORTHORHOMBIC)
# restartsub = ac.RestartSubscriber("waterboxRestartOrthorhombic.res", 1000)

# integrator.subscribe(restartsub)
# integrator.propagate(1000)
# integrator.unsubscribe(restartsub)


# This is how to read a restart file
# ======================================
# Create a new integrator

# integrator2 = ac.LangevinPistonIntegrator(0.002)
# integrator2.setSimulationContext(ctx)
# integrator2.setPistonFriction(12.0)

# integrator2.setCrystalType(ac.CRYSTAL.CUBIC)
# integrator2.setPistonMass([500.0])
# readrestartsub = ac.RestartSubscriber("waterboxRestartCubic.res", 1000)

# integrator2.setCrystalType(ac.CRYSTAL.TETRAGONAL)
# integrator2.setPistonMass([500.0, 500.0])
# readrestartsub = ac.RestartSubscriber("waterboxRestartTetragonal.res", 1000)

# integrator2.setCrystalType(ac.CRYSTAL.ORTHORHOMBIC)
# integrator2.setPistonMass([500.0, 500.0, 500.0])
# readrestartsub = ac.RestartSubscriber("waterboxRestartOrthorhombic.res", 1000)

# integrator2.subscribe(readrestartsub)
# readrestartsub.readRestart()
# integrator2.propagate(100)
