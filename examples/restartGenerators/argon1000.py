# Test python script to generate an argon1000 restart file

import charmm.apocharmm as ac
import numpy as np

testDataPath = "/u/aviatfel/dev/testapo3/test/data/"
prmfile = testDataPath + "argon.prm"
psffile = testDataPath + "argon1000.psf"
crdfile = testDataPath + "argon_1000.crd"
nSteps = int(1e6)

boxDim = [44., 44., 44.]


prm = ac.CharmmParameters([prmfile])
psf = ac.CharmmPSF(psffile)
fm = ac.ForceManager(psf, prm)
fm.setBoxDimensions(boxDim)
ctx = ac.CharmmContext(fm)

crd = ac.CharmmCrd(crdfile)

ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

# integrator = ac.LangevinPistonIntegrator(0.002)
# integrator.setCrystalType(ac.CRYSTAL.CUBIC)
# integrator.setPistonFriction(5.0)
# integrator.setBathTemperature(300.0)
# integrator.setPistonMass(100.0)
# integrator.setSimulationContext(ctx)

integrator = ac.LangevinThermostatIntegrator(0.002, 300.0, 5.0)
integrator.setSimulationContext(ctx)

# integrator = ac.VelocityVerletIntegrator(0.002)
# integrator.setSimulationContext(ctx)

restartsub = ac.RestartSubscriber("argonRestart.rst", 10000)
integrator.subscribe(restartsub)

integrator.propagate(nSteps)
