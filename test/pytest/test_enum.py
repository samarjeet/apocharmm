import charmm.apocharmm as ac   

testDataPath = "/u/aviatfel/dev/testapo3/test/data/"
prmfile = testDataPath + "toppar_water_ions.str"
psffile = testDataPath + "waterbox.psf"
restartFile = testDataPath + "restart/waterbox.npt.restart"
nSteps = int(1e4)
boxDim = [48.9342, 48.9342, 48.9342]



integrator = ac.LangevinPistonIntegrator(0.001)
integrator.setCrystalType(ac.CRYSTAL.CUBIC)

prm = ac.CharmmParameters([prmfile])
psf = ac.CharmmPSF(testDataPath + "waterbox.psf")   
fm = ac.ForceManager(psf, prm)
fm.setVdwType(ac.NonBondedType.VDW_VFSW)
print(ac.NonBondedType.VDW_VFSW)
print(ac.NonBondedType.VDW_DBEXP)

fm.setBoxDimensions(boxDim)
ctx = ac.CharmmContext(fm)
ctx.readRestart(restartFile)

ctx.calculatePotentialEnergy(True, True)

#langevinThermostatNVE = ac.LangevinThermostatIntegrator(0.002)
#langevinThermostatNVE.setFriction(0.0)
#langevinThermostatNVE.setBathTemperature(300.0)
#langevinThermostatNVE.setSimulationContext(ctx)
#
#NVESub = ac.StateSubscriber("waterbox_nve.state", 100)
#NVESub.setReportFlags("potentialenergy, kineticenergy, totalenergy, temperature")
#langevinThermostatNVE.subscribe([NVESub])
#
#langevinThermostatNVE.propagate(nSteps)
