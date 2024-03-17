import charmm.apocharmm as ch

prm = ch.CharmmParameters("../test/data/toppar_water_ions.str")
psf = ch.CharmmPSF("../test/data/waterbox.psf")
crd = ch.CharmmCrd("../test/data/waterbox.crd")

fm = ch.ForceManager(psf, prm)
fm.setBoxDimensions([50.0, 50.0, 50.0])
fm.setFFTGrid(48, 48, 48)
fm.setKappa(0.34)
fm.setCtonnb(9.0)
fm.setCtofnb(10.0)
fm.setCutoff(12.0)
fm.initialize()

ctx = ch.CharmmContext(fm)
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

integrator = ch.LangevinThermostatIntegrator(0.001, 12, 300)
integrator.setSimulationContext(ctx)

minimizer = ch.Minimizer()
minimizer.setSimulationContext(ctx)
minimizer.minimize(1000)

stateSub = ch.StateSubscriber("waterStateSub_script.txt", 1000)

integrator.subscribe(stateSub)

integrator.propagate(10000)