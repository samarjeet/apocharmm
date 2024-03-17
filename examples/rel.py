import apocharmm as ch

prm = ch.CharmmParameters("../../test/data/toppar_water_ions.str")
psf = ch.CharmmPSF("../../test/data/waterbox.psf")
crd = ch.CharmmCrd("../../test/data/waterbox.crd")

fm = ch.ForceManager(psf, prm)
fm.setBoxDimensions([50.0, 50.0, 50.0])
fm.setFFTGrid(48, 48, 48)
fm.setKappa(0.34)
fm.setCutoff(10.0)
fm.setCtonnb(7.0)
fm.setCtofnb(8.0)
fm.initialize()

ctx = ch.CharmmContext(fm)
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

subscriber = ch.NetCDFSubscriber("vv_waterbox.nc", ctx)
ctx.subscribe(subscriber)
stateSub = ch.StateSubscriber("vv_waterbox.txt", ctx)
ctx.subscribe(stateSub)

integrator.setReportSteps(5000)

integrator.propagate(100000)
