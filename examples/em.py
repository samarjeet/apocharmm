import apocharmm as ch

psf = ch.CharmmPSF("solv_em.psf")

prm = ch.CharmmParameters(["../../test/data/toppar_water_ions.str", "par_all36_cgenff.prm", "em.str"])

fm = ch.ForceManager(psf, prm)

fm.setBoxDimensions([31.1032, 31.1032, 31.1032])
fm.setFFTGrid(32, 32, 32)
fm.setKappa(0.34)
fm.setPmeSplineOrder(6)
fm.setCutoff(16.0)
fm.setCtonnb(10.0)
fm.setCtofnb(12.0)

fm.initialize()

ctx = ch.CharmmContext(fm)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("solv_em.cor")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)
print("Initialized ")

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

subscriber = ch.NetCDFSubscriber("vv_etha_meoh.nc", ctx)
ctx.subscribe(subscriber)
stateSub = ch.StateSubscriber("vv_etha_meoh.txt", ctx)
ctx.subscribe(stateSub)

integrator.setReportSteps(100)
print("Starting propagation")
#integrator.propagate(1000)
