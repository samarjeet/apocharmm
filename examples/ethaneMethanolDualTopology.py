import apocharmm as ch

psf1 = ch.CharmmPSF("solv.em.meoh.psf")
psf2 = ch.CharmmPSF("solv.em.etha.psf")

prm = ch.CharmmParameters(["../../test/data/toppar_water_ions.str", "par_all36_cgenff.prm"])

fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)

fmDual = ch.ForceManagerComposite()
fmDual.addForceManager(fm1)
fmDual.addForceManager(fm2)

fmDual.setBoxDimensions([31.1032, 31.1032, 31.1032])
fmDual.setFFTGrid(32, 32, 32)
fmDual.setKappa(0.34)
fmDual.setPmeSplineOrder(6)
fmDual.setCutoff(16.0)
fmDual.setCtonnb(10.0)
fmDual.setCtofnb(12.0)

fmDual.initialize()

# (1-lambda) * psf1 + lambda * psf2
fmDual.setLambda(0.2) 
print("Lambda value is : ", fmDual.getLambda())

ctx = ch.CharmmContext(fmDual)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("solv.em.cor")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
#ctx.calculateForces()
print("Initialized ")

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

subscriber = ch.NetCDFSubscriber("vv_etha_meoh.nc", ctx)
ctx.subscribe(subscriber)
stateSub = ch.StateSubscriber("vv_etha_meoh.txt", ctx)
ctx.subscribe(stateSub)

integrator.setReportSteps(100)
print("Starting propagation")
integrator.propagate(1000)
