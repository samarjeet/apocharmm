import apocharmm as ch

psf = ch.CharmmPSF("../../test/data/waterbox.psf")
prm = ch.CharmmParameters("../../test/data/toppar_water_ions.str")

fm = ch.ForceManager(psf, prm)
#fm.setBoxDimensions([50.0, 50.0, 50.0])
#fm.setFFTGrid(48, 48, 48)
#fm.setKappa(0.34)
#fm.setCutoff(10.0)
#fm.setCutoff(12.0)
#fm.setCtonnb(7.0)
#fm.setCtofnb(8.0)
#fm.initialize()

prm2 = ch.CharmmParameters("../../test/data/toppar_water_ions.str")
fm2 = ch.ForceManager(psf, prm2)
#fm2.setBoxDimensions([50.0, 50.0, 50.0])
#fm2.setFFTGrid(48, 48, 48)
#fm2.setKappa(0.34)
#fm2.setCutoff(10.0)
#fm2.setCutoff(12.0)
#fm2.setCtonnb(7.0)
#fm2.setCtofnb(8.0)
#fm2.initialize()

fmComposite = ch.ForceManagerComposite()
fmComposite.addForceManager(fm)
fmComposite.addForceManager(fm2)

fmComposite.setBoxDimensions([50.0, 50.0, 50.0])
fmComposite.setFFTGrid(48, 48, 48)
fmComposite.setKappa(0.34)
fmComposite.setCutoff(10.0)
fmComposite.setCutoff(12.0)
fmComposite.setCtonnb(7.0)
fmComposite.setCtofnb(8.0)

fmComposite.initialize()

# (1-lambda) * psf1 + lambda * psf2
fmComposite.setLambda(0.2) 
#print("Lambda value is : ", fmComposite.getLambda())

ctx = ch.CharmmContext(fmComposite)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("../../test/data/waterbox.crd")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
#ctx.calculateForces()

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

subscriber = ch.NetCDFSubscriber("vv_waterbox.nc", ctx)
ctx.subscribe(subscriber)
stateSub = ch.StateSubscriber("vv_waterbox.txt", ctx)
ctx.subscribe(stateSub)

integrator.setReportSteps(5000)
print("Starting propagation")
integrator.propagate(100000)
