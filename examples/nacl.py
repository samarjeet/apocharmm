import apocharmm as ch
import utils

psf1 = ch.CharmmPSF("../../test/data/nacl0.psf")
psf2 = ch.CharmmPSF("../../test/data/nacl1.psf")

prm = ch.CharmmParameters(["../../test/data/toppar_water_ions.str"])

fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)

fmDual = ch.ForceManagerComposite()
fmDual.addForceManager(fm1)
fmDual.addForceManager(fm2)
# (1-lambda) * psf1 + lambda * psf2
fmDual.setLambda(1.0) 
print("Lambda value is : ", fmDual.getLambda())

#fmDual = fm2
boxLength = 30.9120
fmDual.setBoxDimensions([boxLength, boxLength, boxLength])
fmDual.setFFTGrid(32, 32, 32)
fmDual.setKappa(0.34)
fmDual.setPmeSplineOrder(6)
fmDual.setCutoff(16.0)
fmDual.setCtonnb(10.0)
fmDual.setCtofnb(12.0)

fmDual.initialize()

ctx = ch.CharmmContext(fmDual)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("../../test/data/nacl.cor")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)
print("Initialized ")

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
#integrator.propagate(3000)

subscriber = ch.NetCDFSubscriber("vv_eds_nacl.nc", ctx)
ctx.subscribe(subscriber)
stateSub = ch.StateSubscriber("vv_eds_nacl.txt", ctx)
ctx.subscribe(stateSub)
dualSub = ch.DualTopologySubscriber("dual_vv_eds_nacl.txt", ctx)
ctx.subscribe(dualSub)

integrator.setReportSteps(1)
print("Starting propagation")
#integrator.propagate(10)
#utils.calculateMeanAndVariance("dual_vv_eds_2cle.txt")
