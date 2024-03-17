import apocharmm as ch
import utils
import time

psf2 = ch.CharmmPSF("../test/data/l0.2cle.psf")
psf1 = ch.CharmmPSF("../test/data/l1.2cle.psf")

prm = ch.CharmmParameters(["../test/data/toppar_water_ions.str", "../test/data/par_all36_cgenff.prm", 
  "../test/data/2cle.str"])

fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)

fmDual = ch.ForceManagerComposite()
fmDual.addForceManager(fm1)
fmDual.addForceManager(fm2)
fmDual.setLambda(0.2) 
print("Lambda value is : ", fmDual.getLambda())

#fmDual = fm1
boxLength = 30.9120
fmDual.setBoxDimensions([boxLength, boxLength, boxLength])
fmDual.setFFTGrid(32, 32, 32)
fmDual.setKappa(0.34)
fmDual.setPmeSplineOrder(4)
fmDual.setCutoff(9.0)
fmDual.setCtonnb(7.0)
fmDual.setCtofnb(8.0)

fmDual.initialize()

ctx = ch.CharmmContext(fmDual)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("../test/data/solv2.2cle.cor")
print("Coords")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
start = time.time()
#integrator.propagate(50000)

#subscriber = ch.NetCDFSubscriber("out/vv_eds_2cle.nc", ctx)
lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
for i in range(len(lambdas)):
  fmDual.setLambda(lambdas[i])
  subscriber = ch.DcdSubscriber("out/vv_eds_2cle_"+str(i)+".dcd", ctx)
  ctx.subscribe(subscriber)
  stateSub = ch.StateSubscriber("out/vv_eds_2cle_"+str(i)+".txt", ctx)
  ctx.subscribe(stateSub)
  dualSub = ch.DualTopologySubscriber("out/dual_vv_eds_2cle_"+str(i)+".txt", ctx)
  ctx.subscribe(dualSub)

  integrator.setReportSteps(5000)
  print("Starting propagation")
  integrator.propagate(50000)
  integrator.propagate(250000)
  end = time.time()
  print("Time : ", end - start)
#utils.calculateMeanAndVariance("dual_vv_eds_2cle.txt")
