import apocharmm as ch
import utils
import time

psf2 = ch.CharmmPSF("../test/data/l0.pert.100k.psf")
psf1 = ch.CharmmPSF("../test/data/l1.pert.100k.psf")

prm = ch.CharmmParameters(["../test/data/toppar_water_ions.str", 
                              "../test/data/par_all36_cgenff.prm"])

fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)

fmDual = ch.ForceManagerComposite()
fmDual.addForceManager(fm1)
fmDual.addForceManager(fm2)

# (1-lambda) * psf1 + lambda * psf2
boxLength = 99.64716
fmDual.setBoxDimensions([boxLength, boxLength, boxLength])
fmDual.setFFTGrid(64, 64, 64)
fmDual.setKappa(0.34)
fmDual.setPmeSplineOrder(4)
fmDual.setCutoff(10.0)
fmDual.setCtonnb(8.0)
fmDual.setCtofnb(9.0)

fmDual.initialize()

ctx = ch.CharmmContext(fmDual)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("../test/data/nvt_equil.100k.cor")
ctx.setCoordinates(crd)

ctx.assignVelocitiesAtTemperature(300)
energy = ctx.calculateForces(False, True, True)
print(energy)
#quit()

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
start = time.time()

lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
for i in range(len(lambdas)):
  fmDual.setLambda(lambdas[i])
  subscriber = ch.DcdSubscriber("out/vv_pert_100k_"+str(i)+".dcd", ctx)
  ctx.subscribe(subscriber)
  stateSub = ch.StateSubscriber("out/vv_pert_100k_"+str(i)+".txt", ctx)
  ctx.subscribe(stateSub)
  dualSub = ch.DualTopologySubscriber("out/dual_vv_pert_100k_"+str(i)+".txt", ctx)
  ctx.subscribe(dualSub)

  print("equilibration...")
  integrator.propagate(50000)
  print("production...")
  integrator.propagate(250000)
  end = time.time()
  print("Time : ", end - start)

  ctx.unsubscribe(subscriber)
  ctx.unsubscribe(stateSub)
  ctx.unsubscribe(dualSub)
#utils.calculateMeanAndVariance("dual_vv_pert_100k.txt")
