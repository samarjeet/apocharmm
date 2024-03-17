#import apocharmm as ch
#import charmm as ch
from charmm import apocharmm as ch
import utils

psf2 = ch.CharmmPSF("../test/data/l0.pert.25k.psf")
psf1 = ch.CharmmPSF("../test/data/l0.pert.25k.psf")

prm = ch.CharmmParameters(["../test/data/toppar_water_ions.str", "../test/data/par_all36_cgenff.prm"])

fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)

#fmEDS = ch.ForceManagerComposite()
#fmEDS.addForceManager(fm1)
#fmEDS.addForceManager(fm2)
# (1-lambda) * psf1 + lambda * psf2
#fmEDS.setLambda(0.2) 
#print("Lambda value is : ", fmEDS.getLambda())




fmEDS = ch.EDSForceManager(fm1, fm2)
#fmEDS.addForceManager(fm1)
#fmEDS.addForceManager(fm2)


#fmEDS = fm1
boxLength = 62.79503
fmEDS.setBoxDimensions([boxLength, boxLength, boxLength])
fmEDS.setFFTGrid(64, 64, 64)
fmEDS.setKappa(0.34)
fmEDS.setPmeSplineOrder(4)
fmEDS.setCutoff(10.0)
fmEDS.setCtonnb(8.0)
fmEDS.setCtofnb(9.0)

fmEDS.initialize()

s=0.3
eOff1 = -82481.3 
eOff2 = -74366.5
#fmEDS.setSValue(s)
#fmEDS.setEnergyOffsets([eOff1, eOff2])
s = 1.0
fmEDS.setSValue(s)
fmEDS.setEnergyOffsets([0.0, 0.0])
#fmEDS.initialize()

print("init done")
ctx = ch.CharmmContext(fmEDS)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("../test/data/nvt_equil.25k.cor")
ctx.setCoordinates(crd)

ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)


print("starting the run")
integrator = ch.LangevinThermostatIntegrator(0.001)
integrator.setSimulationContext(ctx)
integrator.setFriction(5.0)
integrator.setBathTemperature(300.0)


print("equilibration...")
integrator.propagate(50000)
print("production...")
integrator.propagate(250000)




lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
for i in range(len(lambdas)):
  fmEDS.setLambda(lambdas[i])
  subscriber = ch.DcdSubscriber("out/vv_pert_25k_"+str(i)+".dcd", ctx)
  ctx.subscribe(subscriber)
  stateSub = ch.StateSubscriber("out/vv_pert_25k_"+str(i)+".txt", ctx)
  ctx.subscribe(stateSub)
  dualSub = ch.DualTopologySubscriber("out/dual_vv_pert_25k_"+str(i)+".txt", ctx)
  ctx.subscribe(dualSub)

  print("equilibration...")
  integrator.propagate(50000)
  print("production...")
  integrator.propagate(250000)
#Vutils.calculateMeanAndVariance("dual_vv_pert_25k.txt")
