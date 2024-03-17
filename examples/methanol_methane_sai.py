import apocharmm as ch
import utils
import os

base_path = "/u/samar/Documents/git/transformato-systems/methanol-methane-solvation-free-energy/methanol/intst"
path_1 = base_path + "1/"

psf1 = ch.CharmmPSF(base_path+"1/lig_in_waterbox.psf")
psf2 = ch.CharmmPSF(base_path+"2/lig_in_waterbox.psf")
psf3 = ch.CharmmPSF(base_path+"3/lig_in_waterbox.psf")

#psf2 = ch.CharmmPSF("../test/data/l0.pert.25k.psf")
#psf1 = ch.CharmmPSF("../test/data/l1.pert.25k.psf")

prm = ch.CharmmParameters(["../test/data/toppar_water_ions.str", "../test/data/par_all36_cgenff.prm"])

fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)

fmsai = ch.ForceManagerComposite()
fmsai.addForceManager(fm1)
fmsai.addForceManager(fm2)
# (1-lambda) * psf1 + lambda * psf2
fmsai.setLambda(0.2) 
print("Lambda value is : ", fmsai.getLambda())




fmEDS = ch.EDSForceManager()
fmEDS.addForceManager(fm1)
emEDS.addForceManager(fm2)

s=0.3
eOff1 = -82481.3 
eOff2 = -74366.5
fmEDS.setSValue(s)
fmEDS.setEnergyOffsets([eOff1, eOff2])


#fmsai = fm1
boxLength = 62.79503
fmsai.setBoxDimensions([boxLength, boxLength, boxLength])
fmsai.setFFTGrid(64, 64, 64)
fmsai.setKappa(0.34)
fmsai.setPmeSplineOrder(4)
fmsai.setCutoff(10.0)
fmsai.setCtonnb(8.0)
fmsai.setCtofnb(9.0)

fmsai.initialize()

ctx = ch.CharmmContext(fmsai)
#ctx = ch.CharmmContext(fm)
crd = ch.CharmmCrd("../test/data/nvt_equil.25k.cor")
ctx.setCoordinates(crd)

ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
for i in range(len(lambdas)):
  fmsai.setLambda(lambdas[i])
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
