import apocharmm as ch
import utils

psf0 = ch.CharmmPSF("../../test/data/l0.pert.25k.psf")
psf1 = ch.CharmmPSF("../../test/data/l0.pert.25k.psf")
psf2 = ch.CharmmPSF("../../test/data/l1.pert.25k.psf")
psf3 = ch.CharmmPSF("../../test/data/l1.pert.25k.psf")
psf4 = ch.CharmmPSF("../../test/data/l1.pert.25k.psf")

prm = ch.CharmmParameters(["../../test/data/toppar_water_ions.str", "par_all36_cgenff.prm"])

fm0 = ch.ForceManager(psf0, prm)
fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)
fm3 = ch.ForceManager(psf3, prm)
fm4 = ch.ForceManager(psf4, prm)

fmMBAR = ch.MBARForceManager()
fmMBAR.addForceManager(fm0)
fmMBAR.addForceManager(fm1)
fmMBAR.addForceManager(fm2)
fmMBAR.addForceManager(fm3)
fmMBAR.addForceManager(fm4)

fmMBAR.setLambdaVec([0.0, 0.0, 1.0, 0.0, 0.0]) 

boxLength = 62.79503
fmMBAR.setBoxDimensions([boxLength, boxLength, boxLength])
fmMBAR.setFFTGrid(64, 64, 64)
fmMBAR.setKappa(0.34)
fmMBAR.setPmeSplineOrder(4)
fmMBAR.setCutoff(10.0)
fmMBAR.setCtonnb(8.0)
fmMBAR.setCtofnb(9.0)

fmMBAR.initialize()

ctx = ch.CharmmContext(fmMBAR)
crd = ch.CharmmCrd("../../test/data/nvt_equil.25k.cor")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
#integrator.propagate(2000)
#quit()
integrator.propagate(10000)

#subscriber = ch.NetCDFSubscriber("vv_mbar_2cle.nc", ctx)
#ctx.subscribe(subscriber)
#stateSub = ch.StateSubscriber("vv_mbar_2cle.txt", ctx)
#ctx.subscribe(stateSub)
mbarSub = ch.MBARSubscriber("pe_vv_mbar_25k_10.txt", ctx)
ctx.subscribe(mbarSub)

integrator.setReportSteps(10)
print("Starting propagation")
integrator.propagate(100000)
#utils.calculateMeanAndVarianceExt("pe_vv_mbar_eds_2cle.txt")
