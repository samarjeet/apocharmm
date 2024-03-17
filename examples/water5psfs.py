import apocharmm as ch
import utils

psf0 = ch.CharmmPSF("../../test/data/waterbox_0.psf")
psf1 = ch.CharmmPSF("../../test/data/waterbox_1.psf")
psf2 = ch.CharmmPSF("../../test/data/waterbox_2.psf")
psf3 = ch.CharmmPSF("../../test/data/waterbox_3.psf")
psf4 = ch.CharmmPSF("../../test/data/waterbox_4.psf")

prm = ch.CharmmParameters(["../../test/data/toppar_water_ions.str"])

fm0 = ch.ForceManager(psf0, prm)
fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)
fm3 = ch.ForceManager(psf3, prm)
fm4 = ch.ForceManager(psf4, prm)

fmMul = ch.ForceManagerComposite()
fmMul.addForceManager(fm0)
fmMul.addForceManager(fm1)
fmMul.addForceManager(fm2)
fmMul.addForceManager(fm3)
fmMul.addForceManager(fm4)

fmMul.setBoxDimensions([50.0, 50.0, 50.0]);
fmMul.setFFTGrid(48, 48, 48);
fmMul.setKappa(0.34);
fmMul.setCutoff(10.0);
fmMul.setCtonnb(7.0);
fmMul.setCtofnb(8.0);

fmMul.initialize();

fmMul.setLambdaVec([0.0, 0.0, 0.0, 0., 1.0])

ctx = ch.CharmmContext(fmMul)
crd = ch.CharmmCrd("../../test/data/waterbox.crd")
print("Coords")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
ctx.calculateForces(False, True, True)

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
integrator.propagate(3000)

subscriber = ch.NetCDFSubscriber("vv_waterbox_5psfs.nc", ctx)
ctx.subscribe(subscriber)
stateSub = ch.StateSubscriber("vv_waterbox_5psfs.txt", ctx)
ctx.subscribe(stateSub)
dualSub = ch.DualTopologySubscriber("pe_vv_waterbox_5psfs.txt", ctx)
ctx.subscribe(dualSub)

integrator.setReportSteps(100)
print("Starting propagation")
integrator.propagate(4000)
utils.calculateMeanAndVarianceExt("pe_vv_waterbox_5psfs.txt", 5)

