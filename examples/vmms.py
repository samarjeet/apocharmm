import apocharmm as ch
import utils

psf0 = ch.CharmmPSF("../../test/data/waterbox_0.psf")
psf1 = ch.CharmmPSF("../../test/data/waterbox_1.psf")
psf2 = ch.CharmmPSF("../../test/data/waterbox_2.psf")
#psf3 = ch.CharmmPSF("../../test/data/waterbox_3.psf")
#psf4 = ch.CharmmPSF("../../test/data/waterbox_4.psf")

prm = ch.CharmmParameters(["../../test/data/toppar_water_ions.str"])

fm0 = ch.ForceManager(psf0, prm)
fm1 = ch.ForceManager(psf1, prm)
fm2 = ch.ForceManager(psf2, prm)
#fm3 = ch.ForceManager(psf3, prm)
#fm4 = ch.ForceManager(psf4, prm)

fm0.setBoxDimensions([50.0, 50.0, 50.0]);
fm0.setFFTGrid(48, 48, 48);
fm0.setKappa(0.34);
fm0.setCutoff(10.0);
fm0.setCtonnb(7.0);
fm0.setCtofnb(8.0);
fm0.initialize();

# TODO : create VMMS context - contains molar fraction
ctx0 = ch.CharmmContext(fm0)

fm1.setBoxDimensions([50.0, 50.0, 50.0]);
fm1.setFFTGrid(48, 48, 48);
fm1.setKappa(0.34);
fm1.setCutoff(10.0);
fm1.setCtonnb(7.0);
fm1.setCtofnb(8.0);
fm1.initialize();
ctx1 = ch.CharmmContext(fm1)

fm2.setBoxDimensions([50.0, 50.0, 50.0]);
fm2.setFFTGrid(48, 48, 48);
fm2.setKappa(0.34);
fm2.setCutoff(10.0);
fm2.setCtonnb(7.0);
fm2.setCtofnb(8.0);
fm2.initialize();
ctx2 = ch.CharmmContext(fm2)

crd = ch.CharmmCrd("../../test/data/waterbox.crd")

ctx0.setCoordinates(crd)
ctx0.assignVelocitiesAtTemperature(300)
ctx0.calculateForces(False, True, True)

ctx1.setCoordinates(crd)
ctx1.assignVelocitiesAtTemperature(300)
ctx1.calculateForces(False, True, True)

ctx2.setCoordinates(crd)
ctx2.assignVelocitiesAtTemperature(300)
ctx2.calculateForces(False, True, True)

integrator = ch.VMMSVelocityVerletIntegrator(0.001)
integrator.setSimulationContexts([ctx0, ctx1, ctx2])
integrator.setSoluteAtoms([0,1,2])

#integrator.propagate(1000)

#subscriber = ch.NetCDFSubscriber("vv_waterbox_5psfs.nc", ctx)
#ctx.subscribe(subscriber)
#stateSub = ch.StateSubscriber("vv_waterbox_5psfs.txt", ctx)
#ctx.subscribe(stateSub)
#dualSub = ch.DualTopologySubscriber("pe_vv_waterbox_5psfs.txt", ctx)
#ctx.subscribe(dualSub)
#
#integrator.setReportSteps(100)
#print("Starting propagation")
#integrator.propagate(4000)
#utils.calculateMeanAndVarianceExt("pe_vv_waterbox_5psfs.txt", 5)

