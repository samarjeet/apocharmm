import apocharmm as ch

prm = ch.CharmmParameters(["../../test/data/par_all36_prot.prm", "../../test/data/toppar_water_ions.str"])
#psf = ch.CharmmPSF("~/Documents/git/test_gpu/factorix/factorix.psf")
psf = ch.CharmmPSF("factorix.psf")
crd = ch.CharmmCrd("factorix.cor")

fm = ch.ForceManager(psf, prm)
boxLength = 90
fm.setBoxDimensions([boxLength, boxLength, boxLength]);
fm.setFFTGrid(128, 128, 128);
fm.setPmeSplineOrder(6)
fm.setKappa(0.34);
fm.setCutoff(9.0);
fm.setCtonnb(7.0);
fm.setCtofnb(8.0);
fm.initialize();


ctx = ch.CharmmContext(fm)
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

#subscriber = ch.NetCDFSubscriber("vv_walp.nc", ctx)
#ctx.subscribe(subscriber)
#stateSub = ch.StateSubscriber("vv_walp.txt", ctx)
#ctx.subscribe(stateSub)
#
#subscriber = ch.NetCDFSubscriber("vv_walp.nc", ctx)
#ctx.subscribe(subscriber)
#integrator.setReportSteps(5000)

integrator.propagate(1000)
