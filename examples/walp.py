import apocharmm as ch

prm = ch.CharmmParameters(["../test/data/par_all36_prot.prm", "../test/data/par_all36_lipid.prm", "../test/data/toppar_water_ions.str"])
psf = ch.CharmmPSF("../test/data/walp.psf")
crd = ch.CharmmCrd("../test/data/walp.crd")

fm = ch.ForceManager(psf, prm)
fm.setBoxDimensions([53.4630707, 53.4630707, 80.4928487]);
fm.setFFTGrid(48, 48, 48);
fm.setKappa(0.34);
fm.setCutoff(9.0);
fm.setCtonnb(7.0);
fm.setCtofnb(8.0);
fm.initialize();


ctx = ch.CharmmContext(fm)
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

#quit()

integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)

subscriber = ch.DcdSubscriber("out/vv_walp.dcd", ctx)
ctx.subscribe(subscriber)
#stateSub = ch.StateSubscriber("vv_walp.txt", ctx)
#ctx.subscribe(stateSub)
#
#subscriber = ch.NetCDFSubscriber("vv_walp.nc", ctx)
#ctx.subscribe(subscriber)
integrator.setReportSteps(10000)

integrator.propagate(1000000)
