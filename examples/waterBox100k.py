import charmm.apocharmm as ch
import utils

print('Reading PSF...')
psf = ch.CharmmPSF("../test/data/l1.pert.100k.psf")
print('Reading parameters...')
prm = ch.CharmmParameters(["../test/data/toppar_water_ions.str", "../test/data/par_all36_cgenff.prm"])
fm = ch.ForceManager(psf, prm)

boxLength = 99.64716
fm.setBoxDimensions([boxLength, boxLength, boxLength])
fm.setFFTGrid(100, 100, 100)
fm.setKappa(0.34)
fm.setPmeSplineOrder(4)
fm.setCutoff(10.0)
fm.setCtonnb(7.0)
fm.setCtofnb(9.0)

fm.initialize()

ctx = ch.CharmmContext(fm)
#ctx = ch.CharmmContext(fm)
print('Redading coordinates...')
crd = ch.CharmmCrd("../test/data/nvt_equil.100k.cor")
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)
#ctx.calculateForces(False, True, True)
#quit()
integrator = ch.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
#integrator.propagate(1000)

#subscriber = ch.NetCDFSubscriber("out/waterbox_100k.nc")
#ctx.subscribe(subscriber)
#stateSub = ch.StateSubscriber("vv_pert_100k.txt", ctx)
#ctx.subscribe(stateSub)
#dualSub = ch.DualTopologySubscriber("dual_vv_pert_100k.txt", ctx)
#ctx.subscribe(dualSub)

#integrator.setReportSteps(1000)
print("Starting propagation")
#integrator.propagate(250000)
integrator.propagate(100000)
