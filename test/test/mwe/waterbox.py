import apocharmm as ac

crd = ac.CharmmCrd('../data/waterbox.crd')
psf = ac.CharmmPSF('../data/waterbox.psf')
prm = ac.CharmmParameters('../data/toppar_water_ions.str')

fm = ac.ForceManager(psf, prm)
fm.setFFTGrid(48,48,48)
fm.setBoxDimensions([50.,50.,50.])
fm.setCutoff(10.0)
fm.setCtonnb(7.0)
fm.setCtofnb(8.0)
fm.initialize()

ctx = ac.CharmmContext(fm)
#ctx.setCoordinates(crd)
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

minim = ac.Minimizer()
#minim.setSimulationContext(ctx)
#print( ctx.calculatePotentialEnergy(True,True))
#
#print('minim starting')
#minim.minimize(100)
#print('minim done')
#print( ctx.calculatePotentialEnergy(True,True))

integrator = ac.VelocityVerletIntegrator(0.001)
integrator.setSimulationContext(ctx)
statesub = ac.StateSubscriber('out.txt', ctx)
ctx.subscribe(statesub)
integrator.setReportSteps(1)
print('starting propag')
integrator.propagate(10)
print('propag done')


