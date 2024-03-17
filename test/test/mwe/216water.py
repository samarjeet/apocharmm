import apocharmm as ac

#pdb = ac.PDB('../../step3_pbcsetup.pdb')
#psf = ac.CharmmPSF('../../step3_pbcsetup.psf')
pdb = ac.PDB('water216.pdb')
psf = ac.CharmmPSF('water216.psf')
prm = ac.CharmmParameters('toppar_water_ions.str')

fm = ac.ForceManager(psf, prm)
fm.setFFTGrid(20,20,20)
fm.setBoxDimensions([20,20,20])
fm.setCutoff(10.0)
fm.setCtonnb(7.0)
fm.setCtofnb(8.0)
fm.initialize()

ctx = ac.CharmmContext(fm)
#ctx.setCoordinates(crd)
ctx.setCoordinates(pdb)
ctx.assignVelocitiesAtTemperature(300.)

minim = ac.Minimizer()
minim.setSimulationContext(ctx)
print( ctx.calculatePotentialEnergy(True,True))

print('minim starting')
minim.minimize(10)
print('minim done')
print( ctx.calculatePotentialEnergy(True,True))

integrator = ac.VelocityVerletIntegrator(0.001)
statesub = ac.StateSubscriber('out.txt', ctx)
integrator.setReportSteps(5000)
print('starting propag')
integrator.propagate(10000)
print('propag done')
