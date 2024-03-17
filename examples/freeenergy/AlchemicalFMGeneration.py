import charmm.apocharmm as ac 

# Idea: verify we can do electrostatic deactivation using apocharmm API
# We'll test this on a simple solvated molecule from the freesolv database

path = "/u/aviatfel/work/apocharmm/freeenergy/freesolv/"
molid = 1034539
boxdim = 35.0
nAtomsInMolecule = 21


prmlist = [ f'/u/aviatfel/toppar/toppar_water_ions.str',
            f'/u/aviatfel/toppar/par_all36_cgenff.prm',
            f'{path}/setup/cgenff/mobley_{molid}.str' 
]
psffile = f'{path}solvated/psf/{molid}.solv.psf'
crdfile = f'{path}solvated/crd/{molid}.solv.crd'

prm = ac.CharmmParameters(prmlist)
psf = ac.CharmmPSF(psffile)
crd = ac.CharmmCrd(crdfile)

fm = ac.ForceManager(psf, prm)
alchemicalFactory = ac.AlchemicalForceManagerGenerator(fm)
alchemicalFactory.setAlchemicalRegion([i for i in range(nAtomsInMolecule)])

electrostaticAnnihilationWindows = [[1.0, 0.0],
                                    [0.5, 0.0],
                                    [0.0, 0.0]]

mbarForceManager = ac.MBARForceManager()
for (lElec, lVdw) in electrostaticAnnihilationWindows:
    fm = alchemicalFactory.generateForceManager(lElec, lVdw)
    mbarForceManager.addForceManager(fm)
selectionVector = [1.0, 0.0, 0.0]
mbarForceManager.setSelectorVec(selectionVector)
mbarForceManager.setBoxDimensions([boxdim, boxdim, boxdim])

ctx = ac.CharmmContext(mbarForceManager)
ctx.setCoordinates(crd)
ctx.assignVelocitiesAtTemperature(300)

integrator = ac.LangevinThermostatIntegrator(0.002, 300, 5.0)
integrator.setSimulationContext(ctx)


integrator.setDebugPrintFrequency(100)
integrator.propagate(1000)


