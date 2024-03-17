# IDEA : benchmark performances of apocharmm

# Benchmark systems: 
# - DHFR
# - one (or several) waterboxes
# - apoE
# - STMV

from charmm import apocharmm as ac
dataPath = "../test/data/"
samarPath = "/u/samar/Documents/git/test_gpu/"


# Run dhfr as a first test (also named JAC)
paramFiles = [dataPath + "par_all36m_prot.prm", dataPath + "toppar_water_ions.str"]
prm = ac.CharmmParameters(paramFiles)
psf = ac.CharmmPSF(samarPath + "JAC.psf")

fm = ac.ForceManager(psf, prm)

boxdims = [61.6447 for i in range(3)]
fm.setBoxDimensions(boxdims)
fm.setFFTGrid(64,64,64)

fm.setCutoff(12.0)
fm.setCtonnb(9.0)
fm.setCtofnb(10.0)

fm.initialize()

crd = ac.CharmmCoordinates(samarPath + "JAC.crd")

ctx = ac.CharmmContext(fm)
ctx.setCoordinates(crd)


mini = ac.Minimizer()
mini.setSimulationContext(ctx)
mini.minimize(1000)


integrator = ac.LangevinThermostatIntegrator(.001, 5.0, 300.0)
integrator.setSimulationContext(ctx)
integrator.propagate(1000)





