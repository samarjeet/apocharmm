# Produce equilibrated (NPT) restart file for the systems I need
import charmm.apocharmm as ac
# setup data location
testDataPath = "/u/aviatfel/dev/testapo3/test/data/"

print("using ", ac.__file__)


###### HERE BE THE VARIABLES TO CHANGE #####
# Input file names
psffile = "waterbox.psf"
# Give empty string to crd if restart used and vice versa
crdfile = ""
restartfile = "restart/heat_waterbox.restart"
# Comment whatever is not needed
prmlist = [
      "toppar_water_ions.str",
      #"par_all36m_prot.prm", 
      #"par_all36_lipid.prm"
]

# Simulation parameters
ensemble = "NPT" # Choose between NVT (Langevin thermostat) and NPT (Langevin piston) 
temperatureVelocities = 300.0
bathTemperature = 300.0
friction = 5.0
boxSize = [50., 50., 50.]
cutoffPair = 12.0
cutoffNonbonded = 10.0
cutonNonbonded = 9.0
pistonMass = 500.0

timestep = 0.002
restartfreq = 1000
outputfreq = 1000
nSteps = int(1e6)
crystalType = ac.CRYSTAL.CUBIC
#################################################



###### EVERYTHING HEREUNDER SHOULD NOT NEED BE TOUCHED #####
prmlist = [ testDataPath + x for x in prmlist]
prm = ac.CharmmParameters(prmlist)
psf = ac.CharmmPSF(testDataPath + psffile)
fm = ac.ForceManager(psf, prm)
fm.setBoxDimensions(boxSize)
fm.setCutoff(cutoffPair)
fm.setCtofnb(cutoffNonbonded)
fm.setCtonnb(cutonNonbonded)

ctx = ac.CharmmContext(fm)
if restartfile == "":
    crd = ac.CharmmCrd(testDataPath + crdfile)
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(temperatureVelocities)
elif crdfile == "":
    ctx.readRestart(testDataPath + restartfile)
else:
      print("ERROR: You can't have both a restart and a coordinate file!")
      exit()

if ensemble == "NVT":
      integrator = ac.LangevinThermostatIntegrator(timestep, bathTemperature, friction)
elif ensemble == "NPT":
      integrator = ac.LangevinPistonIntegrator(timestep)
      integrator.setCrystalType(crystalType)
      integrator.setBathTemperature(bathTemperature)
      integrator.setPistonFriction(friction)
      integrator.setPistonMass(pistonMass)

integrator.setSimulationContext(ctx)
sub = ac.StateSubscriber("equilibratedState.txt", outputfreq)
sub.setReportFlags()
restartsub = ac.RestartSubscriber("equilibratedRestart.res", restartfreq)
integrator.subscribe([sub, restartsub])

integrator.propagate(nSteps)



