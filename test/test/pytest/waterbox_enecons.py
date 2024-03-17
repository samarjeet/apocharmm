# Long runs to test energy conservation

import charmm.apocharmm as ac
import argparse

testDataPath = "/u/aviatfel/dev/testapo3/test/data/"
prmfile = testDataPath + "toppar_water_ions.str"
psffile = testDataPath + "waterbox.psf"
restartFile = testDataPath + "restart/waterbox.npt.restart"

boxDim = [48.9342, 48.9342, 48.9342]
nWarmupSteps = 5000
friction = 0.0
bathTemperature = 300.0
pistonMass = 0.0



# Return fm and ctx for a new run
def setupSimulation():
    prm = ac.CharmmParameters([prmfile])
    psf = ac.CharmmPSF(testDataPath + "waterbox.psf")   
    fm = ac.ForceManager(psf, prm)
    fm.setBoxDimensions(boxDim)
    ctx = ac.CharmmContext(fm)
    ctx.readRestart(restartFile)
    ctx.assignVelocitiesAtTemperature(bathTemperature)
    return fm, ctx


## Get args
parser = argparse.ArgumentParser(description="Run a simulation")
parser.add_argument("integrator", type=str, help="Integrator to use (langevinthermostat, velocityverlet or langevinpiston)")
parser.add_argument("timestep", type=float, help="Timestep to use (in ps)")
parser.add_argument("nsteps", type=int, help="Number of steps to run")
parser.add_argument("-f", "--friction", type=float, help="Friction coefficient to use (in ps^-1). By default: 0.0")
parser.add_argument("-t", "--temperature", type=float, help="Bath temperature to use (in K). By default: 300.0")
parser.add_argument("-pm", "--pistonmass", type=float, help="Piston mass to use (in amu). By default: 0.0")
args = parser.parse_args()

if args.temperature:
    bathTemperature = args.temperature
if args.friction:
    friction = args.friction
if args.pistonmass:
    pistonMass = args.pistonmass


fm,ctx = setupSimulation()
if args.integrator == "langevinthermostat":
    integrator = ac.LangevinThermostatIntegrator(args.timestep)
    integrator.setFriction(friction)
    integrator.setBathTemperature(bathTemperature)
    integrator.setSimulationContext(ctx)
elif args.integrator == "langevinpiston": 
    integrator = ac.LangevinThermostatIntegrator(args.timestep)
    integrator.setFriction(friction)
    integrator.setBathTemperature(bathTemperature)
    integrator.setSimulationContext(ctx)
    integrator.propagate(nWarmupSteps)
    print("Warmup complete")

    integrator = ac.LangevinPistonIntegrator(args.timestep)
    integrator.setCrystalType(ac.CRYSTAL.CUBIC)
    integrator.setPistonMass([pistonMass])
    integrator.setPistonFriction(friction)
    if friction == 0.0:
        integrator.setNoseHooverFlag(False)
    else: 
        integrator.setPistonMass([pistonMass])
        integrator.setBathTemperature(bathTemperature)
    ctx.assignVelocitiesAtTemperature(bathTemperature)
    integrator.setSimulationContext(ctx)


outfilenamebase = f"waterbox.{args.integrator}.{args.timestep*1000}fs.pg{friction}.nocom"
sub = ac.StateSubscriber(f"out/{outfilenamebase}.state", 1000)
dcdSub = ac.DcdSubscriber(f"out/{outfilenamebase}.dcd", 1000)
integrator.subscribe([sub, dcdSub])

integrator.setDebugPrintFrequency(1000)
integrator.propagate(args.nsteps)
