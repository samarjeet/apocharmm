# Objective: compute solvation free energy for the molecules of the FreeSolv database

# Consists in 2 parts : 
# - compute the annihilation free energy in gas phase (turning off elec and vdw for molecule in vacuum)
# - compute the annihilation free energy in water (turning off elec and vdw for molecule solvated in water)


# we have the molecules psf and crd files, prm files. 
# we need to generate a solvated system (using charmm ?)
# 

import charmm.apocharmm as ac
import argparse as ap 

def getNumAtomsInMolecule(molid):
    crdname = f"/u/aviatfel/work/apocharmm/freeenergy/freesolv/vacuum/crd/{molid}.vacuum.crd"
    with open(crdname, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("*"):
            continue
        l = l.split()
        return int(l[0])

parser = ap.ArgumentParser(description="Run FreeSolv database molecules simulations to compute solvation free energy")
parser.add_argument("molid", type=int, help="Molecule ID in FreeSolv database")
parser.add_argument("runtype", type=str, help="aq or vac (both types will have to be run)")

args = parser.parse_args()
molid = args.molid
runtype = args.runtype
numAtomsInMol = getNumAtomsInMolecule(molid)

#HARDCODING FOR NOW
#molid = 1034539 # This should be taken as an argument
#numAtomsInMol = 22 # We'll need a function for that !
#runtype = "aq" # "aq" or "vac"

# paths
path = "/u/aviatfel/work/apocharmm/freeenergy/freesolv/"
prmpath = path + "setup/cgenff/"

#### Free energy computation parameters ####
lambdaElectrostaticSchedule = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#lambdaVdwSchedule = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, .95, .975, .989, .996, .999, 1.0]
#lambdaVdwSchedule = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, .95, .975,  0.993, .999,1.0]
#lambdaVdwSchedule = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.975, 0.997, 0.999, 0.9995, 1.0]
lambdaVdwSchedule = [0.0, 0.2878, 0.7782, 0.9141, 0.9658, 0.9872, 0.996, 0.999, 0.9995, 0.9999, 1.0] 

####  Parameters that probably won't change ####
temperature = 300.0
friction = 5.0
boxdim = 35.0
nOutputFreq = 1000


if runtype == "aq":
    prmlist = [ f'/u/aviatfel/toppar/toppar_water_ions.str',
                f'/u/aviatfel/toppar/par_all36_cgenff.prm',
                f'{prmpath}mobley_{molid}.str' 
            ]
    prmListNoNonBonded = [
        f'/u/aviatfel/toppar/toppar_water_ions.str',
        f'{prmpath}par_all36_cgenff_{molid}_novdw.prm',
        f'{prmpath}mobley_{molid}.str']

    psffile = f'{path}solvated/psf/{molid}.solv.psf'
    crdfile = f'{path}solvated/crd/{molid}.solv.crd'

elif runtype == "vac":
    prmlist = [ f'/u/aviatfel/toppar/par_all36_cgenff.prm',
                f'{prmpath}mobley_{molid}.str' ]
    prmListNoNonBonded = [ f'{prmpath}par_all36_cgenff_{molid}_novdw.prm',
                           f'{prmpath}mobley_{molid}.str']
    psffile = f'{path}vacuum/psf/{molid}.vacuum.psf'
    crdfile = f'{path}vacuum/crd/{molid}.vacuum.crd'


if runtype == "vac": 
    boxdim = 520.0
    friction = 0.0


crd = ac.CharmmCrd(crdfile)

print("Current run type : ", runtype)

def switchOffElectrostatics(nWarmupSteps = 10000, nProdSteps = 100000):
    psf = ac.CharmmPSF(psffile)
    prm = ac.CharmmParameters(prmlist)
    fmOn = ac.ForceManager(psf, prm)
    fmFactory = ac.AlchemicalForceManagerGenerator(fmOn)
    fmFactory.setAlchemicalRegion([i for i in range(numAtomsInMol)])
    fmOff = fmFactory.generateForceManager(0.0, 1.0)

    FEPEIfm = ac.FEPEIForceManager()
    FEPEIfm.addForceManager(fmOff)
    FEPEIfm.addForceManager(fmOn)
    FEPEIfm.setBoxDimensions([boxdim, boxdim, boxdim])
    FEPEIfm.setLambdas(lambdaElectrostaticSchedule)

    if runtype == "vac":
        FEPEIfm.setCutoff(255.)
        FEPEIfm.setCtofnb(254.)
        FEPEIfm.setCtonnb(250.)
        FEPEIfm.setFFTGrid(4,4,4)
        FEPEIfm.setKappa(0.)

    ctx = ac.CharmmContext(FEPEIfm)
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(temperature)

    for l in lambdaElectrostaticSchedule:
        print(f"Elec : Lambda = {l}")
        FEPEIfm.setLambda(l)
        langevinThermostat = ac.LangevinThermostatIntegrator(0.002, temperature, friction)
        langevinThermostat.setSimulationContext(ctx)
        # Equilibrate a bit ? 
        langevinThermostat.propagate(nWarmupSteps)


        # Production
        outputname = f"out/fep_elec_{l}.{molid}.{runtype}"
        if suffix != "":
            outputname += f".{suffix}"
        outputname += ".dat"
        FEPSubscriber = ac.FEPSubscriber(outputname, 1000)
        langevinThermostat.subscribe([FEPSubscriber])
        langevinThermostat.propagate(nProdSteps)

def switchOffVdw(nWarmupSteps=10000, nProdSteps=100000, suffix=""):
# Starting from electrostatics already off !!!
    psf = ac.CharmmPSF(psffile)
    prmVdW = ac.CharmmParameters(prmlist)
    prmNoVdW = ac.CharmmParameters(prmListNoNonBonded)
    fmVdw = ac.ForceManager(psf, prmVdW)
    fmVdw.setVdwType(ac.NonBondedType.VDW_DBEXP)
    fmNoVdw = ac.ForceManager(psf, prmNoVdW)
    fmNoVdw.setVdwType(ac.NonBondedType.VDW_DBEXP)
    # Scale charges to 0 in both cases
    fmFactory = ac.AlchemicalForceManagerGenerator(fmVdw)
    fmFactory.setAlchemicalRegion([i for i in range(numAtomsInMol)])
    fmVdwNoElec = fmFactory.generateForceManager(0.0, 1.0)
    fmFactory = ac.AlchemicalForceManagerGenerator(fmNoVdw)
    fmFactory.setAlchemicalRegion([i for i in range(numAtomsInMol)])
    fmNoVdwNoElec = fmFactory.generateForceManager(0.0, 1.0)

    FEPEIfm = ac.FEPEIForceManager()
    FEPEIfm.addForceManager(fmVdwNoElec)
    FEPEIfm.addForceManager(fmNoVdwNoElec)
    FEPEIfm.setBoxDimensions([boxdim, boxdim, boxdim])
    FEPEIfm.setLambdas(lambdaVdwSchedule)

    if runtype == "vac":
        FEPEIfm.setCutoff(255.)
        FEPEIfm.setCtofnb(254.)
        FEPEIfm.setCtonnb(250.)
        FEPEIfm.setKappa(0.0)
        FEPEIfm.setFFTGrid(6,6,6)

    ctx = ac.CharmmContext(FEPEIfm)
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(temperature)

    for l in lambdaVdwSchedule:
        print(f"Vdw : Lambda = {l}")
        FEPEIfm.setLambda(l)
        ctx.setCoordinates(crd)
        langevinThermostat = ac.LangevinThermostatIntegrator(0.002, temperature, friction)
        langevinThermostat.setSimulationContext(ctx)
        # Equilibrate a bit ? 
        langevinThermostat.propagate(nWarmupSteps)

        # Production
        outputname = f"out/fep_noelec_vdw_{l}.{molid}.{runtype}"
        if suffix != "":
            outputname += f".{suffix}"
        outputname += ".dat"
        FEPSubscriber = ac.FEPSubscriber(outputname, nOutputFreq)
        langevinThermostat.subscribe([FEPSubscriber])
        langevinThermostat.propagate(nProdSteps)

## MAIN RUN ##
#switchOffElectrostatics()
switchOffVdw(nProdSteps=int(1e6), suffix="1mkoenig")













