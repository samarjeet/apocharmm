# SCript to run a single SAI (Stefan's approach) trajectory
import charmm.apocharmm as ac
from charmm.experimental import *
import argparse as ap
import os

# Next: run this for real !

# Input parsing
parser = ap.ArgumentParser(
    description="Given a molID from the FreeSolv database, a phase (solvated or vacuum), an alchemical window number, runs the simulation for the given alch. window, generates the MBAR-parsable file.\n")
parser.add_argument("-id", "--molid", type=int, default=9979854,
                    help="MolID from FreeSolv. Default: 9979854")
parser.add_argument("--phase", type=str, default="solvated",
                    help="Phase of the simulation (vacuum or solvated). Default: solvated")
args = parser.parse_args()

molid = args.molid
phase = args.phase
if phase not in ["solvated", "vacuum"]:
    raise ValueError(f"Phase must be either 'solvated' or 'vacuum'")

###################
# To be modified
###################
# Run prms
boxdim = 35.0 if phase == "solvated" else 520.0
ctonnb = 10.0 if phase == "solvated" else 250.0
ctofnb = 12.0 if phase == "solvated" else 254.0
cutoff = 14.0 if phase == "solvated" else 255.0
nSteps = 5000  # Number of production steps
nEquilSteps = 2000  # Number of equilibration steps
temperature = 303.15
langevinThermostatFriction = 12.0
# Paths
prmParentDir = "sai_params"
if phase == "vacuum":
    prmParentDir += "_vac"
outputDir = "out/sai"
dataDir = "../test/data"
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)
################################

#################################
# Cycle preparation
#################################
# Find number of atoms of ligand and number of heavy atoms
crd = CRD(f"{dataDir}/{molid}.{phase}.crd")
hydrogenCount, heavyAtomCount = 0, 0
for (i, atom) in enumerate(crd.getAtoms()):
    if i == 0:
        resName = atom.getResidueName()
    if atom.getResidueName() != resName:
        break
    if atom.getAtomName()[0] == "H":
        hydrogenCount += 1
    else:
        heavyAtomCount += 1
ligandAtoms = [i for i in range(0, heavyAtomCount + hydrogenCount)]
# Find number of alchemical states (4 for elec deactiv, 2 for vdw deactiv of all
# hydrogens, 1 per vdw deactiv of heavy atom)
numberOfAlchemicalStates = 4 + 2 + heavyAtomCount

# Print summary
print(
    f"MolID: {molid}, phase: {phase}\n" +
    f"ligandAtoms: {ligandAtoms}, \nhydrogenCount: {hydrogenCount}, heavyAtomCount: {heavyAtomCount}" +
    f"\nnumber of alchemical states: {numberOfAlchemicalStates}")

#########################################
# Prepare compositeFM and MBAR outputs
###########################################
# Create all alch windows' force managers
fmList = []
for stateNumber in range(0, numberOfAlchemicalStates):
    psf = ac.CharmmPSF(f"{prmParentDir}/intst{stateNumber}/ligand.psf")
    prmlist = [f"{dataDir}/par_all36_cgenff.prm", f"{dataDir}/toppar_water_ions.str",
               f"{dataDir}/mobley_{molid}.str"]
    if phase == "vacuum":
        prmlist.pop(1)
    if stateNumber > 3:  # For the electrostatic deactivation steps, no dummy parameter has been generated
        prmlist.insert(0,
                       f"{prmParentDir}/intst{stateNumber}/dummy_parameters.prm")
    prm = ac.CharmmParameters(prmlist)
    fm = ac.ForceManager(psf, prm)
    fmList.append(fm)


# Create composite force manager
fmMbar = ac.MBARForceManager(fmList)
fmMbar.setBoxDimensions([boxdim, boxdim, boxdim])
fmMbar.setCtonnb(ctonnb)
fmMbar.setCtofnb(ctofnb)
fmMbar.setCutoff(cutoff)
if phase == "vacuum":
    fmMbar.setFFTGrid(4, 4, 4)
    fmMbar.setKappa(0.0)

ctx = ac.CharmmContext(fmMbar)
crd = ac.CharmmCrd(
    f"{dataDir}/{molid}.{phase}.crd")
ctx.setCoordinates(crd)

selectorVec = [0 for i in range(0, numberOfAlchemicalStates)]
selectorVec[0] = 1
fmMbar.setSelectorVec(selectorVec)

# Prepare minimizer
minimizer = ac.Minimizer()
minimizer.setSimulationContext(ctx)

# Prepare integrators
integrator = ac.LangevinThermostatIntegrator(
    .001, temperature, langevinThermostatFriction)
integrator.setSimulationContext(ctx)


for stateNumber in range(numberOfAlchemicalStates):
    print(f"Running state {stateNumber}")
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(temperature)
    minimizer.minimize()
    selectorVec = [0 for i in range(numberOfAlchemicalStates)]
    selectorVec[stateNumber] = 1
    print("Setting selector vec...")
    fmMbar.setSelectorVec(selectorVec)

    # NPT equilibration
    # ====================
    print("  NPT equilibration...")
    # NPT integrator for equilibration
    isobaricIntegrator = ac.LangevinPistonIntegrator(.002)
    isobaricIntegrator.setPistonFriction(12.0)
    isobaricIntegrator.setSimulationContext(ctx)
    isobaricIntegrator.setBathTemperature(temperature)
    isobaricIntegrator.setCrystalType(ac.CRYSTAL.CUBIC)
    isobaricIntegrator.setPistonMass([500.0])
    isobaricIntegrator.propagate(nEquilSteps)

    # Production run
    # ==================
    print("  Production run...")
    dcdsub = ac.DcdSubscriber(
        f"{outputDir}/traj.{molid}.{stateNumber}.{phase}.dcd", 1000)
    mbarsub = ac.MBARSubscriber(
        f"{outputDir}/mbar.{molid}.{stateNumber}.{phase}.out", 100)
    integrator.subscribe([mbarsub, dcdsub])
    integrator.propagate(nSteps)
    integrator.unsubscribe(mbarsub)
    integrator.unsubscribe(dcdsub)
#    del mbarsub

# Different approach: run ALL equilibrations, then rull ALL production, and all of that independently
