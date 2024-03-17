# Idea : test file for various NPT simulations
#
# We should test :
# - For P1 PBC,
#    * for each geometry (cubic, orthorhombic, triclinic)
#       _ the energy conservation (NPH)
#       _ the pressure/volume conservation (NPH, NPT)
#       _ the temperature conservation (NPT)
#
# These same tests should then (later) be performed for P2 PBC; for NPAT and NPgammaT


from io import TextIOWrapper, BytesIO
import numpy as np
import pytest
import sys
import charmm.apocharmm as apo

testDataPath = "../data/"
prmfile = testDataPath + "toppar_water_ions.str"
psffile = testDataPath + "waterbox.psf"
restartFile = testDataPath + "restart/waterbox.npt.restart"

nSteps = int(101)
boxDim = [48.9342, 48.9342, 48.9342]
temperature = 300.0
timestepPs = 0.002
subscriberPrintFrequency = 10
debugPrintFrequency = 100

simulationCellGeometries = [apo.CRYSTAL.CUBIC,
                            apo.CRYSTAL.TETRAGONAL, apo.CRYSTAL.ORTHORHOMBIC]
simulationCellGeometriesIDs = ["CUBIC", "TETRAGONAL", "ORTHORHOMBIC"]

simulationCellGeometries = [simulationCellGeometries[0]]
simulationCellGeometriesIDs = [simulationCellGeometriesIDs[0]]


# Prepare context, force manager using waterbox inputs.
# Uses restart file.
def setupSimulation():
    prm = apo.CharmmParameters([prmfile])
    psf = apo.CharmmPSF(testDataPath + "waterbox.psf")
    fm = apo.ForceManager(psf, prm)
    fm.setBoxDimensions(boxDim)
    ctx = apo.CharmmContext(fm)
    ctx.readRestart(restartFile)
    ctx.assignVelocitiesAtTemperature(temperature)
    return fm, ctx


def runSimulation(capsys, integrator, outputFileName):
    fm, ctx = setupSimulation()
    integrator.setSimulationContext(ctx)
    integrator.setPistonFriction(0.0)
    integrator.setNoseHooverFlag(False)

    stateSub = apo.StateSubscriber(outputFileName, subscriberPrintFrequency)
    stateSub.setReportFlags(
        "potentialenergy, kineticenergy, totalenergy, temperature, volume, boxsizecomponents, pressurescalar")
    integrator.subscribe([stateSub])
    integrator.propagate(nSteps)
    captured = capsys.readouterr()
    print("here is the output\n\n\n\n\n", captured.out)

    with open("out.txt", 'w') as f:
        f.write(captured.out)

    return captured.out


@pytest.fixture(params=simulationCellGeometries, ids=simulationCellGeometriesIDs)
def setupIntegrator(request):
    integrator = apo.LangevinPistonIntegrator(timestepPs)
    integrator.setCrystalType(request.param)
    integrator.setDebugPrintFrequency(debugPrintFrequency)
    return integrator, request.param


'''
# Arrange = prepare the integrator, the simulation context, the force manager. 
#          This is where several cases (=all three geometries) will be tested
# Act = run the simulation
# Assert = load output file, check energy conservation
def testNPHruns(capsys,setupIntegrator):
    integrator = setupIntegrator[0]
    geometry = str(setupIntegrator[1]).split(".")[-1].lower()
    outputFileName = f"waterbox.nph.{geometry}.state"
    runSimulation(capsys, integrator, outputFileName)




    # Check all quantities that should be conserved. 
    # If they aren't, add to errorList
    errors = []
    # Check energy conservation
    # Check pressure conservation
    # Check box dimension
    assert not errors, "errors occured:\n{}".format("\n".join(errors))



#def test_myoutput(capsys):  # or use "capfd" for fd-level
#    print("hello")
#    sys.stderr.write("world\n")
#    captured = capsys.readouterr()
#    assert captured.out == "hello\n"
#    assert captured.err == "world\n"
#    print("next")
#    captured = capsys.readouterr()
#    assert captured.out == "next\n"
'''


# ALRIGHT This is taking me way too long.
# 1. Run all you need with good output file names
# 2. Perform assertions.
# To make sure this approach will work, I'll run a simple stupid thing, with file and stdout output, capture stdout, see if I can make tests on both.


def generateData():
    x = np.arange(10)
    for i in range(10):
        print(x[i])
    xlines = [str(i*2) + "\n" for i in x]
    with open("double.txt", 'w') as f:
        f.writelines(xlines)

    print("DATA GENERATED")


# setup the environment
old_stdout = sys.stdout
sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)

# do some writing (indirectly)
generateData()
write('blub')

# get output
sys.stdout.seek(0)      # jump to the start
out = sys.stdout.read()  # read output

# restore stdout
sys.stdout.close()
sys.stdout = old_stdout

# do stuff with the output
print(out.upper())
