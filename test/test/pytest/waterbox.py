# Test file for the waterbox example

import charmm.apocharmm as ac
import numpy as np
import pytest

testDataPath = "/u/aviatfel/dev/testapo3/test/data/"
prmfile = testDataPath + "toppar_water_ions.str"
psffile = testDataPath + "waterbox.psf"
restartFile = testDataPath + "restart/waterbox.npt.restart"
nSteps = int(1e4)

boxDim = [48.9342, 48.9342, 48.9342]

# Objective: monitor several things over a waterbox simulation, starting from a
# restart file (already equilibrated):
# - Energy conservation (NVE)
# - Temperature conservation (NVT)
# - Pressure, Temperature, density conservation (NPT)


@pytest.fixture
def setupSimulation(request):
    prm = ac.CharmmParameters([prmfile])
    psf = ac.CharmmPSF(testDataPath + "waterbox.psf")   
    fm = ac.ForceManager(psf, prm)
    fm.setBoxDimensions(boxDim)
    ctx = ac.CharmmContext(fm)
    ctx.readRestart(restartFile)
    return fm, ctx

# NVE simulation
def testEnergyConservation(setupSimulation):
    langevinThermostatNVE = ac.LangevinThermostatIntegrator(0.002)
    langevinThermostatNVE.setFriction(0.0)
    langevinThermostatNVE.setBathTemperature(300.0)
    langevinThermostatNVE.setSimulationContext(setupSimulation[1])

    NVESub = ac.StateSubscriber("waterbox_nve.state", 100)
    NVESub.setReportFlags("potentialenergy, kineticenergy, totalenergy, temperature")
    langevinThermostatNVE.subscribe([NVESub])

    langevinThermostatNVE.propagate(nSteps)

    # Analyse output
    data = np.loadtxt("waterbox_nve.state", unpack=True)
    etot = data[3]
    etotAve = np.average(etot)
    etotStd = np.std(etot)
    print(f"etotAve = {etotAve}, etotStd = {etotStd}")
    assert etotStd/etotAve < 1e-2

# NVT simulation
def testTemperatureConservation(setupSimulation):
    langevinThermostatNVT = ac.LangevinThermostatIntegrator(0.002)
    langevinThermostatNVT.setFriction(5.0)
    langevinThermostatNVT.setBathTemperature(300.0)
    langevinThermostatNVT.setSimulationContext(setupSimulation[1])

    NVTSub = ac.StateSubscriber("waterbox_nvt.state", 100)
    NVTSub.setReportFlags("potentialenergy, kineticenergy, totalenergy, temperature")
    langevinThermostatNVT.subscribe([NVTSub])

    langevinThermostatNVT.propagate(nSteps)

    # Analyse output
    data = np.loadtxt("waterbox_nvt.state", unpack=True)
    temperature = data[4]
    temperatureAve = np.average(temperature)
    temperatureStd = np.std(temperature)
    print(f"temperatureAve = {temperatureAve}, temperatureStd = {temperatureStd}")
    assert temperatureStd/temperatureAve < 1e-2
    assert abs(temperatureAve - 300.0) < 1.0

# NPT simulation
def testPressureConservation(setupSimulation):
    langevinPistonNPT = ac.LangevinPistonIntegrator(0.002)
    langevinPistonNPT.setPistonFriction(5.0)
    langevinPistonNPT.setBathTemperature(300.0)
    langevinPistonNPT.setSimulationContext(setupSimulation[1])

    NPTSub = ac.StateSubscriber("waterbox_npt.state", 100)
    NPTSub.setReportFlags("potentialenergy, kineticenergy, totalenergy, temperature, pressurescalar, volume, density")
    langevinPistonNPT.subscribe([NPTSub])

    langevinPistonNPT.propagate(10*nSteps)

    # Analyse output
    data = np.loadtxt("waterbox_npt.state", unpack=True)
    pressure = data[5]
    pressureAve = np.average(pressure)
    pressureStd = np.std(pressure)
    print(f"pressureAve = {pressureAve}, pressureStd = {pressureStd}")
    assert abs(pressureAve - 1.0) < 4.

def testDensityValue():
    data = np.loadtxt("waterbox_npt.state", unpack=True)
    densityAverage = np.average(data[6])
    assert abs(densityAverage - 1.0) < 1e-2


    


