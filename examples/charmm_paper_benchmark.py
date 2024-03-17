import charmm.apocharmm as ch
import time


def dhfr():
    prm = ch.CharmmParameters(
        ["../test/data/par_all22_prot.prm", "../test/data/toppar_water_ions.str"])
    # prm = ch.CharmmParameters(["../test/data/par_all36_prot.prm", "../test/data/toppar_water_ions.str"])
    psf = ch.CharmmPSF("../test/data/jac_5dhfr.psf")

    fm = ch.ForceManager(psf, prm)
    fm.setBoxDimensions([62.23, 62.23, 62.23])
    fm.setFFTGrid(64, 64, 64)
    fm.setKappa(0.34)
    fm.setCutoff(9.0)
    fm.setCtonnb(7.0)
    fm.setCtofnb(7.5)

    ctx = ch.CharmmContext(fm)
    crd = ch.CharmmCrd("../test/data/jac_5dhfr.crd")
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300)

    timeStep = 0.002
    numSteps = 10000

    integrator = ch.LangevinThermostatIntegrator(0.002)
    integrator.setFriction(0.0)
    integrator.setBathTemperature(300)

    integrator.setSimulationContext(ctx)

    startTime = time.time()
    integrator.propagate(numSteps)
    endTime = time.time()

    print("Speed in ns/day: ", (numSteps * timeStep) /
          (endTime - startTime) * 86400 / 1000)


def dmpg():
    prm = ch.CharmmParameters(
        ["../test/data/par_all36_lipid.prm", "../test/data/toppar_water_ions.str"])
    psf = ch.CharmmPSF("../test/data/dmpg.psf")

    fm = ch.ForceManager(psf, prm)
    fm.setBoxDimensions([62.23, 62.23, 62.23])
    fm.setFFTGrid(64, 64, 64)
    fm.setKappa(0.34)
    fm.setCutoff(9.0)
    fm.setCtonnb(7.0)
    fm.setCtofnb(7.5)

    ctx = ch.CharmmContext(fm)
    crd = ch.CharmmCrd("../test/data/dmpg.crd")
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300)

    timeStep = 0.002
    numSteps = 10000

    integrator = ch.LangevinThermostatIntegrator(0.002)
    integrator.setFriction(0.0)
    integrator.setBathTemperature(300)

    integrator.setSimulationContext(ctx)

    startTime = time.time()
    integrator.propagate(numSteps)
    endTime = time.time()

    print("Speed in ns/day: ", (numSteps * timeStep) /
          (endTime - startTime) * 86400 / 1000)
