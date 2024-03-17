from charmm import apocharmm as ch
import argparse


def pore_simulation(args):
    paramPath = "/u/samar/toppar/"
    filePath = "/u/samar/Documents/git/test_gpu/pore/"
    prmlist = ['par_all36_lipid.prm',
               'toppar_all36_lipid_bacterial.str', 'toppar_water_ions.str']
    prmlist = [paramPath + p for p in prmlist]

    prm = ch.CharmmParameters(prmlist)
    psf = ch.CharmmPSF(filePath + "jmin_pore_20p_min.psf")
    # crd = ch.CharmmCrd(filePath + "jmin_pore_20p_min.crd")
    crd = ch.CharmmCrd(filePath + "equil.cor")

    fm = ch.ForceManager(psf, prm)
    # fm.setBoxDimensions([160.0, 160.0, 100.0])
    fm.setBoxDimensions([147.09442, 147.09442, 106.35873])
    fm.setCutoff(12.0)
    fm.setCtonnb(8.0)
    fm.setCtofnb(10.0)

    ctx = ch.CharmmContext(fm)

    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300)

  # An initial NVT simulation
  langevinThermostat = ch.LangevinThermostatIntegrator(0.002)
  langevinThermostat.setFriction(5.0)
  langevinThermostat.setBathTemperature(298.17)
  langevinThermostat.setSimulationContext(ctx)
  restartsubscriber = ch.RestartSubscriber("out/jiyeon_nvt.res", 10000)
  langevinThermostat.subscribe([restartsubscriber])
  numSteps = int(1e5)
  langevinThermostat.propagate(numSteps)
  print("NVT done")

  # An additional short NPT simulation to get the right area
  langevinPiston = ch.LangevinPistonIntegrator(0.002)
  langevinPiston.setCrystalType(ch.CRYSTAL.TETRAGONAL)
  #langevinPiston.setPistonMass(500.0)
  langevinPiston.setPistonFriction(12.0)
  langevinPiston.setBathTemperature(298.17)
  langevinPiston.setSimulationContext(ctx)
  subscriber = ch.DcdSubscriber("out/jiyeon_npt.dcd", 1000)
  restartsubscriber = ch.RestartSubscriber("out/jiyeon_npt.res", 10000)
  #stateSub = ch.StateSubscriber("out/jiyeon_npt.state", 100)
  #stateSub.setReportFlags("all")
  langevinPiston.subscribe([subscriber, restartsubscriber])
  nptSteps = int(1e5)
  langevinPiston.propagate(nptSteps)
  print("Short NPT done")

  # A production NPT simulation
  langevinPiston = ch.LangevinPistonIntegrator(0.002)
  langevinPiston.setCrystalType(ch.CRYSTAL.TETRAGONAL)
  langevinPiston.setPistonMass([0.0, 500.0])
  langevinPiston.setPistonFriction(12.0)
  langevinPiston.setBathTemperature(298.17)
  langevinPiston.setSimulationContext(ctx)
  subscriber = ch.DcdSubscriber("out/jiyeon_npat.dcd", 10000)
  restartsubscriber = ch.RestartSubscriber("out/jiyeon_npat.res", 50000)
  #stateSub = ch.StateSubscriber("out/jiyeon_npt.state", 100)
  #stateSub.setReportFlags("all")
  langevinPiston.subscribe([subscriber, restartsubscriber])
  nptSteps = int(1e7)
  langevinPiston.setDebugPrintFrequency(1000)
  langevinPiston.propagate(nptSteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pore simulation')
    args = parser.parse_args()
    pore_simulation(args)
