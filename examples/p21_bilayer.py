from charmm import apocharmm as ch
import argparse


def p21_bilayer(args):
    prm = ch.CharmmParameters(["../test/data/par_all36_prot.prm", "../test/data/par_all36_lipid.prm",
                              "../test/data/toppar_all36_lipid_cholesterol.str", "../test/data/toppar_water_ions.str"])
    psf = ch.CharmmPSF("../test/data/fp.psf")
    crd = ch.CharmmCrd("../test/data/min_p21.crd")

    fm = ch.ForceManager(psf, prm)
    fm.setBoxDimensions([64.52, 64.52, 102.02])
    fm.setFFTGrid(72, 72, 108)
    fm.setKappa(0.34)
    fm.setCutoff(12.5)
    fm.setCtonnb(9.0)
    fm.setCtofnb(11.0)
    fm.setPeriodicBoundaryCondition(ch.P21)

    ctx = ch.CharmmContext(fm)

    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300)

    langevinThermostat = ch.LangevinThermostatIntegrator(0.002)
    langevinThermostat.setFriction(12.0)
    langevinThermostat.setBathTemperature(298.17)
    langevinThermostat.setSimulationContext(ctx)

    subscriber = ch.DcdSubscriber("out/p21_bilayer_nvt.dcd", 1000)
    restartsubscriber = ch.RestartSubscriber("out/p21_bilayer_nvt.res", 10000)

    langevinThermostat.subscribe([subscriber, restartsubscriber])

    numSteps = int(5e3)
    langevinThermostat.propagate(numSteps)

    langevinPiston = ch.LangevinPistonIntegrator(args.timeStep * 0.001)
    langevinPiston.setCrystalType(ch.CRYSTAL.TETRAGONAL)

    pressurePistonMass = langevinPiston.getPistonMass()
    langevinPiston.setPistonMass(
        args.pressurePistonMassFactor * pressurePistonMass)
    langevinPiston.setPistonFriction(args.frictionCoefficient)
    langevinPiston.setBathTemperature(298.17)
    langevinPiston.setSimulationContext(ctx)

    nhPistonMass = langevinPiston.getNoseHooverPistonMass()
    langevinPiston.setNoseHooverPistonMass(
        nhPistonMass*args.temeperaturePistonMassFactor)

    baseName = f'out/p21_bilayer_npt_{args.timeStep}_{args.frictionCoefficient}_{args.pressurePistonMassFactor}_{args.temeperaturePistonMassFactor}'

    dcdSubscriber = ch.DcdSubscriber(f'{baseName}.dcd', 1000)
    restartsubscriber = ch.RestartSubscriber(f'{baseName}.res', 10000)
    #stateSub = ch.StateSubscriber(f'{baseName}.state', 100)
    #stateSub.setReportFlags("all")

    #langevinPiston.subscribe([subscriber, restartsubscriber, stateSub])

    langevinPiston.subscribe([dcdSubscriber, restartsubscriber])

    nptSteps = int(1e7)
    langevinPiston.propagate(nptSteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P21 bilayer simulation")
    parser.add_argument("timeStep", type=float, help="Time step value")
    parser.add_argument("frictionCoefficient", type=float,
                        help="Friction coefficient value")
    parser.add_argument("pressurePistonMassFactor", type=float,
                        help="Pressure piston mass factor : X default value")
    parser.add_argument("temeperaturePistonMassFactor", type=float,
                        help="Temperature piston mass factor : X value")
    args = parser.parse_args()

    p21_bilayer(args)
