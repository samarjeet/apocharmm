from charmm import apocharmm as ch
import argparse


def p21_bilayer():
    prm = ch.CharmmParameters(
        [
            "../test/data/par_all36_lipid.prm",
            "../test/data/toppar_all36_lipid_cholesterol.str",
            "../test/data/toppar_water_ions.str",
        ]
    )

    filePath = "/v/gscratch/mbs/spark/induced-order/test_p21_apo_charmm/sq2x2/low/"
    psf = ch.CharmmPSF(filePath + "init.psf")
    # minimized for P21
    crd = ch.CharmmCrd(filePath + "step6.6_equilibration.crd")

    fm = ch.ForceManager(psf, prm)
    fm.setBoxDimensions([144.96, 144.96, 98.65])
    fm.setFFTGrid(150, 150, 108)
    fm.setKappa(0.34)
    fm.setCutoff(14.0)  # cutnb of CHARMM
    fm.setCtonnb(10.0)
    fm.setCtofnb(12.0)
    fm.setPeriodicBoundaryCondition(ch.P21)  # Only for P21 simulations

    ctx = ch.CharmmContext(fm)

    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300)

    langevinThermostat = ch.LangevinThermostatIntegrator(0.002)
    langevinThermostat.setFriction(12.0)
    langevinThermostat.setBathTemperature(298.17)
    langevinThermostat.setSimulationContext(ctx)

    subscriber = ch.DcdSubscriber("out/p21_soohyung_nvt_new.dcd", 1000)
    # restartsubscriber = ch.RestartSubscriber("out/p21_soohyung_nvt.res", 10000)

    langevinThermostat.subscribe([subscriber])  # , restartsubscriber])

    numSteps = int(5e4)
    langevinThermostat.propagate(numSteps)

    langevinPiston = ch.LangevinPistonIntegrator(0.002)
    langevinPiston.setCrystalType(ch.CRYSTAL.TETRAGONAL)
    systemMass = psf.getMass()
    pistonMass = 0.05 * systemMass

    langevinPiston.setPistonMass(
        [pistonMass, pistonMass])  # 2 values for x/y and z
    langevinPiston.setPistonFriction(20.0)
    langevinPiston.setBathTemperature(298.17)
    langevinPiston.setSimulationContext(ctx)

    """
    restartsubscriber = ch.RestartSubscriber("old.res", 10000) # an already existing restart file
    langevinPiston.subscribe([restartsubscriber])
    restartsubscriber.readRestart()
    """

    baseName = f"out/p21_soohyung_npt_new_{0.002}_{20.0}_{pistonMass}"

    dcdSubscriber = ch.DcdSubscriber(f"{baseName}.dcd", 1000)
    restartsubscriber = ch.RestartSubscriber(f"{baseName}.res", 10000)
    # stateSub = ch.StateSubscriber(f'{baseName}.state', 1000)
    # stateSub.setReportFlags("all")

    # langevinPiston.subscribe([subscriber, restartsubscriber, stateSub])

    langevinPiston.subscribe([dcdSubscriber])  # , restartsubscriber])

    nptSteps = int(5e6)
    langevinPiston.propagate(nptSteps)


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="P21 bilayer simulation")
    parser.add_argument("timeStep", type=float, help="Time step value")
    parser.add_argument(
        "frictionCoefficient", type=float, help="Friction coefficient value"
    )
    parser.add_argument(
        "pressurePistonMassFactor",
        type=float,
        help="Pressure piston mass factor : X default value",
    )
    parser.add_argument(
        "temeperaturePistonMassFactor",
        type=float,
        help="Temperature piston mass factor : X value",
    )
    args = parser.parse_args()
    """

    p21_bilayer()
