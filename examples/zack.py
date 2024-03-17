from charmm import apocharmm as ch
import argparse


def simulation(args):
    paramPath = "/u/zjarin/apob/toppar/"
    filePath = "/u/zjarin/for_felix/"
    prmlist = ['par_all36_lipid.prm',
               'toppar_water_ions.str', 'toppar_all36_lipid_tag_vanni.str']
    prmlist = [paramPath + p for p in prmlist]

    prm = ch.CharmmParameters(prmlist)
    psf = ch.CharmmPSF(filePath + "trilayer.psf")
    # crd = ch.CharmmCrd(filePath + "jmin_pore_20p_min.crd")
    crd = ch.PDB(filePath + "trilayer.pdb")

    fm = ch.ForceManager(psf, prm)
    fm.setBoxDimensions([154.178, 154.178, 112.266])
    fm.setCutoff(14.0)
    fm.setCtonnb(10.0)
    fm.setCtofnb(12.0)

    ctx = ch.CharmmContext(fm)

    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300)

    print("Minimization : start")
    minimizer = ch.Minimizer()
    minimizer.setSimulationContext(ctx)
    #minimizer.minimize(10)

    # An initial NVT simulation
    langevinThermostat = ch.LangevinThermostatIntegrator(0.002)
    langevinThermostat.setFriction(5.0)
    langevinThermostat.setBathTemperature(300.0)
    langevinThermostat.setSimulationContext(ctx)
    restartsubscriber = ch.RestartSubscriber("out/zack.res", 10000)

    subscriber = ch.DcdSubscriber("out/zack_nvt.dcd", 10000)
    langevinThermostat.subscribe([restartsubscriber, subscriber])
    numSteps = int(1e5)

    print("NVT : start")
    langevinThermostat.propagate(numSteps)

    # NPT simulation
    langevinPiston = ch.LangevinPistonIntegrator(0.002)
    langevinPiston.setCrystalType(ch.CRYSTAL.TETRAGONAL)
    # langevinPiston.setPistonMass(500.0)
    langevinPiston.setPistonFriction(12.0)
    langevinPiston.setBathTemperature(300.0)
    langevinPiston.setSimulationContext(ctx)
    subscriber = ch.DcdSubscriber("out/zack_npt.dcd", 1000)
    restartsubscriber = ch.RestartSubscriber("out/zack_npt.res", 10000)
    # stateSub = ch.StateSubscriber("out/jiyeon_npt.state", 100)
    # stateSub.setReportFlags("all")
    langevinPiston.subscribe([subscriber, restartsubscriber])
    nptSteps = int(1e5)

    print("NPT : start")
    langevinPiston.propagate(nptSteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zack bilayer simulation')
    args = parser.parse_args()
    simulation(args)
