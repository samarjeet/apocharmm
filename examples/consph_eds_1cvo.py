from charmm import apocharmm as ac
import argparse


def run_reference():
    # setup data location
    testDataPath = "/v/gscratch/cbs/adachen/EDS_1CVO_apoCHARMM/ark/model_compound_GLU/"

    # Import the parameters anf the system's topology (prm and psf)
    psf0 = ac.CharmmPSF(testDataPath + "glu_depr.psf")
    psf1 = ac.CharmmPSF(testDataPath + "glu_prot.psf")

    prm = ac.CharmmParameters(
        [testDataPath + "toppar_water_ions.str", testDataPath + "par_all36m_prot.prm"]
    )

    # Set up the ForceManager that will drive the simulation.
    # Create the ForceManager object using the psf and prm
    fm0 = ac.ForceManager(psf0, prm)
    fm1 = ac.ForceManager(psf1, prm)

    fmEDS = ac.EDSForceManager(fm0, fm1)

    # Setup box size, FFT options, cutoffs...
    fmEDS.setBoxDimensions([30.0, 30.0, 30.0])
    fmEDS.setFFTGrid(30, 30, 30)
    fmEDS.setCtonnb(9.0)
    fmEDS.setCtofnb(10.0)
    fmEDS.setCutoff(12.0)
    # Finally, initialize the ForceManager object !
    fmEDS.initialize()

    s = 0.003
    eOffset0 = 0
    eOffset1 = 44

    fmEDS.setSValue(s)
    fmEDS.setEnergyOffsets([eOffset0, eOffset1])

    # The simulation state will be handled by a CharmmContext object, created from the ForceManager.

    ctx = ac.CharmmContext(fmEDS)

    # This CharmmContext handles the coordinates and velocities.

    crd = ac.CharmmCrd(testDataPath + "glu.crd")
    ctx.setCoordinates(crd)
    ctx.assignVelocitiesAtTemperature(300.0)

    # We start by a short minimization of our system.
    minimizer = ac.Minimizer()
    # minimizer.setSimulationContext(ctx)
    # minimizer.minimize(1000)

    # Here we will integrate using a Langevin thermostat. We create the Integrator object, then attach it to the CharmmContext. Finally, we propagate for 10 steps.

    integrator = ac.LangevinThermostatIntegrator(0.002)  # , 300, 12)
    integrator.setFriction(12.0)
    integrator.setBathTemperature(300.0)
    integrator.setSimulationContext(ctx)
    # integrator.propagate(1000)

    # The simulation can be monitored using various Susbcribers, responsible for creating output files at a given frequency:
    # * StateSubscribers (time, energy, temperature, pressure...);
    # * RestartSubscriber: outputs a Charmm-like restart file
    # * DcdSubscriber: saves the trajectory to a .dcd file
    # * ...
    #
    integrator.setDebugPrintFrequency(100)
    # stateSub = ac.StateSubscriber(testDataPath + "gluState.txt", 500)
    dcdSub = ac.DcdSubscriber(testDataPath + "gluTraj.dcd", 1000)
    # restartSub = ac.RestartSubscriber(testDataPath + "gluRestart.res", 2000)
    mbarSub = ac.BEDSSubscriber(testDataPath + "glu_state_energies.txt", 1000)

    # Once created, a Subscriber needs to be subscribed to an Integrator.
    # integrator.subscribe(stateSub)
    # integrator.subscribe([dcdSub, restartSub, mbarSub])
    integrator.subscribe(mbarSub)

    numSteps = 5000
    integrator.propagate(numSteps)


def run_constant_pH():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a EDS constant pH MD simulation.")
    parser.add_argument(
        "--simulation",
        help="choose the simulation to run",
        default="reference",
        type=str,
        choices=["reference", "constant_pH"],
    )

    args = parser.parse_args()
    print(args)
    if args.simulation == "reference":
        run_reference()
