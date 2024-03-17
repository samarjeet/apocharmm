// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "ForceManager.h"
#include "helper.h"

#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaLeapFrogIntegrator.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "CudaVMMSVelocityVerletIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "MBARForceManager.h"
#include "StateSubscriber.h"
#include "XYZSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>

/* IDEA: Very basic test here. Run 100 steps for some example systems (for now,
 * waterbox, dhfr). Check that it didn't explode. Do this for every integrator
 * implemented. */

// Nose-Hoover integrator : has not been fully implemented yet
// Verlet : obsolete
// VMMSVV : not tested yet

std::shared_ptr<CudaLangevinPistonIntegrator>
setupLangevinPistonIntegrator(std::shared_ptr<CharmmContext> ctx) {
  std::shared_ptr<CudaLangevinPistonIntegrator> integrator =
      std::make_shared<CudaLangevinPistonIntegrator>(0.001);
  integrator->setPistonFriction(10.0);
  integrator->setSimulationContext(ctx);
  integrator->setCrystalType(CRYSTAL::CUBIC);
  integrator->setPistonMass({500.0});
  return integrator;
}

std::shared_ptr<CudaLangevinThermostatIntegrator>
setupLangevinThermostatIntegrator(std::shared_ptr<CharmmContext> ctx) {
  std::shared_ptr<CudaLangevinThermostatIntegrator> integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(0.001);
  integrator->setSimulationContext(ctx);
  integrator->setBathTemperature(300.0);
  integrator->setFriction(12.0);
  return integrator;
}

std::shared_ptr<CudaLeapFrogIntegrator>
setupLeapFrogIntegrator(std::shared_ptr<CharmmContext> ctx) {
  std::shared_ptr<CudaLeapFrogIntegrator> integrator =
      std::make_shared<CudaLeapFrogIntegrator>(0.001);
  integrator->setSimulationContext(ctx);
  return integrator;
}

std::shared_ptr<CudaVelocityVerletIntegrator>
setupVelocityVerletIntegrator(std::shared_ptr<CharmmContext> ctx) {
  std::shared_ptr<CudaVelocityVerletIntegrator> integrator =
      std::make_shared<CudaVelocityVerletIntegrator>(0.001);
  integrator->setSimulationContext(ctx);
  return integrator;
}

std::shared_ptr<CudaVerletIntegrator>
setupVerletIntegrator(std::shared_ptr<CharmmContext> ctx) {
  std::shared_ptr<CudaVerletIntegrator> integrator =
      std::make_shared<CudaVerletIntegrator>(0.001);
  integrator->setSimulationContext(ctx);
  return integrator;
}

TEST_CASE("Basic functions", "[unittest]") {
  std::string dataPath = getDataPath();
  SECTION("Context") {
    auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();
    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "argon_10.crd");
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(300);

    auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
    CHECK_THROWS(integrator->propagate(10));
    CHECK_NOTHROW(integrator->setSimulationContext(ctx));

    // exception if setting a context to integrator already having one
    CHECK_THROWS(integrator->setSimulationContext(ctx));

    integrator->setDebugPrintFrequency(10);

    std::cout << "You should see two successive debug prints hereunder..."
              << std::endl;

    CHECK_NOTHROW(integrator->propagate(20));
  }

  SECTION("Basics") {
    auto integ = std::make_shared<CudaVelocityVerletIntegrator>(0.002);
    // timestep intialization
    REQUIRE(integ->getTimeStep() == 0.002);
    // timestep change
    integ->setTimeStep(0.001);
    REQUIRE(integ->getTimeStep() == 0.001);
  }

  SECTION("Subscribers handling") {
    auto integ = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
    auto statesub =
        std::make_shared<StateSubscriber>("Integrator.statesub.txt", 10);
    auto dcdsub = std::make_shared<DcdSubscriber>("Integrator.dcdsub.txt", 12);
    auto xyzsub = std::make_shared<XYZSubscriber>("Integrator.xyzsub.txt", 14);
    // add a single sub
    CHECK_NOTHROW(integ->subscribe(statesub));

    // add a list of sub
    std::vector<std::shared_ptr<Subscriber>> sublist{dcdsub, xyzsub};
    CHECK_NOTHROW(integ->subscribe(sublist));

    // remove a sub
    CHECK_NOTHROW(integ->unsubscribe(dcdsub));
    // try to remove a non-subbed subscriber
    CHECK_THROWS(integ->unsubscribe(dcdsub));
  }

  // Check that NaNs will throw an error and kill the simulation
  // To create a failing sim on purpose, load bad coordinates from another
  // file's.
  SECTION("NaN error") {
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox_NaN.crd");
    ctx->setCoordinates(crd);
    auto integr =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002, 300, 12.);
    integr->setSimulationContext(ctx);
    // This should throw ! That's what we're testing.
    CHECK_THROWS(integr->propagate(10));
  }

  SECTION("langevinPiston") {
    //  Test initialize function sets things to 0
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx->setCoordinates(crd);
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(12.0);
    integrator->setSimulationContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setPistonMass({500.0});
    integrator->propagate(100);
    // internal variables are now messed up

    integrator->initialize();
    // internal variables for the pistons (Nose-Hoover and langevin) should now
    // be zeroed out
    CHECK(integrator->getPistonNoseHooverForce() == 0.0);
    CHECK(integrator->getPistonNoseHooverForcePrevious() == 0.0);
    CHECK(integrator->getPistonNoseHooverVelocity() == 0.0);
    CHECK(integrator->getPistonNoseHooverVelocityPrevious() == 0.0);
    integrator->getOnStepPistonVelocity().transferFromDevice();
    CHECK(integrator->getOnStepPistonVelocity()[0] == 0.0);
    integrator->getHalfStepPistonVelocity().transferFromDevice();
    CHECK(integrator->getHalfStepPistonVelocity()[0] == 0.0);
    integrator->getOnStepPistonPosition().transferFromDevice();
    CHECK(integrator->getOnStepPistonPosition()[0] == 0.0);
    integrator->getHalfStepPistonPosition().transferFromDevice();
    CHECK(integrator->getHalfStepPistonPosition()[0] == 0.0);
  }
}

TEST_CASE("waterbox") {
  int nsteps = 100;
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({50.0, 50.0, 50.0});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);
  fm->initialize();

  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  SECTION("LangevinPiston") {
    auto integrator = setupLangevinPistonIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  SECTION("LangevinThermostat") {
    auto integrator = setupLangevinThermostatIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  SECTION("leapfrog") {
    auto integrator = setupLeapFrogIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  SECTION("velocityVerlet") {
    auto integrator = setupVelocityVerletIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  // VERLET IS BUGGED -- yields a Segfault
  //    SECTION("Verlet") {
  //        auto integrator = setupVerletIntegrator(ctx);
  //        CHECK_NOTHROW(integrator.propagate(nsteps));
  //    }
  // Add remaining ones there...
}

TEST_CASE("dhfr") {
  int nsteps = 100;
  std::string dataPath = getDataPath();
  std::vector<std::string> prmlist = {dataPath + "par_all22_prot.prm",
                                      dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmlist);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "jac_5dhfr.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  double boxdim = 62.23;
  fm->setBoxDimensions({boxdim, boxdim, boxdim});
  fm->setFFTGrid(64, 64, 64);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtofnb(8.0);
  fm->setCtonnb(7.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(
      dataPath + "jac_5dhfr.crd"); //"dhfr.charmm_gui.crd");
  ctx->setCoordinates(crd);

  SECTION("LangevinPiston") {
    auto integrator = setupLangevinPistonIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  SECTION("LangevinThermostat") {
    auto integrator = setupLangevinThermostatIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  SECTION("leapfrog") {
    auto integrator = setupLeapFrogIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  SECTION("velocityVerlet") {
    auto integrator = setupVelocityVerletIntegrator(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }

  // VERLET IS BUGGED -- yields a Segfault
  //    SECTION("Verlet") {
  //        auto integrator = setupVerletIntegrator(ctx);
  //        CHECK_NOTHROW(integrator.propagate(nsteps));
  //    }
  //    // Copy paste above here !
}

// Make sure that gas phase simulations run
TEST_CASE("waterDimer", "[gasphase]") {
  int nsteps = 100;
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterDimer.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  double dim = 520.;
  fm->setBoxDimensions({dim, dim, dim});
  // IS SETFFTGRID THE NECESSARY PART ?
  fm->setFFTGrid(4, 4, 4);
  fm->setCutoff(255.0);
  fm->setCtonnb(250.0);
  fm->setCtofnb(254.0);
  fm->setKappa(0.0);

  fm->setPrintEnergyDecomposition(true);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterDimer.crd");

  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  SECTION("LangevinThermostat") {
    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002, 300, 0.0);
    integrator->setDebugPrintFrequency(100);
    integrator->setSimulationContext(ctx);
    CHECK_NOTHROW(integrator->propagate(nsteps));
  }
}

// Test that two identical integrators with identical contexts and identical
// random generator states will produce identical trajectories
TEST_CASE("deterministic") {
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  auto fm1 = std::make_shared<ForceManager>(psf, prm);
  auto fm2 = std::make_shared<ForceManager>(psf, prm);
  fm1->setBoxDimensions({50.0, 50.0, 50.0});
  fm2->setBoxDimensions({50.0, 50.0, 50.0});

  auto ctx1 = std::make_shared<CharmmContext>(fm1);
  auto ctx2 = std::make_shared<CharmmContext>(fm2);
  ctx1->setRandomSeedForVelocities(144);
  ctx2->setRandomSeedForVelocities(ctx1->getRandomSeedForVelocities());

  // ctx1->useHolonomicConstraints(false);
  // ctx2->useHolonomicConstraints(false);

  ctx1->setCoordinates(crd);
  ctx2->setCoordinates(crd);
  ctx1->assignVelocitiesAtTemperature(300);
  ctx2->assignVelocitiesAtTemperature(300);

  // Check that initial velocities are identical
  CudaContainer<double4> vel1 = ctx1->getVelocityMass(),
                         vel2 = ctx2->getVelocityMass(),
                         pos1 = ctx1->getCoordinatesCharges(),
                         pos2 = ctx2->getCoordinatesCharges();
  vel1.transferFromDevice();
  vel2.transferFromDevice();
  pos1.transferFromDevice();
  pos2.transferFromDevice();

  std::cout << "vel1: " << vel1.getHostArray()[0].x << " "
            << vel1.getHostArray()[0].y << " " << vel1.getHostArray()[0].z
            << std::endl;
  std::cout << "vel2: " << vel2.getHostArray()[0].x << " "
            << vel2.getHostArray()[0].y << " " << vel2.getHostArray()[0].z
            << std::endl;

  CHECK(compareCudaVectorsTriples(vel1, vel2, 0.00001));
  CHECK(compareCudaVectorsTriples(pos1, pos2, 0.00001));

  SECTION("velocityVerlet") {
    int nsteps = 5;
    std::shared_ptr<CudaVelocityVerletIntegrator>
        integrator1 = std::make_shared<CudaVelocityVerletIntegrator>(0.001),
        integrator2 = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
    integrator1->setDebugPrintFrequency(1);
    integrator2->setDebugPrintFrequency(1);
    integrator1->setSimulationContext(ctx1);
    integrator2->setSimulationContext(ctx2);

    // Check that initial velocities are identical
    CudaContainer<double4> vel1 = ctx1->getVelocityMass(),
                           vel2 = ctx2->getVelocityMass(),
                           pos1 = ctx1->getCoordinatesCharges(),
                           pos2 = ctx2->getCoordinatesCharges();
    vel1.transferFromDevice();
    vel2.transferFromDevice();
    pos1.transferFromDevice();
    pos2.transferFromDevice();
    CHECK(compareCudaVectorsTriples(vel1, vel2, 0.00001));
    CHECK(compareCudaVectorsTriples(pos1, pos2, 0.00001));

    // Doing one step each is deterministic ! (they follow each other)
    // for (int i = 0; i < nsteps; i++) {
    //  std::cout << "--Step " << i << "--" << std::endl;
    //  integrator1->propagate(1);
    //  integrator2->propagate(1);
    //}

    // Doing ten steps for 1, then ten steps for 2, then comparing
    std::cout << "--integrator1--" << std::endl;
    integrator1->propagate(nsteps);
    std::cout << "--integrator2--" << std::endl;
    integrator2->propagate(nsteps);

    // looking at the aftermath
    vel1 = ctx1->getVelocityMass();
    pos1 = ctx1->getCoordinatesCharges();
    vel1.transferFromDevice();
    pos1.transferFromDevice();
    std::cout << "vel1: " << vel1.getHostArray()[0].x << " "
              << vel1.getHostArray()[0].y << " " << vel1.getHostArray()[0].z
              << std::endl;

    vel2 = ctx2->getVelocityMass();
    pos2 = ctx2->getCoordinatesCharges();
    vel2.transferFromDevice();
    pos2.transferFromDevice();
    std::cout << "vel2: " << vel2.getHostArray()[0].x << " "
              << vel2.getHostArray()[0].y << " " << vel2.getHostArray()[0].z
              << std::endl;

    CHECK(compareCudaVectorsTriples(vel1, vel2, 0.00001));
    CHECK(compareCudaVectorsTriples(pos1, pos2, 0.00001));
  }

  // For now, tests with 0 friction (aka no randomness).
  // We should also test with friction and fix the rng seed, to check that also
  // is deterministic
  SECTION("langevinThermostatNoFriction") {
    int nsteps = 1000;
    auto integrator1 = std::make_shared<CudaLangevinThermostatIntegrator>(
             0.001, 300, 0.0),
         integrator2 = std::make_shared<CudaLangevinThermostatIntegrator>(
             0.001, 300, 0.0);
    integrator1->setSimulationContext(ctx1);
    integrator2->setSimulationContext(ctx2);

    // Check that initial velocities are identical
    CudaContainer<double4> vel1 = ctx1->getVelocityMass(),
                           vel2 = ctx2->getVelocityMass(),
                           pos1 = ctx1->getCoordinatesCharges(),
                           pos2 = ctx2->getCoordinatesCharges();
    vel1.transferFromDevice();
    vel2.transferFromDevice();
    pos1.transferFromDevice();
    pos2.transferFromDevice();
    CHECK(compareCudaVectorsTriples(vel1, vel2, 0.00001));
    CHECK(compareCudaVectorsTriples(pos1, pos2, 0.00001));

    // Doing ten steps for 1, then ten steps for 2, then comparing
    integrator1->propagate(nsteps);
    integrator2->propagate(nsteps);

    // looking at the aftermath
    vel1 = ctx1->getVelocityMass();
    pos1 = ctx1->getCoordinatesCharges();
    vel1.transferFromDevice();
    pos1.transferFromDevice();
    vel2 = ctx2->getVelocityMass();
    pos2 = ctx2->getCoordinatesCharges();
    vel2.transferFromDevice();
    pos2.transferFromDevice();
    CHECK(compareCudaVectorsTriples(vel1, vel2, 0.00001));
    CHECK(compareCudaVectorsTriples(pos1, pos2, 0.00001));
  }
}

TEST_CASE("debug") {
  // Trying downcasting fun
  // std::shared_ptr<CudaIntegrator> baseIntegrator;
  std::shared_ptr<CudaVelocityVerletIntegrator> baseIntegrator;
  try {
    std::shared_ptr<CudaLangevinPistonIntegrator> lpIntegrator =
        std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(baseIntegrator);
  } catch (std::bad_cast &e) {
    std::cout << "Downcasting failed" << std::endl;
  }

  std::shared_ptr<CudaLangevinPistonIntegrator> lpi =
      std::make_shared<CudaLangevinPistonIntegrator>(0.001);
  try {
    std::shared_ptr<CudaVelocityVerletIntegrator> vvi =
        std::dynamic_pointer_cast<CudaVelocityVerletIntegrator>(lpi);
  } catch (std::bad_cast &e) {
    std::cout << "Downcasting 2 failed" << std::endl;
  }
}
TEST_CASE("casting") {
  // see how downcasting works
  auto baseintegrator = std::make_shared<CudaIntegrator>(0.001);
  auto vvintegrator = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
  auto ltintegrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(0.001, 300, 12.0);
  auto lpintegrator = std::make_shared<CudaLangevinPistonIntegrator>(0.001);

  // Can I downcast base to LP
  auto basetolpintegrator =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(baseintegrator);
  CHECK(basetolpintegrator == nullptr);

  // Can I downcast LT to LP
  auto lttolpintegrator =
      std::dynamic_pointer_cast<CudaLangevinPistonIntegrator>(ltintegrator);
  CHECK(lttolpintegrator == nullptr);
}
