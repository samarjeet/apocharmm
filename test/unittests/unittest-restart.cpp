// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:   James E. Gonzales II, Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "ForceManager.h"
#include "RestartSubscriber.h"
#include "catch.hpp"
#include "compare.h"
#include "test_paths.h"
#include <iostream>

TEST_CASE("restart") {
  // In this test case, three integrators are created. 1.) Runs nsteps, writes a
  // restart file, and then runs nsteps again. 2.) Reads the restart file
  // generated from integrator 1 and runs nsteps. 3.) Runs 2 * nsteps. The
  // coordinates and velocities are all compared at the end to ensure that the
  // same trajectory is generated.
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = true;
  const int nsteps = 1000;
  const double timeStep = 0.002;

  // Setup CHARMM parameters, PSF, and coordinates
  auto prm1 =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf1 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd1 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  auto prm2 =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf2 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd2 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  auto prm3 =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf3 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd3 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  // Setup force managers
  auto fm1 = std::make_shared<ForceManager>(psf1, prm1);
  fm1->setBoxDimensions(boxDims);

  auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
  fm2->setBoxDimensions(boxDims);

  auto fm3 = std::make_shared<ForceManager>(psf3, prm3);
  fm3->setBoxDimensions(boxDims);

  // Setup CHARMM contexts
  auto ctx1 = std::make_shared<CharmmContext>(fm1);
  ctx1->setCoordinates(crd1);
  ctx1->setRandomSeedForVelocities(randomSeed);
  ctx1->assignVelocitiesAtTemperature(temperature);
  ctx1->useHolonomicConstraints(useHolonomicConstraints);

  auto ctx2 = std::make_shared<CharmmContext>(fm2);
  ctx2->setCoordinates(crd2);
  ctx2->setRandomSeedForVelocities(randomSeed);
  ctx2->assignVelocitiesAtTemperature(temperature);
  ctx2->useHolonomicConstraints(useHolonomicConstraints);

  auto ctx3 = std::make_shared<CharmmContext>(fm3);
  ctx3->setCoordinates(crd3);
  ctx3->setRandomSeedForVelocities(randomSeed);
  ctx3->assignVelocitiesAtTemperature(temperature);
  ctx3->useHolonomicConstraints(useHolonomicConstraints);

  // Check that coordinates match
  CudaContainer<double4> coordinatesCharges1 = ctx1->getCoordinatesCharges();
  CudaContainer<double4> coordinatesCharges2 = ctx2->getCoordinatesCharges();
  CudaContainer<double4> coordinatesCharges3 = ctx3->getCoordinatesCharges();
  coordinatesCharges1.transferToHost();
  coordinatesCharges2.transferToHost();
  coordinatesCharges3.transferToHost();
  CHECK(CompareVectors1<double4>(coordinatesCharges1.getHostArray(),
                                 coordinatesCharges2.getHostArray(), 0.0,
                                 true));
  CHECK(CompareVectors1<double4>(coordinatesCharges1.getHostArray(),
                                 coordinatesCharges3.getHostArray(), 0.0,
                                 true));
  CHECK(CompareVectors1<double4>(coordinatesCharges2.getHostArray(),
                                 coordinatesCharges3.getHostArray(), 0.0,
                                 true));

  CudaContainer<double4> velocitiesMasses1 = ctx1->getVelocityMass();
  CudaContainer<double4> velocitiesMasses2 = ctx2->getVelocityMass();
  CudaContainer<double4> velocitiesMasses3 = ctx3->getVelocityMass();
  velocitiesMasses1.transferToHost();
  velocitiesMasses2.transferToHost();
  velocitiesMasses3.transferToHost();
  CHECK(CompareVectors1<double4>(velocitiesMasses1.getHostArray(),
                                 velocitiesMasses2.getHostArray(), 0.0, true));
  CHECK(CompareVectors1<double4>(velocitiesMasses1.getHostArray(),
                                 velocitiesMasses3.getHostArray(), 0.0, true));
  CHECK(CompareVectors1<double4>(velocitiesMasses2.getHostArray(),
                                 velocitiesMasses3.getHostArray(), 0.0, true));

  SECTION("noseHoover") {
    std::cout << "**** Nose-Hoover Thermostat ****" << std::endl;
    std::cout << "First integrator..." << std::flush;

    // Setup first integrator
    auto integrator1 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator1->setCharmmContext(ctx1);

    // Setup restart subscriber
    auto rst = std::make_shared<RestartSubscriber>("tmpNoseHoover.rst", nsteps);
    integrator1->subscribe(rst);

    // Run first integrator
    integrator1->propagate(nsteps);
    integrator1->unsubscribe(rst);

    // Run first integrator second set of steps
    integrator1->propagate(nsteps);

    std::cout << "\rFirst integrator Done." << std::endl;
    std::cout << "Second integrator..." << std::flush;

    // Setup second integrator from restart file
    auto integrator2 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator2->setCharmmContext(ctx2);
    // integrator2->setupFromRestartFile("tmpNoseHoover.rst");

    // Run second integrator
    integrator2->propagate(nsteps);
    integrator2->propagate(nsteps);

    std::cout << "\rSecond integrator Done." << std::endl;
    std::cout << "Third integrator..." << std::flush;

    // Setup third integrator
    auto integrator3 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator3->setCharmmContext(ctx3);

    // Run third integrator
    integrator3->propagate(2 * nsteps);

    std::cout << "\rThird integrator Done." << std::endl;
  }

  // Check that the coordinates and velocities match
  coordinatesCharges1 = ctx1->getCoordinatesCharges();
  coordinatesCharges2 = ctx2->getCoordinatesCharges();
  coordinatesCharges3 = ctx3->getCoordinatesCharges();
  coordinatesCharges1.transferToHost();
  coordinatesCharges2.transferToHost();
  coordinatesCharges3.transferToHost();
  CHECK(CompareVectors1<double4>(coordinatesCharges1.getHostArray(),
                                 coordinatesCharges2.getHostArray(), 0.0,
                                 true));
  CHECK(CompareVectors1<double4>(coordinatesCharges1.getHostArray(),
                                 coordinatesCharges3.getHostArray(), 0.0,
                                 true));
  CHECK(CompareVectors1<double4>(coordinatesCharges2.getHostArray(),
                                 coordinatesCharges3.getHostArray(), 0.0,
                                 true));

  velocitiesMasses1 = ctx1->getVelocityMass();
  velocitiesMasses2 = ctx2->getVelocityMass();
  velocitiesMasses3 = ctx3->getVelocityMass();
  velocitiesMasses1.transferToHost();
  velocitiesMasses2.transferToHost();
  velocitiesMasses3.transferToHost();
  CHECK(CompareVectors1<double4>(velocitiesMasses1.getHostArray(),
                                 velocitiesMasses2.getHostArray(), 0.0, true));
  CHECK(CompareVectors1<double4>(velocitiesMasses1.getHostArray(),
                                 velocitiesMasses3.getHostArray(), 0.0, true));
  CHECK(CompareVectors1<double4>(velocitiesMasses2.getHostArray(),
                                 velocitiesMasses3.getHostArray(), 0.0, true));
}
