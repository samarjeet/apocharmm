// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "ForceManager.h"
#include "catch.hpp"
// #include "compare.h"
#include "HarmonicRestraintForce.h"
#include "test_paths.h"
#include <iostream>
#include <memory>

TEST_CASE("harmonicRestraintForce") {
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = true;
  const int nstep = 10000;
  const double timeStep = (useHolonomicConstraints) ? 0.002 : 0.001;

  SECTION("nacl") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "nacl_pair.psf");
    auto crd = std::make_shared<CharmmCrd>(dataPath + "nacl_pair.cor");
    // auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    // auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    for (int i = 0; i < crd->getNumAtoms(); i++) {
      std::cout << "crd->getCoordinatesD()[" << i << "] = {"
                << crd->getCoordinatesD()[i].x << ", "
                << crd->getCoordinatesD()[i].y << ", "
                << crd->getCoordinatesD()[i].z << ", "
                << crd->getCoordinatesD()[i].w << "}" << std::endl;
      std::cout << "crd->getCoordinatesF()[" << i << "] = {"
                << crd->getCoordinatesF()[i].x << ", "
                << crd->getCoordinatesF()[i].y << ", "
                << crd->getCoordinatesF()[i].z << ", "
                << crd->getCoordinatesF()[i].w << "}" << std::endl;
    }

    // Setup force manager
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions(boxDims);

    // Setup CHARMM context
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);
    // ctx->setRandomSeedForVelocities(randomSeed);
    // ctx->assignVelocitiesAtTemperature(temperature);
    // ctx->useHolonomicConstraints(useHolonomicConstraints);

    // // Setup integrator
    // auto integrator =
    //     std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    // integrator->setThermostatFriction(1.0);
    // integrator->setThermostatRngSeed(randomSeed);
    // integrator->setCharmmContext(ctx);
  }
}

/* *
TEST_CASE("restraintForce", "[energy]") {
  std::string dataPath = getDataPath();

  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "water2_1.psf");

  auto numAtoms = psf->getNumAtoms();

  auto restraintForceValues = std::make_shared<Force<long long int>>();
  restraintForceValues->realloc(numAtoms, 1.5f);

  CudaEnergyVirial restraintEnergyVirial;
  auto restraint = std::make_shared<GeometricRestraintForce<long long,
  float>>( restraintEnergyVirial); restraint->setForce(restraintForceValues);
  // put this force on a stream

  restraint->addRestraint(RestraintShape::PLANE, PotentialFunction::HARMONIC,
                          false, {0.0, 0.0, 0.0}, true, {1.0, 0.0, 0.0}, false,
                          1.0, 0.0, {0, 1, 2});




  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({50.0, 50.0, 50.0});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(12.0);
  fm->setCtonnb(8.0);
  fm->setCtofnb(10.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "water2.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);
  ctx->calculatePotentialEnergy(true, true);

  auto xyzq = ctx->getXYZQ();

  restraint->calc_force(xyzq->getDeviceXYZQ(), true, true);
  cudaDeviceSynchronize();

  // restraintForceValues->transferFromDevice();

  // assert that all the forces are correct
  // REQUIRE(ctx->getPotentialEnergy() == Approx(-1.041e+04).epsilon(0.01));
  // REQUIRE(ctx->getForces()[0] == Approx(1.0).epsilon(0.01));
  // REQUIRE(ctx->getForces()[1] == Approx(1.0).epsilon(0.01));
  INFO("No assertion performed !!");
  // CHECK(false);
}
* */
