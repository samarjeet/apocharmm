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
#include "compare.h"
#include "test_paths.h"
#include <iostream>

TEST_CASE("langevinThermostat", "[dynamics]") {
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = true;
  const int nsteps = 10000;
  const double timeStep = (useHolonomicConstraints) ? 0.002 : 0.001;

  SECTION("waterbox") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    // Setup force manager
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions(boxDims);

    // Setup CHARMM context
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);
    ctx->setRandomSeedForVelocities(randomSeed);
    ctx->assignVelocitiesAtTemperature(temperature);
    ctx->useHolonomicConstraints(useHolonomicConstraints);

    // Setup integrator
    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    integrator->setThermostatFriction(1.0);
    integrator->setThermostatRngSeed(randomSeed);
    integrator->setCharmmContext(ctx);

    if (useHolonomicConstraints)
      CHECK(ctx->computeTemperature() == 1839.70496f);
    else
      CHECK(ctx->computeTemperature() == 304.5621f);

    integrator->propagate(nsteps);
    integrator->resetAverageTemperature();
    integrator->propagate(nsteps);

    CudaContainer<double> averageTemperature =
        integrator->getAverageTemperature();
    averageTemperature.transferToHost();

    if (useHolonomicConstraints) {
      CHECK(ctx->computeTemperature() == 299.96588f);
      CHECK(averageTemperature[0] == Approx(301.3690901918).margin(1e-8));
      CHECK(averageTemperature[1] == Approx(300.1985645927).margin(1e-8));
    } else {
      CHECK(ctx->computeTemperature() == 294.9382f);
      CHECK(averageTemperature[0] == Approx(305.4079733868).margin(1e-8));
      CHECK(averageTemperature[1] == Approx(299.7495496752).margin(1e-8));
    }
  }

  SECTION("deterministic") {
    // Setup CHARMM parameters, PSF, and coordinates
    auto prm1 =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf1 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd1 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    auto prm2 =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf2 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd2 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    // Setup force managers
    auto fm1 = std::make_shared<ForceManager>(psf1, prm1);
    fm1->setBoxDimensions(boxDims);

    auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
    fm2->setBoxDimensions(boxDims);

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

    // Check that coordinates and velocities match
    CudaContainer<double4> initialCoordinatesCharges1 =
        ctx1->getCoordinatesCharges();
    CudaContainer<double4> initialCoordinatesCharges2 =
        ctx2->getCoordinatesCharges();
    initialCoordinatesCharges1.transferToHost();
    initialCoordinatesCharges2.transferToHost();
    CHECK(CompareVectors1<double4>(initialCoordinatesCharges1.getHostArray(),
                                   initialCoordinatesCharges2.getHostArray(),
                                   0.0, true));

    ctx1->getVelocityMass().transferToHost();
    ctx2->getVelocityMass().transferToHost();
    CHECK(CompareVectors1<double4>(ctx1->getVelocityMass().getHostArray(),
                                   ctx2->getVelocityMass().getHostArray(), 0.0,
                                   true));

    // Setup integrators
    auto integrator1 =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    integrator1->setThermostatFriction(1.0);
    integrator1->setThermostatRngSeed(randomSeed);
    integrator1->setCharmmContext(ctx1);

    auto integrator2 =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    integrator2->setThermostatFriction(1.0);
    integrator2->setThermostatRngSeed(randomSeed);
    integrator2->setCharmmContext(ctx2);

    // Propagate integrators
    integrator1->propagate(nsteps);
    integrator2->propagate(nsteps);

    // Check that coordinates match
    ctx1->getCoordinatesCharges().transferToHost();
    ctx2->getCoordinatesCharges().transferToHost();
    CHECK(CompareVectors1<double4>(ctx1->getCoordinatesCharges().getHostArray(),
                                   ctx2->getCoordinatesCharges().getHostArray(),
                                   0.0, true));

    // Sanity check that the coordinates are actually different
    CHECK(ctx1->getCoordinatesCharges()[0].x !=
          initialCoordinatesCharges1[0].x);
    CHECK(ctx1->getCoordinatesCharges()[0].y !=
          initialCoordinatesCharges1[0].y);
    CHECK(ctx1->getCoordinatesCharges()[0].z !=
          initialCoordinatesCharges1[0].z);
    CHECK(ctx2->getCoordinatesCharges()[0].x !=
          initialCoordinatesCharges2[0].x);
    CHECK(ctx2->getCoordinatesCharges()[0].y !=
          initialCoordinatesCharges2[0].y);
    CHECK(ctx2->getCoordinatesCharges()[0].z !=
          initialCoordinatesCharges2[0].z);

    // Check that velocities match
    ctx1->getVelocityMass().transferToHost();
    ctx2->getVelocityMass().transferToHost();
    CHECK(CompareVectors1<double4>(ctx1->getVelocityMass().getHostArray(),
                                   ctx2->getVelocityMass().getHostArray(), 0.0,
                                   true));
  }
}
