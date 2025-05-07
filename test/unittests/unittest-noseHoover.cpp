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
#include "CudaNoseHooverThermostatIntegrator.h"
#include "ForceManager.h"
#include "catch.hpp"
#include "compare.h"
#include "test_paths.h"
#include <iostream>

TEST_CASE("noseHooverThermostat", "[dynamics]") {
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = false;
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
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator->setCharmmContext(ctx);

    if (useHolonomicConstraints)
      CHECK(ctx->computeTemperature() == 1841.79761f);
    else
      CHECK(ctx->computeTemperature() == 304.5621f);

    integrator->propagate(nsteps);

    CudaContainer<double> averageTemperature =
        integrator->getAverageTemperature();
    averageTemperature.transferToHost();

    if (useHolonomicConstraints) {
      CHECK(ctx->computeTemperature() == 302.81256f);
      CHECK(averageTemperature[0] == Approx(299.851426339).margin(1e-8));
    } else {
      CHECK(ctx->computeTemperature() == 300.1882f);
      CHECK(averageTemperature[0] == Approx(299.9776116026).margin(1e-8));
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

    // Check that coordinates match
    CudaContainer<double4> initialCoordinatesCharges1 =
        ctx1->getCoordinatesCharges();
    CudaContainer<double4> initialCoordinatesCharges2 =
        ctx2->getCoordinatesCharges();
    initialCoordinatesCharges1.transferToHost();
    initialCoordinatesCharges2.transferToHost();
    CHECK(CompareVectors1<double4>(initialCoordinatesCharges1.getHostArray(),
                                   initialCoordinatesCharges2.getHostArray(),
                                   0.0, true));

    CudaContainer<double4> velocitiesMasses1 = ctx1->getVelocityMass();
    CudaContainer<double4> velocitiesMasses2 = ctx2->getVelocityMass();
    velocitiesMasses1.transferToHost();
    velocitiesMasses2.transferToHost();
    CHECK(CompareVectors1<double4>(velocitiesMasses1.getHostArray(),
                                   velocitiesMasses2.getHostArray(), 0.0,
                                   true));

    // Setup integrators
    auto integrator1 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator1->setCharmmContext(ctx1);

    auto integrator2 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator2->setCharmmContext(ctx2);

    // Ensure that integrator variables match
    double referenceTemperature1 = integrator1->getReferenceTemperature();
    CudaContainer<double> noseHooverPistonMass1 =
        integrator1->getNoseHooverPistonMass();
    CudaContainer<double> noseHooverPistonVelocity1 =
        integrator1->getNoseHooverPistonVelocity();
    CudaContainer<double> noseHooverPistonVelocityPrevious1 =
        integrator1->getNoseHooverPistonVelocityPrevious();
    CudaContainer<double> noseHooverPistonForce1 =
        integrator1->getNoseHooverPistonForce();
    CudaContainer<double> noseHooverPistonForcePrevious1 =
        integrator1->getNoseHooverPistonForcePrevious();
    CudaContainer<double> kineticEnergy1 = integrator1->getKineticEnergy();
    CudaContainer<double> averageTemperature1 =
        integrator1->getAverageTemperature();

    noseHooverPistonMass1.transferToHost();
    noseHooverPistonVelocity1.transferToHost();
    noseHooverPistonVelocityPrevious1.transferToHost();
    noseHooverPistonForce1.transferToHost();
    noseHooverPistonForcePrevious1.transferToHost();
    kineticEnergy1.transferToHost();
    averageTemperature1.transferToHost();

    double referenceTemperature2 = integrator2->getReferenceTemperature();
    CudaContainer<double> noseHooverPistonMass2 =
        integrator2->getNoseHooverPistonMass();
    CudaContainer<double> noseHooverPistonVelocity2 =
        integrator2->getNoseHooverPistonVelocity();
    CudaContainer<double> noseHooverPistonVelocityPrevious2 =
        integrator2->getNoseHooverPistonVelocityPrevious();
    CudaContainer<double> noseHooverPistonForce2 =
        integrator2->getNoseHooverPistonForce();
    CudaContainer<double> noseHooverPistonForcePrevious2 =
        integrator2->getNoseHooverPistonForcePrevious();
    CudaContainer<double> kineticEnergy2 = integrator2->getKineticEnergy();
    CudaContainer<double> averageTemperature2 =
        integrator2->getAverageTemperature();

    noseHooverPistonMass2.transferToHost();
    noseHooverPistonVelocity2.transferToHost();
    noseHooverPistonVelocityPrevious2.transferToHost();
    noseHooverPistonForce2.transferToHost();
    noseHooverPistonForcePrevious2.transferToHost();
    kineticEnergy2.transferToHost();
    averageTemperature2.transferToHost();

    CHECK(referenceTemperature1 == referenceTemperature2);
    CHECK(noseHooverPistonMass1[0] == noseHooverPistonMass2[0]);
    CHECK(noseHooverPistonVelocity1[0] == noseHooverPistonVelocity2[0]);
    CHECK(noseHooverPistonVelocityPrevious1[0] ==
          noseHooverPistonVelocityPrevious2[0]);
    CHECK(noseHooverPistonForce1[0] == noseHooverPistonForce2[0]);
    CHECK(noseHooverPistonForcePrevious1[0] ==
          noseHooverPistonForcePrevious2[0]);
    CHECK(kineticEnergy1[0] == kineticEnergy2[0]);
    CHECK(averageTemperature1[0] == averageTemperature2[0]);

    // Propagate integrators
    integrator1->propagate(nsteps);
    integrator2->propagate(nsteps);

    // Check that coordinates match
    CudaContainer<double4> coordinatesCharges1 = ctx1->getCoordinatesCharges();
    CudaContainer<double4> coordinatesCharges2 = ctx2->getCoordinatesCharges();
    coordinatesCharges1.transferToHost();
    coordinatesCharges2.transferToHost();
    CHECK(CompareVectors1<double4>(coordinatesCharges1.getHostArray(),
                                   coordinatesCharges2.getHostArray(), 0.0,
                                   true));

    // Sanity check that the coordinates are actually different
    CHECK(coordinatesCharges1[0].x != initialCoordinatesCharges1[0].x);
    CHECK(coordinatesCharges1[0].y != initialCoordinatesCharges1[0].y);
    CHECK(coordinatesCharges1[0].z != initialCoordinatesCharges1[0].z);
    CHECK(coordinatesCharges2[0].x != initialCoordinatesCharges2[0].x);
    CHECK(coordinatesCharges2[0].y != initialCoordinatesCharges2[0].y);
    CHECK(coordinatesCharges2[0].z != initialCoordinatesCharges2[0].z);

    // Check that velocities match
    velocitiesMasses1 = ctx1->getVelocityMass();
    velocitiesMasses2 = ctx2->getVelocityMass();
    velocitiesMasses1.transferToHost();
    velocitiesMasses2.transferToHost();
    CHECK(CompareVectors1<double4>(velocitiesMasses1.getHostArray(),
                                   velocitiesMasses2.getHostArray(), 0.0,
                                   true));

    // Compare integrator variables again
    referenceTemperature1 = integrator1->getReferenceTemperature();
    noseHooverPistonMass1 = integrator1->getNoseHooverPistonMass();
    noseHooverPistonVelocity1 = integrator1->getNoseHooverPistonVelocity();
    noseHooverPistonVelocityPrevious1 =
        integrator1->getNoseHooverPistonVelocityPrevious();
    noseHooverPistonForce1 = integrator1->getNoseHooverPistonForce();
    noseHooverPistonForcePrevious1 =
        integrator1->getNoseHooverPistonForcePrevious();
    kineticEnergy1 = integrator1->getKineticEnergy();
    averageTemperature1 = integrator1->getAverageTemperature();

    noseHooverPistonMass1.transferToHost();
    noseHooverPistonVelocity1.transferToHost();
    noseHooverPistonVelocityPrevious1.transferToHost();
    noseHooverPistonForce1.transferToHost();
    noseHooverPistonForcePrevious1.transferToHost();
    kineticEnergy1.transferToHost();
    averageTemperature1.transferToHost();

    referenceTemperature2 = integrator2->getReferenceTemperature();
    noseHooverPistonMass2 = integrator2->getNoseHooverPistonMass();
    noseHooverPistonVelocity2 = integrator2->getNoseHooverPistonVelocity();
    noseHooverPistonVelocityPrevious2 =
        integrator2->getNoseHooverPistonVelocityPrevious();
    noseHooverPistonForce2 = integrator2->getNoseHooverPistonForce();
    noseHooverPistonForcePrevious2 =
        integrator2->getNoseHooverPistonForcePrevious();
    kineticEnergy2 = integrator2->getKineticEnergy();
    averageTemperature2 = integrator2->getAverageTemperature();

    noseHooverPistonMass2.transferToHost();
    noseHooverPistonVelocity2.transferToHost();
    noseHooverPistonVelocityPrevious2.transferToHost();
    noseHooverPistonForce2.transferToHost();
    noseHooverPistonForcePrevious2.transferToHost();
    kineticEnergy2.transferToHost();
    averageTemperature2.transferToHost();

    CHECK(referenceTemperature1 == referenceTemperature2);
    CHECK(noseHooverPistonMass1[0] == noseHooverPistonMass2[0]);
    CHECK(noseHooverPistonVelocity1[0] == noseHooverPistonVelocity2[0]);
    CHECK(noseHooverPistonVelocityPrevious1[0] ==
          noseHooverPistonVelocityPrevious2[0]);
    CHECK(noseHooverPistonForce1[0] == noseHooverPistonForce2[0]);
    CHECK(noseHooverPistonForcePrevious1[0] ==
          noseHooverPistonForcePrevious2[0]);
    CHECK(kineticEnergy1[0] == kineticEnergy2[0]);
    CHECK(averageTemperature1[0] == averageTemperature2[0]);
  }
}
