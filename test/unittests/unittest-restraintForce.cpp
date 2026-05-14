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
#include "DcdSubscriber.h"
#include "ForceManager.h"
#include "HarmonicRestraintForce.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <memory>

TEST_CASE("harmonicRestraintForce") {
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const int nstep = 10000;
  const double timeStep = 0.002;

  SECTION("nacl") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "nacl_pair.psf");
    auto crd = std::make_shared<CharmmCrd>(dataPath + "nacl_pair.cor");

    // Setup force manager
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions(boxDims);

    // Setup CHARMM context
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);
    ctx->setRandomSeedForVelocities(randomSeed);
    ctx->assignVelocitiesAtTemperature(temperature);

    // Setup integrator
    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    integrator->setThermostatFriction(1.0);
    integrator->setThermostatRngSeed(randomSeed);
    integrator->setCharmmContext(ctx);

    // DCD subscriber to visualize effects of restraints
    // Uncomment the lines below if you want to write out the trajectory
    // auto dcd = std::make_shared<DcdSubscriber>("tmpHarmRestraint.dcd", 1);
    // integrator->subscribe(dcd);

    // Setup harmonic restraint
    auto harm =
        std::make_shared<HarmonicRestraintForce<long long int, float>>();
    harm->initialize(ctx->getNumAtoms(), boxDims);
    harm->setReferenceCoordinates(crd->getCoordinatesD());
    // harm->setMasses(psf->getMasses());
    fm->subscribe(harm, "HarmonicRestraint", harm->getStream(),
                  harm->getForce(), harm->getEnergyVirial());

    // Compute initial distance between Na-Cl before any dynamics
    CudaContainer<double4> &xyzq = ctx->getCoordinatesCharges();
    xyzq.transferToHost();
    const double dx0 = xyzq[1].x - xyzq[0].x;
    const double dy0 = xyzq[1].y - xyzq[0].y;
    const double dz0 = xyzq[1].z - xyzq[0].z;
    const double r0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
    std::cout << "r0 = " << r0 << std::endl;

    // Run dynamics with progressively stronger harmonic restraints.
    // (NOT ACTUALLY GOOD PRACTICE, THIS IS ONLY FOR DEMO/TESTING)
    double forceConstant = 1e-8;
    for (int i = 0; i < 14; i++) {
      harm->setForceConstant(forceConstant);
      integrator->propagate(1000);
      forceConstant *= 10.0;
    }

    xyzq.transferToHost();
    const double dx = xyzq[1].x - xyzq[0].x;
    const double dy = xyzq[1].y - xyzq[0].y;
    const double dz = xyzq[1].z - xyzq[0].z;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    std::cout << "r = " << r << std::endl;

    CHECK(std::abs(r0 - r) < 0.01);
  }
}
