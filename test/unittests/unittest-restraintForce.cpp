// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaRestraintForce.h"
#include "ForceManager.h"
#include "GeometricRestraintForce.h"
#include "catch.hpp"
#include "test_paths.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

TEST_CASE("restraintForce", "[energy]") {
  std::string dataPath = getDataPath();

  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "water2_1.psf");

  auto numAtoms = psf->getNumAtoms();

  auto restraintForceValues = std::make_shared<Force<long long int>>();
  restraintForceValues->realloc(numAtoms, 1.5f);

  CudaEnergyVirial restraintEnergyVirial;
  /*auto restraint = std::make_shared<GeometricRestraintForce<long long,
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
  */
}
