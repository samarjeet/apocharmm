// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include <iostream>

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaLeapFrogIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "StateSubscriber.h"
#include "catch.hpp"

#include "test_paths.h"

std::vector<std::vector<float>> readTableFromFile(std::string fileName) {
  std::ifstream fin;
  fin.open(fileName);
  std::vector<std::vector<float>> vel;

  while (fin.good()) {
    float vx, vy, vz;
    fin >> vx >> vy >> vz;
    vel.push_back({vx, vy, vz});
  }
  fin.close();
  return vel;
}

// Verify energy conservation with different integrator->, over 100k steps
// - Velocity Verlet
// - Langevin thermostat
TEST_CASE("argon1000", "[energy conservation]") {
  float integratorMargin;
  std::string dataPath = getDataPath();
  int nFrames = 500, nStepsPerFrame = 100, averageOver = nFrames / 4;
  CudaContainer<double> epotCC;
  double ekin;
  std::vector<double> etotVals;
  auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "argon1000.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  double dim = 50.0;

  fm->setBoxDimensions({dim, dim, dim});
  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->readRestart(dataPath + "restart/argon1000.restart");
  std::cout << "restart read." << std::endl;
  ctx->assignVelocitiesAtTemperature(300.0);

  SECTION("velocityVerlet") {
    integratorMargin = .01;
    auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
    integrator->setCharmmContext(ctx);
    for (int i = 0; i < nFrames; i++) {
      integrator->propagate(nStepsPerFrame);
      ctx->calculateForces(true, true);
      epotCC = ctx->getPotentialEnergy();
      epotCC.transferToHost();
      double epot = epotCC[0];
      ekin = ctx->getKineticEnergy();
      etotVals.push_back(epot + ekin);
    }
  }

  // Use no friction to test energy conservation
  SECTION("LangevinThermostat") {
    integratorMargin = .1;
    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.001, 300., 0.0);
    integrator->setCharmmContext(ctx);
    for (int i = 0; i < nFrames; i++) {
      integrator->propagate(nStepsPerFrame);
      ctx->calculateForces(true, true);
      epotCC = ctx->getPotentialEnergy();
      epotCC.transferToHost();
      double epot = epotCC[0];
      ekin = ctx->getKineticEnergy();
      etotVals.push_back(epot + ekin);
    }
  }

  // Check energy conservation
  double eAverageInitial = 0, etotAverageFinal = 0, etotAverage = 0;
  for (int i = 0; i < averageOver; i++) {
    eAverageInitial += etotVals[averageOver + i];
    etotAverageFinal += etotVals[nFrames - averageOver + i];
  }

  for (int i = 0; i < nFrames; i++) {
    etotAverage += etotVals[i];
  }
  etotAverage /= nFrames;
  eAverageInitial /= averageOver;
  etotAverageFinal /= averageOver;
  double eDiff = eAverageInitial - etotAverageFinal;
  double stddev = 0;
  for (int i = 0; i < nFrames; i++) {
    stddev += (etotVals[i] - etotAverage) * (etotVals[i] - etotAverage);
  }

  stddev /= nFrames;
  stddev = std::sqrt(stddev);

  std::cout << "Average initial energy: " << eAverageInitial << std::endl;
  std::cout << "Average final energy: " << etotAverageFinal << std::endl;
  std::cout << "Average en ergy : " << etotAverage << std::endl;
  std::cout << "Stddev: " << std::endl;
  CHECK(eAverageInitial == Approx(etotAverageFinal).margin(integratorMargin));
  CHECK(std::abs(eDiff) <= 1.0);
}
