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
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaLeapFrogIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "RestartSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>

// REVISIT ONCE WE HAVE AN ACTUAL RESTART FILE WORKING

TEST_CASE("waterbox", "[energy conservation]") {
  float integratorMargin;
  std::string dataPath = getDataPath();
  int nFrames = 1000, nStepsPerFrame = 100, nStepsEquilibration = 2000,
      averageOver = nFrames / 4;
  CudaContainer<double> epotCC;
  double ekin;
  std::vector<double> etotVals;
  float bathTemperature = 300.0, frictionCoeff = 12.0;

  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  // std::string path = "/u/samar/projects/benchmark/waterbox98/";
  //  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  double cubeLength1 = 98.0;
  fm->setBoxDimensions({cubeLength1, cubeLength1, cubeLength1});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(9.0);
  fm->setCtonnb(8.0);
  fm->setCtofnb(8.5);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  // auto crd = std::make_shared<CharmmCrd>(path + "step3_pbcsetup.crd");
  ctx->setCoordinates(crd);
  ctx->calculatePotentialEnergy(true, true);
  ctx->getPotentialEnergy();

  CudaMinimizer minimizer;
  minimizer.setSimulationContext(ctx);
  minimizer.minimize(10000);

  ctx->calculatePotentialEnergy(true, true);
  ctx->getPotentialEnergy();

  ctx->assignVelocitiesAtTemperature(300);
  auto restartSub =
      std::make_shared<RestartSubscriber>("waterboxRestart.res", 10000);

  auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(
      0.001, bathTemperature, frictionCoeff);
  integrator->setFriction(0.0);
  // integrator.setDebugPrintFrequency(100);
  integrator->setSimulationContext(ctx);
  // integrator.subscribe(restartSub);

  integrator->propagate(100);
}

/*
   SECTION("water2") {

  auto prm =
std::make_shared<CharmmParameters>("../test/data/toppar_water_ions.str"); auto
psf = std::make_shared<CharmmPSF>("../test/data/water2.psf");

  //auto fm = std::make_unique<ForceManager>(psf, prm);
  auto fm = std::make_shared<ForceManager>(psf, prm);
  double dim = 50.0;

  fm->setBoxDimensions({dim, dim, dim});
  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->readRestart(dataPath + "restart/waterbox.npt.restart");
  ctx->assignVelocitiesAtTemperature(300.0);

  // THIS SHOULD BE TESTED IN THE CONTEXT FILE !
  REQUIRE(ctx->getVolume() == Approx(50.0 * 50.0 * 50.0));

  SECTION("velocity Verlet") {
    integratorMargin = .01;
    auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.001);
    integrator->setSimulationContext(ctx);
    for (int i = 0; i < nFrames; i++) {
      integrator->propagate(nStepsPerFrame);
      epotCC = ctx->getPotentialEnergy();
      epotCC.transferToHost();
      double epot = epotCC[0];
      ekin = ctx->getKineticEnergy();
      etotVals.push_back(epot + ekin);
    }
  }
  SECTION("Langevin thermostat") {
    integratorMargin = .01;
    auto integrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.001, 300, 0.0);
    integrator->setSimulationContext(ctx);
    for (int i = 0; i < nFrames; i++) {
      integrator->propagate(nStepsPerFrame);
      epotCC = ctx->getPotentialEnergy();
      epotCC.transferToHost();
      double epot = epotCC[0];
      ekin = ctx->getKineticEnergy();
      etotVals.push_back(epot + ekin);
    }
  }
  SECTION("Langevin piston") {
    integratorMargin = .01;
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.001);
    integrator->setPistonFriction(0.0);
    integrator->setSimulationContext(ctx);
    integrator->setBathTemperature(300.0);
    for (int i = 0; i < nFrames; i++) {
      integrator->propagate(nStepsPerFrame);
      epotCC = ctx->getPotentialEnergy();
      epotCC.transferToHost();
      double epot = epotCC[0];
      ekin = ctx->getKineticEnergy();
      etotVals.push_back(epot + ekin);
    }
  }
}
*/