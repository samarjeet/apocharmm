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
#include "CudaContainer.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaMinimizer.h"
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "PDB.h"
#include "RestartSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "compare.h"
#include "helper.h"
#include "test_paths.h"
#include <iomanip>
#include <iostream>

/* Move this to a longer test
TEST_CASE("Pressure", "[unit]") {
  std::string dataPath = getDataPath();
  const float boxDim = 50.0f;

  // Read topology and parameters
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");

  // Read PSF and coordinates
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  // Set up force manager
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({boxDim, boxDim, boxDim});
  fm->setCutoff(12.0);
  fm->setCtonnb(10.0);
  fm->setCtofnb(9.0);

  // Set up CHARMM context
  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->setCoordinates(crd);

  // Set up integrator
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(
      0.002); //, CRYSTAL::CUBIC);
  integrator->setCrystalType(CRYSTAL::CUBIC);
  integrator->setPistonFriction(12.0);
  integrator->setPistonMass({500.0});
  integrator->setCharmmContext(ctx);

  // Heat from 30 -> 300 K
  float temp = 28.0f;
  while (temp < 300.0f) {
    temp += 2.0f;
    ctx->assignVelocitiesAtTemperature(temp);
    integrator->propagate(100); // 0.2 ps
  }

  // Equilibration
  integrator->propagate(5000); // 10 ps

  integrator->setDebugPrintFrequency(1000);

  SECTION("1") { integrator->setPressure({1.0, 0.0, 1.0, 0.0, 0.0, 1.0}); }
  SECTION("10") { integrator->setPressure({10.0, 0.0, 10.0, 0.0, 0.0, 10.0}); }
  SECTION("100") {
    integrator->setPressure({100.0, 0.0, 100.0, 0.0, 0.0, 100.0});
  }
  SECTION("1000") {
    integrator->setPressure({1000.0, 0.0, 1000.0, 0.0, 0.0, 1000.0});
  }
  // Simulate for 1 ns. After this, the pressure should be equilibrated
  // integrator->propagate(500000);
  integrator->propagate(5000);

  // Simulate
  long long unsigned int totNumSteps = 50000000; // 10 ns -> 100ns
  long long unsigned int numSteps = 10000;
  long long unsigned int numFrames = totNumSteps / numSteps;
  std::vector<std::vector<double>> pressureTensors(numFrames,
                                                   std::vector<double>(6, 0.0));
  std::vector<double> pressureScalars(numFrames, 0.0);
  for (long long unsigned int frame = 0; frame < numFrames; frame++) {
    integrator->propagate(numSteps);
    // Get pressure scalar and tensor
    double ps = integrator->getPressureScalar();
    pressureScalars[frame] = ps;
    std::cout << std::setw(6) << frame + 1 << "/" << numFrames << ": "
              << pressureScalars[frame] << ", ";
    std::vector<double> pt = integrator->getPressureTensor();
    for (int i = 0; i < 6; i++) {
      pressureTensors[frame][i] = pt[i];
      std::cout << pressureTensors[frame][i];
      if (i < 5)
        std::cout << ", ";
    }
    std::cout << std::endl;
  }

  // Compute averages of pressures
  double avgPs = 0.0;
  std::array<double, 6> avgPt{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (long long unsigned int frame = 0; frame < numFrames; frame++) {
    avgPs += pressureScalars[frame];
    for (int i = 0; i < 6; i++)
      avgPt[i] += pressureTensors[frame][i];
  }
  avgPs /= static_cast<double>(numFrames);
  for (int i = 0; i < 6; i++)
    avgPt[i] /= static_cast<double>(numFrames);

  // Compute standard deviations of pressures
  double stdPs = 0.0;
  std::array<double, 6> stdPt{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (long long unsigned int frame = 0; frame < numFrames; frame++) {
    double sdiff = pressureScalars[frame] - avgPs;
    double sdiff2 = sdiff * sdiff;
    stdPs += sdiff2;
    for (int i = 0; i < 6; i++) {
      double tdiff = pressureTensors[frame][i] - avgPt[i];
      double tdiff2 = tdiff * tdiff;
      stdPt[i] += tdiff2;
    }
  }
  stdPs = std::sqrt(stdPs / static_cast<double>(numFrames));
  for (int i = 0; i < 6; i++)
    stdPt[i] = std::sqrt(stdPt[i] / static_cast<double>(numFrames));

  // Print the results
  std::cout << " s: " << std::fixed << std::setprecision(4) << avgPs << " ("
            << std::fixed << std::setprecision(4) << stdPs << ")" << std::endl;
  for (int i = 0; i < 6; i++) {
    std::cout << "t" << i << ": " << std::fixed << std::setprecision(4)
              << avgPt[i] << " (" << std::fixed << std::setprecision(4)
              << stdPt[i] << ")" << std::endl;
  }
}
*/
TEST_CASE("unittest", "[basic]") {
  std::string dataPath = getDataPath();
  SECTION("NHPistonMass") {
    // Check that the mass of the Nose-Hoover is computed correctly
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50., 50., 50.});
    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx->setCoordinates(crd);

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(12.0);
    integrator->setCharmmContext(ctx);

    double nhpm = integrator->getNoseHooverPistonMass();
    // waterbox = 3916 waters
    // 1 water = 2*1.008 + 15.9994 = 18.0154
    // CHARMM-GUI heuristic : 2% of total mass
    double waterMass = 2 * 1.008 + 15.9994;
    double expectedNHPistonMass = waterMass * 3916 * 0.02;
    CHECK(nhpm == Approx(expectedNHPistonMass));
  }
  // Check that RNG seeds are handled well.
  SECTION("seed") {
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    auto integratorbis = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    // Check that seeds are different by default
    CHECK(integrator->getSeedForPistonFriction() !=
          integratorbis->getSeedForPistonFriction());

    auto firstSeed = integrator->getSeedForPistonFriction();
    integratorbis->setSeedForPistonFriction(firstSeed);
    CHECK(integrator->getSeedForPistonFriction() ==
          integratorbis->getSeedForPistonFriction());
  }
}

TEST_CASE("waterbox", "[dynamics]") {
  std::string dataPath = getDataPath();
  /* Move to extended tests
  SECTION("waterbox") {
    float expectedBoxDim = 48.9342, approxBoxDim = 50.;
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({approxBoxDim, approxBoxDim, approxBoxDim});
    fm->setCutoff(12.0);
    fm->setCtonnb(10.0);
    fm->setCtofnb(9.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    // ctx->readRestart(dataPath + "restart/waterbox.npt.restart");
    ctx->readRestart(
        "/u/aviatfel/work/apocharmm/restartgen/equilibratedRestart.res");

    auto dimFromRestart = fm->getBoxDimensions();
    std::cout << "Box dimensions from restart: " << dimFromRestart[0] << " "
              << dimFromRestart[1] << " " << dimFromRestart[2] << std::endl;

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(12.);
    integrator->setCharmmContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);

    auto densitySub =
        std::make_shared<StateSubscriber>("waterbox_lp.density", 100);
    densitySub->readReportFlags("density");
    integrator->subscribe(densitySub);

    // Every 10 steps save pressure and box size values
    int nStepsPerFrame = 10, nFrames = 1000;
    std::vector<float4> pressureValues, boxDimValues;
    float4 pressureValue, boxDimValue;
    for (int i = 0; i < nFrames; i++) {
      integrator->propagate(nStepsPerFrame);
      ctx->computePressure();
      auto pressureContainer = ctx->getPressure();
      pressureContainer.transferFromDevice();
      auto pressure = pressureContainer.getHostArray();

      pressureValue.x = pressure[0];
      pressureValue.y = pressure[4];
      pressureValue.z = pressure[8];
      pressureValues.push_back(pressureValue);

      std::vector<double> boxSize = ctx->getBoxDimensions();
      boxDimValue.x = boxSize[0];
      boxDimValue.y = boxSize[1];
      boxDimValue.z = boxSize[2];
      boxDimValues.push_back(boxDimValue);

      // std::cout << "Pressure: " << std::endl;
      // int k = 0;
      // for (int i = 0; i < 3; i++) {
      //   for (int j = 0; j < 3; j++) {
      //     std::cout << pressure[k++] << " ";
      //   }
      //   std::cout << std::endl;
      // }
    }

    // Compute averages
    float4 pressureAverage, boxDimAverage;

    for (int i = 0; i < nFrames; i++) {
      pressureAverage.x += pressureValues[i].x;
      pressureAverage.y += pressureValues[i].y;
      pressureAverage.z += pressureValues[i].z;
      boxDimAverage.x += boxDimValues[i].x;
      boxDimAverage.y += boxDimValues[i].y;
      boxDimAverage.z += boxDimValues[i].z;
    }
    boxDimAverage.x /= nFrames;
    boxDimAverage.y /= nFrames;
    boxDimAverage.z /= nFrames;
    pressureAverage.x /= nFrames;
    pressureAverage.y /= nFrames;
    pressureAverage.z /= nFrames;

    pressureAverage.w =
        (pressureAverage.x + pressureAverage.y + pressureAverage.z) / 3.0;

    CHECK(pressureAverage.w == Approx(1.0));
    CHECK(boxDimAverage.x == Approx(boxDimAverage.y).margin(0.1));
    CHECK(boxDimAverage.x == Approx(boxDimAverage.y).margin(0.1));
    CHECK(boxDimAverage.x == Approx(expectedBoxDim).margin(0.1));
  }
  */

  // Check propagation is same with same seeds
  SECTION("seed") {
    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
    std::vector<double> boxDim = {50.0, 50.0, 50.0};
    int rdmSeed = 314159, nsteps = 5000;
    double pistonMass = 0.0;
    double pistonFriction = 12.0;
    bool useHolonomicConstraints = true;
    bool useNoseHoover = true;

    // Topology, parameters, PSF, and coordinates
    auto prm1 = std::make_shared<CharmmParameters>(prmFiles);
    auto psf1 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd1 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    // Setup force manager
    auto fm1 = std::make_shared<ForceManager>(psf1, prm1);
    fm1->setBoxDimensions(boxDim);

    // Setup CHARMM context
    auto ctx1 = std::make_shared<CharmmContext>(fm1);
    ctx1->setCoordinates(crd1);
    ctx1->assignVelocitiesAtTemperature(300.0);

    // fm1->setPrintEnergyDecomposition(true);
    ctx1->useHolonomicConstraints(useHolonomicConstraints);
    ctx1->calculatePotentialEnergy(true, true);
    fm1->setPrintEnergyDecomposition(false);

    // Setup integrator
    auto integrator1 = std::make_shared<CudaLangevinPistonIntegrator>(0.001);
    integrator1->setCrystalType(CRYSTAL::CUBIC);
    integrator1->setPistonMass({pistonMass}); // setCrystalType resets this to 0
    integrator1->setPistonFriction(pistonFriction);
    integrator1->setSeedForPistonFriction(rdmSeed);
    integrator1->setNoseHooverFlag(useNoseHoover);
    integrator1->setCharmmContext(ctx1);
    // integrator1->setDebugPrintFrequency(1);

    // ================================================

    // Create a duplicate system the exact same way to ensure that the
    // integrator being used is deterministic

    // Topology, parameters, PSF, and coordinates
    auto prm2 = std::make_shared<CharmmParameters>(prmFiles);
    auto psf2 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd2 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    // Setup force manager
    auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
    fm2->setBoxDimensions(boxDim);

    // Setup CHARMM context
    auto ctx2 = std::make_shared<CharmmContext>(fm2);
    ctx2->setCoordinates(crd2);
    ctx2->assignVelocitiesAtTemperature(300.0);

    // fm2->setPrintEnergyDecomposition(true);
    ctx2->useHolonomicConstraints(useHolonomicConstraints);
    ctx2->calculatePotentialEnergy(true, true);
    fm2->setPrintEnergyDecomposition(false);

    // Setup integrator
    auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(0.001);
    integrator2->setCrystalType(CRYSTAL::CUBIC);
    integrator2->setPistonMass({pistonMass}); // setCrystalType resets this to 0
    integrator2->setPistonFriction(pistonFriction);
    integrator2->setSeedForPistonFriction(rdmSeed);
    integrator2->setNoseHooverFlag(useNoseHoover);
    integrator2->setCharmmContext(ctx2);
    // integrator2->setDebugPrintFrequency(1);

    // Both systems should match at this point

    // Check that coordinates match
    auto coordinatesCharges1 = ctx1->getCoordinatesCharges();
    auto coordinatesCharges2 = ctx2->getCoordinatesCharges();
    coordinatesCharges1.transferFromDevice();
    coordinatesCharges2.transferFromDevice();
    CHECK(CompareVectors1(coordinatesCharges1.getHostArray(),
                          coordinatesCharges2.getHostArray(), 0.0, true));

    // Check that velocities match
    auto velocityMass1 = ctx1->getVelocityMass();
    auto velocityMass2 = ctx2->getVelocityMass();
    velocityMass1.transferFromDevice();
    velocityMass2.transferFromDevice();
    CHECK(CompareVectors1(velocityMass1.getHostArray(),
                          velocityMass2.getHostArray(), 0.0, true));

    // auto coordsDeltaPrevious1 = integrator1->getCoordsDeltaPrevious();
    // auto coordsDeltaPrevious2 = integrator2->getCoordsDeltaPrevious();
    // coordsDeltaPrevious1.transferFromDevice();
    // coordsDeltaPrevious2.transferFromDevice();
    // CHECK(CompareVectors1(coordsDeltaPrevious1.getHostArray(),
    //                       coordsDeltaPrevious2.getHostArray(), 0.0, true));

    // Ensure that pressure piston variables are the same
    auto onStepPistonPosition1 = integrator1->getOnStepPistonPosition();
    auto onStepPistonPosition2 = integrator2->getOnStepPistonPosition();
    onStepPistonPosition1.transferFromDevice();
    onStepPistonPosition2.transferFromDevice();
    CHECK(CompareVectors1(onStepPistonPosition1.getHostArray(),
                          onStepPistonPosition2.getHostArray(), 0.0, true));

    auto halfStepPistonPosition1 = integrator1->getHalfStepPistonPosition();
    auto halfStepPistonPosition2 = integrator2->getHalfStepPistonPosition();
    halfStepPistonPosition1.transferFromDevice();
    halfStepPistonPosition2.transferFromDevice();
    CHECK(CompareVectors1(halfStepPistonPosition1.getHostArray(),
                          halfStepPistonPosition2.getHostArray(), 0.0, true));

    // Ensure that Nose-Hoover variables are the same
    CHECK(integrator1->getNoseHooverPistonMass() ==
          integrator2->getNoseHooverPistonMass());

    CHECK(integrator1->getNoseHooverPistonPosition() ==
          integrator2->getNoseHooverPistonPosition());

    CHECK(integrator1->getNoseHooverPistonVelocity() ==
          integrator2->getNoseHooverPistonVelocity());

    CHECK(integrator1->getNoseHooverPistonVelocityPrevious() ==
          integrator2->getNoseHooverPistonVelocityPrevious());

    CHECK(integrator1->getNoseHooverPistonForce() ==
          integrator2->getNoseHooverPistonForce());

    CHECK(integrator1->getNoseHooverPistonForcePrevious() ==
          integrator2->getNoseHooverPistonForcePrevious());

    // Propagate simulation
    // nsteps = 10;
    integrator1->propagate(nsteps);
    integrator2->propagate(nsteps);

    // Check that coordinates match
    coordinatesCharges1 = ctx1->getCoordinatesCharges();
    coordinatesCharges2 = ctx2->getCoordinatesCharges();
    coordinatesCharges1.transferFromDevice();
    coordinatesCharges2.transferFromDevice();
    CHECK(CompareVectors1(coordinatesCharges1.getHostArray(),
                          coordinatesCharges2.getHostArray(), 0.0, true));

    // Check that velocities match
    velocityMass1 = ctx1->getVelocityMass();
    velocityMass2 = ctx2->getVelocityMass();
    velocityMass1.transferFromDevice();
    velocityMass2.transferFromDevice();
    CHECK(CompareVectors1(velocityMass1.getHostArray(),
                          velocityMass2.getHostArray(), 0.0, true));

    /*
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50., 50., 50.});
    auto context1 = std::make_shared<CharmmContext>(fm);
    auto context2 = std::make_shared<CharmmContext>(fm);
    // fix seed for velocities initialization
    context2->setRandomSeedForVelocities(
        context1->getRandomSeedForVelocities());
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    context1->setCoordinates(crd);
    context2->setCoordinates(crd);
    context1->assignVelocitiesAtTemperature(300);
    context2->assignVelocitiesAtTemperature(300);

    auto integrator1 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator2->setSeedForPistonFriction(
        integrator1->getSeedForPistonFriction());
    integrator1->setPistonFriction(00.);
    integrator2->setPistonFriction(00.);
    integrator1->setCharmmContext(context1);
    integrator2->setCharmmContext(context2);
    integrator1->setCrystalType(CRYSTAL::CUBIC);
    integrator2->setCrystalType(CRYSTAL::CUBIC);

    std::shared_ptr<RestartSubscriber> restartSub1 =
                                           std::make_shared<RestartSubscriber>(
                                               "crd1.res", 1000),
                                       restartSub2 =
                                           std::make_shared<RestartSubscriber>(
                                               "crd2.res", 1000);
    integrator1->subscribe(restartSub1);
    integrator2->subscribe(restartSub2);

    integrator1->propagate(1000);
    integrator2->propagate(1000);

    auto crd1cc = context1->getCoordinatesCharges();
    auto crd2cc = context2->getCoordinatesCharges();
    crd1cc.transferFromDevice();
    crd2cc.transferFromDevice();
    std::vector<double> crd1, crd2;
    for (int i = 0; i < crd1cc.size(); i++) {
      crd1.push_back(crd1cc.getHostArray()[i].x);
      crd1.push_back(crd1cc.getHostArray()[i].y);
      crd1.push_back(crd1cc.getHostArray()[i].z);
      crd2.push_back(crd2cc.getHostArray()[i].x);
      crd2.push_back(crd2cc.getHostArray()[i].y);
      crd2.push_back(crd2cc.getHostArray()[i].z);
    }
    CHECK(compareVectors(crd1, crd2));
    */
  }
}

TEST_CASE("nve", "[dynamics]") {
  /*
  Testing NPT as a NVE.
  Debugging for box net translation.
  */
  std::string dataPath = getDataPath();
  SECTION("waterbox") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setCutoff(12.0);
    fm->setCtonnb(10.0);
    fm->setCtofnb(9.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(300.0);

    // ctx->useHolonomicConstraints(false);

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(0.0);
    integrator->setCharmmContext(ctx);
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setPistonMass({0.0, 0.0});
    integrator->setNoseHooverFlag(false);

    // integrator->setDebugPrintFrequency(1000);

    auto dcdSubscriber =
        std::make_shared<DcdSubscriber>("waterbox_lp_nve.dcd", 1000);
    integrator->subscribe(dcdSubscriber);

    integrator->propagate(1e3); // You get to propagate for 1e7 steps when
                                // you're actually asserting something
  }
}

TEST_CASE("argon") {
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "argon1000.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  fm->setBoxDimensions(boxDim);
  fm->setCutoff(12.0);
  fm->setCtonnb(10.0);
  fm->setCtofnb(9.0);

  auto ctx = std::make_shared<CharmmContext>(fm);

  auto crd = std::make_shared<CharmmCrd>(dataPath + "argon_1000.crd");
  ctx->setCoordinates(crd);

  ctx->assignVelocitiesAtTemperature(100);
  // ctx->useHolonomicConstraints(false);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(0.0);
  integrator->setCharmmContext(ctx);
  integrator->setNoseHooverFlag(false);

  SECTION("orthorhombic") {
    integrator->setCrystalType(CRYSTAL::ORTHORHOMBIC);
    integrator->setPistonMass({50000.0, 5000.0, 5000.0});
  }
  SECTION("tetragonal") {
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setPistonMass({50000.0, 5000.0});
  }
  SECTION("cubic") {
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setPistonMass({50000.0});
  }
  int dof = integrator->getPistonDegreesOfFreedom();
  // std::cout << "Piston degrees of freedom: " << dof << std::endl;
  //  integrator->setDebugPrintFrequency(1000);
  integrator->propagate(1e3); // You'll get to propagate for 1e7 steps when
                              // you're actually asserting something
}

TEST_CASE("p1crystalTypes_nph") {
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  fm->setBoxDimensions(boxDim);
  fm->setCutoff(12.0);
  fm->setCtonnb(10.0);
  fm->setCtofnb(9.0);

  auto ctx = std::make_shared<CharmmContext>(fm);

  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);

  ctx->assignVelocitiesAtTemperature(300);
  // ctx->useHolonomicConstraints(false);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.001);
  integrator->setPistonFriction(0.0);
  integrator->setCharmmContext(ctx);
  integrator->setNoseHooverFlag(false);

  integrator->setMaxPredictorCorrectorSteps(10);

  SECTION("orthorhombic") {
    integrator->setCrystalType(CRYSTAL::ORTHORHOMBIC);
    integrator->setPistonMass({500.0, 500.0, 500.0});
  }
  SECTION("tetragonal") {
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setPistonMass({500.0, 500.0});
  }
  SECTION("cubic") {
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setPistonMass({500.0});
  }

  int dof = integrator->getPistonDegreesOfFreedom();
  // std::cout << "Piston degrees of freedom: " << dof << std::endl;
  //  integrator->setDebugPrintFrequency(100);
  integrator->propagate(1e3);
}

// Test case with friction set at 12.0 and Nose-Hoover thermostat
TEST_CASE("p1crystalTypes_12") {
  std::string dataPath = getDataPath();
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  fm->setBoxDimensions(boxDim);
  fm->setCutoff(12.0);
  fm->setCtonnb(10.0);
  fm->setCtofnb(9.0);

  auto ctx = std::make_shared<CharmmContext>(fm);

  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);

  ctx->assignVelocitiesAtTemperature(300);

  auto preIntegrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  preIntegrator->setPistonFriction(12.0);
  preIntegrator->setCharmmContext(ctx);
  preIntegrator->setBathTemperature(300.0);
  preIntegrator->setCrystalType(CRYSTAL::CUBIC);
  preIntegrator->setPistonMass({10000.0});
  // preIntegrator->propagate(5e4);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(12.0);
  integrator->setCharmmContext(ctx);
  integrator->setBathTemperature(300.0);

  SECTION("orthorhombic") {
    integrator->setCrystalType(CRYSTAL::ORTHORHOMBIC);
    // integrator->setPistonMass({500.0, 500.0, 500.0});
  }
  SECTION("tetragonal") {
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    // integrator->setPistonMass({500.0, 500.0});
  }
  SECTION("cubic") {
    integrator->setCrystalType(CRYSTAL::CUBIC);
    // integrator->setPistonMass({500.0});
  }

  // integrator->setNoseHooverFlag(false);
  integrator->setPressure({1000.0, 0.0, 1000.0, 0.0, 0.0, 1000.0});
  integrator->setPistonFriction(2.0);

  int dof = integrator->getPistonDegreesOfFreedom();
  // std::cout << "Piston degrees of freedom: " << dof << std::endl;
  //  integrator->setDebugPrintFrequency(100);
  integrator->propagate(1e3);
}

TEST_CASE("p21crystalTypes_nph") {
  std::string dataPath = getDataPath();

  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);
  std::shared_ptr<CharmmPSF> psf =
      std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({50.0, 50.0, 50.0});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setCutoff(12.0);

  fm->setCtonnb(8.0);
  fm->setCtofnb(10.0);

  fm->setPeriodicBoundaryCondition(PBC::P21);

  auto ctx = std::make_shared<CharmmContext>(fm);

  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox_p21_min.crd");
  ctx->setCoordinates(crd);

  ctx->assignVelocitiesAtTemperature(300);

  CudaMinimizer minimizer;
  minimizer.setCharmmContext(ctx);
  minimizer.minimize();

  CudaLangevinThermostatIntegrator langevinThermostat(0.002);
  langevinThermostat.setFriction(5.0);
  langevinThermostat.setBathTemperature(300.0);
  langevinThermostat.setCharmmContext(ctx);
  langevinThermostat.propagate(5e3);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(5.0);
  integrator->setCharmmContext(ctx);
  // integrator->setNoseHooverFlag(false);

  SECTION("orthorhombic") {
    integrator->setCrystalType(CRYSTAL::ORTHORHOMBIC);
    integrator->setPistonMass({500.0, 500.0, 500.0});
  }
  SECTION("tetragonal") {
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setPistonMass({500.0, 500.0});
  }
  SECTION("cubic") {
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setPistonMass({500.0});
  }

  int dof = integrator->getPistonDegreesOfFreedom();
  // std::cout << "Piston degrees of freedom: " << dof << std::endl;
  //  integrator->setDebugPrintFrequency(100);
  integrator->propagate(1e3);
}

TEST_CASE("lipidBilayer") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "par_all36_lipid.prm",
                                    dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "bilayer.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  std::vector<double> boxDim = {65.72835, 65.72835, 79.90151};
  fm->setBoxDimensions(boxDim);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd =
      std::make_shared<CharmmCrd>("/u/aviatfel/work/charmm/goldenstandardruns/"
                                  "lipidbilayer/bilayer.equil.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(20.0);
  integrator->setCharmmContext(ctx);
  integrator->setCrystalType(CRYSTAL::TETRAGONAL);
  CHECK_NOTHROW(integrator->propagate(1000));
}

TEST_CASE("zack") {
  std::string dataPath = "/u/zjarin/apob/toppar/";
  std::string prmPath = "/u/zjarin/apob/toppar/";

  SECTION("beta") {
    std::vector<std::string> prmFiles{
        prmPath + "par_all36m_prot.prm",
        prmPath + "par_all36_na.prm",
        prmPath + "par_all36_carb.prm",
        prmPath + "par_all36_lipid.prm",
        prmPath + "par_all36_cgenff.prm",
        prmPath + "toppar_all36_prot_model.str",
        prmPath + "toppar_all36_prot_modify_res.str",
        prmPath + "toppar_all36_lipid_cholesterol.str",
        prmPath + "toppar_all36_lipid_tag_vanni.str",
        prmPath + "toppar_water_ions.str"};

    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(
        "/u/zjarin/for_samar/beta_sheet/fullsystem.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({144.814, 144.814, 147.789});
    // fm->setFFTGrid(48, 48, 48);
    fm->setFFTGrid(128, 128, 128);
    fm->setKappa(0.32);
    fm->setCutoff(14.0);
    fm->setCtonnb(12.0);
    fm->setCtofnb(8.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd =
        std::make_shared<PDB>("/u/zjarin/for_samar/beta_sheet/input.pdb");
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(310);

    ctx->calculatePotentialEnergy(true, true);
    auto pe = ctx->getPotentialEnergy();
    pe.transferFromDevice();
    std::cout << "Potential energy before min:" << pe[0] << std::endl;

    auto energyComponents = fm->getEnergyComponents();
    for (auto energyComponent : energyComponents) {
      std::cout << energyComponent.first << " " << energyComponent.second
                << "\n";
    }

    CudaMinimizer minimizer;
    minimizer.setCharmmContext(ctx);
    // minimizer.minimize();

    pe = ctx->getPotentialEnergy();
    pe.transferFromDevice();
    std::cout << "Potential energy after min:" << pe[0] << std::endl;

    // CudaLangevinThermostatIntegrator langevinThermostat(0.002);
    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    langevinThermostat->setFriction(5.0);
    langevinThermostat->setBathTemperature(310.0);
    langevinThermostat->setCharmmContext(ctx);
    langevinThermostat->propagate(5e3);

    std::cout << "Setting up NPT integrator" << std::endl;

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setBathTemperature(310.0);

    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setPistonMass({500.0, 500.0});
    integrator->setSurfaceTension(20.0);
    integrator->setPistonFriction(12.0);
    integrator->setCharmmContext(ctx);

    // integrator->setNoseHooverFlag(false);

    integrator->setDebugPrintFrequency(1000);
    auto dcdSubscriber =
        std::make_shared<DcdSubscriber>("zack_npgt.20.dcd", 10000);
    integrator->subscribe(dcdSubscriber);
    integrator->propagate(5e3);
    // integrator->propagate(5e6); // You'll get to propagate for 5e6 when you
    //                             // actually assert something.

    // YOU SHOULD MONITOR SOMETHING HERE
  }
}
