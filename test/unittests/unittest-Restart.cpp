// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:   Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "ForceManager.h"
#include "RestartSubscriber.h"
#include "catch.hpp"
#include "compare.h"
#include "helper.h"
#include "test_paths.h"
#include <iostream>

// Get single-line entry frm restart file (boxdim, On step piston velocity... )
std::vector<double> getRestartFileEntry(std::string fname,
                                        std::string entryName) {
  std::ifstream f(fname);
  std::string line;
  bool found = false;
  while (std::getline(f, line)) {
    if (line.find(entryName) != std::string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::cout << "Entry " << entryName << " not found in " << fname
              << std::endl;
    throw std::invalid_argument("Entry not found in restart file.");
  }
  std::getline(f, line);
  std::stringstream ss(line);
  // Could be one, two or three values on the line
  std::vector<double> outVec;
  double x;
  while (ss >> x)
    outVec.push_back(x);
  f.close();
  return outVec;
}

std::shared_ptr<CudaLangevinThermostatIntegrator>
setupLangevinThermostatIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
  integrator->setSimulationContext(ctx);
  integrator->setFriction(12.0);
  integrator->setBathTemperature(300.0);
  return integrator;
}

std::shared_ptr<CudaLangevinPistonIntegrator>
setupLangevinPistonIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(20.0);
  integrator->setSimulationContext(ctx);
  integrator->setCrystalType(CRYSTAL::CUBIC);
  return integrator;
}

std::shared_ptr<CudaVelocityVerletIntegrator>
setupVelocityVerletIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.002);
  integrator->setSimulationContext(ctx);
  return integrator;
}

// Check that the integrator runs and creates a restart file
void checkRuns(std::shared_ptr<CudaIntegrator> integrator,
               std::string fileName) {
  auto restartSub = std::make_shared<RestartSubscriber>(fileName, 10);
  integrator->subscribe(restartSub);
  CHECK_NOTHROW(integrator->propagate(20));
  // Check it creates and fills up a restart file
  integrator->unsubscribe(restartSub);
  std::ifstream restartFile(fileName);
  CHECK(restartFile);
  CHECK(restartFile.peek() != std::ifstream::traits_type::eof());
  restartFile.close();
}

// Check that the integrator can use previously created restart file
void checkRestart(std::shared_ptr<CudaIntegrator> integrator,
                  std::string fileName) {
  auto restartSub = std::make_shared<RestartSubscriber>(fileName, 10);
  integrator->subscribe(restartSub);
  CHECK_NOTHROW(restartSub->readRestart());
  CHECK_NOTHROW(integrator->propagate(10));
}

// Test that for each integrator, we're capable of generating a restart file and
// reading it. Only tests if it runs, no testing is done regarding the actual
// content quality.
TEST_CASE("integratorCoverage") {
  // Setup system
  std::string dataPath = getDataPath();
  std::string restartFileName;
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50., 50., 50.});
  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300.0);

  auto fm2 = std::make_shared<ForceManager>(psf, prm);
  fm2->setBoxDimensions({50., 50., 50.});
  auto ctx2 = std::make_shared<CharmmContext>(fm2);
  ctx2->setCoordinates(crd);

  // Prepare one integrator for the propagation and restart saving, and
  // another for the restart reading
  SECTION("langevinPiston") {
    std::shared_ptr<CudaLangevinPistonIntegrator> integrator =
        setupLangevinPistonIntegrator(ctx);
    restartFileName = "langevinPistonRestartTest.res";
    checkRuns(integrator, restartFileName);

    auto integrator2 = setupLangevinPistonIntegrator(ctx2);
    checkRestart(integrator2, restartFileName);
  }

  SECTION("velocityVerlet") {
    auto integrator = setupVelocityVerletIntegrator(ctx);
    restartFileName = "velocityVerletRestartTest.res";
    checkRuns(integrator, restartFileName);
    auto integrator2 = setupVelocityVerletIntegrator(ctx2);
    checkRestart(integrator2, restartFileName);
  }

  SECTION("langevinThermostat") {
    auto integrator = setupLangevinThermostatIntegrator(ctx);
    restartFileName = "langevinThermostatRestartTest.res";
    checkRuns(integrator, restartFileName);
    auto integrator2 = setupLangevinThermostatIntegrator(ctx2);
    checkRestart(integrator2, restartFileName);
  }
}

TEST_CASE("restart") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  fm->setBoxDimensions(boxDim);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  CudaContainer<double> onStepPistonPosition, halfStepPistonPosition,
      onStepPistonVelocity, halfStepPistonVelocity;

  double pistonNHposition, pistonNHvelocity, pistonNHforce;

  std::string fileName = "restartWater.res";

  SECTION("save") {
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setSimulationContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->propagate(20);

    auto myRestartSubWrite = std::make_shared<RestartSubscriber>(fileName, 30);
    integrator->subscribe(myRestartSubWrite);
    // Propagating for 30 (resp 60) steps only, doesnt work
    // ? 31 (resp 61) seems to fix. Must be the order in which variables are
    // updated. Not sure why/how
    integrator->propagate(30);

    // Save values as references to compare later them with the subscriber
    // reading functions results
    auto refBoxDim = integrator->getBoxDimensions();
    auto ctxBoxDim = ctx->getBoxDimensions();
    onStepPistonPosition = integrator->getOnStepPistonPosition();
    halfStepPistonPosition = integrator->getHalfStepPistonPosition();
    onStepPistonVelocity = integrator->getOnStepPistonVelocity();
    halfStepPistonVelocity = integrator->getHalfStepPistonVelocity();
    CudaContainer<double4> coords = ctx->getCoordinatesCharges();
    CudaContainer coordsDeltaPrevious = integrator->getCoordsDeltaPrevious();
    coords.transferFromDevice();
    coordsDeltaPrevious.transferFromDevice();
    int numAtoms = coords.size();
    std::vector<double> refCoordFirst = {coords[0].x, coords[0].y, coords[0].z},
                        refCoordLast = {coords[numAtoms - 1].x,
                                        coords[numAtoms - 1].y,
                                        coords[numAtoms - 1].z},
                        refCoordDeltaFirst = {coordsDeltaPrevious[0].x,
                                              coordsDeltaPrevious[0].y,
                                              coordsDeltaPrevious[0].z},
                        refCoordDeltaLast = {
                            coordsDeltaPrevious[numAtoms - 1].x,
                            coordsDeltaPrevious[numAtoms - 1].y,
                            coordsDeltaPrevious[numAtoms - 1].z};

    pistonNHposition = integrator->getPistonNoseHooverPosition();
    pistonNHvelocity = integrator->getPistonNoseHooverVelocityPrevious();
    pistonNHforce = integrator->getPistonNoseHooverForcePrevious();

    // Weird bug: using NoseHoover (setNoseHooverFlag(true))
    // causes the gdb-run version to crash (velocities explode).

    // Testing the reading part
    // Assert that content of LP variables (piston vel...) is the
    // same as calculated above
    auto myRestartSub = std::make_shared<RestartSubscriber>(fileName, 30);

    auto restartFileBoxDim = myRestartSub->readBoxDimensions();
    CHECK(compareVectors(restartFileBoxDim, refBoxDim, true));

    auto restartFileOnStepPistonPosition =
        myRestartSub->readOnStepPistonPosition();
    CHECK(compareVectors(restartFileOnStepPistonPosition,
                         onStepPistonPosition.getHostArray(), true));

    auto restartFileHalfStepPistonPosition =
        myRestartSub->readHalfStepPistonPosition();
    CHECK(compareVectors(restartFileHalfStepPistonPosition,
                         halfStepPistonPosition.getHostArray()));

    auto restartFileOnStepPistonVelocity =
        myRestartSub->readOnStepPistonVelocity();
    CHECK(compareVectors(restartFileOnStepPistonVelocity,
                         onStepPistonVelocity.getHostArray()));

    auto restartFilePositions = myRestartSub->readPositions();
    REQUIRE(restartFilePositions.size() == coords.size());
    CHECK(compareVectors(restartFilePositions[0], refCoordFirst));
    CHECK(compareVectors(restartFilePositions[numAtoms - 1], refCoordLast));

    auto restartFileCoordsDeltaPrevious =
        myRestartSub->readCoordsDeltaPrevious();
    REQUIRE(restartFileCoordsDeltaPrevious.size() ==
            coordsDeltaPrevious.size());
    CHECK(
        compareVectors(restartFileCoordsDeltaPrevious[0], refCoordDeltaFirst));
    CHECK(compareVectors(restartFileCoordsDeltaPrevious[numAtoms - 1],
                         refCoordDeltaLast));

    // Nose-Hoover variables
    auto restartFileNHposition = myRestartSub->readNoseHooverPistonPosition();
    auto restartFileNHvelocity = myRestartSub->readNoseHooverPistonVelocity();
    auto restartFileNHforce = myRestartSub->readNoseHooverPistonForce();
    CHECK(restartFileNHposition == Approx(pistonNHposition));
    CHECK(restartFileNHvelocity == Approx(pistonNHvelocity));
    CHECK(restartFileNHforce == Approx(pistonNHforce));
  }

  SECTION("io") {
    // Make sure we can write stuff to a restart file but also read stuff from
    // it
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setSimulationContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    auto writeSub = std::make_shared<RestartSubscriber>(fileName, 10);

    integrator->subscribe(writeSub);
    integrator->propagate(20);
    integrator->unsubscribe(writeSub); // should close file

    // Check that the restart file has been created. Check it is not empty.
    std::fstream ifile(fileName, std::ios::in);
    CHECK(ifile); // Fails if file was not created.
    CHECK(ifile.peek() !=
          std::ifstream::traits_type::eof()); // Fails if file is empty"
    ifile.close();

    // Open and read that restart file through another sub
    auto readSub = std::make_shared<RestartSubscriber>(fileName, 10);
    // Check this hasn't emptied the file. Fails if file is empty
    ifile.open(fileName);
    CHECK(ifile.peek() != std::ifstream::traits_type::eof());

    integrator->subscribe(readSub);
    CHECK_NOTHROW(readSub->readRestart());
  }

  // This ensures that the restart file contains the coordinates of step
  // N*saveFreq-1 (in the code), rather than N*saveFreq. This ensures that, for
  // example, for a simulation of 200 steps and a restart save frequency of 100,
  // the restart file is updated at steps "100" (99 for the code) and "200" (199
  // for the code), rather than steps 0 and 100 for the code.
  // In other words, it makes sure the very last step of the propagation, if
  // nStepsTotal = N*saveFreq, is saved.
  SECTION("savingPoint") {
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setSimulationContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    auto savingpointres1 =
        std::make_shared<RestartSubscriber>("savingPoint1.res", 10);
    integrator->subscribe(savingpointres1);
    integrator->propagate(10);

    // Coords in the subscriber should correspond to the ones in the context
    // (after 10 step)
    auto ctxCoords = ctx->getCoordinatesCharges();
    ctxCoords.transferFromDevice();
    auto savingpointCoords = savingpointres1->readPositions();
    CHECK(compareVectors(savingpointCoords[0],
                         {ctxCoords[0].x, ctxCoords[0].y, ctxCoords[0].z}));
  }

  // Test that the wrapper (for all reading + ctx/integrator setting) works.
  // Requires the "save" section to have run first
  SECTION("readWrapped") {
    auto readRestartSub = std::make_shared<RestartSubscriber>(fileName, 30);
    // Check the restart file content is the expected number of lines
    REQUIRE(lineCounter(fileName) == ctx->getNumAtoms() * 3 + 10 + 7 + 9);

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setSimulationContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);

    integrator->subscribe(readRestartSub);

    CHECK_NOTHROW(readRestartSub->readRestart());
    // Assert the integrator AND context have the right values distributes
    // Context: boxdim, coords, velocities,
    // Integrator: coordsdeltaprevious, on-step piston pos, on-step piston
    // vel, half-step piston pos, N-H piston pos, N-H piston vel, N-H piston
    // force
    auto boxDimRef = getRestartFileEntry(fileName, "!BOXX");
    auto firstCoordRef = getRestartFileEntry(fileName, "!X,");
    auto firstVelRef = getRestartFileEntry(fileName, "!VX");
    auto firstCoordsDeltaPreviousRef = getRestartFileEntry(fileName, "!XOLD");
    auto osppref = getRestartFileEntry(fileName, "!onStepPistonPosition");
    auto ospvref = getRestartFileEntry(fileName, "!onStepPistonVelocity");
    auto hspref = getRestartFileEntry(fileName, "!halfStepPistonPosition");
    auto nhpref = getRestartFileEntry(fileName, "!pistonNoseHooverPosition");
    auto nhvref = getRestartFileEntry(fileName, "!pistonNoseHooverVelocity");
    auto nhfref = getRestartFileEntry(fileName, "!pistonNoseHooverForce");

    std::vector<double> firstCoord, firstVel, firstCoordsDeltaPrevious;
    CudaContainer<double4> ctxCoords = ctx->getCoordinatesCharges(),
                           ctxVel = ctx->getVelocityMass(),
                           coordsDeltaPrevious =
                               integrator->getCoordsDeltaPrevious();
    CudaContainer<double> ospp = integrator->getOnStepPistonPosition(),
                          ospv = integrator->getOnStepPistonVelocity(),
                          hsp = integrator->getHalfStepPistonPosition();
    std::vector<double> nhpp = {integrator->getPistonNoseHooverPosition()},
                        nhpv =
                            {integrator->getPistonNoseHooverVelocityPrevious()},
                        nhpf = {integrator->getPistonNoseHooverForcePrevious()};
    ctxCoords.transferFromDevice();
    ctxVel.transferFromDevice();
    coordsDeltaPrevious.transferFromDevice();
    firstCoord = {ctxCoords[0].x, ctxCoords[0].y, ctxCoords[0].z};
    firstVel = {ctxVel[0].x, ctxVel[0].y, ctxVel[0].z};
    firstCoordsDeltaPrevious = {coordsDeltaPrevious[0].x,
                                coordsDeltaPrevious[0].y,
                                coordsDeltaPrevious[0].z};

    CHECK(compareVectors(boxDimRef, ctx->getBoxDimensions(), true));
    CHECK(compareVectors(boxDimRef, integrator->getBoxDimensions(), true));

    CHECK(compareVectors(firstCoordRef, firstCoord));
    CHECK(compareVectors(firstVelRef, firstVel));
    CHECK(
        compareVectors(firstCoordsDeltaPreviousRef, firstCoordsDeltaPrevious));
    CHECK(compareVectors(osppref, ospp.getHostArray()));
    CHECK(compareVectors(ospvref, ospv.getHostArray()));
    CHECK(compareVectors(hspref, hsp.getHostArray()));
    CHECK(compareVectors(nhpp, nhpref));
    CHECK(compareVectors(nhpv, nhvref));
    CHECK(compareVectors(nhpf, nhfref));

    // Check that you can get going from there
    CHECK_NOTHROW(integrator->propagate(10));

    // Verify that giving wrong Crystal type throws an error
    auto wrongCrystalIntegrator =
        std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    wrongCrystalIntegrator->setPistonFriction(20.0);
    wrongCrystalIntegrator->setSimulationContext(ctx);
    wrongCrystalIntegrator->setCrystalType(CRYSTAL::TETRAGONAL);
    auto wrongRestartSub = std::make_shared<RestartSubscriber>(fileName, 30);
    wrongCrystalIntegrator->subscribe(wrongRestartSub);
    CHECK_THROWS(wrongRestartSub->readRestart());
  }

  /* *
  // Test that (with a no-randomness integrator) the result of a propagation
  // starting from the restart file is the same as the one starting from the
  // initial conditions.
  // For now, only tests it for deterministic prop (no friction in the
  // langevin piston)
  SECTION("identicalPropagation") {
    int rdmSeed = 144, nsteps = 1000;

    // Setup integrator
    auto integrator1 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator1->setCrystalType(CRYSTAL::CUBIC);
    // integrator1->setPistonFriction(12.0);
    integrator1->setPistonFriction(0.0);
    integrator1->setSeedForPistonFriction(rdmSeed);
    integrator1->setNoseHooverFlag(false);
    integrator1->setSimulationContext(ctx);
    // auto integrator1 =
    //     std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    // integrator1->setFriction(0.0);
    // integrator1->setSimulationContext(ctx);

    // Create a restart file
    auto restartsub = std::make_shared<RestartSubscriber>("idprop.res", nsteps);
    integrator1->subscribe(restartsub);
    integrator1->propagate(nsteps);
    integrator1->unsubscribe(restartsub);

    auto xyzq1 = ctx->getCoordinatesCharges();
    xyzq1.transferFromDevice();

    // Make sure that the integrator is actually deterministic
    auto prm2 = std::make_shared<CharmmParameters>(prmFiles);
    auto psf2 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

    auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
    fm2->setBoxDimensions(boxDim);

    auto ctx2 = std::make_shared<CharmmContext>(fm2);
    auto crd2 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    ctx2->setCoordinates(crd2);
    ctx2->assignVelocitiesAtTemperature(300);

    // Setup second integrator and run dynamics to ensure that trajectories are
    // deterministic
    auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator2->setCrystalType(CRYSTAL::CUBIC);
    // integrator2->setPistonFriction(12.0);
    integrator2->setPistonFriction(0.0);
    integrator2->setSeedForPistonFriction(rdmSeed);
    integrator2->setNoseHooverFlag(false);
    integrator2->setSimulationContext(ctx2);
    // auto integrator2 =
    //     std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    // integrator2->setFriction(0.0);
    // integrator2->setSimulationContext(ctx2);

    integrator2->propagate(nsteps);
    auto xyzq2 = ctx2->getCoordinatesCharges();
    xyzq2.transferFromDevice();
    CHECK(
        CompareVectors1(xyzq1.getHostArray(), xyzq2.getHostArray(), 0.0, true));

    // Setup third integrator and run dynamics using the restart file
    auto prm3 = std::make_shared<CharmmParameters>(prmFiles);
    auto psf3 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

    auto fm3 = std::make_shared<ForceManager>(psf3, prm3);
    fm3->setBoxDimensions(boxDim);

    auto ctx3 = std::make_shared<CharmmContext>(fm3);
    auto crd3 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx3->setCoordinates(crd3);

    auto integrator3 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator3->setCrystalType(CRYSTAL::CUBIC);
    // integrator3->setPistonFriction(12.0);
    integrator3->setPistonFriction(0.0);
    integrator3->setSeedForPistonFriction(rdmSeed);
    integrator3->setNoseHooverFlag(false);
    integrator3->setSimulationContext(ctx3);
    // auto integrator3 =
    //     std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    // integrator3->setFriction(0.0);
    // integrator3->setSimulationContext(ctx2);

    auto restartsub3 = std::make_shared<RestartSubscriber>("idprop.res");
    integrator3->subscribe(restartsub3);
    restartsub3->readRestart();

    auto xyzq3 = ctx3->getCoordinatesCharges();
    xyzq3.transferFromDevice();

    // Ensure that after reading the restart file coordinates match
    CHECK(CompareVectorsPBC1(xyzq1.getHostArray(), xyzq2.getHostArray(),
                             {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0,
                             true));
    CHECK(CompareVectorsPBC1(xyzq1.getHostArray(), xyzq3.getHostArray(),
                             {boxDim[0], boxDim[1], boxDim[2], 0.0}, 1.0e-6,
                             true));
    CHECK(CompareVectorsPBC1(xyzq2.getHostArray(), xyzq3.getHostArray(),
                             {boxDim[0], boxDim[1], boxDim[2], 0.0}, 1.0e-6,
                             true));

    {
      auto tmp1 = integrator1->getCoordsDelta();
      auto tmp2 = integrator2->getCoordsDelta();
      auto tmp3 = integrator3->getCoordsDelta();
      tmp1.transferFromDevice();
      tmp2.transferFromDevice();
      tmp3.transferFromDevice();
      CHECK(CompareVectorsPBC1(tmp1.getHostArray(), tmp2.getHostArray(),
                               {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0,
                               true));
      CHECK(CompareVectorsPBC1(tmp1.getHostArray(), tmp3.getHostArray(),
                               {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0,
                               true));
      CHECK(CompareVectorsPBC1(tmp2.getHostArray(), tmp3.getHostArray(),
                               {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0,
                               true));
    }

    // Ensure that Nose-Hoover variables are the same
    CHECK(integrator1->getNoseHooverPistonMass() ==
          integrator2->getNoseHooverPistonMass());
    CHECK(integrator1->getNoseHooverPistonMass() ==
          integrator3->getNoseHooverPistonMass());
    CHECK(integrator2->getNoseHooverPistonMass() ==
          integrator3->getNoseHooverPistonMass());

    CHECK(integrator1->getPistonNoseHooverPosition() ==
          integrator2->getPistonNoseHooverPosition());
    CHECK(integrator1->getPistonNoseHooverPosition() ==
          integrator3->getPistonNoseHooverPosition());
    CHECK(integrator2->getPistonNoseHooverPosition() ==
          integrator3->getPistonNoseHooverPosition());

    CHECK(integrator1->getPistonNoseHooverVelocity() ==
          integrator2->getPistonNoseHooverVelocity());
    CHECK(integrator1->getPistonNoseHooverVelocity() ==
          integrator3->getPistonNoseHooverVelocity());
    CHECK(integrator2->getPistonNoseHooverVelocity() ==
          integrator3->getPistonNoseHooverVelocity());

    CHECK(integrator1->getPistonNoseHooverForce() ==
          integrator2->getPistonNoseHooverForce());
    CHECK(integrator1->getPistonNoseHooverForce() ==
          integrator3->getPistonNoseHooverForce());
    CHECK(integrator2->getPistonNoseHooverForce() ==
          integrator3->getPistonNoseHooverForce());

    // integrator1->setSeedForPistonFriction(rdmSeed + 1);
    // integrator2->setSeedForPistonFriction(rdmSeed + 1);
    // integrator3->setSeedForPistonFriction(rdmSeed + 1);

    integrator1->propagate(nsteps);
    integrator2->propagate(nsteps);
    integrator3->propagate(nsteps);

    xyzq1 = ctx->getCoordinatesCharges();
    xyzq1.transferFromDevice();
    xyzq2 = ctx2->getCoordinatesCharges();
    xyzq2.transferFromDevice();
    xyzq3 = ctx3->getCoordinatesCharges();
    xyzq3.transferFromDevice();

    // Ensure that after propagating more steps the coordinates match
    CHECK(
        CompareVectors1(xyzq1.getHostArray(), xyzq2.getHostArray(), 0.0, true));
    CHECK(
        CompareVectors1(xyzq1.getHostArray(), xyzq3.getHostArray(), 0.0, true));
    CHECK(
        CompareVectors1(xyzq2.getHostArray(), xyzq3.getHostArray(), 0.0, true));

    // // Using an independent sim with same seed, initialize from restart file,
    // // run for nsteps, and finally compare ctx1 and ctx2 variables
    // auto prm2 = std::make_shared<CharmmParameters>(prmFiles);
    // auto psf2 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    // auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
    // fm2->setBoxDimensions(boxDim);

    // auto ctx2 = std::make_shared<CharmmContext>(fm2);
    // auto crd2 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    // ctx2->setCoordinates(crd2);

    // auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    // integrator2->setCrystalType(CRYSTAL::CUBIC);
    // integrator2->setPistonFriction(0.0);
    // integrator2->setSimulationContext(ctx2);
    // integrator2->setSeedForPistonFriction(rdmSeed);
    // auto restartsub2 =
    //     std::make_shared<RestartSubscriber>("idprop.res", 4 * nsteps);
    // integrator2->subscribe(restartsub2);
    // // restartsub2->readRestart();

    // auto xyzq2 = ctx2->getCoordinatesCharges();
    // xyzq2.transferFromDevice();
    // std::cout
    //     << "!X, Y, Z (atom 0): " << xyzq2.getHostArray()[0].x << " "
    //     << xyzq2.getHostArray()[0].y << " " << xyzq2.getHostArray()[0].z
    //     << " > ctx2, after   0 steps (compare with idprop.res, first entry)"
    //     << std::endl;

    // integrator2->propagate(nsteps);
    // integrator2->propagate(nsteps);

    // xyzq2 = ctx2->getCoordinatesCharges();
    // xyzq2.transferFromDevice();
    // std::cout << "!X, Y, Z (atom 0): " << xyzq2.getHostArray()[0].x << " "
    //           << xyzq2.getHostArray()[0].y << " " <<
    //           xyzq2.getHostArray()[0].z
    //           << " > ctx2, after 100 steps (should be identical to ctx1 after
    //           "
    //              "200 steps)"
    //           << std::endl;

    // CHECK(xyzq1.getHostArray()[0].x == xyzq2.getHostArray()[0].x);
    // CHECK(xyzq1.getHostArray()[0].y == xyzq2.getHostArray()[0].y);
    // CHECK(xyzq1.getHostArray()[0].z == xyzq2.getHostArray()[0].z);
  }
  * */
}

// This test_case should be a SECTION in the "restart" test_case, but for some
// reason (i/o ?) it fails when run along the other sections.
TEST_CASE("restartRead") {
  std::string dataPath = getDataPath();
  std::string fileName = "restartWater.res";
  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  // auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  fm->setBoxDimensions(boxDim);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(300);

  CudaContainer<double> onStepPistonPosition, halfStepPistonPosition,
      onStepPistonVelocity, halfStepPistonVelocity;
  double pistonNHposition, pistonNHvelocity, pistonNHforce;
  // test that we can initialize each variable read from the restart file into
  // the context and/or integrator. This requires a "restart.out" file ready
  // (run the previous section first)
  SECTION("read") {
    // auto readRestartSub = std::make_shared<RestartSubscriber>(
    // dataPath + "restart/restartDebugger.out", 30);
    auto readRestartSub = std::make_shared<RestartSubscriber>(fileName, 30);

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setSimulationContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);

    // setting positions [->context]
    auto positionsRead = readRestartSub->readPositions();
    ctx->setCoordinates(positionsRead);
    // setting velocities [->context]
    auto velocitiesRead = readRestartSub->readVelocities();
    ctx->assignVelocities(velocitiesRead);
    // setting coordsdeltaprevious [->integrator], testing
    auto cdp = readRestartSub->readCoordsDeltaPrevious();
    CHECK_NOTHROW(integrator->setCoordsDeltaPrevious(cdp));
    auto getcc = integrator->getCoordsDeltaPrevious();
    getcc.transferFromDevice();
    std::vector<double> cdp0 = {getcc.getHostArray()[0].x,
                                getcc.getHostArray()[0].y,
                                getcc.getHostArray()[0].z};
    CHECK(CompareVectors(cdp0, cdp[0]));
    // setting box dimensions [->context->force manager]
    auto boxDimRead = readRestartSub->readBoxDimensions();
    ctx->setBoxDimensions(boxDimRead);
    // setting on-step piston velocity [->integrator], testing
    auto onStepPistonVelocityRead = readRestartSub->readOnStepPistonVelocity();
    integrator->setOnStepPistonVelocity(onStepPistonVelocityRead);
    auto ccospv = integrator->getOnStepPistonVelocity();
    CHECK(compareVectors(onStepPistonVelocityRead, ccospv.getHostArray()));
    // setting on-step piston position
    auto onStepPistonPositionRead = readRestartSub->readOnStepPistonPosition();
    integrator->setOnStepPistonPosition(onStepPistonPositionRead);
    auto ccosp = integrator->getOnStepPistonPosition();
    CHECK(compareVectors(onStepPistonPositionRead, ccosp.getHostArray()));
    // setting half-step piston position
    auto halfStepPistonPositionRead =
        readRestartSub->readHalfStepPistonPosition();
    integrator->setHalfStepPistonPosition(halfStepPistonPositionRead);
    auto cchsp = integrator->getHalfStepPistonPosition();
    CHECK(compareVectors(halfStepPistonPositionRead, cchsp.getHostArray()));

    // Need to also get and retrieve the nose-hoover variables (vel, pos)
    double pistonNHpositionRead =
        readRestartSub->readNoseHooverPistonPosition();
    double pistonNHvelocityRead =
        readRestartSub->readNoseHooverPistonVelocity();
    double pistonNHforceRead = readRestartSub->readNoseHooverPistonForce();
    integrator->setPistonNoseHooverPosition(pistonNHpositionRead);
    integrator->setPistonNoseHooverVelocityPrevious(pistonNHvelocityRead);
    integrator->setPistonNoseHooverForcePrevious(pistonNHforceRead);
    CHECK(integrator->getPistonNoseHooverPosition() ==
          Approx(pistonNHpositionRead));
    CHECK(integrator->getPistonNoseHooverVelocityPrevious() ==
          Approx(pistonNHvelocityRead));
    CHECK(integrator->getPistonNoseHooverForcePrevious() ==
          Approx(pistonNHforceRead));

    // Check if the integrator can run after this
    CHECK_NOTHROW(integrator->propagate(10));
  }
}
