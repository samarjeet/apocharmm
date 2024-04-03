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
  integrator->setCharmmContext(ctx);
  integrator->setFriction(12.0);
  integrator->setBathTemperature(300.0);
  return integrator;
}

std::shared_ptr<CudaLangevinPistonIntegrator>
setupLangevinPistonIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator->setPistonFriction(20.0);
  integrator->setCharmmContext(ctx);
  integrator->setCrystalType(CRYSTAL::CUBIC);
  return integrator;
}

std::shared_ptr<CudaVelocityVerletIntegrator>
setupVelocityVerletIntegrator(std::shared_ptr<CharmmContext> ctx) {
  auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.002);
  integrator->setCharmmContext(ctx);
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
    restartFileName = "langevinPistonRestartTest.rst";
    checkRuns(integrator, restartFileName);

    auto integrator2 = setupLangevinPistonIntegrator(ctx2);
    checkRestart(integrator2, restartFileName);
  }

  SECTION("velocityVerlet") {
    auto integrator = setupVelocityVerletIntegrator(ctx);
    restartFileName = "velocityVerletRestartTest.rst";
    checkRuns(integrator, restartFileName);
    auto integrator2 = setupVelocityVerletIntegrator(ctx2);
    checkRestart(integrator2, restartFileName);
  }

  SECTION("langevinThermostat") {
    auto integrator = setupLangevinThermostatIntegrator(ctx);
    restartFileName = "langevinThermostatRestartTest.rst";
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

  std::string fileName = "restartWater.rst";

  SECTION("save") {
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setCharmmContext(ctx);
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

    pistonNHposition = integrator->getNoseHooverPistonPosition();
    pistonNHvelocity = integrator->getNoseHooverPistonVelocityPrevious();
    pistonNHforce = integrator->getNoseHooverPistonForcePrevious();

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
    auto restartFileNHvelocity =
        myRestartSub->readNoseHooverPistonVelocityPrevious();
    auto restartFileNHforce = myRestartSub->readNoseHooverPistonForcePrevious();
    CHECK(restartFileNHposition == Approx(pistonNHposition));
    CHECK(restartFileNHvelocity == Approx(pistonNHvelocity));
    CHECK(restartFileNHforce == Approx(pistonNHforce));
  }

  SECTION("io") {
    // Make sure we can write stuff to a restart file but also read stuff from
    // it
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setCharmmContext(ctx);
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
    integrator->setCharmmContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    auto savingpointres1 =
        std::make_shared<RestartSubscriber>("savingPoint1.rst", 10);
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
    REQUIRE(lineCounter(fileName) == ctx->getNumAtoms() * 3 + 10 + 7 + 9 + 13);

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setCharmmContext(ctx);
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
    auto nhpref = getRestartFileEntry(fileName, "!noseHooverPistonPosition");
    auto nhvref =
        getRestartFileEntry(fileName, "!noseHooverPistonVelocityPrevious");
    auto nhfref =
        getRestartFileEntry(fileName, "!noseHooverPistonForcePrevious");

    std::vector<double> firstCoord, firstVel, firstCoordsDeltaPrevious;
    CudaContainer<double4> ctxCoords = ctx->getCoordinatesCharges(),
                           ctxVel = ctx->getVelocityMass(),
                           coordsDeltaPrevious =
                               integrator->getCoordsDeltaPrevious();
    CudaContainer<double> ospp = integrator->getOnStepPistonPosition(),
                          ospv = integrator->getOnStepPistonVelocity(),
                          hsp = integrator->getHalfStepPistonPosition();
    std::vector<double> nhpp = {integrator->getNoseHooverPistonPosition()},
                        nhpv =
                            {integrator->getNoseHooverPistonVelocityPrevious()},
                        nhpf = {integrator->getNoseHooverPistonForcePrevious()};
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
    wrongCrystalIntegrator->setCharmmContext(ctx);
    wrongCrystalIntegrator->setCrystalType(CRYSTAL::TETRAGONAL);
    auto wrongRestartSub = std::make_shared<RestartSubscriber>(fileName, 30);
    wrongCrystalIntegrator->subscribe(wrongRestartSub);
    CHECK_THROWS(wrongRestartSub->readRestart());
  }
}

// Test that (with a no-randomness integrator) the result of a propagation
// starting from the restart file is the same as the one starting from the
// initial conditions.
// For now, only tests it for deterministic prop (no friction in the
// langevin piston)
TEST_CASE("identicalPropagation") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};
  std::vector<double> boxDim = {50.0, 50.0, 50.0};
  int rdmSeed = 314159, nsteps = 1; // 5000;
  std::string fileName = "idprop.rst";

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

  // Setup integrator
  auto integrator1 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator1->setCrystalType(CRYSTAL::CUBIC);
  integrator1->setPistonMass({500.0}); // setCrystalType resets this
  // integrator1->setPistonFriction(12.0);
  integrator1->setPistonFriction(0.0);
  integrator1->setSeedForPistonFriction(rdmSeed);
  integrator1->setNoseHooverFlag(false);
  integrator1->setCharmmContext(ctx1);
  integrator1->setDebugPrintFrequency(1);

  // Create a restart file
  auto restartsub = std::make_shared<RestartSubscriber>(fileName, nsteps);
  integrator1->subscribe(restartsub);

  // Propagate first simulation
  integrator1->propagate(nsteps);
  integrator1->unsubscribe(restartsub);

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

  // Setup integrator
  auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator2->setCrystalType(CRYSTAL::CUBIC);
  integrator2->setPistonMass({500.0}); // setCrystalType resets this
  // integrator2->setPistonFriction(12.0);
  integrator2->setPistonFriction(0.0);
  integrator2->setSeedForPistonFriction(rdmSeed);
  integrator2->setNoseHooverFlag(false);
  integrator2->setCharmmContext(ctx2);
  integrator2->setDebugPrintFrequency(1);

  // Propagate second simulation
  integrator2->propagate(nsteps);

  /* *
  // Create a triplicate system that uses the restart file from the first
  // simulation to ensure that using a restart file to start a new simulation
  // results in the same trajectories

  // Topology, parameters, PSF, and coordinates
  auto prm3 = std::make_shared<CharmmParameters>(prmFiles);
  auto psf3 = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto crd3 = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

  // Setup force manager
  auto fm3 = std::make_shared<ForceManager>(psf3, prm3);
  fm3->setBoxDimensions(boxDim);

  // Setup CHARMM context
  auto ctx3 = std::make_shared<CharmmContext>(fm3);
  ctx3->setCoordinates(crd3);

  auto integrator3 = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
  integrator3->setCrystalType(CRYSTAL::CUBIC);
  integrator3->setPistonMass({500.0}); // setCrystalType resets this
  // integrator3->setPistonFriction(12.0);
  integrator3->setPistonFriction(0.0);
  integrator3->setSeedForPistonFriction(rdmSeed);
  // integrator3->setNoseHooverFlag(false);
  integrator3->setCharmmContext(ctx3);

  auto restartsub3 = std::make_shared<RestartSubscriber>(fileName);
  integrator3->subscribe(restartsub3);
  restartsub3->readRestart();
  integrator3->unsubscribe(restartsub3);
  * */

  // All three systems should match at this point

  // Check that coordinates match
  auto coordinatesCharges1 = ctx1->getCoordinatesCharges();
  auto coordinatesCharges2 = ctx2->getCoordinatesCharges();
  // auto coordinatesCharges3 = ctx3->getCoordinatesCharges();
  coordinatesCharges1.transferFromDevice();
  coordinatesCharges2.transferFromDevice();
  // coordinatesCharges3.transferFromDevice();

  // Use PBC compare here because when we set the coordinates for the CHARMM
  // context, it rebuilds the neighbor list, which performs image centering
  CHECK(CompareVectorsPBC1(coordinatesCharges1.getHostArray(),
                           coordinatesCharges2.getHostArray(),
                           {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0, true));
  // CHECK(CompareVectorsPBC1(coordinatesCharges1.getHostArray(),
  //                          coordinatesCharges3.getHostArray(),
  //                          {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0,
  //                          true));
  // CHECK(CompareVectorsPBC1(coordinatesCharges2.getHostArray(),
  //                          coordinatesCharges3.getHostArray(),
  //                          {boxDim[0], boxDim[1], boxDim[2], 0.0}, 0.0,
  //                          true));

  // Check that velocities match
  auto velocityMass1 = ctx1->getVelocityMass();
  auto velocityMass2 = ctx2->getVelocityMass();
  // auto velocityMass3 = ctx3->getVelocityMass();
  velocityMass1.transferFromDevice();
  velocityMass2.transferFromDevice();
  // velocityMass3.transferFromDevice();
  CHECK(CompareVectors1(velocityMass1.getHostArray(),
                        velocityMass2.getHostArray(), 0.0, true));
  // CHECK(CompareVectors1(velocityMass1.getHostArray(),
  //                       velocityMass3.getHostArray(), 0.0, true));
  // CHECK(CompareVectors1(velocityMass2.getHostArray(),
  //                       velocityMass3.getHostArray(), 0.0, true));

  auto coordsDeltaPrevious1 = integrator1->getCoordsDeltaPrevious();
  auto coordsDeltaPrevious2 = integrator2->getCoordsDeltaPrevious();
  // auto coordsDeltaPrevious3 = integrator3->getCoordsDeltaPrevious();
  coordsDeltaPrevious1.transferFromDevice();
  coordsDeltaPrevious2.transferFromDevice();
  // coordsDeltaPrevious3.transferFromDevice();
  CHECK(CompareVectors1(coordsDeltaPrevious1.getHostArray(),
                        coordsDeltaPrevious2.getHostArray(), 0.0, true));
  // CHECK(CompareVectors1(coordsDeltaPrevious1.getHostArray(),
  //                       coordsDeltaPrevious3.getHostArray(), 0.0, true));
  // CHECK(CompareVectors1(coordsDeltaPrevious2.getHostArray(),
  //                       coordsDeltaPrevious3.getHostArray(), 0.0, true));

  // Ensure that Nose-Hoover variables are the same
  CHECK(integrator1->getNoseHooverPistonMass() ==
        integrator2->getNoseHooverPistonMass());
  // CHECK(integrator1->getNoseHooverPistonMass() ==
  //       integrator3->getNoseHooverPistonMass());
  // CHECK(integrator2->getNoseHooverPistonMass() ==
  //       integrator3->getNoseHooverPistonMass());

  CHECK(integrator1->getNoseHooverPistonPosition() ==
        integrator2->getNoseHooverPistonPosition());
  // CHECK(integrator1->getNoseHooverPistonPosition() ==
  //       integrator3->getNoseHooverPistonPosition());
  // CHECK(integrator2->getNoseHooverPistonPosition() ==
  //       integrator3->getNoseHooverPistonPosition());

  CHECK(integrator1->getNoseHooverPistonVelocity() ==
        integrator2->getNoseHooverPistonVelocity());
  // CHECK(integrator1->getNoseHooverPistonVelocity() ==
  //       integrator3->getNoseHooverPistonVelocity());
  // CHECK(integrator2->getNoseHooverPistonVelocity() ==
  //       integrator3->getNoseHooverPistonVelocity());

  CHECK(integrator1->getNoseHooverPistonVelocityPrevious() ==
        integrator2->getNoseHooverPistonVelocityPrevious());
  // CHECK(integrator1->getNoseHooverPistonVelocityPrevious() ==
  //       integrator3->getNoseHooverPistonVelocityPrevious());
  // CHECK(integrator2->getNoseHooverPistonVelocityPrevious() ==
  //       integrator3->getNoseHooverPistonVelocityPrevious());

  CHECK(integrator1->getNoseHooverPistonForce() ==
        integrator2->getNoseHooverPistonForce());
  // CHECK(integrator1->getNoseHooverPistonForce() ==
  //       integrator3->getNoseHooverPistonForce());
  // CHECK(integrator2->getNoseHooverPistonForce() ==
  //       integrator3->getNoseHooverPistonForce());

  CHECK(integrator1->getNoseHooverPistonForcePrevious() ==
        integrator2->getNoseHooverPistonForcePrevious());
  // CHECK(integrator1->getNoseHooverPistonForcePrevious() ==
  //       integrator3->getNoseHooverPistonForcePrevious());
  // CHECK(integrator2->getNoseHooverPistonForcePrevious() ==
  //       integrator3->getNoseHooverPistonForcePrevious());

  /* *
  integrator1->propagate(nsteps);
  integrator2->propagate(nsteps);
  integrator3->propagate(nsteps);
  // integrator1->propagate(20 * nsteps);
  // integrator2->propagate(20 * nsteps);
  // integrator3->propagate(20 * nsteps);

  coordinatesCharges1 = ctx1->getCoordinatesCharges();
  coordinatesCharges2 = ctx2->getCoordinatesCharges();
  coordinatesCharges3 = ctx3->getCoordinatesCharges();
  coordinatesCharges1.transferFromDevice();
  coordinatesCharges2.transferFromDevice();
  coordinatesCharges3.transferFromDevice();

  // Ensure that after propagating more steps the coordinates match
  CHECK(CompareVectors1(coordinatesCharges1.getHostArray(),
                        coordinatesCharges2.getHostArray(), 0.0, true));
  CHECK(CompareVectors1(coordinatesCharges1.getHostArray(),
                        coordinatesCharges3.getHostArray(), 0.0, true));
  CHECK(CompareVectors1(coordinatesCharges2.getHostArray(),
                        coordinatesCharges3.getHostArray(), 0.0, true));
  coordsDeltaPrevious1 = integrator1->getCoordsDeltaPrevious();
  coordsDeltaPrevious2 = integrator2->getCoordsDeltaPrevious();
  coordsDeltaPrevious3 = integrator3->getCoordsDeltaPrevious();
  coordsDeltaPrevious1.transferFromDevice();
  coordsDeltaPrevious2.transferFromDevice();
  coordsDeltaPrevious3.transferFromDevice();
  CHECK(CompareVectors1(coordsDeltaPrevious1.getHostArray(),
                        coordsDeltaPrevious2.getHostArray(), 0.0, true));
  CHECK(CompareVectors1(coordsDeltaPrevious1.getHostArray(),
                        coordsDeltaPrevious3.getHostArray(), 0.0, true));
  CHECK(CompareVectors1(coordsDeltaPrevious2.getHostArray(),
                        coordsDeltaPrevious3.getHostArray(), 0.0, true));

  // Ensure that Nose-Hoover variables are the same
  CHECK(integrator1->getNoseHooverPistonMass() ==
        integrator2->getNoseHooverPistonMass());
  CHECK(integrator1->getNoseHooverPistonMass() ==
        integrator3->getNoseHooverPistonMass());
  CHECK(integrator2->getNoseHooverPistonMass() ==
        integrator3->getNoseHooverPistonMass());

  CHECK(integrator1->getNoseHooverPistonPosition() ==
        integrator2->getNoseHooverPistonPosition());
  CHECK(integrator1->getNoseHooverPistonPosition() ==
        integrator3->getNoseHooverPistonPosition());
  CHECK(integrator2->getNoseHooverPistonPosition() ==
        integrator3->getNoseHooverPistonPosition());

  CHECK(integrator1->getNoseHooverPistonVelocity() ==
        integrator2->getNoseHooverPistonVelocity());
  CHECK(integrator1->getNoseHooverPistonVelocity() ==
        integrator3->getNoseHooverPistonVelocity());
  CHECK(integrator2->getNoseHooverPistonVelocity() ==
        integrator3->getNoseHooverPistonVelocity());

  CHECK(integrator1->getNoseHooverPistonVelocityPrevious() ==
        integrator2->getNoseHooverPistonVelocityPrevious());
  CHECK(integrator1->getNoseHooverPistonVelocityPrevious() ==
        integrator3->getNoseHooverPistonVelocityPrevious());
  CHECK(integrator2->getNoseHooverPistonVelocityPrevious() ==
        integrator3->getNoseHooverPistonVelocityPrevious());

  CHECK(integrator1->getNoseHooverPistonForce() ==
        integrator2->getNoseHooverPistonForce());
  CHECK(integrator1->getNoseHooverPistonForce() ==
        integrator3->getNoseHooverPistonForce());
  CHECK(integrator2->getNoseHooverPistonForce() ==
        integrator3->getNoseHooverPistonForce());

  CHECK(integrator1->getNoseHooverPistonForcePrevious() ==
        integrator2->getNoseHooverPistonForcePrevious());
  CHECK(integrator1->getNoseHooverPistonForcePrevious() ==
        integrator3->getNoseHooverPistonForcePrevious());
  CHECK(integrator2->getNoseHooverPistonForcePrevious() ==
        integrator3->getNoseHooverPistonForcePrevious());
  * */
}

// This test_case should be a SECTION in the "restart" test_case, but for some
// reason (i/o ?) it fails when run along the other sections.
TEST_CASE("restartRead") {
  std::string dataPath = getDataPath();
  std::string fileName = "restartWater.rst";
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
    integrator->setCharmmContext(ctx);
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
    integrator->setNoseHooverPistonPosition(pistonNHpositionRead);
    integrator->setNoseHooverPistonVelocityPrevious(pistonNHvelocityRead);
    integrator->setNoseHooverPistonForcePrevious(pistonNHforceRead);
    CHECK(integrator->getNoseHooverPistonPosition() ==
          Approx(pistonNHpositionRead));
    CHECK(integrator->getNoseHooverPistonVelocityPrevious() ==
          Approx(pistonNHvelocityRead));
    CHECK(integrator->getNoseHooverPistonForcePrevious() ==
          Approx(pistonNHforceRead));

    // Check if the integrator can run after this
    CHECK_NOTHROW(integrator->propagate(10));
  }
}
