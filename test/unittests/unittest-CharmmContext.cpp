// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, Félix Aviat
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaLangevinPistonIntegrator.h"
#include "StateSubscriber.h"
#include "Subscriber.h"
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h" // has the global variable test_path

TEST_CASE("CharmmContext", "[unit]") {
  std::string dataPath = getDataPath();
  SECTION("Basics") {
    // std::string test_path = "..";
    auto psf = std::make_shared<CharmmPSF>(dataPath + "argon_10.psf");
    auto prm = std::make_shared<CharmmParameters>(dataPath + "argon.prm");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    // Temperature values
    float t1 = 100.0, t2 = 200.0, t3 = 300.0;
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    // Check temperature getter/setter
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setTemperature(t1);
    // std::cout << "Set temperature to " << ctx->getTemperature() << std::endl;
    REQUIRE(ctx->getTemperature() == t1);

    // Checking copy constructor
    auto ctx2 = std::make_shared<CharmmContext>(*ctx);
    // temperature copied
    REQUIRE(ctx2->getTemperature() == t1);
    ctx->setTemperature(t2);
    // changing original does not change copy
    REQUIRE(ctx2->getTemperature() == t1);
    // changing copy does not change original
    ctx2->setTemperature(t3);
    REQUIRE(ctx->getTemperature() == t2);

    auto crd = std::make_shared<CharmmCrd>(dataPath + "argon_10.crd");

    // Check that the coordinate setter using vect(vect(double)) works
    std::vector<double> singleCoord = {-1., -2., -3.};
    std::vector<std::vector<double>> coords;
    for (int i = 0; i < 10; i++) {
      std::vector<double> currentCoord = {
          singleCoord[0] + i, singleCoord[1] + i, singleCoord[2] + i};
      coords.push_back(currentCoord);
    }
    CHECK_NOTHROW(ctx->setCoordinates(coords));
    auto cccc = ctx->getCoordinatesCharges();
    cccc.transferFromDevice();
    std::vector<double> coordToTest = {cccc.getHostArray()[0].x,
                                       cccc.getHostArray()[0].y,
                                       cccc.getHostArray()[0].z};
    CHECK(compareVectors(coordToTest, singleCoord));

    // Check the velocities setter using vect(vect(double))
    CHECK_NOTHROW(ctx->assignVelocities(coords));
    auto ccvm = ctx->getVelocityMass();
    ccvm.transferFromDevice();
    std::vector<double> velToTest = {ccvm.getHostArray()[0].x,
                                     ccvm.getHostArray()[0].y,
                                     ccvm.getHostArray()[0].z};
    CHECK(compareVectors(velToTest, singleCoord));
  }

  SECTION("Temperature") {
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    std::vector<std::string> prmlist = {dataPath + "toppar_water_ions.str"};
    auto prm = std::make_shared<CharmmParameters>(prmlist);
    auto fm = std::make_shared<ForceManager>(psf, prm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    fm->setBoxDimensions({50., 50., 50.});

    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(300.);
    float tout = ctx->computeTemperature();
    /*std::cout << "Computed temperature: " << tout
              << " DOFs: " << ctx->getDegreesOfFreedom()
              << " kinetic energy: " << ctx->getKineticEnergy() << std::endl;
    */
    // THIS SHOULD RUN A BIT !
    int nWatersInWaterbox = 3916;
    // For a pure water system using SHAKE constraints, we have 6 dofs per
    // molecule
    CHECK(ctx->getDegreesOfFreedom() == nWatersInWaterbox * 6 - 3);
    // INFO("This requires equilibration to work anyways");
    // CHECK(tout == Approx(300.).margin(1.));
  }
  SECTION("Pressure") {
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50., 50., 50.});
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->readRestart(dataPath + "restart/heat_waterbox.restart");

    // make sure the computePressure function returns the same value as the
    // Langevin piston one

    // pressure from integrator ?
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(0.0);
    integrator->setCharmmContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    // integrator->setDebugPrintFrequency(1);
    ctx->computePressure();
    integrator->propagate(1);
    ctx->computePressure();
    integrator->propagate(1);
    // ctx->computePressure();

    // INFO("Currently, the CharmmContext::computePressure function does NOT "
    //      "work.");
    //  CHECK(false);
  }
}

TEST_CASE("CompositeForceManager", "[unit]") {
  std::string dataPath = getDataPath();

  // When creating a CharmmContext based on a CompositeForceManager, the
  // CharmmContext's FM should still be initialized if the original FM is
  SECTION("Initialization issue") {
    auto fmc = std::make_shared<ForceManagerComposite>();
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm1 = std::make_shared<ForceManager>(psf, prm);
    auto fm2 = std::make_shared<ForceManager>(psf, prm);

    fmc->addForceManager(fm1);
    fmc->addForceManager(fm2);
    fmc->setBoxDimensions({32., 32., 32.});
    fmc->setFFTGrid(12, 12, 12);
    fmc->initialize();
    REQUIRE(fmc->isInitialized());
    auto ctx = std::make_shared<CharmmContext>(fmc);
    CHECK(ctx->getForceManager()->isInitialized());
  }
}

// Check random velocity assignment, and seed fixing
TEST_CASE("randomSeed") {
  std::string dataPath = getDataPath();
  int seednum = 144;
  auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
  auto prm =
      std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50., 50., 50.});
  auto ctx = std::make_shared<CharmmContext>(fm);
  auto ctxbis = std::make_shared<CharmmContext>(fm);

  auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
  ctx->setCoordinates(crd);
  ctxbis->setCoordinates(crd);

  // Check that: 1. without a fixed seed, initializing velocities twice gives
  // two different results.
  // 2. With a fixed seed, initializing velocities twice
  // gives the same result
  SECTION("assignVelocitiesFixedSeed") {
    ctx->setRandomSeedForVelocities(seednum);
    ctxbis->setRandomSeedForVelocities(seednum);
    ctx->assignVelocitiesAtTemperature(300.);
    ctxbis->assignVelocitiesAtTemperature(300.);
    // get velocity values (extract to a vector)
    auto velmasscc = ctx->getVelocityMass();
    auto velmassccbis = ctxbis->getVelocityMass();
    velmasscc.transferFromDevice();
    velmassccbis.transferFromDevice();
    std::vector<double> velvec, velvecbis;
    for (int i = 0; i < ctx->getNumAtoms(); i++) {
      velvec.push_back(velmasscc.getHostArray()[i].x);
      velvec.push_back(velmasscc.getHostArray()[i].y);
      velvec.push_back(velmasscc.getHostArray()[i].z);
      velvecbis.push_back(velmassccbis.getHostArray()[i].x);
      velvecbis.push_back(velmassccbis.getHostArray()[i].y);
      velvecbis.push_back(velmassccbis.getHostArray()[i].z);
    }
    // ensure they are the same
    CHECK(compareVectors(velvec, velvecbis));
  }
  SECTION("assignVelocitiesDifftSeeds") {
    // REQUIRE(ctx->getRandomSeedForVelocities() !=
    //         ctxbis->getRandomSeedForVelocities());

    ctx->assignVelocitiesAtTemperature(300.);
    ctxbis->assignVelocitiesAtTemperature(300.);
    // get velocity values (extract to a vector)
    auto velmasscc = ctx->getVelocityMass();
    auto velmassccbis = ctxbis->getVelocityMass();
    velmasscc.transferFromDevice();
    velmassccbis.transferFromDevice();

    std::vector<double> velvec, velvecbis;

    for (int i = 0; i < ctx->getNumAtoms(); i++) {
      velvec.push_back(velmasscc.getHostArray()[i].x);
      velvec.push_back(velmasscc.getHostArray()[i].y);
      velvec.push_back(velmasscc.getHostArray()[i].z);
      velvecbis.push_back(velmassccbis.getHostArray()[i].x);
      velvecbis.push_back(velmassccbis.getHostArray()[i].y);
      velvecbis.push_back(velmassccbis.getHostArray()[i].z);
    }
    // ensure they are different
    // CHECK(!compareVectors(velvec, velvecbis));
  }
}

// Functions remaining to be tested
//------------------------
// Constructors
// CharmmContext; usual cons
// CharmmContext basecons (useless)
// setCoordinates;
// others
// computeTemperature;
// setPeriodicBoundaryCondition;
// getPeriodicBoundaryCondition
// setCoords;
// getCoords;
// setCharges;
// setMasses;
// calculatePotentialEnergyPr;
// calculateKineticEnergy;
// getKineticEnergy;
// getKineticEnergy_;
// getForces;
// getNumAtoms ;
// *getXYZQ;
// *get_loc2glo ;
// getForceStride ;
// setNumAtoms;
// setCoordsCharges;
// setCoordsCharges;
// resetNeighborList;
// calculatePotentialEnergy;
// calculateForces
// setMasses;
// assignVelocitiesAtTemperature;
// assignVelocitiesFromCHARMMVelocityFile;
// assignVelocities;
// &getVelocityMass;
// &getCoordinatesCharges;
// &getBoxDimensions;
// getBonds;
// minimize;
// getDegreesOfFreedom;
// imageCentering;
// getForceManager
// getPotentialEnergies;
// getPotentialEnergy;
// getVolume ;
// calculatePressure;
// getWaterMolecules;
// useHolonomicConstras;
// isUsingHolonomicConstras
// orient;
// setForceManager;
// linkBackForceManager;
// writeCrd;
