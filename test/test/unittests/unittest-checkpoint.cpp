// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "Checkpoint.h"
#include "CheckpointSubscriber.h"
#include "CudaContainer.h"
#include "CudaLangevinPistonIntegrator.h"
#include "ForceManager.h"
#include "catch.hpp"
#include "compare.h"
#include "test_paths.h"

TEST_CASE("checkpint", "[basic][extra]") {
  std::string dataPath = getDataPath();
  std::vector<double> boxDims = {50.0, 50.0, 50.0};
  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};

  SECTION("write") {
    // Read topology, parameters, PSF, and coordinates
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    // Setup force manager
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions(boxDims);

    // Setup CHARMM context
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);
    ctx->assignVelocitiesAtTemperature(300);

    // Setup integrator
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setPistonFriction(12.0);
    integrator->setSimulationContext(ctx);

    // Create checkpoint object and checkpoint the current state of the
    // simulation
    auto checkpoint = std::make_shared<Checkpoint>("test.chk");
    checkpoint->writeCheckpoint(ctx);
    // checkpoint->writeCheckpoint(integrator);
    // checkpoint->writeCheckpoint(ctx, integrator);

    // // Run simulation
    // integrator->propagate(nsteps);
  }

  SECTION("read") {}

  /* *
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
