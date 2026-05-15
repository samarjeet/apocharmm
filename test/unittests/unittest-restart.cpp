// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:   James E. Gonzales II, Félix Aviat, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaLangevinPistonIntegrator.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "ForceManager.h"
#include "RestartSubscriber.h"
#include "catch.hpp"
#include "compare.h"
#include "test_paths.h"
#include <iomanip>
#include <iostream>
#include <vector>

TEST_CASE("restart") {
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = true;
  const double timeStep = (useHolonomicConstraints) ? 0.002 : 0.001;
  const int nsteps = (useHolonomicConstraints) ? 1000 : 2000;

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
  ctx2->useHolonomicConstraints(useHolonomicConstraints);

  SECTION("noseHoover") {
    std::cout << "**** Nose-Hoover ****" << std::endl;

    // Setup first integrator
    auto integrator1 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator1->setCharmmContext(ctx1);

    // Setup restart subscriber
    auto rst = std::make_shared<RestartSubscriber>("tmpNoseHoover.rst", nsteps);
    integrator1->subscribe(rst);

    integrator1->propagate(nsteps);
    integrator1->unsubscribe(rst);
    integrator1->propagate(1);

    // Setup second integrator from restart file
    auto integrator2 =
        std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
    integrator2->setCharmmContext(ctx2);
    integrator2->initializeFromRestartFile("tmpNoseHoover.rst");

    integrator2->propagate(1);

    // Check Nose-Hoover variables
    integrator1->getNoseHooverPistonVelocity().transferToHost();
    integrator1->getNoseHooverPistonVelocityPrevious().transferToHost();
    integrator1->getNoseHooverPistonForce().transferToHost();
    integrator1->getNoseHooverPistonForcePrevious().transferToHost();
    integrator2->getNoseHooverPistonVelocity().transferToHost();
    integrator2->getNoseHooverPistonVelocityPrevious().transferToHost();
    integrator2->getNoseHooverPistonForce().transferToHost();
    integrator2->getNoseHooverPistonForcePrevious().transferToHost();
    CHECK(integrator1->getNoseHooverPistonVelocity()[0] ==
          Approx(integrator2->getNoseHooverPistonVelocity()[0]).margin(1e-12));
    CHECK(integrator1->getNoseHooverPistonVelocityPrevious()[0] ==
          Approx(integrator2->getNoseHooverPistonVelocityPrevious()[0])
              .margin(1e-12));
    CHECK(integrator1->getNoseHooverPistonForce()[0] ==
          Approx(integrator2->getNoseHooverPistonForce()[0]).margin(1e-12));
    CHECK(integrator1->getNoseHooverPistonForcePrevious()[0] ==
          Approx(integrator2->getNoseHooverPistonForcePrevious()[0])
              .margin(1e-12));
  }

  SECTION("langevinPiston") {
    std::cout << "**** Langevin-Piston ****" << std::endl;

    // Setup first integrator
    auto integrator1 = std::make_shared<CudaLangevinPistonIntegrator>(timeStep);
    integrator1->setCrystalType(CRYSTAL::CUBIC);
    integrator1->setLangevinPistonFrictionSeed(randomSeed);
    integrator1->setCharmmContext(ctx1);

    // Setup restart subscriber
    auto rst =
        std::make_shared<RestartSubscriber>("tmpLangevinPiston.rst", nsteps);
    integrator1->subscribe(rst);

    integrator1->propagate(nsteps);
    integrator1->unsubscribe(rst);
    integrator1->propagate(1);

    // Setup second integrator from restart file
    auto integrator2 = std::make_shared<CudaLangevinPistonIntegrator>(timeStep);
    integrator2->setCrystalType(
        CRYSTAL::ORTHORHOMBIC); // Intentionally set incorrectly
    integrator2->setCharmmContext(ctx2);
    integrator2->initializeFromRestartFile("tmpLangevinPiston.rst");

    integrator2->propagate(1);

    // Check Nose-Hoover variables
    integrator1->getNoseHooverPistonVelocity().transferToHost();
    integrator1->getNoseHooverPistonVelocityPrevious().transferToHost();
    integrator1->getNoseHooverPistonForce().transferToHost();
    integrator1->getNoseHooverPistonForcePrevious().transferToHost();
    integrator2->getNoseHooverPistonVelocity().transferToHost();
    integrator2->getNoseHooverPistonVelocityPrevious().transferToHost();
    integrator2->getNoseHooverPistonForce().transferToHost();
    integrator2->getNoseHooverPistonForcePrevious().transferToHost();
    CHECK(integrator1->getNoseHooverPistonVelocity()[0] ==
          Approx(integrator2->getNoseHooverPistonVelocity()[0]).margin(1e-12));
    CHECK(integrator1->getNoseHooverPistonVelocityPrevious()[0] ==
          Approx(integrator2->getNoseHooverPistonVelocityPrevious()[0])
              .margin(1e-12));
    CHECK(integrator1->getNoseHooverPistonForce()[0] ==
          Approx(integrator2->getNoseHooverPistonForce()[0]).margin(0.0));
    CHECK(integrator1->getNoseHooverPistonForcePrevious()[0] ==
          Approx(integrator2->getNoseHooverPistonForcePrevious()[0])
              .margin(1e-12));

    // Check Langevin-Piston variables
    integrator1->getLangevinPistonOnStepPosition().transferToHost();
    integrator1->getLangevinPistonHalfStepPosition().transferToHost();
    integrator1->getLangevinPistonOnStepVelocity().transferToHost();
    integrator1->getLangevinPistonHalfStepVelocity().transferToHost();
    integrator1->getLangevinPistonDeltaPosition().transferToHost();
    integrator1->getLangevinPistonDeltaPositionPrevious().transferToHost();
    integrator2->getLangevinPistonOnStepPosition().transferToHost();
    integrator2->getLangevinPistonHalfStepPosition().transferToHost();
    integrator2->getLangevinPistonOnStepVelocity().transferToHost();
    integrator2->getLangevinPistonHalfStepVelocity().transferToHost();
    integrator2->getLangevinPistonDeltaPosition().transferToHost();
    integrator2->getLangevinPistonDeltaPositionPrevious().transferToHost();
    CHECK(integrator1->getCrystalType() == integrator2->getCrystalType());
    CHECK(integrator1->getLangevinPistonOnStepPosition()[0] ==
          Approx(integrator2->getLangevinPistonOnStepPosition()[0])
              .margin(1e-15));
    CHECK(integrator1->getLangevinPistonHalfStepPosition()[0] ==
          Approx(integrator2->getLangevinPistonHalfStepPosition()[0])
              .margin(1e-15));
    CHECK(integrator1->getLangevinPistonOnStepVelocity()[0] ==
          Approx(integrator2->getLangevinPistonOnStepVelocity()[0])
              .margin(1e-15));
    CHECK(integrator1->getLangevinPistonHalfStepVelocity()[0] ==
          Approx(integrator2->getLangevinPistonHalfStepVelocity()[0])
              .margin(1e-15));
    CHECK(
        integrator1->getLangevinPistonDeltaPosition()[0] ==
        Approx(integrator2->getLangevinPistonDeltaPosition()[0]).margin(1e-15));
    CHECK(integrator1->getLangevinPistonDeltaPositionPrevious()[0] ==
          Approx(integrator2->getLangevinPistonDeltaPositionPrevious()[0])
              .margin(1e-15));
  }

  SECTION("langevinThermostat") {
    std::cout << "**** Langevin Thermostat ****" << std::endl;

    // Setup first integrator
    auto integrator1 =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    integrator1->setThermostatFriction(1.0);
    integrator1->setThermostatRngSeed(randomSeed);
    integrator1->setCharmmContext(ctx1);

    // Setup restart subscrber
    auto rst = std::make_shared<RestartSubscriber>("tmpLangevinThermostat.rst",
                                                   nsteps);
    integrator1->subscribe(rst);

    integrator1->propagate(nsteps);
    integrator1->unsubscribe(rst);
    integrator1->propagate(1);

    // Setup second integrator from restart file
    auto integrator2 =
        std::make_shared<CudaLangevinThermostatIntegrator>(timeStep);
    integrator2->setThermostatFriction(1.0);
    integrator2->setCharmmContext(ctx2);
    integrator2->initializeFromRestartFile("tmpLangevinThermostat.rst");

    integrator2->propagate(1);
  }

  // Check that the coordinates and velocities match enough
  ctx1->getCoordinatesCharges().transferToHost();
  ctx2->getCoordinatesCharges().transferToHost();
  CHECK(CompareVectors1<double4>(ctx1->getCoordinatesCharges().getHostArray(),
                                 ctx2->getCoordinatesCharges().getHostArray(),
                                 1e-12, true));

  ctx1->getVelocityMass().transferToHost();
  ctx2->getVelocityMass().transferToHost();
  CHECK(CompareVectors1<double4>(ctx1->getVelocityMass().getHostArray(),
                                 ctx2->getVelocityMass().getHostArray(), 1e-12,
                                 true));
}

/* *
TEST_CASE("restartCHARMM") {
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = true;
  const double timeStep = (useHolonomicConstraints) ? 0.002 : 0.001;
  const int nsteps = (useHolonomicConstraints) ? 10000 : 20000;

  // 1LVZ
  const std::vector<double> boxDims(3, 48.0169234);
  const int nfftx = 50, nffty = 50, nfftz = 50;
  const std::vector<std::string> prmFiles = {
      "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36m_prot.prm",
      "/u/jeg/charmm_dev/c49a1jeg/toppar/toppar_water_ions.str"};
  const std::string psfName =
      "/u/jeg/Documents/git/CHARMM-Tutorial-Dev/1lvz/dyn/1lvzi.psf";
  const std::string corName =
      "/u/jeg/Documents/git/CHARMM-Tutorial-Dev/1lvz/init_heat_eq/1lvzm0.cor";
  const std::string rstName =
      "/u/jeg/Documents/git/CHARMM-Tutorial-Dev/1lvz/dyn/rst/1lvzd2.rst";

  // // 3KWG
  // const std::vector<double> boxDims(3, 71.590617);
  // const int nfftx = 80, nffty = 80, nfftz = 80;
  // const std::vector<std::string> prmFiles = {
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36m_prot.prm",
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/toppar_water_ions.str"};
  // const std::string psfName =
  //     "/u/jeg/biomed/ns1/bc/3kwg/init_heat_eq/3kwgi.psf";
  // const std::string corName =
  //     "/u/jeg/biomed/ns1/bc/3kwg/init_heat_eq/3kwgm0.cor";
  // const std::string rstName =
  //     "/u/jeg/biomed/ns1/bc/3kwg/init_heat_eq/rst/3kwge0.rst";

  // // 3F5T Umbrella
  // const std::vector<double> boxDims = {128.918671, 71.8888814, 71.8888814};
  // const int nfftx = 144, nffty = 72, nfftz = 72;
  // const std::vector<std::string> prmFiles = {
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36m_prot.prm",
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/toppar_water_ions.str"};
  // const std::string psfName =
  //     "/u/jeg/biomed/ns1/umbrella/3f5t/init_heat_eq0/3f5t_edai.psf";
  // const std::string corName =
  //     "/u/jeg/biomed/ns1/umbrella/3f5t/init_heat_eq0/3f5t_edam0.cor";
  // const std::string rstName =
  //     "/u/jeg/biomed/ns1/umbrella/3f5t/init_heat_eq0/rst/3f5t_edae0.rst";

  // // 3RYI 2MER
  // const std::vector<double> boxDims = {210.229501, 104.171506, 104.171506};
  // const int nfftx = 216, nffty = 108, nfftz = 108;
  // const std::vector<std::string> prmFiles = {
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36m_prot.prm",
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36_na.prm",
  // "/u/jeg/charmm_dev/c49a1jeg/toppar/stream/na/toppar_all36_na_nad_ppi.str",
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/toppar_water_ions.str"};
  // const std::string psfName =
  //     "/u/jeg/biomed/mt/2mer/3ryi/init_heat_eq/3ryii.psf";
  // const std::string corName =
  //     "/u/jeg/biomed/mt/2mer/3ryi/init_heat_eq/3ryim0.cor";
  // const std::string rstName =
  //     "/u/jeg/biomed/mt/2mer/3ryi/init_heat_eq/rst/3ryie0.rst";

  // // 3RYI 4MER
  // const std::vector<double> boxDims = {365.681337, 144.883412, 144.883412};
  // const int nfftx = 384, nffty = 150, nfftz = 150;
  // const std::vector<std::string> prmFiles = {
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36m_prot.prm",
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/par_all36_na.prm",
  // "/u/jeg/charmm_dev/c49a1jeg/toppar/stream/na/toppar_all36_na_nad_ppi.str",
  //     "/u/jeg/charmm_dev/c49a1jeg/toppar/toppar_water_ions.str",
  // };
  // const std::string psfName =
  //     "/u/jeg/biomed/mt/4mer/3ryi/init_heat_eq/3ryii.psf";
  // const std::string corName =
  //     "/u/jeg/biomed/mt/4mer/3ryi/init_heat_eq/3ryim0.cor";
  // const std::string rstName =
  //     "/u/jeg/biomed/mt/4mer/3ryi/init_heat_eq/rst/3ryie0.rst";

  // Setup CHARMM parameters, PSF, and coordinates
  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(psfName);
  auto crd = std::make_shared<CharmmCrd>(corName);

  // Setup force manager
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions(boxDims);
  fm->setFFTGrid(nfftx, nffty, nfftz);
  fm->setPmeSplineOrder(6);

  // Setup CHARMM context
  auto ctx = std::make_shared<CharmmContext>(fm);
  ctx->setCoordinates(crd);
  ctx->setRandomSeedForVelocities(randomSeed);
  ctx->assignVelocitiesAtTemperature(temperature);
  ctx->useHolonomicConstraints(useHolonomicConstraints);

  auto integrator =
      std::make_shared<CudaNoseHooverThermostatIntegrator>(timeStep);
  integrator->setCharmmContext(ctx);
  integrator->initializeFromRestartFile(rstName);

  integrator->propagate(nsteps);

  integrator->getAverageTemperature().transferToHost();
  std::cout << "integrator->getAverageTemperature()[0] = "
            << std::setprecision(12) << integrator->getAverageTemperature()[0]
            << std::endl;
}
* */
