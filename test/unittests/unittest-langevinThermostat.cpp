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
#include "CudaLangevinThermostatIntegrator.h"
#include "CudaMinimizer.h"
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "RestartSubscriber.h"
#include "StateSubscriber.h"

#include "catch.hpp"
#include "test_paths.h"
#include <iostream>

/* For several systems,
  - verify conservation of energy if we set friction to 0,
  - verify conservation of temperature
  - verify heating-up capability ?
  */
TEST_CASE("waterbox", "[dynamics]") {
  auto path = getDataPath();
  auto dataPath = getDataPath();
  auto prm = std::make_shared<CharmmParameters>(path + "toppar_water_ions.str");
  auto psf = std::make_shared<CharmmPSF>(path + "waterbox.psf");
  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({50.0, 50.0, 50.0});
  fm->setKappa(0.34);
  fm->setFFTGrid(48, 48, 48);
  fm->initialize();
  auto ctx = std::make_shared<CharmmContext>(fm);
  double friction = 5.0;

  SECTION("TemperatureConservation") {
    float tempFinal = 300.;

    ctx->readRestart(path + "restart/heat_waterbox.restart");
    // CudaLangevinThermostatIntegrator integrator(0.002, tempFinal, friction);
    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setFriction(friction);
    integrator->setBathTemperature(tempFinal);

    integrator->setSimulationContext(ctx);

    REQUIRE(integrator->getFriction() == friction);

    auto sub = std::make_shared<StateSubscriber>("heat_waterbox.txt", 1000);
    integrator->subscribe(sub);
    std::vector<float> tempvals;
    for (int i = 0; i < 100; i++) {
      integrator->propagate(1000);
      double temp = ctx->computeTemperature();
      tempvals.push_back(temp);
    }

    float tempAvg;
    for (auto &t : tempvals) {
      tempAvg += t;
    }
    tempAvg /= tempvals.size();

    CHECK(tempAvg == Approx(tempFinal).margin(0.1));
  }
}
/*SECTION("EnergyConservation") {

  ctx->readRestart(path + "restart/heat_waterbox.restart");
  ctx->assignVelocitiesAtTemperature(300);

  std::shared_ptr<CudaLangevinThermostatIntegrator> integrator =
      std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
  integrator->setFriction(12.0);
  integrator->setBathTemperature(300.0);
  integrator->setSimulationContext(ctx);

  std::cout << "DOF:" << ctx->getDegreesOfFreedom() << "\n";
  // auto subscriber =
  //     std::make_shared<DcdSubscriber>("lang_5_waterbox.dcd", ctx);
  // auto stateSubscriber =
  //     std::make_shared<StateSubscriber>("lang_5_waterbox.txt", ctx);

  // ctx->subscribe(subscriber);
  // ctx->subscribe(stateSubscriber);

  // integrator.setReportSteps(5000);

  // integrator.propagate(1000);
  integrator->propagate(100000);
}
*/
//
/*SECTION("cpz bcd") {
  std::vector<std::string> prmFiles{
      "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/cpz.om.prm",
      "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/bcd.prm",
      "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/toppar_water.str"};

  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);

  auto psf = std::make_shared<CharmmPSF>(
      "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/cpz.bcd.syst.psf");
  auto result = prm->getBondedParamsAndLists(psf);
  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({36.0, 36.0, 36.0});
  fm->setFFTGrid(48, 48, 48);
  fm->setKappa(0.34);
  fm->setPmeSplineOrder(6);
  fm->setCutoff(12.0);
  fm->setCtonnb(8.0);
  fm->setCtofnb(10.0);
  // fm->initialize();

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(
      "/u/aviatfel/work/sampl9/revisit/cleanstart/setup/cpz.bcd.mini2.crd");
  ctx->setCoordinates(crd);

  std::cout << "Energy : " << ctx->calculatePotentialEnergy(true, true);
  ctx->assignVelocitiesAtTemperature(300);
  ctx->useHolonomicConstraints(true);

  double timeStep = 0.001;
  auto integrator = CudaLangevinThermostatIntegrator(timeStep);
  integrator.setFriction(5.0);
  integrator.setBathTemperature(298.17);
  integrator.setSimulationContext(ctx);

  auto integrator2 = CudaVelocityVerletIntegrator(timeStep);
  integrator2.setSimulationContext(ctx);

  // auto subscriber = std::make_shared<DcdSubscriber>("lang_cpz_bcd.dcd",
  // ctx);
  //  auto stateSubscriber =
  //      std::make_shared<StateSubscriber>("lang_cpz_bcd.txt", ctx);
  //  auto xyzSubscriber =
  //      std::make_shared<XYZSubscriber>("lang_cpz_bcd.xyz", ctx);
  //   ctx->subscribe(subscriber);
  //   ctx->subscribe(stateSubscriber);
  //    ctx->subscribe(xyzSubscriber);

  // integrator.setReportSteps(1);
  integrator.propagate(10000);
  // integrator2.setReportSteps(1);
  //  integrator2.propagate(10000);
}
*/
/*
SECTION("5dfr") {
  std::string path1 = "/u/samar/Documents/git/test_gpu/charmm-gui-5dfr/";
  std::vector<std::string> prmFiles{path1 + "toppar/par_all36m_prot.prm",
                                    path1 + "toppar/toppar_water_ions.str"};
  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);

  std::shared_ptr<CharmmPSF> psf =
      std::make_shared<CharmmPSF>(path1 + "step3_pbcsetup.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({62.23, 62.23, 62.23});
  fm->setFFTGrid(64, 64, 64);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(path1 + "step3_pbcsetup.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);

  ctx->assignVelocitiesAtTemperature(300);

  CudaLangevinThermostatIntegrator integrator(0.002);
  integrator.setFriction(5.0);
  integrator.setBathTemperature(300.0);
  integrator.setSimulationContext(ctx);

  std::cout << "DOF:" << ctx->getDegreesOfFreedom() << "\n";
  integrator.propagate(1000);

  // integrator.propagate(100000);
}

//
SECTION("jac") {
  std::vector<std::string> prmFiles{path + "par_all22_prot.prm",
                                    path + "toppar_water_ions.str"};
  std::shared_ptr<CharmmParameters> prm =
      std::make_shared<CharmmParameters>(prmFiles);

  std::shared_ptr<CharmmPSF> psf =
      std::make_shared<CharmmPSF>(path + "jac_5dhfr.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({62.23, 62.23, 62.23});
  fm->setFFTGrid(64, 64, 64);
  fm->setKappa(0.34);
  fm->setCutoff(10.0);
  fm->setCtonnb(7.0);
  fm->setCtofnb(8.0);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(path + "jac_5dhfr.crd");
  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);

  ctx->assignVelocitiesAtTemperature(300);

  CudaLangevinThermostatIntegrator integrator(0.002);
  integrator.setFriction(5.0);
  integrator.setBathTemperature(300.0);
  integrator.setSimulationContext(ctx);

  std::cout << "DOF:" << ctx->getDegreesOfFreedom() << "\n";
  integrator.propagate(1000);

  // integrator.propagate(100000);

  // Add some assertions there !
}

*/
//
