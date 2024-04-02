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
#include "CudaMinimizer.h"
#include "CudaVelocityVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "RestartSubscriber.h"
#include "catch.hpp"
#include "helper.h"
#include "test_paths.h"
#include <iomanip>
#include <iostream>

TEST_CASE("waterbox", "[all]") {
  int nSteps = 2000;
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
  ctx->calculatePotentialEnergy(true, true);

  // assert that all the forces are correct
  // upto third place in decimal and (importantly) in the direction!
  auto charmmForces =
      std::make_shared<CharmmCrd>(dataPath + "waterbox_p21_min_force.crd");
  int stride = ctx->getForceStride();
  auto force = ctx->getForces()->xyz();

  CudaContainer<double> forcesContainer;
  forcesContainer.allocate(stride * 3);
  forcesContainer.setDeviceArray(force);
  forcesContainer.transferFromDevice();
  compareP21Forces(ctx->getNumAtoms(), stride, forcesContainer.getHostArray(),
                   charmmForces->getCoordinates());

  SECTION("nve") {
    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    langevinThermostat->setFriction(0.0);
    langevinThermostat->setCharmmContext(ctx);
    langevinThermostat->propagate(nSteps);
  }

  SECTION("nvt") {
    auto langevinThermostat =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    langevinThermostat->setFriction(12.0);
    langevinThermostat->setBathTemperature(300.0);
    langevinThermostat->setCharmmContext(ctx);
    langevinThermostat->propagate(nSteps);
  }

  SECTION("npt") {
    auto virial = fm->getVirial();
    int k = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        std::cout << virial[k++] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(5.0);
    integrator->setCharmmContext(ctx);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setPistonMass({500.0});

    std::string subfile = "p21_waterbox_npt.dcd";
    auto subscriber = std::make_shared<DcdSubscriber>(subfile, 1000);
    // integrator->setDebugPrintFrequency(100);
    integrator->propagate(nSteps);
  }
}

TEST_CASE("bilayer", "[all]") {
  int nSteps = 2000;
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{
      dataPath + "par_all36_prot.prm", dataPath + "par_all36_lipid.prm",
      dataPath + "toppar_all36_lipid_cholesterol.str",
      dataPath + "toppar_water_ions.str"};

  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "fp.psf");
  // psf->setHydrogenMass(4.0320);

  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions({64.52, 64.52, 102.02});
  fm->setFFTGrid(72, 72, 108);
  fm->setKappa(0.34);
  fm->setCutoff(12.5);
  fm->setCtonnb(9.0);
  fm->setCtofnb(11.0);
  fm->setPeriodicBoundaryCondition(PBC::P21);

  auto ctx = std::make_shared<CharmmContext>(fm);
  auto crd = std::make_shared<CharmmCrd>(dataPath + "min_p21.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(298.17);

  // ctx->useHolonomicConstraints(false);

  std::cout << "Potential energy : \n"
            << ctx->calculatePotentialEnergy(true, true) << "\n";

  /*
  auto energyComponents = fm->getEnergyComponents();
  for (auto energyComponent : energyComponents) {
    std::cout << energyComponent.first << " " << energyComponent.second << "\n";
    */

  auto langevinThermostat =
      std::make_shared<CudaLangevinThermostatIntegrator>(0.001);

  langevinThermostat->setBathTemperature(300.0);
  langevinThermostat->setCharmmContext(ctx);

  SECTION("nve") {
    langevinThermostat->setFriction(0.0);
    langevinThermostat->propagate(nSteps);
  }

  SECTION("nvt") {
    langevinThermostat->setFriction(12.0);
    langevinThermostat->propagate(nSteps);
  }

  SECTION("npt") {

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(0.0);
    integrator->setCharmmContext(ctx);
    integrator->setPistonMass({500.0});
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setNoseHooverFlag(false);

    integrator->propagate(nSteps);
    std::string subfile = "p21_bilayer_npt.dcd";
    // auto subscriber = std::make_shared<DcdSubscriber>(subfile, 1000, ctx);
    auto subscriber = std::make_shared<DcdSubscriber>(subfile, 1000);
    // subscriber->setCharmmContext(ctx);
    integrator->subscribe(subscriber);
    // fm->setPrintEnergyDecomposition(true);

    // integrator->setDebugPrintFrequency(100);
    integrator->propagate(nSteps);
  }
  SECTION("nvt_npt") {

    langevinThermostat->setFriction(12.0);
    langevinThermostat->propagate(nSteps);

    std::cout << "nvt_npt: nvt done\n";

    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(0.002);
    integrator->setPistonFriction(20.0);
    integrator->setCharmmContext(ctx);
    integrator->setCrystalType(CRYSTAL::TETRAGONAL);
    integrator->setPistonMass({5000.0});
    integrator->setBathTemperature(300.0);
    // integrator.setNoseHooverFlag(false);

    integrator->propagate(nSteps);

    std::string subfile = "p21_bilayer_npt.dcd";
    auto subscriber = std::make_shared<DcdSubscriber>(subfile, 1000);
    integrator->subscribe(subscriber);

    // integrator->setDebugPrintFrequency(100);
    integrator->propagate(nSteps);
  }
}

TEST_CASE("virial") {
  std::string dataPath = getDataPath();
  std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str"};

  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  // auto psf = std::make_shared<CharmmPSF>(dataPath + "nacl_2ions.psf");
  auto psf = std::make_shared<CharmmPSF>(dataPath + "nacl3.psf");
  // auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");

  auto fm = std::make_shared<ForceManager>(psf, prm);
  fm->setBoxDimensions({50.0, 50.0, 50.0});
  // fm->setKappa(0.0);
  fm->setPrintEnergyDecomposition(true);

  std::shared_ptr<CharmmContext> ctx;
  std::shared_ptr<CharmmCrd> crd;

  SECTION("p1") {
    ctx = std::make_shared<CharmmContext>(fm);
    // crd = std::make_shared<CharmmCrd>(dataPath + "nacl_2ions.cor");
    crd = std::make_shared<CharmmCrd>(dataPath + "nacl.cor");

    // ctx->assignVelocitiesAtTemperature(298.17);
  }

  SECTION("p21") {
    fm->setPeriodicBoundaryCondition(PBC::P21);
    ctx = std::make_shared<CharmmContext>(fm);
    // crd = std::make_shared<CharmmCrd>(dataPath + "nacl_2ions.cor");
    crd = std::make_shared<CharmmCrd>(dataPath + "nacl.cor");
    // crd = std::make_shared<CharmmCrd>(dataPath + "waterbox_p21_min.crd");
  }

  ctx->setCoordinates(crd);

  ctx->calculatePotentialEnergy(true, true);
  int stride = ctx->getForceStride();
  auto force = ctx->getForces()->xyz();

  CudaContainer<double> forcesContainer;
  forcesContainer.allocate(stride * 3);
  forcesContainer.setDeviceArray(force);
  forcesContainer.transferFromDevice();

  // std::cout << "Forces :  \n";

  /*for (int i = 0; i < ctx->getNumAtoms(); i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << std::setw(16) << forcesContainer[i + j * stride] << " ";
    }
    std::cout << "\n";
  }
  */

  std::cout << "Virial : \n";

  auto virial = ctx->getVirial().getHostArray();

  int k = 0;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << std::setw(16) << virial[k++] << " ";
    }
    std::cout << "\n";
  }
}

TEST_CASE("dimer_bilayer", "[all]") {
  std::string dataPath = "/u/arice/Projects/fusion_peptide_influenza/dimer_PMF/"
                         "POPC/P21_test/";
  std::vector<std::string> prmFiles{dataPath + "toppar/par_all36m_prot.prm",
                                    dataPath + "toppar/par_all36_lipid.prm",
                                    dataPath + "toppar/toppar_water_ions.str"};

  auto prm = std::make_shared<CharmmParameters>(prmFiles);
  auto psf = std::make_shared<CharmmPSF>(dataPath + "dimer.psf");
  // psf->setHydrogenMass(4.0320);

  auto fm = std::make_shared<ForceManager>(psf, prm);

  fm->setBoxDimensions(
      {103.12799835205078, 103.00199890136719, 79.9280014038086});
  fm->setFFTGrid(96, 96, 96);
  fm->setKappa(0.34);
  fm->setCutoff(12.0);
  fm->setCtonnb(8.0);
  fm->setCtofnb(10.0);
  fm->setPeriodicBoundaryCondition(PBC::P21);

  auto ctx = std::make_shared<CharmmContext>(fm);
  // auto crd = std::make_shared<CharmmCrd>(dataPath + "dimer.crd");
  auto crd = std::make_shared<CharmmCrd>(
      "/u/samar/Documents/git/test_gpu/minimize/min_p21.crd");
  ctx->setCoordinates(crd);
  ctx->assignVelocitiesAtTemperature(310.0);

  std::cout << "Potential energy : \n"
            << ctx->calculatePotentialEnergy(true, true) << "\n";

  auto energyComponents = fm->getEnergyComponents();
  for (auto energyComponent : energyComponents) {
    std::cout << energyComponent.first << " " << energyComponent.second << "\n";
  }
  /*
  CudaMinimizer minimizer;
  minimizer.setCharmmContext(ctx);
  std::cout << "Minimizing...\n";
  minimizer.minimize(100);
  */

  CudaLangevinThermostatIntegrator langevinThermostat(0.001);

  langevinThermostat.setBathTemperature(310.0);
  langevinThermostat.setCharmmContext(ctx);

  SECTION("nve") {
    CudaMinimizer minimizer;
    minimizer.setCharmmContext(ctx);
    std::cout << "Minimizing...\n";
    minimizer.minimize(100);

    langevinThermostat.setFriction(0.0);
    langevinThermostat.propagate(2e5);
    // fm->setPrintEnergyDecomposition(true);
    langevinThermostat.setDebugPrintFrequency(100);
    langevinThermostat.propagate(2e5);
  }

  SECTION("nvt") {
    langevinThermostat.setFriction(12.0);
    langevinThermostat.propagate(2e5);
    // fm->setPrintEnergyDecomposition(true);
    langevinThermostat.setDebugPrintFrequency(100);
    langevinThermostat.propagate(2e5);
  }

  SECTION("npt") {

    auto integrator = CudaLangevinPistonIntegrator(0.002);
    integrator.setPistonFriction(0.0);
    integrator.setCharmmContext(ctx);
    integrator.setPistonMass({500.0});
    integrator.setCrystalType(CRYSTAL::TETRAGONAL);
    integrator.setNoseHooverFlag(false);
    // integrator.setNoseHooverPistonMass(1000.0);
    // integrator.setDebugPrintFrequency(1);

    integrator.propagate(1e4);
    std::string subfile = "p21_bilayer_npt.dcd";
    // auto subscriber = std::make_shared<DcdSubscriber>(subfile, 1000, ctx);
    auto subscriber = std::make_shared<DcdSubscriber>(subfile, 1000);
    // subscriber->setCharmmContext(ctx);
    integrator.subscribe(subscriber);
    // fm->setPrintEnergyDecomposition(true);

    integrator.setDebugPrintFrequency(100);
    integrator.propagate(1e6);
  }
  SECTION("nvt_npt") {

    langevinThermostat.setFriction(12.0);
    langevinThermostat.propagate(5e4);

    auto integrator = CudaLangevinPistonIntegrator(0.002);
    integrator.setPistonFriction(0.0);
    integrator.setCharmmContext(ctx);
    integrator.setCrystalType(CRYSTAL::TETRAGONAL);
    integrator.setPistonMass({500.0});
    integrator.setBathTemperature(300.0);
    integrator.setRemoveCenterOfMassFrequency(1000);
    integrator.setNoseHooverFlag(false);

    integrator.propagate(1e4);

    std::string subfile = "p21_bilayer_npt.dcd";
    auto subscriber = std::make_shared<DcdSubscriber>(subfile, 10000);
    // integrator.subscribe(subscriber);

    integrator.setDebugPrintFrequency(1000);
    integrator.propagate(1e7);
  }
}
