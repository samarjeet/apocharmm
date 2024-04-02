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
#include "CompositeSubscriber.h"
#include "Constants.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "EDSForceManager.h"
#include "ForceManagerGenerator.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"

#include <iostream>
#include <vector>

TEST_CASE("basic", "[unit]") {
  std::string dataPath = getDataPath();
  SECTION("Initialization") {
    auto psf0 = std::make_shared<CharmmPSF>(dataPath + "l0.pert.25k.psf");
    auto psf1 = std::make_shared<CharmmPSF>(dataPath + "l1.pert.25k.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_cgenff.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm0 = std::make_shared<ForceManager>(psf0, prm);
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);

    auto fmEDS = std::make_shared<EDSForceManager>();
    fmEDS->addForceManager(fm0);
    fmEDS->addForceManager(fm1);

    double boxLength = 62.79503;
    int fftDim = 64;
    fmEDS->setBoxDimensions({boxLength, boxLength, boxLength});
    fmEDS->setFFTGrid(fftDim, fftDim, fftDim);
    fmEDS->initialize();
    fmEDS->setSValue(0.05);
    fmEDS->setEnergyOffsets({-81037.0, -81040.0});

    // Check that the EDS ForceManager is initialized
    CHECK(fmEDS->isInitialized());
    // Check that the EDS ForceManager LINKED TO THE CONTEXT is initialized (has
    // been an issue)
    auto ctx = std::make_shared<CharmmContext>(fmEDS);
    CHECK(ctx->getForceManager()->isInitialized());
  }
}

TEST_CASE("consph", "[unit]") {
  std::string dataPath =
      "/v/gscratch/cbs/adachen/EDS_1CVO_apoCHARMM/1CVO/02_test_eds/";
  std::string paramPath = "../test/data/";
  SECTION("base") {
    auto psf_111 = std::make_shared<CharmmPSF>(dataPath + "1CVO_111.psf");
    auto psf_011 = std::make_shared<CharmmPSF>(dataPath + "1CVO_011.psf");
    auto psf_101 = std::make_shared<CharmmPSF>(dataPath + "1CVO_101.psf");
    auto psf_110 = std::make_shared<CharmmPSF>(dataPath + "1CVO_110.psf");
    auto psf_100 = std::make_shared<CharmmPSF>(dataPath + "1CVO_100.psf");
    auto psf_010 = std::make_shared<CharmmPSF>(dataPath + "1CVO_010.psf");
    auto psf_001 = std::make_shared<CharmmPSF>(dataPath + "1CVO_001.psf");
    auto psf_000 = std::make_shared<CharmmPSF>(dataPath + "1CVO_000.psf");

    std::vector<std::string> prmFiles{paramPath + "toppar_water_ions.str",
                                      paramPath + "par_all36m_prot.prm"};

    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm_111 = std::make_shared<ForceManager>(psf_111, prm);
    auto fm_011 = std::make_shared<ForceManager>(psf_011, prm);
    auto fm_101 = std::make_shared<ForceManager>(psf_101, prm);
    auto fm_110 = std::make_shared<ForceManager>(psf_110, prm);
    auto fm_100 = std::make_shared<ForceManager>(psf_100, prm);
    auto fm_010 = std::make_shared<ForceManager>(psf_010, prm);
    auto fm_001 = std::make_shared<ForceManager>(psf_001, prm);
    auto fm_000 = std::make_shared<ForceManager>(psf_000, prm);

    auto fmEDS = std::make_shared<EDSForceManager>();
    fmEDS->addForceManager(fm_111);
    fmEDS->addForceManager(fm_011);
    fmEDS->addForceManager(fm_101);
    fmEDS->addForceManager(fm_110);
    fmEDS->addForceManager(fm_100);
    fmEDS->addForceManager(fm_010);
    fmEDS->addForceManager(fm_001);
    fmEDS->addForceManager(fm_000);

    double boxLength = 60.0;
    int fftDim = 72;
    fmEDS->setBoxDimensions({boxLength, boxLength, boxLength});
    fmEDS->setFFTGrid(fftDim, fftDim, fftDim);
    fmEDS->setPmeSplineOrder(4);
    fmEDS->setKappa(0.34);
    fmEDS->setCutoff(10.0);
    fmEDS->setCtonnb(8.0);
    fmEDS->setCtofnb(9.0);
    fmEDS->initialize();

    double pH = 7.0;
    double T = 300.0;
    double ln10 = std::log(10.0);
    double kT = charmm::constants::kBoltz * T;

    double offsetASP = 43.60 + kT * ln10 * (pH - 4.0);
    double offsetGLU = 46.15 + kT * ln10 * (pH - 4.4);

    double offset_111 = 0.0;
    double offset_011 = offsetGLU;
    double offset_101 = offsetASP;
    double offset_110 = offsetASP;
    double offset_100 = offsetASP * 2.0;
    double offset_010 = offsetGLU + offsetASP;
    double offset_001 = offsetGLU + offsetASP;
    double offset_000 = offsetGLU + offsetASP * 2.0;

    double baseOffset = 0.0;
    fmEDS->setEnergyOffsets({baseOffset + offset_111, baseOffset + offset_011,
                             baseOffset + offset_101, baseOffset + offset_110,
                             baseOffset + offset_100, baseOffset + offset_010,
                             baseOffset + offset_001, baseOffset + offset_000});
    fmEDS->setSValue(0.007);

    auto ctx = std::make_shared<CharmmContext>(fmEDS);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "1CVO.crd");
    ctx->setCoordinates(crd);

    ctx->assignVelocitiesAtTemperature(T);

    ctx->calculatePotentialEnergy(true, true);
    auto potentialEnergy = ctx->getPotentialEnergy();
    potentialEnergy.transferFromDevice();
    for (int i = 0; i < potentialEnergy.size(); i++) {
      std::cout << "Potential Energy: " << potentialEnergy[i] << std::endl;
    }

    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setFriction(5.0);
    integrator->setBathTemperature(T);
    integrator->setCharmmContext(ctx);

    auto compositeSub = std::make_shared<CompositeSubscriber>("mbar.out");
    compositeSub->setReportFreq(100);
    integrator->subscribe(compositeSub);
    integrator->setDebugPrintFrequency(1000);

    integrator->propagate(100000);
  }
  SECTION("repd") {
    // s value replica exchange

    auto psf_111 = std::make_shared<CharmmPSF>(dataPath + "1CVO_111.psf");
    auto psf_011 = std::make_shared<CharmmPSF>(dataPath + "1CVO_011.psf");
    auto psf_101 = std::make_shared<CharmmPSF>(dataPath + "1CVO_101.psf");
    auto psf_110 = std::make_shared<CharmmPSF>(dataPath + "1CVO_110.psf");
    auto psf_100 = std::make_shared<CharmmPSF>(dataPath + "1CVO_100.psf");
    auto psf_010 = std::make_shared<CharmmPSF>(dataPath + "1CVO_010.psf");
    auto psf_001 = std::make_shared<CharmmPSF>(dataPath + "1CVO_001.psf");
    auto psf_000 = std::make_shared<CharmmPSF>(dataPath + "1CVO_000.psf");

    std::vector<std::shared_ptr<CharmmPSF>> psfs{
        psf_111, psf_011, psf_101, psf_110, psf_100, psf_010, psf_001, psf_000};

    std::vector<std::string> prmFiles{paramPath + "toppar_water_ions.str",
                                      paramPath + "par_all36m_prot.prm"};

    auto prm = std::make_shared<CharmmParameters>(prmFiles);

    std::vector<std::shared_ptr<ForceManager>> fms_0, fms_1;
    for (auto psf : psfs) {
      auto fm_0 = std::make_shared<ForceManager>(psf, prm);
      fms_0.push_back(fm_0);
      auto fm_1 = std::make_shared<ForceManager>(psf, prm);
      fms_1.push_back(fm_1);
    }

    auto fmEDS_0 = std::make_shared<EDSForceManager>();
    for (auto fm : fms_0) {
      fmEDS_0->addForceManager(fm);
    }
    auto fmEDS_1 = std::make_shared<EDSForceManager>();
    for (auto fm : fms_1) {
      fmEDS_1->addForceManager(fm);
    }

    double boxLength = 60.0;
    int fftDim = 72;
    fmEDS_0->setBoxDimensions({boxLength, boxLength, boxLength});
    fmEDS_0->setFFTGrid(fftDim, fftDim, fftDim);
    fmEDS_0->setPmeSplineOrder(4);
    fmEDS_0->setKappa(0.34);
    fmEDS_0->setCutoff(10.0);
    fmEDS_0->setCtonnb(8.0);
    fmEDS_0->setCtofnb(9.0);
    fmEDS_0->initialize();

    fmEDS_1->setBoxDimensions({boxLength, boxLength, boxLength});
    fmEDS_1->setFFTGrid(fftDim, fftDim, fftDim);
    fmEDS_1->setPmeSplineOrder(4);
    fmEDS_1->setKappa(0.34);
    fmEDS_1->setCutoff(10.0);
    fmEDS_1->setCtonnb(8.0);
    fmEDS_1->setCtofnb(9.0);
    fmEDS_1->initialize();

    double pH = 7.0;
    double T = 300.0;
    double ln10 = std::log(10.0);
    double kT = charmm::constants::kBoltz * T;

    double offsetASP = 43.60 + kT * ln10 * (pH - 4.0);
    double offsetGLU = 46.15 + kT * ln10 * (pH - 4.4);

    double offset_111 = 0.0;
    double offset_011 = offsetGLU;
    double offset_101 = offsetASP;
    double offset_110 = offsetASP;
    double offset_100 = offsetASP * 2.0;
    double offset_010 = offsetGLU + offsetASP;
    double offset_001 = offsetGLU + offsetASP;
    double offset_000 = offsetGLU + offsetASP * 2.0;

    double baseOffset = 0.0;
    fmEDS_0->setEnergyOffsets({baseOffset + offset_111, baseOffset + offset_011,
                               baseOffset + offset_101, baseOffset + offset_110,
                               baseOffset + offset_100, baseOffset + offset_010,
                               baseOffset + offset_001,
                               baseOffset + offset_000});
    fmEDS_1->setEnergyOffsets({baseOffset + offset_111, baseOffset + offset_011,
                               baseOffset + offset_101, baseOffset + offset_110,
                               baseOffset + offset_100, baseOffset + offset_010,
                               baseOffset + offset_001,
                               baseOffset + offset_000});
    fmEDS_0->setSValue(0.007);
    fmEDS_1->setSValue(0.01);

    auto ctx_0 = std::make_shared<CharmmContext>(fmEDS_0);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "1CVO.crd");
    ctx_0->setCoordinates(crd);
    ctx_0->assignVelocitiesAtTemperature(T);
    ctx_0->calculatePotentialEnergy(true, true);

    auto potentialEnergy = ctx_0->getPotentialEnergy();
    potentialEnergy.transferFromDevice();
    for (int i = 0; i < potentialEnergy.size(); i++) {
      std::cout << "Potential Energy: " << potentialEnergy[i] << std::endl;
    }

    auto ctx_1 = std::make_shared<CharmmContext>(fmEDS_1);
    ctx_1->setCoordinates(crd);
    ctx_1->assignVelocitiesAtTemperature(T);
    ctx_1->calculatePotentialEnergy(true, true);

    std::cout << "-----\n";
    potentialEnergy = ctx_1->getPotentialEnergy();
    potentialEnergy.transferFromDevice();
    for (int i = 0; i < potentialEnergy.size(); i++) {
      std::cout << "Potential Energy: " << potentialEnergy[i] << std::endl;
    }

    auto integrator_0 =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator_0->setFriction(5.0);
    integrator_0->setBathTemperature(T);
    integrator_0->setCharmmContext(ctx_0);

    auto integrator_1 =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator_1->setFriction(5.0);
    integrator_1->setBathTemperature(T);
    integrator_1->setCharmmContext(ctx_1);

    auto compositeSub_0 = std::make_shared<CompositeSubscriber>("mbar_0.out");
    compositeSub_0->setReportFreq(100);
    integrator_0->subscribe(compositeSub_0);
    integrator_0->setDebugPrintFrequency(1000);

    auto compositeSub_1 = std::make_shared<CompositeSubscriber>("mbar_1.out");
    compositeSub_1->setReportFreq(100);
    integrator_1->subscribe(compositeSub_1);
    integrator_1->setDebugPrintFrequency(1000);

    for (int i = 0; i < 100; i++) {

      integrator_0->propagate(100);
      integrator_1->propagate(100);

      ctx_0->calculatePotentialEnergy(true, true);
      ctx_1->calculatePotentialEnergy(true, true);

      auto pe00 = ctx_0->getPotentialEnergy();
      pe00.transferFromDevice();
      std::cout << "Potential energy 0:" << pe00[0] << std::endl;

      auto pe11 = ctx_1->getPotentialEnergy();
      pe11.transferFromDevice();
      std::cout << "Potential energy 1:" << pe11[0] << std::endl;

      auto temp_crd0 = ctx_0->getCoordinates();
      auto temp_crd1 = ctx_1->getCoordinates();

      ctx_0->setCoordinates(temp_crd1);
      ctx_1->setCoordinates(temp_crd0);
      std::cout << "\n\nSwapped" << std::endl;

      // Now calculate the energies
      ctx_0->calculatePotentialEnergy(true, true);
      ctx_1->calculatePotentialEnergy(true, true);

      auto pe01 = ctx_0->getPotentialEnergy(); // psf 0 and coords 1
      pe01.transferFromDevice();
      std::cout << "Potential energy 0:" << pe01[0] << std::endl;

      auto pe10 = ctx_1->getPotentialEnergy(); // psf 1 and coords 0
      pe10.transferFromDevice();
      std::cout << "Potential energy 1:" << pe10[0] << std::endl;

      // double T = 300;
      // double kT = charmm::constants::kBoltz * T;
      double beta = 1 / kT;
      auto delta = beta * (pe01[0] + pe10[0] - pe00[0] - pe11[0]);
      std::cout << "Delta: " << delta << std::endl;

      double prob = delta <= 0 ? 1 : exp(-delta);
      std::cout << "Probability: " << prob << std::endl;

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0, 1);

      double r = dis(gen);
      std::cout << "Random number: " << r << std::endl;

      if (r < prob) {
        std::cout << "Accepted" << std::endl;
      } else {
        std::cout << "Rejected" << std::endl;
        ctx_0->setCoordinates(temp_crd0);
        ctx_1->setCoordinates(temp_crd1);
      }
    }
  }
}

/*
TEST_CASE("eds2", "[debug]") {
  std::string dataPath = getDataPath();
  SECTION("fefe") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    auto generator = AlchemicalForceManagerGenerator(fm);
    std::vector<int> alchRegion;
    for (int i = 0; i < 11748; i++) {
      alchRegion.push_back(i);
    }
    generator.setAlchemicalRegion(alchRegion);

    auto fm_generated = generator.generateForceManager(0.7, 0.0);

    auto EDSfm = std::make_shared<EDSForceManager>();
    EDSfm->addForceManager(fm);
    EDSfm->addForceManager(fm_generated);

    auto ctx = std::make_shared<CharmmContext>(EDSfm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    ctx->setCoordinates(crd);

    EDSfm->setEnergyOffsets({-49754.0, -18420.0});

    EDSfm->setSValue(0.05);

    ctx->calculatePotentialEnergy(false);
  }
}
*/

TEST_CASE("eds", "[energy]") {
  std::string dataPath = getDataPath();

  SECTION("25k") {
    auto psf0 = std::make_shared<CharmmPSF>(dataPath + "l0.pert.25k.psf");
    auto psf1 = std::make_shared<CharmmPSF>(dataPath + "l1.pert.25k.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_cgenff.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm0 = std::make_shared<ForceManager>(psf0, prm);
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);

    auto fmEDS = std::make_shared<EDSForceManager>();
    fmEDS->addForceManager(fm0);
    fmEDS->addForceManager(fm1);

    double boxLength = 62.79503;
    int fftDim = 64;
    fmEDS->setBoxDimensions({boxLength, boxLength, boxLength});
    fmEDS->setFFTGrid(fftDim, fftDim, fftDim);
    fmEDS->setPmeSplineOrder(4);
    fmEDS->setKappa(0.34);
    fmEDS->setCutoff(10.0);
    fmEDS->setCtonnb(8.0);
    fmEDS->setCtofnb(9.0);
    fmEDS->initialize();

    auto ctx = std::make_shared<CharmmContext>(fmEDS);
    std::cout << "isinit " << fmEDS->isInitialized()
              << ctx->getForceManager()->isInitialized() << std::endl;

    auto crd = std::make_shared<CharmmCrd>("../test/data/nvt_equil.25k.cor");
    ctx->setCoordinates(crd);
    std::cout << "isinit " << fmEDS->isInitialized()
              << ctx->getForceManager()->isInitialized() << std::endl;

    // ctx->resetNeighborList();
    fmEDS->setSValue(0.05);
    fmEDS->setEnergyOffsets({-81037.0, -81040.0});

    ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();
    std::cout << "isinit " << fmEDS->isInitialized()
              << ctx->getForceManager()->isInitialized() << std::endl;

    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
    std::cout << "isinit " << fmEDS->isInitialized()
              << ctx->getForceManager()->isInitialized() << std::endl;
    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setFriction(5.0);
    integrator->setBathTemperature(300.0);
    std::cout << "isinit " << fmEDS->isInitialized()
              << ctx->getForceManager()->isInitialized() << std::endl;
    integrator->setCharmmContext(ctx);

    auto compositeSub = std::make_shared<CompositeSubscriber>("mbar.out");
    compositeSub->setReportFreq(100);
    integrator->subscribe(compositeSub);
    integrator->propagate(100000);

    // check the content of the BAR header !!!
  }

  /*
    SECTION("2cle"){
      std::cout << "\n\nBegin\n========\n";

      auto psf1 = std::make_shared<CharmmPSF>("../test/data/l0.2cle.psf");
      auto psf2 = std::make_shared<CharmmPSF>("../test/data/l1.2cle.psf");

      //std::vector<std::string>
    prmFiles{"../test/data/toppar_water_ions.str",
    "../test/data/par_all36_cgenff.prm", "../test/data/em.str"};
      std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str",
    "../test/data/par_all36_cgenff.prm", "../test/data/2cle.str"}; auto prm =
    std::make_shared<CharmmParameters>(prmFiles);
      //auto prm =
    std::make_shared<CharmmParameters>("../test/data/par_all36_cgenff.prm");
      auto fm1 = std::make_shared<ForceManager>(psf1, prm);
      auto fm2 = std::make_shared<ForceManager>(psf2, prm);

      //std::shared_ptr<ForceManager> fmEDS =
    std::make_shared<ForceManagerComposite>();

      auto fmEDS = std::make_shared<ForceManagerComposite>();

      fmEDS->addForceManager(fm1);
      fmEDS->addForceManager(fm2);

      //auto fmEDS = fm1;
      float boxLength = 30.9120;
      fmEDS->setBoxDimensions({boxLength, boxLength, boxLength});
      fmEDS->setFFTGrid(32, 32, 32);
      fmEDS->setPmeSplineOrder(6);
      fmEDS->setKappa(0.34);
      fmEDS->setCutoff(16.0);
      fmEDS->setCtonnb(10.0);
      fmEDS->setCtofnb(12.0);
      fmEDS->initialize();

      //fmEDS->setLambda(0.0);

      auto ctx = std::make_shared<CharmmContext>(fmEDS);

      auto crd = std::make_shared<CharmmCrd>("../test/data/solv2.2cle.cor");
      ctx->setCoordinates(crd);
      std::cout << ctx->calculateForces(false, true, true);
      ctx->assignVelocitiesAtTemperature(300);

      CudaVelocityVerletIntegrator integrator(0.001);
      integrator.setCharmmContext(ctx);

      auto subscriber = std::make_shared<NetCDFSubscriber>("vv_eds_2cle.nc",
    ctx); ctx->subscribe(subscriber); auto dualTopologySubscriber =
    std::make_shared<DualTopologySubscriber>("vv_eds_2cle.txt", ctx);
      ctx->subscribe(dualTopologySubscriber);

      integrator.setReportSteps(10);
      //integrator.propagate(1000);
    }
  */
  /*
  SECTION("etha_meoh"){
    std::cout << "\n\nBegin\n========\n";

    auto psf1 = std::make_shared<CharmmPSF>("../test/data/solv.em.meoh.psf");
    auto psf2 = std::make_shared<CharmmPSF>("../test/data/solv.em.etha.psf");

    //std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str",
  "../test/data/par_all36_cgenff.prm", "../test/data/em.str"};
    std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str",
  "../test/data/par_all36_cgenff.prm"};
    //auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto prm =
  std::make_shared<CharmmParameters>("../test/data/par_all36_cgenff.prm");
  auto fm1 = std::make_shared<ForceManager>(psf1, prm); auto fm2 =
  std::make_shared<ForceManager>(psf2, prm);

    //std::shared_ptr<ForceManager> fmEDS =
  std::make_shared<ForceManagerComposite>();

    //fmEDS->addForceManager(fm1);
    //fmEDS->addForceManager(fm2);

    auto fmEDS = fm1;
    fmEDS->setBoxDimensions({31.1032, 31.1032, 31.1032});
    fmEDS->setFFTGrid(32, 32, 32);
    fmEDS->setPmeSplineOrder(6);
    fmEDS->setKappa(0.34);
    fmEDS->setCutoff(16.0);
    fmEDS->setCtonnb(10.0);
    fmEDS->setCtofnb(12.0);
    fmEDS->initialize();

    auto ctx = std::make_shared<CharmmContext>(fmEDS);

    auto crd = std::make_shared<CharmmCrd>("../test/data/solv.em.cor");
    ctx->setCoordinates(crd);
    std::cout << ctx->calculateForces(false, true, true);
    ctx->assignVelocitiesAtTemperature(100);

    //CudaVelocityVerletIntegrator integrator(0.001);
    //integrator.setCharmmContext(ctx);

    //auto subscriber = std::make_shared<NetCDFSubscriber>("vv_eds_water.nc",
  ctx);
    //ctx->subscribe(subscriber);
    //auto dualTopologySubscriber =
  std::make_shared<DualTopologySubscriber>("vv_eds_water.txt", ctx);
    //ctx->subscribe(dualTopologySubscriber);

    //integrator.setReportSteps(10);
    //integrator.propagate(1000);
  }

  */

  /*

    SECTION("2water_1"){
      std::cout << "\n\nBegin\n========\n";

      std::shared_ptr<CharmmPSF> psf1 =
    std::make_shared<CharmmPSF>("../test/data/water2_1.psf");
      std::shared_ptr<CharmmPSF> psf2 =
    std::make_shared<CharmmPSF>("../test/data/water2_2.psf");

      std::shared_ptr<CharmmParameters> prm =
    std::make_shared<CharmmParameters>("../test/data/toppar_water_ions.str");
      auto fm1 = std::make_shared<ForceManager>(psf1, prm);
      fm1->setBoxDimensions({50.0, 50.0, 50.0});
      fm1->setFFTGrid(48, 48, 48);
      fm1->setKappa(0.34);
      fm1->setCutoff(12.0);
      fm1->initialize();


      auto fm2 = std::make_shared<ForceManager>(psf2, prm);
      fm2->setBoxDimensions({50.0, 50.0, 50.0});
      fm2->setFFTGrid(48, 48, 48);
      fm2->setKappa(0.34);
      fm2->setCutoff(12.0);
      fm2->initialize();


      //std::shared_ptr<ForceManager> fmEDS =
    std::make_shared<ForceManagerComposite>(); std::shared_ptr<ForceManager>
    fmEDS = std::make_shared<ForceManagerComposite>();

      fmEDS->addForceManager(fm1);
      fmEDS->addForceManager(fm2);

      auto ctx = std::make_shared<CharmmContext>(fmEDS);

      auto crd = std::make_shared<CharmmCrd>("../test/data/water2.crd");
      ctx->setCoordinates(crd);
      ctx->calculatePotentialEnergy(true);
      //ctx->calculatePotentialEnergy(true, true);
    }

    SECTION("2water_2"){
      std::cout << "\n\nBegin\n========\n";

      std::shared_ptr<CharmmPSF> psf1 =
    std::make_shared<CharmmPSF>("../test/data/water2_1.psf");
      std::shared_ptr<CharmmPSF> psf2 =
    std::make_shared<CharmmPSF>("../test/data/water2_1.psf");

      std::shared_ptr<CharmmParameters> prm1 =
    std::make_shared<CharmmParameters>("../test/data/toppar_water_ions.str");
      std::shared_ptr<CharmmParameters> prm2 =
    std::make_shared<CharmmParameters>("../test/data/toppar_water_ions_2.str");

      auto fm1 = std::make_shared<ForceManager>(psf1, prm1);
      fm1->setBoxDimensions({50.0, 50.0, 50.0});
      fm1->setFFTGrid(48, 48, 48);
      fm1->setKappa(0.34);
      fm1->setCutoff(12.0);
      fm1->initialize();

      auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
      fm2->setBoxDimensions({50.0, 50.0, 50.0});
      fm2->setFFTGrid(48, 48, 48);
      fm2->setKappa(0.34);
      fm2->setCutoff(12.0);
      fm2->initialize();

      auto fmEDS = std::make_shared<ForceManagerComposite>();

      fmEDS->addForceManager(fm1);
      fmEDS->addForceManager(fm2);

      auto ctx = std::make_shared<CharmmContext>(fmEDS);

      auto crd = std::make_shared<CharmmCrd>("../test/data/water2.crd");
      ctx->setCoordinates(crd);
      ctx->calculatePotentialEnergy(true);
      //ctx->calculatePotentialEnergy(true, true);
    }

    SECTION("2water_3"){
      std::cout << "\n\nBegin\n========\n";
      std::shared_ptr<CharmmPSF> psf1 =
    std::make_shared<CharmmPSF>("../test/data/water2_1.psf");
      std::shared_ptr<CharmmPSF> psf2 =
    std::make_shared<CharmmPSF>("../test/data/water2_1.psf");

      std::shared_ptr<CharmmParameters> prm1 =
    std::make_shared<CharmmParameters>("../test/data/toppar_water_ions.str");
      std::shared_ptr<CharmmParameters> prm2 =
    std::make_shared<CharmmParameters>("../test/data/toppar_water_ions.str");

      auto fm1 = std::make_shared<ForceManager>(psf1, prm1);
      fm1->setBoxDimensions({50.0, 50.0, 50.0});
      fm1->setFFTGrid(48, 48, 48);
      fm1->setKappa(0.34);
      fm1->setCutoff(12.0);
      fm1->initialize();


      auto fm2 = std::make_shared<ForceManager>(psf2, prm2);
      fm2->setBoxDimensions({50.0, 50.0, 50.0});
      fm2->setFFTGrid(48, 48, 48);
      fm2->setKappa(0.40);
      fm2->setCutoff(12.0);
      fm2->initialize();


      auto fmEDS = std::make_shared<ForceManagerComposite>();

      fmEDS->addForceManager(fm1);
      fmEDS->addForceManager(fm2);

      auto ctx = std::make_shared<CharmmContext>(fmEDS);

      auto crd = std::make_shared<CharmmCrd>("../test/data/water2.crd");
      ctx->setCoordinates(crd);
      //ctx->calculatePotentialEnergy(true);
      ctx->calculatePotentialEnergy(true, true);
    }
  */
}
