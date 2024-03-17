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
#include "CudaVelocityVerletIntegrator.h"
#include "DualTopologySubscriber.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include <iostream>
#include <vector>

TEST_CASE("rcsb", "[dynamics]") {
  SECTION("1NL0") {

    auto psf = std::make_shared<CharmmPSF>("../test/data/rcsb_1nl0.psf");

    std::vector<std::string> prmFiles{"../test/data/par_all36_prot.prm",
                                      "../tes/data/toppar_water_ions.str"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({88.0, 88.0, 88.0});
    fm->setFFTGrid(96, 96, 96);
    fm->setKappa(0.34);
    fm->setPmeSplineOrder(6);
    fm->setCutoff(14.0);
    fm->setCtonnb(10.0);
    fm->setCtofnb(12.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);

    auto crd = std::make_shared<CharmmCrd>("../test/data/rcsb_1nl0.crd");
    ctx->setCoordinates(crd);
    ctx->calculateForces(false, true, true);
    ctx->assignVelocitiesAtTemperature(300);

    CudaVelocityVerletIntegrator integrator(0.001);
    integrator.setSimulationContext(ctx);

    auto subscriber = std::make_shared<NetCDFSubscriber>("vv_1nl0.nc", ctx);
    ctx->subscribe(subscriber);

    integrator.setReportSteps(10);
    integrator.propagate(1000);
  }

  SECTION("2cle") {
    std::cout << "\n\nBegin\n========\n";

    auto psf1 = std::make_shared<CharmmPSF>("../test/data/l0.2cle.psf");
    auto psf2 = std::make_shared<CharmmPSF>("../test/data/l1.2cle.psf");

    // std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str",
    // "../test/data/par_all36_cgenff.prm", "../test/data/em.str"};
    std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str",
                                      "../test/data/par_all36_cgenff.prm",
                                      "../test/data/2cle.str"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    // auto prm =
    // std::make_shared<CharmmParameters>("../test/data/par_all36_cgenff.prm");
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);
    auto fm2 = std::make_shared<ForceManager>(psf2, prm);

    // std::shared_ptr<ForceManager> fmEDS =
    // std::make_shared<ForceManagerComposite>();

    // auto fmEDS = std::make_shared<ForceManagerComposite>();
    // fmEDS->setLambda(0.0);

    // fmEDS->addForceManager(fm1);
    // fmEDS->addForceManager(fm2);

    auto fmEDS = fm1;
    float boxLength = 30.9120;
    fmEDS->setBoxDimensions({boxLength, boxLength, boxLength});
    fmEDS->setFFTGrid(32, 32, 32);
    fmEDS->setPmeSplineOrder(6);
    fmEDS->setKappa(0.34);
    fmEDS->setCutoff(16.0);
    fmEDS->setCtonnb(10.0);
    fmEDS->setCtofnb(12.0);
    fmEDS->initialize();

    auto ctx = std::make_shared<CharmmContext>(fmEDS);

    auto crd = std::make_shared<CharmmCrd>("../test/data/solv2.2cle.cor");
    ctx->setCoordinates(crd);
    std::cout << ctx->calculateForces(false, true, true);
    ctx->assignVelocitiesAtTemperature(300);

    CudaVelocityVerletIntegrator integrator(0.001);
    integrator.setSimulationContext(ctx);

    auto subscriber = std::make_shared<NetCDFSubscriber>("vv_eds_2cle.nc", ctx);
    ctx->subscribe(subscriber);
    auto dualTopologySubscriber =
        std::make_shared<DualTopologySubscriber>("vv_eds_2cle.txt", ctx);
    ctx->subscribe(dualTopologySubscriber);

    integrator.setReportSteps(10);
    // integrator.propagate(1000);
  }

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
  std::make_shared<CharmmParameters>("../test/data/par_all36_cgenff.prm"); auto
  fm1 = std::make_shared<ForceManager>(psf1, prm); auto fm2 =
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
    //integrator.setSimulationContext(ctx);

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
