// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include <iostream>
#include <vector>
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include<iostream>
#include "catch.hpp"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "DualTopologySubscriber.h"
#include "CudaVelocityVerletIntegrator.h"
#include "NonEquillibriumForceManager.h"

TEST_CASE("new", "[energy and dyna]") {
  SECTION("2water_1_1"){
    std::cout << "\n\nBegin\n========\n"; 
 
    std::shared_ptr<CharmmPSF> psf1 = std::make_shared<CharmmPSF>("../test/data/water2_1.psf");
    std::shared_ptr<CharmmPSF> psf2 = std::make_shared<CharmmPSF>("../test/data/water2_2.psf");

    std::shared_ptr<CharmmParameters> prm = std::make_shared<CharmmParameters>("../test/data/toppar_water_ions.str");
    auto fm2 = std::make_shared<ForceManager>(psf1, prm);
    auto fm1 = std::make_shared<ForceManager>(psf2, prm);

    auto fmNEW = std::make_shared<NonEquillibriumForceManager>();
    fmNEW->setLambda(0.0);
    fmNEW->addForceManager(fm1);
    fmNEW->addForceManager(fm2);

    /*
    auto fmNEW = fm2;
    */

    fmNEW->setBoxDimensions({50.0, 50.0, 50.0});
    fmNEW->setFFTGrid(48, 48, 48);
    fmNEW->setPmeSplineOrder(6);
    fmNEW->setKappa(0.34);
    fmNEW->setCutoff(16.0);
    fmNEW->setCtonnb(10.0);
    fmNEW->setCtofnb(12.0);
    fmNEW->initialize();

    //fmNEW->setLambdaIncrements(0.001);

    auto ctx = std::make_shared<CharmmContext>(fmNEW);

    auto crd = std::make_shared<CharmmCrd>("../test/data/water2.crd");
    ctx->setCoordinates(crd);
    ctx->calculateForces(false, true, true);
    ctx->assignVelocitiesAtTemperature(100);

    CudaVelocityVerletIntegrator integrator(0.001);
    integrator.setSimulationContext(ctx);

    auto subscriber = std::make_shared<NetCDFSubscriber>("vv_new_water.nc", ctx);
    ctx->subscribe(subscriber);
    auto dualTopologySubscriber = std::make_shared<DualTopologySubscriber>("vv_new_water.txt", ctx);
    ctx->subscribe(dualTopologySubscriber);

    integrator.setReportSteps(2000);

    for (int i=0; i < 100; ++i){
      fmNEW->setLambda(0.0);
      integrator.propagate(1000); 
    }
  }

}
