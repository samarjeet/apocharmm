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
#include<iostream>

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "catch.hpp"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "DualTopologySubscriber.h"
#include "CudaVelocityVerletIntegrator.h"


TEST_CASE("eds", "[energy]") {
  SECTION("2cle"){
    std::cout << "\n\nBegin\n========\n"; 
 
    auto psf1 = std::make_shared<CharmmPSF>("../test/data/nacl0.psf");
    auto psf2 = std::make_shared<CharmmPSF>("../test/data/nacl1.psf");
    
    //std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str", "../test/data/par_all36_cgenff.prm", "../test/data/em.str"};  
    std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str"};  
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    //auto prm = std::make_shared<CharmmParameters>("../test/data/par_all36_cgenff.prm");
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);
    auto fm2 = std::make_shared<ForceManager>(psf2, prm);

    auto fmEDS = std::make_shared<ForceManagerComposite>();
    fmEDS->setLambda(0.0);
    fmEDS->addForceManager(fm1);
    fmEDS->addForceManager(fm2);

    /*
    auto fmEDS = fm1;
    */

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

    auto crd = std::make_shared<CharmmCrd>("../test/data/nacl.cor");
    ctx->setCoordinates(crd);
    std::cout << ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();
    std::cout << ctx->calculateForces(false, true, true);
    forces = ctx->getForces();
    std::cout << ctx->calculateForces(false, true, true);
    forces = ctx->getForces();
    std::cout << ctx->calculateForces(false, true, true);
    forces = ctx->getForces();
    ctx->assignVelocitiesAtTemperature(300);

    CudaVelocityVerletIntegrator integrator(0.001);
    integrator.setSimulationContext(ctx);

    auto subscriber = std::make_shared<NetCDFSubscriber>("vv_nacl.nc", ctx);
    ctx->subscribe(subscriber);
    auto dualTopologySubscriber = std::make_shared<DualTopologySubscriber>("vv_nacl.txt", ctx);
    ctx->subscribe(dualTopologySubscriber);

    integrator.setReportSteps(1);
    integrator.propagate(1); 
  }

}
