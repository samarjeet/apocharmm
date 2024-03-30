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
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include <iostream>
#include <string>
#include <vector>

TEST_CASE("verlet integrator", "[dynamics]") {
  SECTION("walp") {
    std::vector<std::string> prmFiles{"../test/data/par_all36_prot.prm",
                                      "../test/data/par_all36_lipid.prm",
                                      "../test/data/toppar_water_ions.str"};
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>(prmFiles);
    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>("../test/data/walp.psf");

    // TODO : this isn't how we'll be using it
    // Only charmmcontext will have to be created,
    // we will take the force manager from it.
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({53.4630707, 53.4630707, 80.4928487});
    //fm->setFFTGrid(48, 48, 48);
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(9.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>("../test/data/walp.crd");
    ctx->setCoordinates(crd);
    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
    //std::cout << "KE : " << ctx->calculateKineticEnergyOld() << "\n";
    //int inp;
    //std::cin >> inp;

    CudaVelocityVerletIntegrator integrator(0.001);
    integrator.setCharmmContext(ctx);

    integrator.propagate(50000);
    //  auto subscriber = std::make_shared<DcdSubscriber>("vv_walp.dcd", ctx);
    auto subscriber = std::make_shared<NetCDFSubscriber>("vv_walp.nc", ctx);
    ctx->subscribe(subscriber);
    auto stateSubscriber =
        std::make_shared<StateSubscriber>("vv_walp.txt", ctx);
    ctx->subscribe(stateSubscriber);

    integrator.setReportSteps(5000);
    integrator.propagate(1000000);
  }
}
