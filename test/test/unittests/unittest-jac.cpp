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
#include "XYZSubscriber.h"
#include "catch.hpp"
#include <iostream>
#include <string>
#include <vector>

TEST_CASE("verlet integrator", "[dynamics]") {
  SECTION("jac") {
    std::vector<std::string> prmFiles{"../test/data/par_all22_prot.prm",
                                      "../test/data/toppar_water_ions.str"};
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>(prmFiles);
    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>("../test/data/jac_5dhfr.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({62.23, 62.23, 62.23});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>("../test/data/jac_5dhfr.crd");
    ctx->setCoordinates(crd);

    // ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    CudaVelocityVerletIntegrator integrator(0.001);
    integrator.setSimulationContext(ctx);

    // integrator.propagate(20000);
    // auto subscriber = std::make_shared<NetCDFSubscriber>("vv_jac.nc", ctx);
    // auto subscriber = std::make_shared<XYZSubscriber>("jac_vv_out.xyz", ctx);
    // auto subscriber = std::make_shared<DcdSubscriber>("vv_jac.dcd", ctx);
    // ctx->subscribe(subscriber);
    // auto stateSubscriber = std::make_shared<StateSubscriber>("vv_jac.txt",
    // ctx); ctx->subscribe(stateSubscriber);

    integrator.setReportSteps(5000);
    integrator.propagate(100000);
  }
}
