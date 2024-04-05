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
#include "CudaMinimizer.h"
#include "CudaVelocityVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>

TEST_CASE("waterbox", "[minimize]") {
  SECTION("waterbox") {
    std::string dataPath = getDataPath();
    std::vector<std::string> prmFiles{dataPath + "par_all22_prot.prm",
                                      dataPath + "toppar_water_ions.str"};
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>(prmFiles);
    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>(dataPath + "jac_5dhfr.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({62.23, 62.23, 62.23});
    fm->setFFTGrid(64, 64, 64);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "jac_5dhfr.crd");
    ctx->setCoordinates(crd);

    ctx->assignVelocitiesAtTemperature(300);

    ctx->calculatePotentialEnergy(true);

    float poti, potf, pottmp;

    auto potcontainer = ctx->getPotentialEnergy();
    potcontainer.transferToHost();
    poti = potcontainer.getHostArray()[0];

    CudaMinimizer minimizer;
    minimizer.setVerboseFlag(true);
    minimizer.setCharmmContext(ctx);
    minimizer.minimize(10);

    ctx->calculatePotentialEnergy(true);
    potcontainer = ctx->getPotentialEnergy();
    potcontainer.transferToHost();
    pottmp = potcontainer.getHostArray()[0];

    minimizer.minimize(1000);

    ctx->calculatePotentialEnergy(true);
    potcontainer = ctx->getPotentialEnergy();
    potcontainer.transferToHost();
    potf = potcontainer.getHostArray()[0];

    std::cout << poti << " " << pottmp << " " << potf << std::endl;
    CHECK(poti > pottmp);
    CHECK(pottmp > potf);

    auto integrator = std::make_shared<CudaVelocityVerletIntegrator>(0.002);
    integrator->setCharmmContext(ctx);

    integrator->propagate(50);
  }
}
