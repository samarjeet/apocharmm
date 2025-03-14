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
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>

TEST_CASE("systems", "[minimize]") {

  std::string dataPath = getDataPath();
  SECTION("waterbox") {
    bool waterbox = true;

    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    if (not waterbox)
      psf = std::make_shared<CharmmPSF>(dataPath + "water2.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50., 50., 50.});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(14.0);
    fm->setCtonnb(10.0);
    fm->setCtofnb(12.0);
    // fm->setPrintEnergyDecomposition(true);

    auto ctx = std::make_shared<CharmmContext>(fm);

    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");
    if (not waterbox)
      crd = std::make_shared<CharmmCrd>(dataPath + "water2.crd");
    ctx->setCoordinates(crd);

    ctx->calculatePotentialEnergy(true);
    // ctx->useHolonomicConstraints(false);

    CudaMinimizer minimizer;
    // minimizer.setVerboseFlag(true);
    minimizer.setCharmmContext(ctx);

    std::cout << "Minimizing" << std::endl;
    // minimizer.setVerboseFlag(true);
    // minimizer.setMethod("abnr");
    minimizer.minimize(100);
  }

  SECTION("dhfr") {
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
    fm->setCtonnb(8.0);
    fm->setCtofnb(9.0);

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
    // minimizer.setVerboseFlag(true);
    minimizer.setCharmmContext(ctx);

    std::cout << "Minimizing" << std::endl;
    minimizer.setVerboseFlag(true);
    minimizer.minimize(10);

    ctx->calculatePotentialEnergy(true);
    potcontainer = ctx->getPotentialEnergy();
    potcontainer.transferToHost();
    pottmp = potcontainer.getHostArray()[0];

    // minimizer.minimize(1000);

    ctx->calculatePotentialEnergy(true);
    potcontainer = ctx->getPotentialEnergy();
    potcontainer.transferToHost();
    potf = potcontainer.getHostArray()[0];

    std::cout << poti << " " << pottmp << " " << potf << std::endl;

    // CHECK(poti > pottmp);
    // CHECK(pottmp > potf);

    for (int i = 0; i < 0; i++) {
      minimizer.minimize(1000);
      ctx->calculatePotentialEnergy(true);
      potcontainer = ctx->getPotentialEnergy();
      potcontainer.transferToHost();
      potf = potcontainer.getHostArray()[0];
      std::cout << potf << std::endl;
    }
    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setCharmmContext(ctx);

    integrator->propagate(50);
  }
}
