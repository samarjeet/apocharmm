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
#include "DcdSubscriber.h"
#include "FEPEIForceManager.h"
#include "FEPSubscriber.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <string>

TEST_CASE("sep_benzene_pyrrole", "[free energy]") {

  std::string dataPath = getDataPath();

  SECTION("sep_ben_pyr") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "benz_pyrl4.1.prm");
    auto psf0 = std::make_shared<CharmmPSF>(dataPath + "benz_solv.psf");

    auto psf1 = std::make_shared<CharmmPSF>(dataPath + "ism_solv.psf");

    auto fm0 = std::make_shared<ForceManager>(psf0, prm);
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);

    auto fm = std::make_shared<FEPEIForceManager>();
    fm->addForceManager(fm0);
    fm->addForceManager(fm1);

    double dimX = 25.5765;
    double dimY = 25.5765;
    double dimZ = 25.5765;
    fm->setBoxDimensions({dimX, dimY, dimZ});
    fm->setFFTGrid(24, 24, 24);
    fm->setKappa(0.34);
    fm->setCutoff(12.5);
    fm->setCtonnb(10.0);
    fm->setCtofnb(12.0);

    std::vector<float> lambdas;
    float l = 0;
    for (int i = 0; i < 21; ++i)
      lambdas.push_back(i * 0.05);
    fm->setLambdas(lambdas);

    // fm->setLambda(0.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    // auto crd = std::make_shared<CharmmCrd>(dataPath + "min.0.crd");

    std::string stefanPath =
        "/u/samar/Documents/git/multi_state_paper/stefan/output/b2p1_3_fix/1/";
    auto crd = std::make_shared<CharmmCrd>(stefanPath + "min." +
                                           std::to_string(0) + ".crd");
    ctx->setCoordinates(crd);

    std::cout << ctx->calculatePotentialEnergy(true, true);
    // char c;
    // std::cin >> c;
    //  ctx->assignVelocitiesAtTemperature(0.0);
    ctx->assignVelocitiesAtTemperature(300.0);
    CudaLangevinThermostatIntegrator equilIntegrator(0.001);
    equilIntegrator.setFriction(5.0);
    equilIntegrator.setBathTemperature(300.0);
    equilIntegrator.setSimulationContext(ctx);

    CudaLangevinThermostatIntegrator integrator(0.001);
    integrator.setFriction(5.0);
    integrator.setBathTemperature(300.0);
    integrator.setSimulationContext(ctx);

    int iter = 0;

    while (iter < 21) {
      crd = std::make_shared<CharmmCrd>(stefanPath + "min." +
                                        std::to_string(iter) + ".crd");
      ctx->setCoordinates(crd);

      auto lambda = lambdas[iter];
      std::cout << "Lambda : " << lambda << "\n";
      fm->setLambda(lambda);

      auto fepSub = std::make_shared<FEPSubscriber>(
          "win_" + std::to_string(iter) + "_vfsw_fepEI_benz_ima.out");
      fepSub->setReportFreq(1000);
      integrator.subscribe(fepSub);

      equilIntegrator.propagate(100000);
      integrator.propagate(400000);
      ++iter;
    }
    // integrator.propagate(2000000);
  }
}
