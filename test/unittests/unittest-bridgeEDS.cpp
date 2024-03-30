// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include "BEDSForceManager.h"
#include "BEDSSubscriber.h"
#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "ForceManagerGenerator.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <vector>

TEST_CASE("eds", "[energy]") {
  std::string dataPath = getDataPath();
  SECTION("hexane_vdw") {

    auto psf1 = // std::make_shared<CharmmPSF>(dataPath + "hexane.solv.psf");
        std::make_shared<CharmmPSF>(dataPath + "hexane.solv.discharged.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_cgenff.prm"};
    std::vector<std::string> prmFiles2{dataPath + "toppar_water_ions.str",
                                       dataPath +
                                           "eds/par_all36_cgenff.prm.mod"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto prm2 = std::make_shared<CharmmParameters>(prmFiles2);
    auto fm0 = std::make_shared<ForceManager>(psf1, prm);
    auto fm1 = std::make_shared<ForceManager>(psf1, prm2);

    auto fmlEDS = std::make_shared<BEDSForceManager>();
    fmlEDS->addForceManager(fm0);
    fmlEDS->addForceManager(fm1);

    double boxLength = 37.0;
    int fftDim = 36;
    fmlEDS->setBoxDimensions({boxLength, boxLength, boxLength});
    fmlEDS->setFFTGrid(fftDim, fftDim, fftDim);
    fmlEDS->setPmeSplineOrder(4);
    fmlEDS->setKappa(0.34);
    fmlEDS->setCutoff(10.0);
    fmlEDS->setCtonnb(8.0);
    fmlEDS->setCtofnb(9.0);

    // fmlEDS.initialize(); // not available ?

    auto ctx = std::make_shared<CharmmContext>(fmlEDS);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "hexane.equil.crd");

    ctx->setCoordinates(crd);

    // fmlEDS->setLambdas(
    //    {0.0, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.975, 0.990, 1.0});
    // fmlEDS->setLambdas({0.0000, 0.2878, 0.7782, 0.9141, 0.9658, 0.9872,
    // 0.9960,
    //                    0.9990, 0.9995, 0.9999, 1.0000});
    //  fmlEDS->setLambdas(
    //      {0.0000, 0.9658, 0.9872, 0.9960, 0.9990, 0.9995, 0.9999, 1.0000});

    std::vector<float> lambdas = {0.0000, 0.2878, 0.7782, 0.90, 1.0000};
    // std::vector<float> lambdas = {0.0000, 1.0000};

    fmlEDS->setLambdas(lambdas);
    fmlEDS->setSValue(0.05);
    fmlEDS->setEndStateEnergyOffsets({-16202.8, -16216.0});

    ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();

    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setFriction(5.0);
    integrator->setBathTemperature(300.0);
    integrator->setCharmmContext(ctx);

    integrator->propagate(50000);
    auto bedsSub = std::make_shared<BEDSSubscriber>("dbexp_bridgeEds_vdw.out");
    bedsSub->setReportFreq(1000);
    integrator->subscribe(bedsSub);
    integrator->propagate(
        5000); // 500k ? You can propagate more when you'll assert something !
  }
  /*
  SECTION("hexane_elec") {
    auto psf0 = std::make_shared<CharmmPSF>(dataPath + "hexane.solv.psf");
    auto psf1 =
        std::make_shared<CharmmPSF>(dataPath + "hexane.solv.discharged.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_cgenff.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm0 = std::make_shared<ForceManager>(psf0, prm);
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);

    auto fmlEDS = std::make_shared<BEDSForceManager>();
    fmlEDS->addForceManager(fm0);
    fmlEDS->addForceManager(fm1);

    double boxLength = 37.0;
    int fftDim = 36;
    fmlEDS->setBoxDimensions({boxLength, boxLength, boxLength});
    fmlEDS->setFFTGrid(fftDim, fftDim, fftDim);
    fmlEDS->setPmeSplineOrder(4);
    fmlEDS->setKappa(0.34);
    fmlEDS->setCutoff(10.0);
    fmlEDS->setCtonnb(8.0);
    fmlEDS->setCtofnb(9.0);

    auto ctx = std::make_shared<CharmmContext>(fmlEDS);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "hexane.equil.crd");

    ctx->setCoordinates(crd);

    fmlEDS->setLambdas({0.0, 0.25, 0.5, 0.75, 1.0});
    fmlEDS->setSValue(0.05);
    fmlEDS->setEndStateEnergyOffsets({-16143.8, -16202.8});

    ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();

    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
    CudaLangevinThermostatIntegrator integrator(0.002);
    integrator.setFriction(5.0);
    integrator.setBathTemperature(300.0);
    integrator.setCharmmContext(ctx);

    auto mbarSub = std::make_shared<MBARSubscriber>("bridgeEdsMbar.out");
    mbarSub->setReportFreq(1000);
    integrator.subscribe(mbarSub);
    integrator.propagate(1000000);
  }
  */
}
