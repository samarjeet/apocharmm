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
#include "FEPEIForceManager.h"
#include "FEPSubscriber.h"
#include "ForceManagerGenerator.h"
#include "MBARForceManager.h"
#include "NetCDFSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <vector>

TEST_CASE("TI-PI") {
  std::string dataPath = getDataPath();
  SECTION("vdw") {

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

    auto fmFEP = std::make_shared<FEPEIForceManager>();
    fmFEP->addForceManager(fm0);
    fmFEP->addForceManager(fm1);

    double boxLength = 37.0;
    int fftDim = 36;
    fmFEP->setBoxDimensions({boxLength, boxLength, boxLength});
    fmFEP->setFFTGrid(fftDim, fftDim, fftDim);
    fmFEP->setPmeSplineOrder(4);
    fmFEP->setKappa(0.34);
    fmFEP->setCutoff(10.0);
    fmFEP->setCtonnb(7.0);
    fmFEP->setCtofnb(8.0);

    // fmFEP->setCutoff(14.0);
    // fmFEP->setCtonnb(11.0);
    // fmFEP->setCtofnb(12.0);

    auto ctx = std::make_shared<CharmmContext>(fmFEP);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "hexane.equil.crd");

    ctx->setCoordinates(crd);

    ctx->assignVelocitiesAtTemperature(300);

    /*std::vector<float> lambdas = {0.0000, 0.2878, 0.7782, 0.9141,
                                  0.9658, 0.9872, 0.9960, 0.9990,
                                  0.9995, 0.9999, 1.0000};
    */
    std::vector<float> lambdas = {0.0000, 0.2878, 0.7782, 0.90, 1.0000};
    fmFEP->setLambdas(lambdas);

    fmFEP->setLambda(0.2878);

    ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();
    // int a ;
    // std::cin >>  a;

    auto equilIntegrator =
        std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    equilIntegrator->setFriction(5.0);
    equilIntegrator->setBathTemperature(300.0);
    equilIntegrator->setCharmmContext(ctx);

    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setFriction(5.0);
    integrator->setBathTemperature(300.0);
    integrator->setCharmmContext(ctx);

    auto fepSub = std::make_shared<FEPSubscriber>("dbexp_fepEI_vdw.out");
    fepSub->setReportFreq(1000);
    integrator->subscribe(fepSub);

    int numEquilibrationSteps = 100;
    int numProductionSteps = 1000;

    for (auto &lambda : lambdas) {
      std::cout << "Lambda : " << lambda << "\n";
      fmFEP->setLambda(lambda);
      equilIntegrator->propagate(numEquilibrationSteps);
      integrator->propagate(numProductionSteps);
    }
  }
}
