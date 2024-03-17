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
/*
TEST_CASE("pert", "[energy]") {
  SECTION("pert") {

    auto psf1 = std::make_shared<CharmmPSF>("../test/data/l0.pert.100k.psf");
    auto psf2 = std::make_shared<CharmmPSF>("../test/data/l1.pert.100k.psf");

    std::vector<std::string> prmFiles{"../test/data/toppar_water_ions.str",
                                      "../test/data/par_all36_cgenff.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    // auto prm =
    // std::make_shared<CharmmParameters>("../test/data/par_all36_cgenff.prm");
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);
    auto fm2 = std::make_shared<ForceManager>(psf2, prm);

    // std::shared_ptr<ForceManager> fmPert =
    // std::make_shared<ForceManagerComposite>();

    auto fmPert = std::make_shared<ForceManagerComposite>();

    fmPert->addForceManager(fm1);
    fmPert->addForceManager(fm2);

    // auto fmPert = fm1;
    float boxLength = 99.64716;
    fmPert->setBoxDimensions({boxLength, boxLength, boxLength});
    fmPert->setFFTGrid(64, 64, 64);
    fmPert->setPmeSplineOrder(4);
    fmPert->setKappa(0.34);
    fmPert->setCutoff(10.0);
    fmPert->setCtonnb(8.0);
    fmPert->setCtofnb(9.0);

    fmPert->initialize();

    // fmPert->setLambda(0.0);

    auto ctx = std::make_shared<CharmmContext>(fmPert);

    auto crd = std::make_shared<CharmmCrd>("../test/data/nvt_equil.100k.cor");
    ctx->setCoordinates(crd);

    std::cout << ctx->calculateForces(true, true, true) << "\n";
    std::cout << ctx->calculateForces(true, true, true) << "\n";
    std::cout << ctx->calculateForces(true, true, true) << "\n";
    ctx->assignVelocitiesAtTemperature(0);

    // CudaVelocityVerletIntegrator integrator(0.001);
    // integrator.setSimulationContext(ctx);

    // auto subscriber = std::make_shared<NetCDFSubscriber>("vv_pert.100k.nc",
    // ctx);
    // auto subscriber = std::make_shared<DcdSubscriber>("vv_pert.100k.dcd",
    // ctx); ctx->subscribe(subscriber); auto dualTopologySubscriber =
    //    std::make_shared<DualTopologySubscriber>("vv_pert.100k.txt", ctx);
    // ctx->subscribe(dualTopologySubscriber);

    // integrator.setReportSteps(1);
    // integrator.propagate(10);
  }
}

TEST_CASE("FEP-PI") {
  // Test cases to try and compute FEP eneries with (py)MBAR
  std::string dataPath = getDataPath();
  SECTION("Electrostatic") {
    std::vector<std::string> prmfiles = {dataPath + "toppar_water_ions.str",
                                         dataPath + "par_all36_cgenff.prm"};
    auto psf = std::make_shared<CharmmPSF>(dataPath + "fep/ethanol.solv.psf");
    auto prm = std::make_shared<CharmmParameters>(prmfiles);

    // Base fm, to be modified to prepare alchemical fms
    auto baseFM = std::make_shared<ForceManager>(psf, prm);
    // FMgenerator to generate all alchemically modified FMs
    auto fmgenerator = AlchemicalForceManagerGenerator(baseFM);

    // Composite FM, to drive all alchemicaly modified FMs
    auto fmFEP = std::make_shared<ForceManagerComposite>();
  }
}
*/
TEST_CASE("FEP-EI") {
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

    CudaLangevinThermostatIntegrator equilIntegrator(0.002);
    equilIntegrator.setFriction(5.0);
    equilIntegrator.setBathTemperature(300.0);
    equilIntegrator.setSimulationContext(ctx);

    CudaLangevinThermostatIntegrator integrator(0.002);
    integrator.setFriction(5.0);
    integrator.setBathTemperature(300.0);
    integrator.setSimulationContext(ctx);

    auto fepSub = std::make_shared<FEPSubscriber>("dbexp_fepEI_vdw.out");
    fepSub->setReportFreq(1000);
    integrator.subscribe(fepSub);

    for (auto &lambda : lambdas) {
      std::cout << "Lambda : " << lambda << "\n";
      fmFEP->setLambda(lambda);
      equilIntegrator.propagate(100000);
      integrator.propagate(250000);
    }
  }
}
