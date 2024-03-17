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

TEST_CASE("benzene_pyrrole", "[free energy]") {

  std::string dataPath = getDataPath();

  SECTION("waterbox") {
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

    fm->setLambda(0.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "min.0.crd");
    ctx->setCoordinates(crd);

    std::cout << ctx->calculatePotentialEnergy(true, true);
    // char c;
    // std::cin >> c;
    //  ctx->assignVelocitiesAtTemperature(0.0);
    ctx->assignVelocitiesAtTemperature(300.0);
    /*
        // ctx->calculatePressure();
        auto thermostat = CudaLangevinThermostatIntegrator(0.002);
        thermostat.setBathTemperature(300.0);
        thermostat.setSimulationContext(ctx);
        thermostat.setFriction(12.0);
        thermostat.propagate(10000);

        auto integrator = CudaLangevinPistonIntegrator(0.002);

        integrator.setSimulationContext(ctx);
        integrator.setPistonMass(50.0);
        integrator.setPistonFriction(20.0);

        // integrator.setCrystalType(CRYSTAL::ORTHORHOMBIC);
        // integrator.setCrystalType(CRYSTAL::TETRAGONAL);
        integrator.setCrystalType(CRYSTAL::CUBIC);

        CHECK(ctx->getVolume() == dimX * dimY * dimZ);
        auto refPressure = integrator.getReferencePressure();
        CHECK(refPressure[0] == 1.0);
        CHECK(refPressure[2] == 1.0);
        CHECK(refPressure[5] == 1.0);

        auto particlesDOF = ctx->getDegreesOfFreedom();

        auto degreesOfFreedom = integrator.getPistonDegreesOfFreedom();
        std::cout << "Particle degrees of freedom: " << particlesDOF <<
       std::endl; std::cout << "Piston degrees of freedom: " << degreesOfFreedom
       << std::endl;

        auto dcdSubscriber =
            std::make_shared<DcdSubscriber>("waterbox_lp.dcd", 1000);
        // dcdSubscriber->setCharmmContext(ctx);
        integrator.subscribe(dcdSubscriber);
        */

    CudaLangevinThermostatIntegrator equilIntegrator(0.001);
    equilIntegrator.setFriction(5.0);
    equilIntegrator.setBathTemperature(300.0);
    equilIntegrator.setSimulationContext(ctx);

    CudaLangevinThermostatIntegrator integrator(0.001);
    integrator.setFriction(5.0);
    integrator.setBathTemperature(300.0);
    integrator.setSimulationContext(ctx);

    /*
    auto equilIntegrator = CudaLangevinPistonIntegrator(0.001);
    equilIntegrator.setSimulationContext(ctx);
    equilIntegrator.setPistonMass(204.0);
    equilIntegrator.setPistonFriction(20.0);
    equilIntegrator.setBathTemperature(300.0);
    equilIntegrator.setCrystalType(CRYSTAL::CUBIC);

    auto integrator = CudaLangevinPistonIntegrator(0.001);
    integrator.setSimulationContext(ctx);
    integrator.setPistonMass(204.0);
    integrator.setPistonFriction(20.0);
    integrator.setBathTemperature(300.0);
    integrator.setCrystalType(CRYSTAL::CUBIC);
    */

    auto fepSub = std::make_shared<FEPSubscriber>("vfsw_fepEI_benz_ima.out");
    fepSub->setReportFreq(1000);
    integrator.subscribe(fepSub);

    for (auto &lambda : lambdas) {
      std::cout << "Lambda : " << lambda << "\n";
      fm->setLambda(lambda);
      equilIntegrator.propagate(100000);
      integrator.propagate(400000);
    }

    // integrator.propagate(2000000);
  }
}
