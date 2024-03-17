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
#include "CudaVelocityVerletIntegrator.h"
#include "CharmmCrd.h"
#include "CudaNoseHooverThermostatIntegrator.h"
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "PDB.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "helper.h"
#include <iostream>

TEST_CASE("noseHooverThermostat", "[dynamics]") {

  SECTION("waterbox") {

    auto prm = std::make_shared<CharmmParameters>(
        "../test/data/toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>("../test/data/waterbox.psf");

    auto fm = std::make_shared<ForceManager>(psf, prm);

    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>("../test/data/waterbox.crd");
    ctx->setCoordinates(crd);

    std::cout << ctx->calculatePotentialEnergy(true, true) << std::endl;

    CudaMinimizer min;
    min.setSimulationContext(ctx);
    min.minimize(10);
    std::cout <<"Energy after min : " << ctx->calculatePotentialEnergy(true, true) << std::endl;
    ctx->assignVelocitiesAtTemperature(300);

    ctx->calculateKineticEnergy();
    double kineticEnegy = ctx->getKineticEnergy();
    // std::cout << "Kinetic energy :" << kineticEnegy << std::endl;

    auto velMass = ctx->getVelocityMass();
    velMass.transferFromDevice();
    calculateKineticEnergyTest(velMass.getHostArray(), kineticEnegy);

    auto integrator = CudaNoseHooverThermostatIntegrator(0.001);
    integrator.setSimulationContext(ctx);
    //auto integrator = CudaVelocityVerletIntegrator(0.001);
    //integrator.setSimulationContext(ctx);
    //integrator.propagate(1000);

    /*
    ctx->useHolonomicConstraints(true);

    // ctx->calculatePressure();
    // integrator.initialize();

    auto subscriber =
        std::make_shared<DcdSubscriber>("lang_5_waterbox.dcd", ctx);
    auto stateSubscriber =
        std::make_shared<StateSubscriber>("lang_5_waterbox.txt", ctx);

    ctx->subscribe(subscriber);
    ctx->subscribe(stateSubscriber);

    integrator.setReportSteps(5000);
    integrator.propagate(100000);
    */

    // integrator.propagate(100000);
    // integrator.propagate(20);
  }
}
