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
#include "CudaLeapFrogIntegrator.h"
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"
#include "DcdSubscriber.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>

TEST_CASE("insulin", "[energy]") {
  std::string dataPath = getDataPath();

  SECTION("monomer") {
    auto psf0 = std::make_shared<CharmmPSF>(dataPath + "4insi.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_prot.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm0 = std::make_shared<ForceManager>(psf0, prm);

    double boxLength = 63.542904;
    int fftDim = 64;
    fm0->setBoxDimensions({boxLength, boxLength, boxLength});
    fm0->setFFTGrid(fftDim, fftDim, fftDim);
    fm0->setPmeSplineOrder(4);
    fm0->setKappa(0.34);
    fm0->setCutoff(12.0);
    fm0->setCtonnb(10.0);
    fm0->setCtofnb(11.0);

    auto ctx = std::make_shared<CharmmContext>(fm0);
    auto crd = std::make_shared<CharmmCrd>(dataPath + "4insi.cor");
    ctx->setCoordinates(crd);

    ctx->calculateForces(false, true, true);
    auto forces = ctx->getForces();

    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);

    auto integrator = std::make_shared<CudaLangevinThermostatIntegrator>(0.002);
    integrator->setFriction(5.0);
    integrator->setBathTemperature(300.0);
    integrator->setSimulationContext(ctx);

    auto subscriber = std::make_shared<DcdSubscriber>("insulin1.dcd");
    integrator->subscribe(subscriber);
    // auto mbarSub = std::make_shared<MBARSubscriber>("mbar.out");
    // mbarSub->setReportFreq(1000);
    // integrator.subscribe(mbarSub);
    integrator->propagate(50);
  }
}
