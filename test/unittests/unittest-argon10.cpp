// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#include <iostream>
#include<string>
#include<vector>

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaVelocityVerletIntegrator.h"
#include "CudaVerletIntegrator.h"
#include "CudaLeapFrogIntegrator.h"
#include "DcdSubscriber.h"
#include "catch.hpp"

std::vector<std::vector<float>> readTableFromFile(std::string fileName){
  std::ifstream fin;
  fin.open(fileName);
  std::vector<std::vector<float>> vel;

  while(fin.good()){
    float vx, vy, vz;
    fin >> vx >> vy >> vz;
    vel.push_back({vx, vy, vz});
  }
  fin.close();
  return vel;
}

TEST_CASE("argon10", "[energy]") {
  SECTION("velocity verlet") {
    std::shared_ptr<CharmmParameters> prm =
        std::make_shared<CharmmParameters>("../test/data/argon.prm");
    std::shared_ptr<CharmmPSF> psf =
        std::make_shared<CharmmPSF>("../test/data/argon_10.psf");
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions({50.0, 50.0, 50.0});
    fm->setFFTGrid(48, 48, 48);
    fm->setKappa(0.34);
    fm->setCutoff(10.0);
    fm->setCtonnb(7.0);
    fm->setCtofnb(8.0);
    fm->initialize();

    auto ctx = std::make_shared<CharmmContext>(fm);
    auto crd = std::make_shared<CharmmCrd>("../test/data/argon_10.crd");
    ctx->setCoordinates(crd);

    ctx->calculatePotentialEnergy(true, true);
    ctx->assignVelocitiesAtTemperature(300);
    //auto vel = readTableFromFile("/u/samar/Documents/git/test_gpu/argon/vel_argon_10.txt");
    //ctx->assignVelocities(vel);

    auto integrator = CudaVelocityVerletIntegrator(0.001);
    //auto integrator = CudaVerletIntegrator(0.001);
    integrator.setCharmmContext(ctx);
    //integrator.initialize();
    
    //auto dcdSubscriber = std::make_shared<DcdSubscriber>("vv_argon10.dcd", ctx);
    //ctx->subscribe(dcdSubscriber);
    
    //integrator.setReportSteps(10);
    integrator.propagate(100);
  }

}
