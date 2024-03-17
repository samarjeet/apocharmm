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
#include "CompositeSubscriber.h"
#include "CudaLangevinThermostatIntegrator.h"
#include "EDSForceManager.h"
#include "ForceManagerGenerator.h"
#include "NetCDFSubscriber.h"
#include "StateSubscriber.h"
#include "catch.hpp"
#include "test_paths.h"
#include <iostream>
#include <vector>

TEST_CASE("rex", "[energy]") {
  std::string dataPath = getDataPath();

  SECTION("25k") {
    auto psf0 = std::make_shared<CharmmPSF>(dataPath + "l0.pert.25k.psf");
    auto psf1 = std::make_shared<CharmmPSF>(dataPath + "l1.pert.25k.psf");

    std::vector<std::string> prmFiles{dataPath + "toppar_water_ions.str",
                                      dataPath + "par_all36_cgenff.prm"};
    auto prm = std::make_shared<CharmmParameters>(prmFiles);
    auto fm0 = std::make_shared<ForceManager>(psf0, prm);
    auto fm1 = std::make_shared<ForceManager>(psf1, prm);

    double boxLength = 62.79503;
    int fftDim = 64;
    fm0->setBoxDimensions({boxLength, boxLength, boxLength});
    fm0->setFFTGrid(fftDim, fftDim, fftDim);
    fm0->setPmeSplineOrder(4);
    fm0->setKappa(0.34);
    fm0->setCutoff(10.0);
    fm0->setCtonnb(8.0);
    fm0->setCtofnb(9.0);

    fm1->setBoxDimensions({boxLength, boxLength, boxLength});
    fm1->setFFTGrid(fftDim, fftDim, fftDim);
    fm1->setPmeSplineOrder(4);
    fm1->setKappa(0.34);
    fm1->setCutoff(10.0);
    fm1->setCtonnb(8.0);
    fm1->setCtofnb(9.0);

    auto ctx0 = std::make_shared<CharmmContext>(fm0);
    auto ctx1 = std::make_shared<CharmmContext>(fm1);

    auto crd0 = std::make_shared<CharmmCrd>("../test/data/nvt_equil.25k.cor");
    auto crd1 = std::make_shared<CharmmCrd>("../test/data/nvt_equil.25k_1.cor");

    ctx0->setCoordinates(crd0);
    ctx1->setCoordinates(crd1);

    ctx0->assignVelocitiesAtTemperature(300);
    ctx1->assignVelocitiesAtTemperature(300);

    ctx0->calculateForces(false, true, true);
    ctx0->calculatePotentialEnergy(true, true);
    ctx1->calculateForces(false, true, true);
    ctx1->calculatePotentialEnergy(true, true);

    auto pe00 = ctx0->getPotentialEnergy();
    pe00.transferFromDevice();
    std::cout << "Potential energy 0:" << pe00[0] << std::endl;

    auto pe11 = ctx1->getPotentialEnergy();
    pe11.transferFromDevice();
    std::cout << "Potential energy 1:" << pe11[0] << std::endl;

    // Swap the coordinates and calculate the potential energy

    auto temp_crd0 = ctx0->getCoordinates();
    auto temp_crd1 = ctx1->getCoordinates();

    ctx0->setCoordinates(temp_crd1);
    ctx1->setCoordinates(temp_crd0);
    std::cout << "\n\nSwapped" << std::endl;

    // Now calculate the energies
    // ctx0->calculateForces(false, true, true);
    ctx0->calculatePotentialEnergy(true, true);
    // ctx1->calculateForces(false, true, true);
    ctx1->calculatePotentialEnergy(true, true);

    pe00 = ctx0->getPotentialEnergy();
    pe00.transferFromDevice();
    std::cout << "Potential energy 0:" << pe00[0] << std::endl;

    pe11 = ctx1->getPotentialEnergy();
    pe11.transferFromDevice();
    std::cout << "Potential energy 1:" << pe11[0] << std::endl;
  }
}
