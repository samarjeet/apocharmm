// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  James E. Gonzales II, Samarjeet Prasad
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CudaLangevinPistonIntegrator.h"
#include "DcdSubscriber.h"
#include "catch.hpp"
#include "compare.h"
#include "test_paths.h"
#include <iostream>

TEST_CASE("langevinPiston", "[dynamics]") {
  const std::string dataPath = getDataPath();
  const std::vector<double> boxDims(3, 50.0);
  const int randomSeed = 314159;
  const double temperature = 300.0;
  const bool useHolonomicConstraints = true;
  const bool useNoseHooverThermostat = true;
  const int nsteps = (useHolonomicConstraints) ? 10000 : 20000;
  const double timeStep = (useHolonomicConstraints) ? 0.002 : 0.001;

  SECTION("waterbox") {
    auto prm =
        std::make_shared<CharmmParameters>(dataPath + "toppar_water_ions.str");
    auto psf = std::make_shared<CharmmPSF>(dataPath + "waterbox.psf");
    auto crd = std::make_shared<CharmmCrd>(dataPath + "waterbox.crd");

    // Setup force manager
    auto fm = std::make_shared<ForceManager>(psf, prm);
    fm->setBoxDimensions(boxDims);

    // Setup CHARMM context
    auto ctx = std::make_shared<CharmmContext>(fm);
    ctx->setCoordinates(crd);
    ctx->setRandomSeedForVelocities(randomSeed);
    ctx->useHolonomicConstraints(useHolonomicConstraints);

    { // NPT Heating & Equilibration
      auto integrator =
          std::make_shared<CudaLangevinPistonIntegrator>(timeStep);
      integrator->useNoseHooverThermostat(true);
      integrator->setCrystalType(CRYSTAL::CUBIC);
      integrator->setLangevinPistonFrictionSeed(randomSeed);
      integrator->setLangevinPistonFriction(20.0);
      integrator->setCharmmContext(ctx);

      const int totNumSteps = (useHolonomicConstraints) ? 50000 : 100000;
      const int ihtfrq = 100;
      const double teminc = 2.0;
      const double firstt = 30.0;
      const double finalt = temperature;
      int step = 0;
      double temp = firstt;
      while (temp <= finalt) {
        ctx->assignVelocitiesAtTemperature(temp);
        integrator->propagate(ihtfrq);
        step += ihtfrq;
        temp += teminc;
      }
      if (step < totNumSteps)
        integrator->propagate(totNumSteps - step);

      std::vector<double> box = ctx->getBoxDimensions();
      CudaContainer<double> averageTemperature =
          integrator->getAverageTemperature();
      CudaContainer<double> averagePressureTensor =
          integrator->getAveragePressureTensor();
      CudaContainer<double> averagePressureScalar =
          integrator->getAveragePressureScalar();

      averageTemperature.transferToHost();
      averagePressureTensor.transferToHost();
      averagePressureScalar.transferToHost();

      std::cout << "==== HEATING ====" << std::endl;
      std::cout << "      Box Dims: " << std::setprecision(12) << box[0]
                << " x " << std::setprecision(12) << box[1] << " x "
                << std::setprecision(12) << box[2] << std::endl;
      std::cout << "   Temperature: " << std::setprecision(12)
                << averageTemperature[0] << " K" << std::endl;
      std::cout << "      Pressure: " << std::setprecision(12)
                << averagePressureScalar[0] << std::endl;
      std::cout << "PressureTensor: " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[0] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[1] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[2] << std::endl;
      std::cout << "                " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[3] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[4] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[5] << std::endl;
      std::cout << "                " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[6] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[7] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[8] << std::endl;

      integrator->resetAverages();
      integrator->setLangevinPistonFriction(0.0);
      integrator->propagate((useHolonomicConstraints) ? 100000 : 200000);

      box = ctx->getBoxDimensions();
      averageTemperature = integrator->getAverageTemperature();
      averagePressureTensor = integrator->getAveragePressureTensor();
      averagePressureScalar = integrator->getAveragePressureScalar();

      averageTemperature.transferToHost();
      averagePressureTensor.transferToHost();
      averagePressureScalar.transferToHost();

      std::cout << "==== EQUILIBRATION ====" << std::endl;
      std::cout << "      Box Dims: " << std::setprecision(12) << box[0]
                << " x " << std::setprecision(12) << box[1] << " x "
                << std::setprecision(12) << box[2] << std::endl;
      std::cout << "   Temperature: " << std::setprecision(12)
                << averageTemperature[0] << " K" << std::endl;
      std::cout << "      Pressure: " << std::setprecision(12)
                << averagePressureScalar[0] << std::endl;
      std::cout << "PressureTensor: " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[0] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[1] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[2] << std::endl;
      std::cout << "                " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[3] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[4] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[5] << std::endl;
      std::cout << "                " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[6] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[7] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[8] << std::endl;
    }

    // Setup integrator
    auto integrator = std::make_shared<CudaLangevinPistonIntegrator>(timeStep);
    integrator->useNoseHooverThermostat(useNoseHooverThermostat);
    integrator->setCrystalType(CRYSTAL::CUBIC);
    integrator->setLangevinPistonFrictionSeed(randomSeed);
    // integrator->setLangevinPistonMass({0.0}); // "Turns off" pressure control
    integrator->setCharmmContext(ctx);

    for (int ITER = 1; ITER <= 20; ITER++) {
      integrator->propagate(nsteps);

      std::vector<double> box = ctx->getBoxDimensions();
      CudaContainer<double> averageTemperature =
          integrator->getAverageTemperature();
      CudaContainer<double> averagePressureTensor =
          integrator->getAveragePressureTensor();
      CudaContainer<double> averagePressureScalar =
          integrator->getAveragePressureScalar();

      averageTemperature.transferToHost();
      averagePressureTensor.transferToHost();
      averagePressureScalar.transferToHost();

      std::cout << "======== ITER " << ITER << " ========" << std::endl;
      std::cout << "      Box Dims: " << std::setprecision(12) << box[0]
                << " x " << std::setprecision(12) << box[1] << " x "
                << std::setprecision(12) << box[2] << std::endl;
      std::cout << "   Temperature: " << std::setprecision(12)
                << averageTemperature[0] << " K" << std::endl;
      std::cout << "      Pressure: " << std::setprecision(12)
                << averagePressureScalar[0] << std::endl;
      std::cout << "PressureTensor: " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[0] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[1] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[2] << std::endl;
      std::cout << "                " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[3] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[4] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[5] << std::endl;
      std::cout << "                " << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[6] << " " << std::setw(20)
                << std::setprecision(12) << averagePressureTensor[7] << " "
                << std::setw(20) << std::setprecision(12)
                << averagePressureTensor[8] << std::endl;
    }
  }
}
